"""Disclaimer: Writen with claude code sonnet/opus 5

Training-curve figures from a wandb run (the only place the run data lives).

Two figures:

  1. `<prefix>_episodes.png` - episode length (left axis) and episodic return
     (right axis) over global_step, with the stage switches marked on the x-axis.
  2. `<prefix>_rewards.png`  - reward composition: the share each reward term
     contributes to the episode's total reward mass, at most `--bands` bands
     (largest terms by mean |contribution|, the rest folded into "other").
     Shares, not raw sums, because terms have opposite signs (the time penalty
     is negative) and a signed stack does not read.

Stage switches are read off `charts/stage_id`; they are drawn as unlabeled
markers, the figure never says which stage is which.

History is fetched once and cached as .npz next to the outputs, so re-styling
a plot costs no API call. Delete the cache (or pass --refresh) to re-fetch.

    uv run python evaluation/plot_training.py \
        --run /models-albert-ludwigs-universit-t-freiburg/joshua/runs/2zwe9huo
"""

import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LENGTH_KEY = "charts/episodic_length"
RETURN_KEY = "charts/episodic_return"
STAGE_KEY = "charts/stage_id"
# wandb's `_step` is the global_step the metric was logged at (ppo.py passes
# global_step as the tensorboard step), so no separate x key has to be fetched
X_KEY = "_step"


def _sampled(run, key, samples):
    """(x, y) for one metric, backing off when the API times out.

    One key per request: this run has >100M steps and the multi-key form of the
    sampled-history query times out server-side on it. Sampled rather than
    scan_history because a full scan takes minutes and these curves are
    smoothed anyway - a few thousand points outresolve the figure already.
    """
    while True:
        try:
            rows = run.history(keys=[key], samples=samples, pandas=False)
            break
        except Exception as err:  # CommError and the service-busy wrappers under it
            if samples <= 250:
                raise
            samples //= 2
            print(f"  {key}: {type(err).__name__}, retrying with samples={samples}")
    rows = [r for r in rows if r.get(key) is not None]
    return (
        np.array([r[X_KEY] for r in rows], dtype=np.float64),
        np.array([r[key] for r in rows], dtype=np.float64),
    )


def fetch(run_path, cache, samples, refresh):
    if cache.exists() and not refresh:
        print(f"using cached history {cache}")
        return dict(np.load(cache))

    import wandb

    run = wandb.Api().run(run_path)
    reward_keys = sorted(k for k in run.summary.keys() if k.startswith("rewards/"))
    print(f"fetching {run.name} ({run.state}); reward terms: {len(reward_keys)}")

    data = {}
    for key in [LENGTH_KEY, RETURN_KEY, STAGE_KEY] + reward_keys:
        x, y = _sampled(run, key, samples)
        # each metric keeps its own x: they are logged at different steps
        # (episode ends vs. iteration ends) and are sampled independently
        data[f"{key}/x"], data[key] = x, y
        print(f"  {key}: {len(x)} points")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    print(f"cached history to {cache}")
    return data


def stage_switches(x, stage_id):
    """global_step of each stage change (first sample carrying the new id)."""
    if len(stage_id) == 0:
        return np.zeros(0)
    changed = np.flatnonzero(np.diff(stage_id) != 0) + 1
    return x[changed]


def smooth(y, window):
    """Centered moving average, edges shrunk instead of padded (no flat tails)."""
    if window <= 1 or len(y) <= 2:
        return y
    window = min(window, len(y))
    kernel = np.ones(window)
    return np.convolve(y, kernel, "same") / np.convolve(np.ones_like(y), kernel, "same")


def _mark_switches(ax, switches, color="0.45", on_top=False):
    for s in switches:
        # on_top: over a filled stack, where a background line would be buried
        ax.axvline(s, color=color, ls=(0, (4, 3)), lw=1.2 if on_top else 1.0,
                   zorder=3 if on_top else 0)
    if len(switches):
        # unlabeled ticks on the x-axis itself, so switches stay readable even
        # where a vline is hidden behind a band
        sec = ax.secondary_xaxis("bottom")
        sec.set_xticks(switches, labels=[""] * len(switches))
        sec.tick_params(length=7, width=1.4, color="0.25")
        sec.spines["bottom"].set_visible(False)


def _xscale(ax, x):
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("environment steps (millions)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:g}")


def plot_episodes(data, switches, window, path):
    x_len, length = data[f"{LENGTH_KEY}/x"], smooth(data[LENGTH_KEY], window)
    x_ret, ret = data[f"{RETURN_KEY}/x"], smooth(data[RETURN_KEY], window)

    fig, ax_len = plt.subplots(figsize=(7.5, 4.0))
    c_len, c_ret = "tab:blue", "tab:red"

    ax_len.plot(x_len, length, color=c_len, lw=1.8, label="episode length")
    ax_len.set_ylabel("episode length (steps)", color=c_len)
    ax_len.tick_params(axis="y", labelcolor=c_len)

    ax_ret = ax_len.twinx()
    ax_ret.plot(x_ret, ret, color=c_ret, lw=1.8, label="episodic return")
    ax_ret.set_ylabel("episodic return", color=c_ret)
    ax_ret.tick_params(axis="y", labelcolor=c_ret)

    _mark_switches(ax_len, switches)
    _xscale(ax_len, x_len)
    ax_ret.set_xlim(ax_len.get_xlim())
    ax_len.set_title("Episode length and return (dashed: stage switch)")
    ax_len.set_zorder(ax_ret.get_zorder() + 1)
    ax_len.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def _bin_mean(x, y, edges):
    """Mean of y per step-bin; empty bins are 0 (the term contributed nothing there)."""
    idx = np.digitize(x, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx, y = idx[keep], y[keep]
    counts = np.bincount(idx, minlength=len(edges) - 1)
    sums = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)


def reward_bands(data, n_bands, n_bins, bin_smooth):
    """Per-term share of the total absolute reward mass, over step bins.

    Binned means, not interpolation: each term is logged at its own steps and
    is zero for most logged episodes (a goal fires rarely), so interpolating
    between two sampled points mostly connects zeros. Averaging every sample
    inside a step bin gives the term's mean per-episode contribution there, and
    puts all terms on one grid at the same time. Terms that only exist from a
    later stage on simply have no samples earlier, hence share 0.

    Returns (x, labels, shares) with shares of shape (n_bands, n_bins), rows
    ordered largest-first and the tail folded into a single "other" row.
    """
    keys = sorted(k for k in data if k.startswith("rewards/") and not k.endswith("/x"))
    edges = np.linspace(
        min(data[f"{k}/x"].min() for k in keys),
        max(data[f"{k}/x"].max() for k in keys),
        n_bins + 1,
    )
    x = 0.5 * (edges[:-1] + edges[1:])
    mat = np.stack([_bin_mean(data[f"{k}/x"], data[k], edges) for k in keys])  # (n_terms, n_bins)
    # rare terms (a goal fires in ~1% of episodes) still miss whole bins, which
    # would read as "the term switched off"; averaging over neighbouring bins
    # estimates the local contribution instead
    mat = np.stack([smooth(row, bin_smooth) for row in mat])
    order = np.argsort(-np.abs(mat).mean(axis=1))
    keep = order[: max(1, n_bands - 1)]
    rest = order[max(1, n_bands - 1) :]

    absolute = np.abs(mat)
    total = absolute.sum(axis=0)
    shares = absolute / np.maximum(total, 1e-12)
    # a bin where every term happened to sample only zeros carries no
    # composition; hold the last known one instead of punching a hole in the stack
    known = np.maximum.accumulate(np.where(total > 0, np.arange(len(total)), 0))
    shares = shares[:, known]

    labels = [keys[i].split("/", 1)[1] for i in keep]
    rows = [shares[i] for i in keep]
    if len(rest):
        labels.append("other")
        rows.append(shares[rest].sum(axis=0))
    return x, labels, np.stack(rows)


def plot_rewards(data, switches, n_bands, n_bins, bin_smooth, path):
    x, labels, shares = reward_bands(data, n_bands, n_bins, bin_smooth)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(labels)))
    ax.stackplot(x, shares, labels=labels, colors=colors, edgecolor="none")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of episode reward mass")
    _mark_switches(ax, switches, color="white", on_top=True)
    _xscale(ax, x)
    ax.set_title("Reward composition (dashed: stage switch)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(labels), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="/models-albert-ludwigs-universit-t-freiburg/joshua/runs/2zwe9huo",
                   help="wandb run path (entity/project/runs/id)")
    p.add_argument("--out-prefix", default=str(pathlib.Path(__file__).parent / "results" / "training"))
    p.add_argument("--samples", type=int, default=10000, help="history points to request per metric")
    p.add_argument("--smooth", type=int, default=80, help="moving-average window in samples")
    p.add_argument("--bands", type=int, default=4, help="max reward bands (last one is 'other')")
    p.add_argument("--reward-bins", type=int, default=40, help="step bins the reward shares average over")
    p.add_argument("--reward-smooth", type=int, default=3, help="moving-average window in reward bins")
    p.add_argument("--refresh", action="store_true", help="re-fetch even if the cache exists")
    args = p.parse_args()

    prefix = pathlib.Path(args.out_prefix)
    data = fetch(args.run, prefix.with_name(prefix.name + "_history.npz"),
                 args.samples, args.refresh)
    switches = stage_switches(data[f"{STAGE_KEY}/x"], data[STAGE_KEY])
    print(f"stage switches at steps: {[f'{s / 1e6:.2f}M' for s in switches]}")

    plot_episodes(data, switches, args.smooth, f"{prefix}_episodes.png")
    plot_rewards(data, switches, args.bands, args.reward_bins, args.reward_smooth,
                 f"{prefix}_rewards.png")


if __name__ == "__main__":
    main()
