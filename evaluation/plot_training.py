"""Disclaimer: Writen with claude code sonnet/opus 5

Training-curve figures from a wandb run (the only place the run data lives).

Two figures:

  1. `<prefix>_episodes` - episode length (left axis) and episodic return
     (right axis) over global_step, with the stage switches marked on the x-axis.
  2. `<prefix>_rewards`  - reward composition: the share each reward term
     contributes to the episode's total reward mass, at most `--bands` bands
     (largest terms by mean |contribution|, the rest folded into "other").
     Shares, not raw sums, because terms have opposite signs (the time penalty
     is negative) and a signed stack does not read.

Each is written twice: `<name>.png` on white and `<name>_transparent.png` with
no background, for slides/posters that are not white.

Stage switches are read off `charts/stage_id`; they are drawn as unlabeled
markers, the figure never says which stage is which.

History is fetched once and cached as .npz next to the outputs, so re-styling
a plot costs no API call. Delete the cache (or pass --refresh) to re-fetch.

A refresh reads the run's `run-<id>-history` artifact (the parquet shards the
history actually lives in), not the sampledHistory API: that endpoint scans the
whole history per request and stops returning at all once a run gets long (this
one is 2GB / ~130M rows over 132M steps, and a single-key query ran 600s without
answering). Shards are pulled one at a time and deleted after reading, so peak
disk is one shard rather than the full artifact - and wandb's own artifact cache
makes a second refresh take seconds.

    uv run python evaluation/plot_training.py \
        --run /models-albert-ludwigs-universit-t-freiburg/joshua/runs/laafgmd0
"""

import argparse
import pathlib
import shutil
import time

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
# every logged scalar group. media/video is a dict column and the _-prefixed
# ones are wandb bookkeeping, so neither is a metric; the rest are cached even
# when these two figures do not draw them, because a refresh is the expensive part
PREFIXES = ("charts/", "rewards/", "losses/", "perf/")


def _history_artifact(api, run_path):
    """The `run-<id>-history` artifact for a run path (entity/project/runs/id)."""
    parts = [p for p in run_path.split("/") if p and p != "runs"]
    entity, project, run_id = parts[-3], parts[-2], parts[-1]
    return api.artifact(f"{entity}/{project}/run-{run_id}-history:latest")


def _subsample(x, y, samples):
    """Sort by step and thin to `samples` points, uniformly over the sorted rows."""
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x) <= samples:
        return x, y
    # uniform stride, not the first N: reward_bands re-bins by step anyway, and
    # the curves are smoothed, so an even spread over the run is what matters
    idx = np.linspace(0, len(x) - 1, samples).round().astype(int)
    return x[idx], y[idx]


def fetch(run_path, cache, samples, refresh):
    if cache.exists() and not refresh:
        print(f"using cached history {cache}")
        return dict(np.load(cache))

    import pyarrow.parquet as pq
    import wandb

    api = wandb.Api(timeout=300)
    art = _history_artifact(api, run_path)
    print(f"fetching {art.name} ({art.size / 1e6:.0f}MB, {len(art.manifest.entries)} shards)")

    work = cache.parent / "_history_shards"
    acc = {}  # key -> (list of x chunks, list of y chunks)
    for name in sorted(art.manifest.entries):
        t = time.time()
        path = art.get_entry(name).download(root=str(work))
        pf = pq.ParquetFile(path)
        # per shard, not once: the shards do not all carry the same columns
        # (a reward term that starts in a later stage is absent from the early ones)
        have = sorted(n for n in pf.schema_arrow.names if n.startswith(PREFIXES))
        acc.update({k: ([], []) for k in have if k not in acc})
        # batched: the widest shard here is 49M rows x 32 metrics, ~12GB as one
        # float64 table. Rows are near-empty (each scalar is logged as its own
        # row), so only the non-null values survive a batch.
        for batch in pf.iter_batches(batch_size=2_000_000, columns=[X_KEY] + have):
            step = batch[X_KEY].to_numpy(zero_copy_only=False).astype(np.float64)
            for key in have:
                y = batch[key].to_numpy(zero_copy_only=False).astype(np.float64)
                mask = ~np.isnan(y)
                if not mask.any():
                    continue
                acc[key][0].append(step[mask])
                acc[key][1].append(y[mask])
        pathlib.Path(path).unlink()  # wandb keeps its own cached copy for next time
        print(f"  {name}: {pf.metadata.num_rows} rows, {time.time() - t:.1f}s")

    data = {}
    for key, (xs, ys) in sorted(acc.items()):
        if not xs:
            continue
        # each metric keeps its own x: they are logged at different steps
        # (episode ends vs. iteration ends) and are thinned independently
        x, y = _subsample(np.concatenate(xs), np.concatenate(ys), samples)
        data[f"{key}/x"], data[key] = x, y
        print(f"  {key}: {len(x)} points, steps {x.min():.0f}..{x.max():.0f}")

    shutil.rmtree(work, ignore_errors=True)
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


# both figures are meant to be stacked, so they share a figure size, a data
# x-range and a fixed axes rectangle - tight_layout would size the axes box off
# each figure's own decorations (the twin y-axis on one, the legend on the
# other) and the two x-axes would come out different widths
FIGSIZE = (7.5, 4.0)
AXES_BOX = dict(left=0.11, right=0.89)


def _save(fig, path):
    """Write both variants: `<path>.png` on white, `<path>_transparent.png` without."""
    fig.savefig(f"{path}.png", dpi=200)
    fig.savefig(f"{path}_transparent.png", dpi=200, transparent=True)
    print(f"saved {path}.png and {path}_transparent.png")


def _xscale(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_xlabel("environment steps (millions)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:g}")


def plot_episodes(data, switches, window, xlim, path):
    x_len, length = data[f"{LENGTH_KEY}/x"], smooth(data[LENGTH_KEY], window)
    x_ret, ret = data[f"{RETURN_KEY}/x"], smooth(data[RETURN_KEY], window)

    fig, ax_len = plt.subplots(figsize=FIGSIZE)
    c_len, c_ret = "tab:blue", "tab:red"

    ax_len.plot(x_len, length, color=c_len, lw=1.8, label="episode length")
    ax_len.set_ylabel("episode length (steps)", color=c_len)
    ax_len.tick_params(axis="y", labelcolor=c_len)

    ax_ret = ax_len.twinx()
    ax_ret.plot(x_ret, ret, color=c_ret, lw=1.8, label="episodic return")
    ax_ret.set_ylabel("episodic return", color=c_ret)
    ax_ret.tick_params(axis="y", labelcolor=c_ret)

    _mark_switches(ax_len, switches)
    _xscale(ax_len, xlim)
    ax_ret.set_xlim(*xlim)
    ax_len.set_title("Episode length and return (dashed: stage switch)")
    ax_len.set_zorder(ax_ret.get_zorder() + 1)
    ax_len.patch.set_visible(False)
    fig.subplots_adjust(top=0.90, bottom=0.155, **AXES_BOX)
    _save(fig, path)


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


def plot_rewards(data, switches, n_bands, n_bins, bin_smooth, xlim, path):
    x, labels, shares = reward_bands(data, n_bands, n_bins, bin_smooth)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(labels)))
    # bin centres sit half a bin inside the data range: extend the outer
    # samples to the edges so the stack fills the shared x-range
    x = np.concatenate([[xlim[0]], x, [xlim[1]]])
    shares = np.column_stack([shares[:, 0], shares, shares[:, -1]])
    ax.stackplot(x, shares, labels=labels, colors=colors, edgecolor="none")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of episode reward mass")
    _mark_switches(ax, switches, color="white", on_top=True)
    _xscale(ax, xlim)
    ax.set_title("Reward composition (dashed: stage switch)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(labels), frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.28, **AXES_BOX)
    _save(fig, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="/models-albert-ludwigs-universit-t-freiburg/joshua/runs/2zwe9huo",
                   help="wandb run path (entity/project/runs/id)")
    p.add_argument("--out-prefix", default=str(pathlib.Path(__file__).parent / "results" / "training"))
    p.add_argument("--samples", type=int, default=10000, help="history points kept per metric")
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

    # one x-range for both figures, from the full training run
    xlim = (min(data[f"{k}/x"].min() for k in (LENGTH_KEY, RETURN_KEY)),
            max(data[f"{k}/x"].max() for k in (LENGTH_KEY, RETURN_KEY)))

    plot_episodes(data, switches, args.smooth, xlim, f"{prefix}_episodes")
    plot_rewards(data, switches, args.bands, args.reward_bins, args.reward_smooth, xlim,
                 f"{prefix}_rewards")


if __name__ == "__main__":
    main()
