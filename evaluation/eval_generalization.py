"""Disclaimer: Writen by claude opus 4.8

Zero-shot team-size generalization of a trained transformer policy.

The headline claim: one policy, trained at a single team size, runs *unchanged*
at team sizes it never saw. The backbone is per-entity-type (no positional
encoding, no per-slot weights), so `agent.set_env` just re-points it at a
differently-sized env - same weights, no fine-tuning.

For each requested team size this rolls the policy out for a fixed number of
episodes and reports:

  - goal rate  = fraction of episodes ended by a goal (termination; the only
                 non-truncation end is `_reward_goal > 0`)
  - mean return= mean per-episode summed reward (env's shaped team reward)
  - mean length= mean steps per episode

Run from anywhere; it adds ../src to sys.path for the top-level env/agent modules.

    uv run python eval_generalization.py \
        --checkpoint ../models/config_stage2_2vs2_v1.cleanrl_model \
        --sizes 1 2 3 --n-yellow 0 --episodes 100
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import myenvs  # noqa: F401,E402  (registers SSLDynamicRobots-v0)
from agent import Agent  # noqa: E402

FULL_FIELD = {"min": (-0.8, -0.8), "max": (0.8, 0.8)}


def build_envs(env_id, n_blue, n_yellow, opponent_strategy, num_envs, seed):
    """SyncVectorEnv of unflattened envs at a given team size (per-entity obs)."""
    env_args = dict(
        n_robots_blue=n_blue,
        n_robots_yellow=n_yellow,
        allowed_positions_blue=FULL_FIELD,
        allowed_positions_yellow=FULL_FIELD,
        allowed_positions_ball=FULL_FIELD,
        opponent_strategy=opponent_strategy if n_yellow > 0 else None,
    )

    def thunk():
        env = gym.make(env_id, **env_args)
        env = gym.wrappers.ClipAction(env)
        return env

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(num_envs)])
    envs.reset(seed=seed)
    return envs


def build_agent(envs, ckpt, device, d_model, n_layers, n_heads, ff_dim, critic_pooling, rpo_alpha):
    """Construct the transformer agent (weights are per-entity-type, so any team
    size works to build it) and load the checkpoint."""
    agent = Agent(
        envs, rpo_alpha, agent_type="transformer", d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim, dropout=0.0, critic_pooling=critic_pooling,
    ).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    return agent


@torch.no_grad()
def run_size(agent, envs, device, episodes, max_steps):
    """Roll out until `episodes` episodes finish; return goal-rate, mean return, mean length.

    Episode accounting rides SyncVectorEnv autoreset: on a done step the returned
    obs is already the reset obs, so we only bank the running totals and zero them.

    The base rSoccer env reports both a goal and a timeout as `terminated` (its
    step returns truncated=False always), so we can't read goal vs timeout off the
    done flag. But the env's timeout check has priority over the goal check, so a
    goal never lands on step==max_steps: goal <=> episode ended before max_steps.
    """
    num_envs = envs.num_envs
    ep_ret = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    returns, lengths, goals = [], [], []

    obs, _ = envs.reset()
    while len(returns) < episodes:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
        action, *_ = agent.get_action_and_value(obs_t)
        obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())

        ep_ret += np.asarray(reward, dtype=np.float64)
        ep_len += 1
        done = np.asarray(term) | np.asarray(trunc)
        for i in np.nonzero(done)[0]:
            returns.append(ep_ret[i])
            lengths.append(ep_len[i])
            goals.append(bool(ep_len[i] < max_steps))  # goal-ended, not a timeout
            ep_ret[i] = 0.0
            ep_len[i] = 0

    n = len(returns)
    return {
        "episodes": n,
        "goal_rate": float(np.mean(goals[:n])),
        "mean_return": float(np.mean(returns[:n])),
        "std_return": float(np.std(returns[:n])),
        "mean_length": float(np.mean(lengths[:n])),
    }


def report(rows, trained_size, out_prefix=None):
    print("\n=== Zero-shot team-size generalization ===")
    if trained_size is not None:
        print(f"(policy trained at size {trained_size}; * marks the trained size)")
    print(f"  {'blue':>4} {'episodes':>9} {'goal_rate':>10} {'mean_return':>12} {'mean_len':>9}")
    for r in rows:
        star = " *" if r["n_blue"] == trained_size else "  "
        print(
            f"  {r['n_blue']:>4} {r['episodes']:>9} {r['goal_rate']:>10.3f}"
            f" {r['mean_return']:>12.2f} {r['mean_length']:>9.1f}{star}"
        )

    if out_prefix:
        sizes = np.array([r["n_blue"] for r in rows])
        np.savez(
            f"{out_prefix}.npz",
            n_blue=sizes,
            goal_rate=np.array([r["goal_rate"] for r in rows]),
            mean_return=np.array([r["mean_return"] for r in rows]),
            std_return=np.array([r["std_return"] for r in rows]),
            mean_length=np.array([r["mean_length"] for r in rows]),
            trained_size=np.array([-1 if trained_size is None else trained_size]),
        )
        print(f"\nsaved raw arrays to {out_prefix}.npz")
        _try_plot(rows, trained_size, f"{out_prefix}.png")


def _try_plot(rows, trained_size, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    sizes = [r["n_blue"] for r in rows]
    goal = [r["goal_rate"] for r in rows]
    ret = [r["mean_return"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    c1 = "tab:blue"
    ax1.plot(sizes, goal, "o-", color=c1)
    ax1.set_xlabel("team size (n blue robots)")
    ax1.set_ylabel("goal rate", color=c1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(sizes)
    ax2 = ax1.twinx()
    c2 = "tab:red"
    ax2.plot(sizes, ret, "s--", color=c2)
    ax2.set_ylabel("mean return", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    if trained_size in sizes:
        ax1.axvline(trained_size, color="gray", ls=":", lw=1)
    ax1.set_title("Zero-shot generalization across team sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved plot to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="SSLDynamicRobots-v0")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3],
                   help="team sizes (n blue robots) to evaluate")
    p.add_argument("--trained-size", type=int, default=None,
                   help="team size the checkpoint was trained at (annotates output only)")
    p.add_argument("--n-yellow", type=int, default=0, help="opponents per size (fixed across sizes)")
    p.add_argument("--opponent-strategy", default="Uhlstein")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--episodes", type=int, default=100, help="episodes to average per size")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-prefix", default=None, help="save .npz/.png under this path prefix")
    # architecture (must match the trained checkpoint; defaults mirror config.py)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ff-dim", type=int, default=512)
    p.add_argument("--critic-pooling", default="attention", choices=["mean", "max", "attention"])
    p.add_argument("--rpo-alpha", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build the agent once (its weights don't depend on team size); the first
    # env just seeds the layout, then set_env re-points it per size
    envs = build_envs(args.env_id, args.sizes[0], args.n_yellow, args.opponent_strategy,
                      args.num_envs, args.seed)
    agent = build_agent(
        envs, args.checkpoint, device, args.d_model, args.n_layers, args.n_heads,
        args.ff_dim, args.critic_pooling, args.rpo_alpha,
    )

    rows = []
    for size in args.sizes:
        if size != args.sizes[0]:
            envs.close()
            envs = build_envs(args.env_id, size, args.n_yellow, args.opponent_strategy,
                              args.num_envs, args.seed)
        agent.set_env(envs)
        (max_steps,) = set(envs.get_attr("max_steps"))
        stats = run_size(agent, envs, device, args.episodes, max_steps)
        stats["n_blue"] = size
        rows.append(stats)
        print(f"size {size}: {stats['episodes']} eps, goal_rate {stats['goal_rate']:.3f}, "
              f"return {stats['mean_return']:.2f}")
    envs.close()

    report(rows, args.trained_size, args.out_prefix)


if __name__ == "__main__":
    main()
