"""Disclaimer: Writen with claude code sonnet/opus 5

Per-entity saliency for a trained transformer policy.

Answers "which entity (ball / self / teammates / opponents) drives an output":
gradient of an output w.r.t. each input token, reduced to one scalar per entity
by taking the L2 norm over that token's real feature dims (count features
excluded). Two outputs are attributed:

  - critic value V(s)                -> one importance vector over entities
  - each acting robot's action mean  -> one row per robot (the N x entity matrix)

Sensitivity-based, not attention weights: it reflects how much the actual output
moves when an entity's features change.

    uv run python evaluation/eval_saliency.py --checkpoint models/v1_2-2zwe9huo.cleanrl_model \
        --n-blue 2 --n-yellow 0 --rollout-steps 200

Run from anywhere; it adds ../src to sys.path for the top-level env/agent modules.
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
    """SyncVectorEnv of unflattened envs (transformer keeps per-entity obs).

    A vector env is needed because Agent reads team sizes / feature widths via
    envs.get_attr(...) when it derives its TokenLayout.
    """
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
    agent = Agent(
        envs, rpo_alpha, agent_type="transformer", d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim, dropout=0.0, critic_pooling=critic_pooling,
    ).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    agent.eval()
    return agent


def _leaf_tokens(agent, obs):
    """Tokenize a flat obs batch and return the three token groups as grad-tracking leaves."""
    ball, teammates, opponents = agent._tokenize(obs)
    return [t.detach().requires_grad_(True) for t in (ball, teammates, opponents)]


def _entity_norms(grad, real_dim):
    """Per-entity L2 norm of a token gradient (B, n, feat+2), over real feature
    dims only (the two appended count features are constant, not entity state)."""
    if grad.shape[1] == 0:
        return grad.new_zeros(grad.shape[0], 0)
    return grad[..., :real_dim].norm(dim=-1)  # (B, n_entity)


def critic_saliency(agent, obs):
    """Importance of each entity for V(s), summed-batch backward.

    Returns dict of (B, n_entity) tensors keyed ball/teammates/opponents.
    """
    ball, teammates, opponents = _leaf_tokens(agent, obs)
    value = agent.critic(ball, teammates, opponents)  # (B, 1)
    g_ball, g_tm, g_opp = torch.autograd.grad(value.sum(), [ball, teammates, opponents])
    lay = agent.layout
    return {
        "ball": _entity_norms(g_ball, lay.ball_dim),
        "teammates": _entity_norms(g_tm, lay.teammate_dim),
        "opponents": _entity_norms(g_opp, lay.opponent_dim),
    }


def actor_saliency(agent, obs):
    """N x entity sensitivity: for each acting robot i, how much its action mean
    depends on the ball / itself / other teammates / opponents.

    Returns (B, n_tm, n_tm, ...) collapsed to type columns [ball, self, teammates_other, opponents]
    as a (B, n_tm, 4) tensor.
    """
    lay = agent.layout
    n_tm = lay.n_teammates
    ball, teammates, opponents = _leaf_tokens(agent, obs)
    # tanh, like get_action_and_value: attribute the action actually emitted,
    # not the unbounded pre-squash head output
    action_mean = torch.tanh(agent.actor(ball, teammates, opponents))  # (B, n_tm, act_dim)
    B = action_mean.shape[0]
    cols = action_mean.new_zeros(B, n_tm, 4)  # ball, self, other-teammates, opponents

    for i in range(n_tm):
        # scalarize robot i's action mean (sum of squares over act dims + batch)
        s_i = action_mean[:, i, :].pow(2).sum()
        g_ball, g_tm, g_opp = torch.autograd.grad(
            s_i, [ball, teammates, opponents], retain_graph=(i < n_tm - 1)
        )
        tm_norms = _entity_norms(g_tm, lay.teammate_dim)  # (B, n_tm)
        opp_norms = _entity_norms(g_opp, lay.opponent_dim)  # (B, n_opp)
        cols[:, i, 0] = _entity_norms(g_ball, lay.ball_dim)[:, 0]
        cols[:, i, 1] = tm_norms[:, i]  # own token
        if n_tm > 1:
            other = tm_norms.sum(dim=1) - tm_norms[:, i]
            cols[:, i, 2] = other / (n_tm - 1)
        if opp_norms.shape[1] > 0:
            cols[:, i, 3] = opp_norms.mean(dim=1)
    return cols  # (B, n_tm, 4)


@torch.no_grad()
def _policy_action(agent, obs_t):
    # greedy mean action: saliency should be measured on the states the
    # evaluated (deterministic) policy actually visits, same as simulate.py
    action, *_ = agent.get_action_and_value(obs_t, deterministic=True)
    return action


def collect(agent, envs, device, rollout_steps):
    """Roll the policy out and accumulate critic + actor saliency over visited states."""
    crit_acc = {"ball": [], "teammates": [], "opponents": []}
    act_acc = []

    obs, _ = envs.reset()
    for _ in range(max(1, rollout_steps)):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)

        crit = critic_saliency(agent, obs_t)
        for k, v in crit.items():
            crit_acc[k].append(v.detach())
        act_acc.append(actor_saliency(agent, obs_t).detach())

        action = _policy_action(agent, obs_t).cpu().numpy()
        obs, _, term, trunc, _ = envs.step(action)

    crit_mean = {
        k: (torch.cat(v, dim=0).mean(dim=0) if v[0].shape[1] > 0 else torch.zeros(0))
        for k, v in crit_acc.items()
    }
    act_mean = torch.cat(act_acc, dim=0).mean(dim=0)  # (n_tm, 4)
    return crit_mean, act_mean


def _norm_rows(mat):
    s = mat.sum(dim=-1, keepdim=True)
    return mat / s.clamp_min(1e-8)


def report(crit_mean, act_mean, out_prefix=None):
    has_opp = crit_mean["opponents"].numel() > 0
    ball = crit_mean["ball"].mean().item() if crit_mean["ball"].numel() else 0.0
    tm = crit_mean["teammates"].mean().item() if crit_mean["teammates"].numel() else 0.0
    opp = crit_mean["opponents"].mean().item() if has_opp else 0.0
    total = max(ball + tm + opp, 1e-8)
    print("\n=== Critic value V(s): entity importance (share) ===")
    print(f"  ball       {ball:8.4f}  ({100 * ball / total:5.1f}%)")
    print(f"  teammates  {tm:8.4f}  ({100 * tm / total:5.1f}%)")
    if has_opp:
        print(f"  opponents  {opp:8.4f}  ({100 * opp / total:5.1f}%)")

    cols = ["ball", "self", "teammate", "opponents"][: 4 if has_opp else 3]
    print("\n=== Actor action: per-robot entity sensitivity (row-normalized) ===")
    print("  robot | " + " ".join(f"{c:>16}" for c in cols))
    norm = _norm_rows(act_mean[:, : len(cols)])
    for i in range(act_mean.shape[0]):
        print(f"  {i:>5} | " + " ".join(f"{norm[i, j].item():16.3f}" for j in range(len(cols))))

    if out_prefix:
        np.savez(
            f"{out_prefix}.npz",
            critic_ball=crit_mean["ball"].numpy(),
            critic_teammates=crit_mean["teammates"].numpy(),
            critic_opponents=crit_mean["opponents"].numpy(),
            actor_matrix=act_mean.numpy(),
            actor_columns=np.array(cols),
        )
        print(f"\nsaved raw arrays to {out_prefix}.npz")
        _try_plot(norm.numpy(), cols, f"{out_prefix}.png")
        _try_plot(norm.numpy(), cols, f"{out_prefix}_transparent.png", transparent=True)


def _try_plot(actor_norm, cols, path, transparent=False):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping heatmap")
        return
    vmax = float(actor_norm.max())
    fig, ax = plt.subplots(figsize=(6, 1 + 0.5 * actor_norm.shape[0]))
    im = ax.imshow(actor_norm, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    for i in range(actor_norm.shape[0]):
        for j in range(len(cols)):
            v = actor_norm[i, j]
            ax.text(j + 0.45, i + 0.42, f"{v:.2f}", ha="right", va="bottom",
                    color="white" if v < 0.6 * vmax else "black")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(actor_norm.shape[0]), [f"robot {i}" for i in range(actor_norm.shape[0])])
    ax.set_title("Share of each robot's action sensitivity")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, transparent=transparent)
    print(f"saved heatmap to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-id", default="SSLDynamicRobots-v0")
    p.add_argument("--n-blue", type=int, default=2)
    p.add_argument("--n-yellow", type=int, default=0)
    p.add_argument("--opponent-strategy", default="Uhlstein")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=200)
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

    envs = build_envs(args.env_id, args.n_blue, args.n_yellow, args.opponent_strategy, args.num_envs, args.seed)
    agent = build_agent(
        envs, args.checkpoint, device, args.d_model, args.n_layers, args.n_heads,
        args.ff_dim, args.critic_pooling, args.rpo_alpha,
    )
    crit_mean, act_mean = collect(agent, envs, device, args.rollout_steps)
    report({k: v.cpu() for k, v in crit_mean.items()}, act_mean.cpu(), args.out_prefix)
    envs.close()


if __name__ == "__main__":
    main()
