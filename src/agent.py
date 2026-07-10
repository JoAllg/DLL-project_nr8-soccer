import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import models

# max team size (real soccer); normalizes the count features in _tokenize
N_MAX = 11.0


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        rpo_alpha,
        agent_type="mlp",
        d_model=64,
        n_layers=2,
        n_heads=4,
        ff_dim=256,
        dropout=0.0,
        critic_pooling="mean",
    ):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.agent_type = agent_type
        self.action_shape = envs.single_action_space.shape
        if agent_type == "mlp":
            obs_dim = int(np.array(envs.single_observation_space.shape).prod())
            act_dim_total = int(np.prod(self.action_shape))
            self.critic = models.MLP_critic(obs_dim)
            self.actor_mean = models.MLP_actor(obs_dim, act_dim_total)
        elif agent_type == "transformer":
            layout, act_dim_per_robot = self._derive_layout(envs)
            # two separately-trained transformers, same backbone architecture
            # (why: see TransformerCritic docstring). Actor and critic see the
            # same full state — no privileged critic, since no hidden/sensor
            # information exists here to grant, and identical scope means no
            # input change when self-play starts.
            self.actor = models.TransformerActor(
                layout, act_dim_per_robot, d_model, n_layers, n_heads, ff_dim, dropout
            )
            self.critic = models.TransformerCritic(
                layout, d_model, n_layers, n_heads, ff_dim, dropout, pooling=critic_pooling
            )
            self._apply_layout(envs, layout)
        else:
            raise ValueError(f"unknown agent_type '{agent_type}'")
        # diagonal Gaussian with a learned global (state-independent) logstd,
        # as in the proven base RPO code — simplest parameterization.
        # transformer: one logstd per per-robot action dim, shared across robots, so the
        # parameter shape stays independent of the team size (checkpoint transfer)
        logstd_dim = act_dim_per_robot if agent_type == "transformer" else act_dim_total
        self.actor_logstd = nn.Parameter(torch.zeros(1, logstd_dim))

    def _derive_layout(self, envs):
        """Read a TokenLayout + per-robot action width off `envs`, self-consistency-checked.

        Shared by __init__ (first-time sizing) and set_env (re-pointing) so the
        derivation/validation logic only lives in one place.
        """
        layout = models.token_layout_from_env(envs)
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim_total = int(np.prod(envs.single_action_space.shape))
        assert layout.obs_dim == obs_dim, (
            f"derived token layout expects obs dim {layout.obs_dim}, env has {obs_dim}"
        )
        assert act_dim_total % layout.n_teammates == 0, (
            f"action dim {act_dim_total} is not divisible by {layout.n_teammates} teammates"
        )
        return layout, act_dim_total // layout.n_teammates

    def _apply_layout(self, envs, layout):
        """Point this agent's layout-dependent bookkeeping (not weights) at envs/layout."""
        self.layout = layout
        self.action_shape = envs.single_action_space.shape
        self.actor.n_teammates = layout.n_teammates
        # static scaling to ~[-1, 1] (permutation-safe, unlike NormalizeObservation);
        # near-redundant for envs that already normalize and declare matching bounds
        # (harmless ÷1 in that case), but does the real work for envs returning raw
        # units — requires the declared Box bounds to match the returned values.
        # non-persistent: derived from the env, and its shape depends on the team size
        high = np.asarray(envs.single_observation_space.high, dtype=np.float32).reshape(-1)
        scale = np.where(np.isfinite(high) & (high > 0), high, 1.0)
        device = self.obs_scale.device if hasattr(self, "obs_scale") else "cpu"
        self.register_buffer("obs_scale", torch.from_numpy(scale).to(device), persistent=False)

    def set_env(self, envs):
        """Repoint an already-built transformer agent at a differently-sized env.

        Team/opponent *counts* are free to change — the backbone is per-entity-type,
        not per-slot, so ball_embed/teammate_embed/opponent_embed/encoder/heads keep
        their weights unchanged; only the layout bookkeeping used to tokenize obs and
        reshape actions needs updating. Per-entity feature widths (ball_dim/
        teammate_dim/opponent_dim) and the per-robot action width, by contrast, are
        fixed project conventions baked into those weight shapes — a mismatch means a
        new Agent (or at best a partial load_state_dict), not a live swap.
        """
        assert self.agent_type == "transformer", "set_env only applies to the transformer agent"
        layout, act_dim_per_robot = self._derive_layout(envs)
        for name, old, new in (
            ("per-robot action dim", self.actor_logstd.shape[-1], act_dim_per_robot),
            ("ball_dim", self.layout.ball_dim, layout.ball_dim),
            ("teammate_dim", self.layout.teammate_dim, layout.teammate_dim),
            ("opponent_dim", self.layout.opponent_dim, layout.opponent_dim),
        ):
            assert old == new, (
                f"new env's {name} ({new}) doesn't match this agent's ({old}) - that "
                "changes weight shapes, so it needs a new Agent, not set_env"
            )
        self._apply_layout(envs, layout)

    def _tokenize(self, x):
        """Slice a flat obs batch into (ball, teammates, opponents) token groups,
        each scaled and with the [team_size, opp_size] count features appended."""
        layout = self.layout
        obs = x / self.obs_scale
        batch_size = obs.shape[0]
        # the flat obs is laid out as [ball | teammates... | opponents...];
        # ball_end/teammates_end mark the segment boundaries to slice out of it
        ball_end = layout.ball_dim
        teammates_end = ball_end + layout.n_teammates * layout.teammate_dim
        ball = obs[:, :ball_end].reshape(batch_size, 1, layout.ball_dim)
        teammates = obs[:, ball_end:teammates_end].reshape(batch_size, layout.n_teammates, layout.teammate_dim)
        opponents = obs[:, teammates_end:].reshape(batch_size, layout.n_opponents, layout.opponent_dim)
        # team-size features - the only size signal (no padding/masking).
        # n/N_MAX keeps them in (0, 1] for every team size; raw counts would
        # push embeddings out of distribution at sizes beyond those trained
        # (see wiki/team-size-generalization.md)
        count_features = obs.new_tensor(
            [layout.n_teammates / N_MAX, layout.n_opponents / N_MAX]
        )

        def with_counts(tokens):
            return torch.cat([tokens, count_features.expand(batch_size, tokens.shape[1], 2)], dim=-1)

        return with_counts(ball), with_counts(teammates), with_counts(opponents)

    def get_value(self, x):
        if self.agent_type == "transformer":
            return self.critic(*self._tokenize(x))
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.agent_type == "transformer":
            tokens = self._tokenize(x)
            # (B, n_teammates, act_dim_per_robot) -> flat (B, act_dim_total) for the Gaussian
            action_mean = self.actor(*tokens).reshape(x.shape[0], -1)
            value = self.critic(*tokens)
            # tile the shared per-robot logstd across teammates (flat, robot-major)
            action_logstd = self.actor_logstd.repeat(1, self.layout.n_teammates).expand_as(action_mean)
        else:
            action_mean = self.actor_mean(x)
            value = self.critic(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            action = action.reshape(x.shape[0], -1)
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(x.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        # joint log-prob/entropy: sum over action dims AND teammates →
        # one scalar per env. The whole team is a single PPO "agent", and the
        # joint log-prob factorizes into this sum because robots are
        # conditionally independent given the shared encoding.
        logprob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        action = action.reshape((x.shape[0],) + self.action_shape)
        return action, logprob, entropy, value
