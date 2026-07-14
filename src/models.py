from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

### MLP Model

# CleanRL/PPO convention: orthogonal init preserves activation/gradient norms through the layer
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP_critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.net(x)


class MLP_actor(nn.Module):
    def __init__(self, obs_dim, act_dim_total):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim_total), std=0.01),
        )

    def forward(self, x):
        return self.net(x)


### Transformer model

@dataclass(frozen=True)
class TokenLayout:
    """How a flat obs vector (ball, teammates, opponents) splits into per-entity tokens.

    Weight shapes derived from a layout are independent of entity counts, so
    any team sizes can be used.
    """

    ball_dim: int
    n_teammates: int
    teammate_dim: int
    n_opponents: int
    opponent_dim: int

    @property
    def obs_dim(self) -> int:
        return (
            self.ball_dim
            + self.n_teammates * self.teammate_dim
            + self.n_opponents * self.opponent_dim
        )

    @property
    def num_tokens(self) -> int:
        return 1 + self.n_teammates + self.n_opponents


def token_layout_from_env(envs) -> TokenLayout:
    """Build a TokenLayout from a vector env's live attributes.

    Env class needs to declares BALL_DIM/TEAMMATE_DIM/OPPONENT_DIM as well as n_robots_blue/n_robots_red
    """
    (n_teammates,) = set(envs.get_attr("n_robots_blue"))
    (n_opponents,) = set(envs.get_attr("n_robots_yellow"))
    (ball_dim,) = set(envs.get_attr("BALL_DIM"))
    (teammate_dim,) = set(envs.get_attr("TEAMMATE_DIM"))
    (opponent_dim,) = set(envs.get_attr("OPPONENT_DIM"))
    return TokenLayout(ball_dim, n_teammates, teammate_dim, n_opponents, opponent_dim)


class TransformerBackbone(nn.Module):
    """Per-entity-type embeddings + pre-LN transformer encoder.

    One token per entity (ball/teammate/opponent), so one network handles any
    team size. No positional encoding: robots are interchangeable, index
    identity must not affect the output. Token groups have shape
    (B, n, width + 2), where the last dimension is composed of the raw features plus [team_size, opp_size] counts.
    """

    def __init__(self, layout: TokenLayout, d_model, n_layers, n_heads, ff_dim, dropout):
        super().__init__()
        # per-type projections because feature widths differ (ball/teammate/
        # opponent). All three projections always exist (even with 0
        # opponents) so that state_dict shapes stay identical across
        # curriculum stages.
        self.ball_embed = layer_init(nn.Linear(layout.ball_dim + 2, d_model))
        self.teammate_embed = layer_init(nn.Linear(layout.teammate_dim + 2, d_model))
        self.opponent_embed = layer_init(nn.Linear(layout.opponent_dim + 2, d_model))
        # pre-LN (norm_first) is more stable than post-LN for RL transformers (cleanrl/ppo_trxl.py:211)
        # no padding mask: team size is fixed within a run, never mixed in a batch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, ball, teammates, opponents):
        tokens = torch.cat(
            [
                self.ball_embed(ball),
                self.teammate_embed(teammates),
                self.opponent_embed(opponents),
            ],
            dim=1,
        )
        return self.encoder(tokens)  # (B, num_tokens, d_model)


class TransformerActor(nn.Module):
    """Shared action head applied to every teammate token; ball/opponent tokens
    inform the result via attention only. One shared head keeps weights
    independent of team size."""

    def __init__(self, layout: TokenLayout, act_dim_per_robot, d_model, n_layers, n_heads, ff_dim, dropout):
        super().__init__()
        self.n_teammates = layout.n_teammates
        self.backbone = TransformerBackbone(layout, d_model, n_layers, n_heads, ff_dim, dropout)
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.Tanh(),
            # std=0.01: near-zero initial action means, so exploration starts ~N(0,1)
            layer_init(nn.Linear(d_model, act_dim_per_robot), std=0.01),
        )

    def forward(self, ball, teammates, opponents):
        hidden = self.backbone(ball, teammates, opponents)
        # token order is ball, teammates, opponents -> teammates start at 1
        own = hidden[:, 1 : 1 + self.n_teammates]
        return self.action_head(own)  # (B, n_teammates, act_dim_per_robot)


class TransformerCritic(nn.Module):
    """Separate encoder (not shared with the actor); pools all entity tokens into a scalar value."""

    def __init__(self, layout: TokenLayout, d_model, n_layers, n_heads, ff_dim, dropout, pooling="mean"):
        super().__init__()
        if pooling not in ("mean", "max", "attention"):
            raise ValueError(f"unknown critic pooling '{pooling}'")
        self.pooling = pooling
        self.backbone = TransformerBackbone(layout, d_model, n_layers, n_heads, ff_dim, dropout)
        if pooling == "attention":
            # PMA: single learned query, zero-init starts as ~uniform (mean) pooling
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pool_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.value_head = layer_init(nn.Linear(d_model, 1), std=1.0)

    def forward(self, ball, teammates, opponents):
        hidden = self.backbone(ball, teammates, opponents)
        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden.max(dim=1).values
        else: # self.pooling == "attention"
            query = self.pool_query.expand(hidden.shape[0], -1, -1)
            pooled, _ = self.pool_attn(query, hidden, hidden, need_weights=False)
            pooled = pooled.squeeze(1)
        return self.value_head(pooled)  # (B, 1)
