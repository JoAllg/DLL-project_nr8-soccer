# Transformer Policy Implementation

Concise walkthrough of the transformer actor/critic (`src/models.py`) and what
`src/agent.py` / `src/ppo.py` add on top of the CleanRL baseline to support it. 

The MLP path (`agent_type="mlp"`) is that unchanged baseline and is not covered in detail here.

## Core idea

The team's state is a variable-size **set** of entities (ball, own robots, opponent robots),
not a fixed-length vector. Instead of flattening the observation into one MLP input (which
breaks the moment team size changes), each entity becomes one **token**, and a shared
transformer encoder attends over all of them. Because every weight matrix operates per-entity
(not per-slot), the same checkpoint works for 1v0, 3v3, 5v5, etc.

The whole team is treated as **one PPO agent**: one forward pass consumes all entity tokens
and emits one action per teammate. Cooperation emerges from self-attention across tokens, so
plain single-agent PPO (not MAPPO) applies — see `agent.py:118` where per-robot log-probs/
entropy are summed into a single scalar per env.

## How the classes are wired together

`Agent` (`agent.py`) owns two independent networks, not a shared trunk:

```
Agent (agent.py)
 ├─ .actor  : TransformerActor        (models.py:145)
 │             ├─ .backbone : TransformerBackbone   (own instance)
 │             │              ├─ .ball_embed / .teammate_embed / .opponent_embed
 │             │              └─ .encoder   : nn.TransformerEncoder (n_layers × TransformerEncoderLayer)
 │             └─ .action_head : Linear → Tanh → Linear   (shared across all teammate tokens)
 │
 └─ .critic : TransformerCritic       (models.py:174)
               ├─ .backbone : TransformerBackbone   (separate instance, independent weights)
               │              ├─ .ball_embed / .teammate_embed / .opponent_embed
               │              └─ .encoder   : nn.TransformerEncoder
               └─ .value_head  (+ .pool_query / .pool_attn if pooling="attention")
```

`TransformerBackbone` is *reused as a class*, not as a module instance: `TransformerActor`
and `TransformerCritic` each construct their own `TransformerBackbone(...)` in `__init__`, so
`agent.actor.backbone` and `agent.critic.backbone` never share a weight tensor. Both backbones
have identical architecture (same embeddings + encoder shape) but are trained independently —
this is what the design doc means by "two separately-trained transformers, same backbone
architecture" (`agent.py:41`).

## Data flow

```
flat obs (B, obs_dim)
   │  Agent._tokenize()            agent.py:69
   ▼
(ball, teammates, opponents)       shapes: (B,1,ball_dim+2), (B,n_team,teammate_dim+2), (B,n_opp,opponent_dim+2)
   │
   ├── TransformerActor.backbone / TransformerCritic.backbone   (separate forward passes)
   ├── TransformerActor  → per-teammate action mean              models.py:145
   └── TransformerCritic → scalar value                          models.py:174
```



### 1. Tokenization (`Agent._tokenize`, `agent.py:69`)

- The flat observation is sliced back into per-entity segments using `TokenLayout`
(`models.py:52`: `ball_dim, n_teammates, teammate_dim, n_opponents, opponent_dim`), which
encodes the per-entity feature widths: `ball [x,y,vx,vy]`,
`teammate [x,y,sinθ,cosθ,vx,vy,vθ]`, `opponent [x,y,vx,vy,vθ]`.
- Each token gets `[team_size / N_MAX, opp_size / N_MAX]` **appended** as extra features
(`N_MAX = 11`, `agent.py`). This is how the network learns about team size at all — there's
no other signal for it, since there's no padd*i*ng/masking (see below). Normalizing by
`N_MAX` keeps the counts in `(0, 1]` for every team size (why: `team-size-generalization.md`).
- Observations are divided by `obs_scale` (the env's declared `Box` high, `agent.py:57`)
instead of using `NormalizeObservation` — a running-mean normalizer would be
permutation-*unsafe* per-entity and is redundant since our envs already return bounded
values.
- `TokenLayout` is built **at runtime**, live from the vector env, by `token_layout_from_env`
(`models.py:80`). It reads `n_robots_blue`/`n_robots_yellow` and the
`BALL_DIM`/`TEAMMATE_DIM`/`OPPONENT_DIM` class constants via `envs.get_attr(...)`, mirroring
how `obs_dim`/`act_dim` are derived from `single_observation_space`/`single_action_space` for
the MLP path. Any env qualifies as long as its class declares those three dimension constants
(see `myenvs.SingleRobot.SSLSingleRobot`); entity *counts* alone can't be recovered from a flat
`Box` shape, hence the extra constants.



### 2. Embedding + encoder (`TransformerBackbone`, `models.py:97`)

- Three separate `nn.Linear` projections (ball/teammate/opponent) map each entity's raw
features to a common `d_model`, all three always constructed (even with 0 opponents) so the
state dict shape is stable across curriculum stages.
- **No learned type embedding** — redundant with per-type projections (each already learns its
own additive offset).
- **No positional encoding** — token identity within a type must not matter (homogeneous
robots are interchangeable; index order carries no meaning).
- Encoder is `nn.TransformerEncoderLayer(norm_first=True)` (pre-LN, more stable for RL) ×
`n_layers`, full self-attention, **no attention mask**. Masking is unneeded because team size
is fixed within a run/batch — never mixed — so there's nothing to mask out.
- Output: `(B, num_tokens, d_model)`, order `[ball, teammates…, opponents…]`.



### 3. Actor head (`TransformerActor`, `models.py:145`)

- Reads back **every teammate token** (`hidden[:, 1:1+n_teammates]`, `models.py:170`) — the
whole team is commanded by the shared policy; ball/opponent tokens only ever influence the
result through attention, never read out directly.
- One small shared MLP head (`Linear → Tanh → Linear`) applied per token → produces an action
per teammate regardless of how many there are.
- Output-layer `std=0.01` (via `layer_init`): near-zero action means at init, so the starting
policy is ≈ isotropic Gaussian noise — unbiased exploration, small initial policy-gradient
steps.



### 4. Critic head (`TransformerCritic`, `models.py:174`)

- **Separate weights from the actor** (its own `TransformerBackbone` instance — see the class
diagram above) — sparse reward + non-stationary self-play targets make a shared actor/critic
trunk prone to objective interference, and there's no expensive shared computation to
justify sharing here.
- Pools *all* entity tokens (not just teammate ones) into one vector, then a linear head →
scalar `V`. Pooling is a constructor choice (`pooling="mean"|"max"|"attention"`):
  - `mean` / `max`: straightforward, permutation-invariant.
  - `attention`: pooling-by-multihead-attention (PMA) with a single learned query
  (`pool_query`, zero-initialized so it starts as ~mean pooling).
  - No CLS token — considered too fragile/data-hungry for this regime.
- Output-layer `std=1.0`: unit-variance value predictions at init, a sane starting scale for
regressing returns.



## Agent-level glue (`agent.py`)

- `Agent.__init__` builds `self.actor` / `self.critic` when `agent_type="transformer"`, and a
single `actor_logstd` **per per-robot action dim** (not per total action dim) — shared across
robots so the parameter shape doesn't grow with team size (`agent.py:66`).
- `get_action_and_value` (`agent.py:96`):
  - Tokenizes once, runs actor and critic on the same tokens (identical observation scope —
  not a privileged critic, since there's no hidden info to grant here).
  - Flattens `(B, n_teammates, act_dim_per_robot) → (B, act_dim_total)` for the diagonal
  Gaussian.
  - RPO perturbation (`z ~ Uniform(-rpo_alpha, rpo_alpha)` added to `action_mean` when
  re-evaluating stored actions, `agent.py:112`) applies identically to the transformer path —
  it operates on the already-flattened mean, so it's architecture-agnostic.
  - **Joint log-prob/entropy**: `.sum(1)` sums over the flattened `(action_dims * n_teammates)`
  axis, i.e. over both per-robot action dims and robots at once — valid because robots are
  conditionally independent given the shared attention encoding. This is what makes the team
  "one PPO agent."



## `agent.py` / `ppo.py` vs. the CleanRL baseline

Both files started from CleanRL's `ppo_continuous_action.py` (vendored at
`deps/cleanrl/cleanrl/ppo_continuous_action.py`). The PPO math itself — GAE, clipped
surrogate policy loss, clipped value loss, grad-norm clipping, the minibatch/epoch update
loop — is **untouched**; everything below is either what feeds `get_action_and_value`/
`get_value` or how the optimizer step around it is driven.

**Agent / architecture (**`agent.py` **vs. the reference's single** `Agent` **class)**


|                        | Reference                       | This repo                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture choice    | One hardcoded MLP actor+critic  | `agent_type` param (`"mlp"` or `"transformer"`) dispatches via if/elif (`agent.py:28`)                                                                                                                                                            |
| `actor_logstd` shape   | `(1, act_dim)` fixed            | `(1, act_dim_per_robot)`, tiled across teammates for the transformer path so the parameter is team-size independent (`agent.py:66`, `:103`)                                                                                                       |
| Action sampling        | Sample once, done               | Adds an **RPO** re-perturbation step: when re-evaluating a stored action, `action_mean` gets `z ~ Uniform(-rpo_alpha, rpo_alpha)` added before recomputing `log_prob`/`entropy` (`agent.py:110-117`) — this is what makes it RPO, not vanilla PPO |
| Joint log-prob/entropy | `.sum(1)` over action dims only | Same `.sum(1)`, but for the transformer path the tensor is already flattened over `(action_dims × n_teammates)`, so the sum also collapses across robots — the whole team becomes one PPO "agent" (`agent.py:122-123`)                            |


**Training loop (**`ppo.py` **vs. the reference** `__main__` **block)**

- **Env vectorization**: `gym.vector.AsyncVectorEnv` (subprocess workers; constructed
*before* `utils.get_device()` touches CUDA, since forking after CUDA init can deadlock a
worker, `ppo.py:219`) vs. the reference's `SyncVectorEnv`.
- `make_env`: adds a `flatten` toggle (`ppo.py:152`, `flatten=args.agent_type=="mlp"`) —
the transformer path skips `FlattenObservation` to keep the per-entity structure
re-tokenizable; also drops `NormalizeObservation`/`TransformObservation` entirely (the
reference has both), since the rSoccer envs already return normalized/bounded
observations — a running-stats wrapper on top would just re-normalize an already-bounded
signal for no benefit.
- **Optimizer**: `AdamW` with two param groups — weight decay applied only to matrix params
(`p.ndim >= 2`), excluding `actor_logstd`/`critic.pool_query` (`ppo.py:247-257`) — vs. the
reference's plain `optim.Adam(agent.parameters(), ...)` with no weight decay.
- **LR schedule**: cosine-with-warmup with optional multi-cycle restarts
(`get_cosine_schedule_with_warmup`, `ppo.py:258-271`), stepped **every minibatch**
(`ppo.py:403-404`) vs. the reference's linear `frac = 1 - (iteration-1)/num_iterations` anneal,
stepped once per iteration.
- **Entropy coefficient**: linearly annealed `ent_coef → final_ent_coef` over
`total_timesteps` (`ppo.py:297-298`) vs. the reference's single constant `ent_coef`.
- **Episode logging**: reads `infos["episode"]`/`infos["_episode"]`
(`AsyncVectorEnv`'s "sync-at-index" info API, `ppo.py:319-324`) vs. the reference's
`infos["final_info"]` per-env-dict iteration — a difference in gymnasium vectorization API
version/style, same purpose.
- **Device/seeding**: `utils.get_device`/`utils.set_seed` (cuda → mps → cpu, optional
deterministic mode) vs. the reference's inline `torch.device(...)` +
`random.seed`/`torch.manual_seed` calls.
- **W&B/video**: `monitor_gym=False` plus a manual video step metric and
`utils.log_new_videos` polling (`ppo.py:194-203`, `:423-424`, `:454-457`) vs. the
reference's `monitor_gym=True` (which breaks on gymnasium 1.x) and no polling equivalent.
- **Extra CLI surface** not present in the reference `Args`: transformer hyperparameters
(`d_model`, `n_layers`, `n_heads`, `ff_dim`, `dropout`, `critic_pooling`), `rpo_alpha`,
scheduler knobs (`warmup_ratio`, `min_lr_ratio`, `num_cycles`, `cycle_decay`,
`weight_decay`), and `final_ent_coef`.



## Adding a new environment / team size

1. Declare `BALL_DIM`, `TEAMMATE_DIM`, `OPPONENT_DIM` as class constants on the env (see
  `myenvs.SingleRobot.SSLSingleRobot`). Team/opponent robot *counts* are already exposed via
   `n_robots_blue`/`n_robots_yellow` (set by `SSLBaseEnv`/`VSSBaseEnv`) and read live by
   `token_layout_from_env` (`models.py:80`) — no other `models.py` change is needed.
2. No other code changes needed — embeddings/heads are per-*type*, not per-*slot*, so weight
  shapes are independent of `n_teammates`/`n_opponents`. A checkpoint trained on a smaller
   team size loads directly (`load_state_dict`) into a run with a different team size; this is
   plain fine-tuning, not architecture surgery.

