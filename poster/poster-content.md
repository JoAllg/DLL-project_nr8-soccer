# Poster Content — Research & Planning

Working notes for filling the DIN A0 UFR poster template
(`DLL26 - UFR Poster Template.pdf`). The template has five content boxes:
**Introduction**, **Method**, **Quantitative Results**, **Qualitative
Results**, **References**, plus a header (title / names / supervisor / GitHub
link) and an acknowledgement line.

This document is grounded in the actual code (`src/`, `config.yml`), not in the
exploratory `wiki/*` notes.

---

## 0. What the project actually is

Course project **NR#8 — "Learning Cooperation in Soccer with Transformer-Based
Policies"**. We train **one shared transformer policy for a whole robot-soccer
team** with deep RL and ask two questions:

1. Can a single set of weights control a **variable number of players** without
  retraining?
2. Does reward shaping + curriculum + self-play produce **cooperative** behavior
  (passing, spacing) rather than every robot chasing the ball?

Environment: **rSoccer**, the RoboCup Small-Size-League (SSL) simulator. Our
custom env `SSLDynamicRobots` (`src/myenvs/DynamicRobots.py`) supports a dynamic
number of blue (our team) and yellow (opponent) robots on a scaled field.
Continuous control: each robot outputs `[v_x, v_y, v_theta, kick, dribbler]`.

Algorithm: **PPO with RPO** (Robust Policy Optimization) adapted from CleanRL
(`src/ppo.py`).

### The central hook (headline idea for the poster)

> A transformer treats each entity (ball, teammate, opponent) as a **token**, so
> the same policy handles 1, 2, 3, … players. **No positional encoding** →
> robots are interchangeable → permutation-invariant team control + **zero-shot
> generalization to team sizes never trained on.**

This is the one sentence a passer-by should leave with. Everything else on the
poster supports it.

---



## 1. Introduction box

Content to include (keep it to ~4 short paragraphs / a diagram):

- **Problem.** Team sports need cooperation. Classic multi-agent RL fixes the
number of agents at train time (concatenated observation / one head per
player), so a 2-player policy cannot run a 3-player game. We want **one policy,
any team size.**
- **Why transformers.** Attention over per-entity tokens is naturally
set-based and permutation-invariant. Drop the positional encoding and the
network cannot tell "robot 0" from "robot 1" — exactly the symmetry a team of
identical robots has.
- **Why it's hard.** Sparse reward (goals are rare), continuous action space,
a moving opponent, and cooperation is not directly observable — it has to be
*shaped* through rewards without hand-scripting the tactics.
- **What we contribute.** A shared transformer actor+critic over tokens, trained
through a **curriculum** (1 robot → 2v2 → passing) with **self-play** against
past checkpoints and scripted opponents, and a library of **shaped cooperative
rewards** (passing, spacing, dribbling).

Good visual for this box: a small **field snapshot** with tokens annotated
(ball token, teammate tokens, opponent tokens) → arrows into the encoder. Sets
up the Method box.

---



## 2. Method box

The technical core. Suggested sub-parts, each one figure or a tight bullet:

### 2a. Tokenized observation (`src/models.py`, `src/agent.py`)

- Flat observation split into tokens: **1 ball** (`BALL_DIM=4`: x, y, vx, vy),
**N teammate** (`TEAMMATE_DIM=7`: x, y, sinθ, cosθ, vx, vy, vθ), **M opponent**
(`OPPONENT_DIM=7`) tokens.
- **Per-type linear embedding** (ball/teammate/opponent have different features,
so three projections) → shared **pre-LN Transformer encoder** → tokens.
- **No positional encoding** (permutation invariance). Team-size signal instead
injected as two scalar count features `[n_teammates/11, n_opp/11]` appended to
every token, so the net *knows* the team size without an index identity.
- Observations statically scaled to [-1, 1] (permutation-safe; can't use running
NormalizeObservation because token order/count varies).



### 2b. Shared actor + pooled critic

- **Actor:** one shared action head applied to *every* teammate token →
`(team_size, 5)` actions. Ball/opponent tokens influence output only via
attention. Because the head is shared, weight shapes are independent of team
size.
- **Critic:** separate encoder, **pools** all tokens (attention/mean/max) into a
single scalar value. The whole team is treated as **one PPO agent** (log-probs
summed across robots), i.e. centralized training.
- **RPO:** when re-evaluating stored actions, perturb the action mean with
uniform noise in `[-α, α]` (α=0.5) → keeps exploration alive, more robust than
vanilla PPO.



### 2c. Curriculum + self-play (`config.yml`, staged in `src/ppo.py`)

The single policy is carried across stages (`agent.set_env` re-points it at a
new team size without changing weights):


| Stage             | Setup                                           | Purpose                           |
| ----------------- | ----------------------------------------------- | --------------------------------- |
| 1vs0              | 1 robot, ball near goal                         | learn to kick/score at all        |
| 1vs0random        | 1 robot, random spawn                           | generalize position               |
| 2vs0 / 2vs0game   | 2 robots                                        | scale team, game-like spawns      |
| 2vs2random / 2vs2 | vs Random opponent                              | learn against motion              |
| 2vs2ou            | vs Ornstein-Uhlenbeck opponent                  | smooth temporally-correlated opp. |
| 2vs2block         | vs scripted **Block** defender                  | learn against real defense        |
| 2vs2opponent      | vs **past checkpoint of itself** (mirrored obs) | self-play                         |
| 3vs0passing       | 3 robots, passing rewards                       | emergent cooperation              |


Opponents (`src/myenvs/opponent.py`): `Random`, `Uhlstein` (OU),
`Block` (scripted interposing defender), `Agent` (self-play checkpoint, observations
mirrored so the opponent "attacks the other way").

### 2d. Reward shaping (`SSLDynamicRobots._reward_*`, weights in `config.yml`)

Weighted sum of per-step rewards, each normalized to [-1, 1]:

- **Dense drive:** `proximity` (closest robot approaches ball), `progress` (ball
toward goal).
- **Events:** `kick_forward`, `kick`, `pass` (ball reaches a *different*
teammate), `dribble`, `goal` / `goal_close` (only counts shots from the
attacking third).
- **Cooperation / penalties:** `spacing` (crowding penalty when blues stack on
the ball), `out_of_bounds`.
- **Reward annealing:** `coop` → `reduced` templates fade the shaping rewards
late in training so the sparse goal reward dominates (avoids reward hacking of
the dense shaping terms).



### 2e. Training infra (one line, maybe a small logo strip)

- Cosine LR with warmup (`src/scheduler/`), entropy-coefficient annealing, AdamW.
- Vectorized envs (`AsyncVectorEnv`, 16 envs), multi-GPU **DDP** via `torchrun`.
- Ran on **BwUniCluster 3.0** (Slurm), logged to TensorBoard / Weights & Biases.

Suggested figure: **architecture diagram** — tokens → per-type embed → encoder →
{shared action head per teammate token} + {pooled critic → value}. This is the
single most important graphic on the poster.

---



## 3. Quantitative Results box (left, large)

Metrics we actually log (`writer.add_scalar` in `src/ppo.py`,
`charts/*` and `losses/*` in TensorBoard/W&B). Candidate plots, ranked:

**Must-have**

1. **Episodic return vs. steps, with stage boundaries marked**
  (`charts/episodic_return`, overlay `charts/stage_id`). Shows learning and the
   curriculum transitions in one figure — the money plot. Annotate each stage
   name; the dips/recoveries at stage changes tell the curriculum story.
2. **Goal / scoring rate over training** — if not logged directly, derive from
  episodic return jumps or add an eval metric (goals per episode). Most
   intuitive "did it learn soccer" number for a general audience.

**Strong supporting**
3. **Generalization bar chart (the headline experiment).** Take one trained
   policy, evaluate **zero-shot** at team sizes it *was* and *was not* trained on
   (e.g. train on 2, eval on 1/2/3) → bars of goal-rate / return per team size.
   This is the direct evidence for the central claim and is worth generating
   specifically for the poster even if not yet in the runs.
4. **Self-play / opponent difficulty:** return vs. Random / OU / Block / self
   opponent — shows robustness improves through the curriculum.

**Diagnostics (small, secondary)**
6. `losses/explained_variance` (critic quality), `losses/approx_kl` +
   `losses/clipfrac` (PPO stability), `charts/learning_rate` /
   `entropy_coefficient` (schedule sanity). Include at most one small multi-panel
   of these; they're for the "is the training healthy" question, not the story.

---



## 4. Qualitative Results box (right, tall)

The template's Qualitative box is tall/narrow — good for **stacked field
snapshots or a short trajectory strip**. Ideas:

- **Emergent cooperation storyboard:** 3–4 frames of a `3vs0passing` or `2v2`
episode showing a **pass** — robot A kicks, ball travels, robot B receives,
shoots. Caption the reward events firing (`pass`, `spacing`, `goal`).
- **Spacing behavior:** side-by-side of *without* vs *with* spacing reward — two
robots stacking on the ball vs. spreading out. Directly visualizes the
cooperation reward's effect.
- **Attention heatmap:** which tokens the actor attends to when deciding a
robot's action (e.g. high attention on ball + nearest teammate before a pass).
Strong "transformer" visual, ties architecture to behavior.
- **Zero-shot team size:** one screenshot each at 1 / 2 / 3 robots with the *same
weights* running — visual proof of generalization.
- Include the **GitHub link as a QR code** (header already has a `<Github link>`
slot) so people can watch the rendered videos (`capture_video` produces them).

We already capture videos during training (`--capture-video`); pull the best
episode frames from `videos/`.

---



## 5. References box

Pick 3–5, matching what the method leans on:

1. **PPO** — Schulman et al. 2017, *Proximal Policy Optimization Algorithms*,
  arXiv:1707.06347.
2. **RPO** — Rahman & Xue, *Robust Policy Optimization* (the α action-mean
  perturbation trick).
3. **CleanRL** — Huang et al., *CleanRL: High-quality Single-file
  Implementations…*, JMLR 2022 (our PPO base).
4. **Transformer** — Vaswani et al. 2017, *Attention Is All You Need*.
5. **rSoccer** — Martins et al., *rSoccer: A Framework for Studying RL in SSL and
  VSS*, RoboCup 2021.

---



## 6. Header / meta

- **Title options** (pick punchy, mention the two selling points — transformer +
variable team size):
  - "One Policy, Any Team: Transformer Reinforcement Learning for Cooperative
  Robot Soccer"
  - "Permutation-Invariant Team Play: A Single Transformer Policy for Variable-Size
  Robot Soccer"
- Names, supervisor (Julien), University of Freiburg, GitHub link/QR.
- Acknowledgement: BwUniCluster 3.0 compute, supervisor, rSoccer authors.

---



## 7. How to approach building the poster (workflow)

1. **Freeze the story first, then design.** One sentence (Section 0 hook) → three
  supporting claims: (a) transformer tokens enable variable team size, (b)
   curriculum+self-play makes it learn, (c) shaped rewards make it cooperative.
   Every box serves one of these; cut anything that doesn't.
2. **Lead with the architecture diagram and the generalization bar chart** — they
  are the two figures unique to this project. If only two things get read, these.
3. **Generate the missing evidence now**, not at the end: a zero-shot
  team-size eval (train at one size, test at others: 1/2/3) → the headline bar
   chart. Needs a short eval script over saved checkpoints (`runs/*.cleanrl_model`).
4. **Curate visuals from existing runs:** best `charts/episodic_return` curve with
  stage boundaries; pull pass/goal frames from `videos/`.
5. **A0 layout discipline:** big fonts, few words, figures do the talking. The
  template's box sizes already hint at weighting — Quantitative box is the
   largest, so the return curve + generalization chart live there.
6. **Dry-run the 60-second pitch** against the poster: intro problem → point at
  architecture → point at return curve → point at generalization chart → point at
   a qualitative pass frame. If the poster supports that walk, it's done.



## 8. Open gaps to close before printing

- [ ] **Generalization experiment (the headline result).** Measure how well one
  ```
  trained policy generalizes to team sizes it never saw: load a checkpoint
  trained at size N, run it *unchanged* (no fine-tuning, via `agent.set_env`)
  at other sizes (1 / 2 / 3), and report goal-rate / mean return per size.
  High bars on unseen sizes = the core claim proven; degradation = an honest,
  still-reportable finding. Needs a short eval script over
  `runs/*.cleanrl_model`.
  ```
- [ ] Attention-visualization script (qualitative box), if time.
- [ ] Confirm final trained checkpoint / best stage to showcase.