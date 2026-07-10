# Team-Size Generalization

How the transformer policy (`src/models.py`, `src/agent.py`) transfers across team sizes —
e.g. a checkpoint from a 5v5 curriculum stage deployed or fine-tuned at 11v11. Companion to
`transformer-implementation.md`.

Context: training uses a **size curriculum** — the robot count increases stepwise over one
long run; each stage sees a single fixed team size, never mixed within a batch.

## What generalizes by construction

- **Per-type projections shared across slots**: `teammate_embed` is one
  `Linear(teammate_dim + 2, d_model)` applied to every teammate token, so no weight shape
  depends on entity counts; same for the shared `action_head` and per-robot `actor_logstd`.
  Checkpoints `load_state_dict` unchanged across sizes; `Agent.set_env` re-points a live agent.
- **No positional encoding**: dodges the classic transformer length-generalization failure —
  learned positional embeddings would have no trained entry for slots beyond the trained team
  size. As a *set*, teammate #7 of an 11-robot team is just another token from the same
  distribution.
- **Static bound-based observation scaling** (`obs_scale` from the env's `Box` bounds):
  unlike running-stats normalization, it has no shape tied to team size and is recomputed from
  the new env, so it transfers cleanly.

## Count features

`Agent._tokenize` appends `[team_size / N_MAX, opp_size / N_MAX]` (`N_MAX = 11`) to every
token — the only team-size signal, since there is no padding/masking. Normalization keeps
every stage and deployment size in `(0, 1]`. Within a stage the feature is constant; it only
becomes informative through the variation the curriculum supplies across stages. A size the
curriculum never visits therefore remains an off-training-point input — mild, but the more
distinct sizes trained, the better the interpolation.

## Attention dilution (mild)

5v5 has 11 tokens (1 ball + 5 + 5); 11v11 has 23. Softmax attention mass per token roughly
halves, shifting attention-output statistics relative to earlier stages. Attention output is
a convex combination (scale-invariant in token count) and pre-LN normalizes the residual
stream, so degradation is graceful — but a jump far beyond the last trained size is still a
distribution shift.

## Critic pooling

- `mean` (default): size-invariant — the right choice for transfer.
- `max`: biased upward with more tokens (a max over 23 samples exceeds one over 11 in
  expectation).
- `attention` (PMA): a convex combination like `mean`, behaves comparably.

Only matters when fine-tuning at the new size; zero-shot deployment runs the actor alone.

## PPO-side scale effects at a stage transition

- **Entropy / joint log-prob magnitude**: both sum over `n_teammates × act_dim_per_robot`
  dims, so they grow linearly with team size (5 → 11 is ~2.2×). An `ent_coef` tuned at one
  stage is mis-calibrated at the next, and the KL / clip-fraction operating point shifts.
- **Return distribution**: rewards change with team size, so the critic re-adapts its output
  scale each stage — expect a transient value-loss spike.
- **`NormalizeReward` running stats** restart with the env each stage; no transfer concern.

## Environment-level shift (outside the model's control)

- **Crowding dynamics**: 22 robots interact very differently from 10.
- **Field geometry**: if positions are normalized by a field that grows with team size, the
  same normalized coordinate means a different physical position/speed.
- **Opponent distribution**: self-play opponents from a smaller stage don't represent play at
  the larger size.

The curriculum is the mitigation for all three: each stage fine-tunes the same checkpoint
under the new dynamics instead of asking for zero-shot transfer.
