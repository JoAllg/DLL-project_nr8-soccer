"""Self-check for the action-saturation and pass-catchability fixes.

Run from src/:  uv run python test_action_and_pass.py
"""

import numpy as np
import torch
import gymnasium as gym
from rsoccer_gym.Entities import Ball, Frame, Robot

import myenvs  # noqa: F401  (registers the envs)
from myenvs.DynamicRobots import SSLDynamicRobots
from agent import Agent

AREAS = dict(
    allowed_positions_blue={"min": (-1, -1), "max": (1, 1)},
    allowed_positions_yellow={"min": (-1, -1), "max": (1, 1)},
    allowed_positions_ball={"min": (-1, -1), "max": (1, 1)},
)


def make_env(n_blue=2, **rewards):
    return SSLDynamicRobots(
        field_type=1,
        n_robots_blue=n_blue,
        n_robots_yellow=0,
        rewards=rewards or {"pass": 1.0},
        **AREAS,
    )


def test_step_returns_metrics_and_splits_truncation():
    """The merged step() must inject episode metrics AND still split truncated."""
    env = make_env(n_blue=1, progress=0.8)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros((1, 5), dtype=np.float32))
    for key in ("episode_pass_count", "episode_goal_count", "episode_reward_breakdown"):
        assert key in info, (
            f"{key} missing from info -- ppo.py's logging guards need it"
        )
    assert not (terminated and truncated), "terminated and truncated must be exclusive"

    # drive to the time limit: it must come back as truncated, not terminated
    env.episode_steps = env.max_steps - 1
    _, _, terminated, truncated, _ = env.step(np.zeros((1, 5), dtype=np.float32))
    assert truncated and not terminated, (terminated, truncated)
    env.close()


def _reception(env, ball_v_x, gap=0.05):
    """Place a receiver so the ball crosses it this step at speed `ball_v_x`.

    `gap` is the ball-to-receiver distance on the scoring step; it must be inside
    _detect_reception's recv_radius while the previous frame's ball is outside it.
    Both are locals in the env, so the callers assert the detector actually fired
    rather than trusting the geometry (see _probe_max_catch_speed).
    """
    d_min = 2.0 * env.field_scale
    passer_x = -3.0
    recv_x = passer_x + d_min + 1.0
    env.reset(seed=0)
    env.last_touch_id = 0
    env.last_touch_pos = np.array([passer_x, 0.0])
    env.episode_steps = 1
    env._pass_counted_step = -1
    # last_frame: ball still outside the catch radius; frame: ball inside it
    last, now = Frame(), Frame()
    last.ball = Ball(x=recv_x - gap - 1.0, y=0.0, v_x=ball_v_x)
    now.ball = Ball(x=recv_x - gap, y=0.0, v_x=ball_v_x)
    for f in (last, now):
        f.robots_blue[0] = Robot(x=passer_x, y=0.0)
        f.robots_blue[1] = Robot(x=recv_x, y=0.0)
    env.last_frame, env.frame = last, now
    return env


def _probe_max_catch_speed(env, hi_factor=8.0):
    """Bisect for the arrival speed at which `catch` reaches 0.

    max_catch_speed is a tuning knob local to _detect_reception, so the tests read
    it back off the reward instead of duplicating the value and going stale every
    time it is retuned.
    """
    at_rest = _reception(env, ball_v_x=0.0)._reward_pass()
    assert at_rest > 0.99, (
        f"a ball arriving at rest scored {at_rest}, so the detector never fired -- "
        "_reception's `gap` is probably outside recv_radius; every other pass "
        "assertion would pass vacuously"
    )
    lo, hi = 0.0, hi_factor * env.max_v
    assert _reception(env, ball_v_x=hi)._reward_pass() == 0.0, "no upper bound on catch"
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _reception(env, ball_v_x=mid)._reward_pass() > 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def test_pass_reward_grades_by_arrival_speed():
    """`catch` must fall monotonically with arrival speed and hit 0 at the cap."""
    env = make_env()
    cap = _probe_max_catch_speed(env)

    speeds = [0.0, 0.25 * cap, 0.5 * cap, 0.75 * cap]
    scores = [_reception(env, ball_v_x=v)._reward_pass() for v in speeds]
    assert all(b < a for a, b in zip(scores, scores[1:])), dict(zip(speeds, scores))
    assert scores[0] > 0.99 and scores[-1] > 0.0, scores

    assert _reception(env, ball_v_x=cap)._reward_pass() == 0.0
    assert _reception(env, ball_v_x=1.5 * cap)._reward_pass() == 0.0
    # measured on models/v1_stage5_2vs0game: passes arrived at 2.41 m/s
    assert _reception(env, ball_v_x=2.41)._reward_pass() == 0.0
    env.close()
    return cap


def test_pass_forward_combines_catch_and_direction():
    env = make_env(pass_forward=1.0)
    cap = _probe_max_catch_speed(env)
    fwd = _reception(env, ball_v_x=0.25 * cap)._reward_pass_forward()
    assert 0.0 < fwd <= 1.0, fwd
    # backward pass: receiver behind the passer -> direction_score 0
    env2 = _reception(make_env(pass_forward=1.0), ball_v_x=0.25 * cap)
    env2.last_touch_pos = np.array([env2.frame.ball.x + 5.0, 0.0])
    assert env2._reward_pass_forward() == 0.0
    # uncatchable forward pass -> catch 0 kills it
    assert (
        _reception(
            make_env(pass_forward=1.0), ball_v_x=2.0 * cap
        )._reward_pass_forward()
        == 0.0
    )
    env.close()


def test_catchable_pass_is_physically_reachable():
    """max_catch_speed must not make the reward unearnable. Rolls the ball in the
    real sim (analytic friction estimates are well off in the first metre) and
    reports how wide the window of receiver distances scoring catch>0 is.

    The window is narrow by construction -- that narrowness IS the kick-strength
    gradient -- but it must not be empty.
    """
    env = make_env()
    d_min, cap = 2.0 * env.field_scale, _probe_max_catch_speed(env)

    best = (0.0, 0.0, 0.0)  # (band width, a3, best catch)
    for a3 in np.arange(0.30, 1.01, 0.05):
        frame = Frame()
        frame.ball = Ball(x=-4.0, y=0.0)
        frame.robots_blue[0] = Robot(x=-4.10, y=0.0, theta=0.0)
        frame.robots_blue[1] = Robot(x=4.0, y=3.0, theta=0.0)  # parked, out of the way
        env.reset()
        env.rsim.reset(frame)
        env.frame = env.rsim.get_frame()
        x0 = env.frame.ball.x
        traj = []
        for t in range(600):
            act = np.zeros((2, 5), dtype=np.float32)
            act[0, 3] = a3 if t < 2 else -1.0
            act[0, 0] = -1.0 if t >= 2 else 0.0  # back the passer off the ball
            env.step(act)
            ball = env.frame.ball
            traj.append((ball.x - x0, np.hypot(ball.v_x, ball.v_y)))
            if traj[-1][1] < 0.003:
                break
        window = [(d, v) for d, v in traj if d >= d_min and 0.02 < v <= cap]
        if window:
            width = window[-1][0] - window[0][0]
            if width > best[0]:
                best = (width, float(a3), max(1.0 - v / cap for _, v in window))

    width, a3, catch = best
    assert width > 0.0, (
        f"no kick strength can deliver a pass of at least d_min={d_min:.2f} m arriving "
        f"at or below max_catch_speed={cap:.3f} m/s -- the pass reward is unearnable"
    )
    assert catch > 0.5, f"best achievable catch is only {catch:.2f}"
    env.close()
    return a3, width, catch


def test_warmstart_pulls_runaway_logstd_back_in_range():
    """A pre-fix checkpoint's logstd must land on the cap, not above it: clamp()
    passes zero gradient outside its range, so a value above LOGSTD_MAX would be
    frozen at the sigma cap for the rest of training."""
    env = gym.vector.SyncVectorEnv([lambda: make_env(progress=0.8)])
    agent = Agent(env, rpo_alpha=0.2, agent_type="transformer")
    sd = {k: v.clone() for k, v in agent.state_dict().items()}
    sd["actor_logstd"] = torch.full_like(
        sd["actor_logstd"], 2.065
    )  # v1_stage5_2vs0game

    agent.load_state_dict(sd)
    assert agent.actor_logstd.max().item() <= agent.LOGSTD_MAX, agent.actor_logstd

    # and gradient must actually reach it again
    obs, _ = env.reset(seed=0)
    _, logprob, _, _ = agent.get_action_and_value(torch.Tensor(obs))
    logprob.sum().backward()
    grad = agent.actor_logstd.grad
    assert grad is not None and torch.any(grad != 0), f"logstd still frozen: {grad}"
    env.close()


def test_reception_counted_once_when_both_rewards_configured():
    env = _reception(make_env(**{"pass": 1.0, "pass_forward": 1.0}), ball_v_x=0.1)
    env._reward_pass()
    env._reward_pass_forward()
    assert env.episode_pass_count == 1, f"double counted: {env.episode_pass_count}"
    env.close()


def test_action_mean_and_std_are_bounded():
    """The whole action range must stay reachable: |mean| <= 1 and sigma <= 1."""
    env = gym.vector.SyncVectorEnv([lambda: make_env(progress=0.8)])
    agent = Agent(env, rpo_alpha=0.2, agent_type="transformer")

    # simulate the runaway measured on models/v1_stage5_2vs0game
    with torch.no_grad():
        agent.actor_logstd.fill_(2.065)
        for p in agent.actor.action_head[-1].parameters():
            p.mul_(500.0)

    obs, _ = env.reset(seed=0)
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs))
        tokens = agent._tokenize(torch.Tensor(obs))
        raw_mean = agent.actor(*tokens).reshape(1, -1)
        std = torch.exp(agent.actor_logstd.clamp(agent.LOGSTD_MIN, agent.LOGSTD_MAX))

    assert raw_mean.abs().max() > 1.0, "test is vacuous: raw head did not blow up"
    assert torch.tanh(raw_mean).abs().max() <= 1.0
    assert std.max().item() <= 1.0, std
    assert std.min().item() >= np.exp(agent.LOGSTD_MIN) - 1e-9, std

    # with sigma <= 1 and a bounded mean, intermediate kick strengths are reachable
    kicks = []
    for _ in range(2000):
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(torch.Tensor(obs))
        kicks.append(np.clip(a.numpy().reshape(-1, 5)[:, 3], -1, 1))
    kicks = np.concatenate(kicks)
    interior = float(np.mean((kicks > 0.05) & (kicks < 0.95)))
    assert interior > 0.15, (
        f"kick strength still effectively bang-bang: interior={interior:.3f}"
    )

    # logprob/entropy must stay finite with the clamp in the graph
    _, logprob, entropy, _ = agent.get_action_and_value(torch.Tensor(obs))
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()
    env.close()
    return interior


if __name__ == "__main__":
    test_step_returns_metrics_and_splits_truncation()
    print("ok  step() injects metrics and splits truncated")
    cap = test_pass_reward_grades_by_arrival_speed()
    print(
        f"ok  pass reward grades by arrival speed, 0 at max_catch_speed={cap:.3f} m/s"
    )
    test_pass_forward_combines_catch_and_direction()
    print("ok  pass_forward = catch * direction")
    test_reception_counted_once_when_both_rewards_configured()
    print("ok  reception counted once per step")
    a3, width, catch = test_catchable_pass_is_physically_reachable()
    print(
        f"ok  catchable pass reachable at kick strength ~{a3:.2f} "
        f"(receiver-distance window {width:.2f} m, best catch {catch:.2f})"
    )
    test_warmstart_pulls_runaway_logstd_back_in_range()
    print("ok  warm-started runaway logstd clamped and still trainable")
    interior = test_action_mean_and_std_are_bounded()
    print(
        f"ok  action mean/std bounded, kick interior fraction {interior:.3f} (was 0.049)"
    )
    print("\nall checks passed")
