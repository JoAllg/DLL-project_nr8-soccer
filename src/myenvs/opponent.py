"""Define different strategies for Opponents"""

from typing import Protocol, Optional, TYPE_CHECKING
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from gymnasium.spaces import Box
import numpy as np
import torch
from agent import Agent

if TYPE_CHECKING:  # avoid circular import
    from DynamicRobots import SSLDynamicRobots


class OpponentPolicy(Protocol):
    def act(self, env: "SSLDynamicRobots") -> np.ndarray: ...


class RandomOpponentPolicy:
    def act(self, env) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(env.n_robots_yellow, 5))


class UhlsteinOpponentPolicy:
    """Temporally-correlated opponent: one OU process per yellow robot,
    sampling smooth 5-dim actions in [-1, 1] (matches env's per-robot action)."""

    def __init__(self):
        self.ou_actions: Optional[list[OrnsteinUhlenbeckAction]] = None

    def act(self, env) -> np.ndarray:
        # init OU for each yellow robot, n_robots_yello only becomes known when env is defined
        if self.ou_actions is None:
            action_space = Box(low=-1, high=1, shape=(5,))
            self.ou_actions = [
                OrnsteinUhlenbeckAction(action_space, dt=env.time_step)
                for _ in range(env.n_robots_yellow)
            ]
        actions = np.stack([ou.sample() for ou in self.ou_actions]).clip(-1, 1)
        # never kick or dribble
        actions[:, 3:5] = 0.0
        return actions

    def reset(self):
        if self.ou_actions is not None:
            for ou in self.ou_actions:
                ou.reset()


class BlockOpponentPolicy:
    """Scripted defender

    Yellow robots interpose/block between threats and the (middle of the) goal they defend.
    The yellow robot closest to the ball protects interposes between the ball and the goal.

    Up to num_blockers - 1 other yellow robots interpose between blue robots and the goal, prioritising the blues closest (x-coordinate) to the goal.

    All other yellows get Uhlstein random actions

    Recomputed every step (no state)

    speed_scale set the speed of the robots of adapting to the opponents and the ball position.
    """

    def __init__(
        self,
        num_blockers: int = 2,
        block_frac: float = 0.4,
        approach_dist: float = 0.7,
        speed_scale: float = 0.3,
    ):
        self.num_blockers = num_blockers  # how many yellows to use blocking strategy
        self.block_frac = block_frac  # standoff: fraction from threat toward goal
        self.approach_dist = (
            approach_dist  # P-controller: distance at which speed saturates
        )
        self.speed_scale = (
            speed_scale  # cap top speed (<1 = slower, gives blue time to act)
        )
        self.ou_actions: Optional[list[OrnsteinUhlenbeckAction]] = None

    def act(self, env) -> np.ndarray:
        if self.ou_actions is None:
            action_space = Box(low=-1, high=1, shape=(5,))
            self.ou_actions = [
                OrnsteinUhlenbeckAction(action_space, dt=env.time_step)
                for _ in range(env.n_robots_yellow)
            ]

        goal_center = np.array([env.field.length / 2, 0.0])
        ball = np.array([env.frame.ball.x, env.frame.ball.y])
        yellow = np.array(
            [
                [env.frame.robots_yellow[i].x, env.frame.robots_yellow[i].y]
                for i in range(env.n_robots_yellow)
            ]
        )
        blue = np.array(
            [
                [env.frame.robots_blue[i].x, env.frame.robots_blue[i].y]
                for i in range(env.n_robots_blue)
            ]
        )

        defender = int(np.argmin(np.linalg.norm(yellow - ball, axis=1)))
        # blues ranked most-dangerous first (closest to goal in x)
        blues_by_danger = sorted(
            range(len(blue)), key=lambda b: goal_center[0] - blue[b, 0]
        )

        actions = np.zeros((env.n_robots_yellow, 5), dtype=np.float32)
        marker_k = 0
        for i in range(env.n_robots_yellow):
            if i >= self.num_blockers:
                actions[i] = self.ou_actions[i].sample()
                continue
            if i == defender:
                threat = ball
            elif marker_k < len(blue):
                threat = blue[blues_by_danger[marker_k]]
                marker_k += 1
            else:  # i <= num_blockers but no more blues left
                actions[i] = self.ou_actions[i].sample()
                continue
            target = threat + self.block_frac * (goal_center - threat)
            actions[i, 0:2] = (
                np.clip((target - yellow[i]) / self.approach_dist, -1.0, 1.0)
                * self.speed_scale
            )
        return actions

    def reset(self):
        if self.ou_actions is not None:
            for ou in self.ou_actions:
                ou.reset()


class AgentOpponentPolicy:
    def __init__(self, agent: Optional[Agent] = None, mirror=True):
        self.agent = agent
        self.mirror = mirror

    def act(self, env) -> np.ndarray:
        obs = env._build_obs_for(
            env.frame.robots_yellow, env.frame.robots_blue, mirror=True
        )
        if self.agent is None:
            raise RuntimeError(
                "AgentOpponentPolicy.act() called before an agent was set "
                "-- call env.set_opponent_agent(agent) after constructing it."
            )

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(
            0
        )  # (obs_dim,) -> (1, obs_dim)
        with torch.no_grad():
            action, *_ = self.agent.get_action_and_value(obs_t)  # (1, action_dim)
            action = action.squeeze(1).cpu().numpy().reshape(env.n_robots_yellow, 5)
            if self.mirror:
                action[:, 0] *= -1.0  # v_x
                action[:, 1] *= -1.0  # v_y
        return action.clip(-1, 1)


OPPONENT_POLICIES = {
    "Random": RandomOpponentPolicy,
    "Uhlstein": UhlsteinOpponentPolicy,
    "Block": BlockOpponentPolicy,
    "Agent": AgentOpponentPolicy,
}
