"""Define different strategies for Opponents"""

from typing import Protocol, Optional, TYPE_CHECKING
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from gymnasium.spaces import Box
import numpy as np
import torch
from agent import Agent

if TYPE_CHECKING: #avoid circular import
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

class AgentOpponentPolicy:
    def __init__(self, agent: Optional[Agent] = None, mirror=True ):
        self.agent = agent
        self.mirror = mirror

    def act(self, env) -> np.ndarray:
        obs = env._build_obs_for(env.frame.robots_yellow,
                                  env.frame.robots_blue, mirror=True)
        if self.agent is None:
            raise RuntimeError(
                "AgentOpponentPolicy.act() called before an agent was set "
                "-- call env.set_opponent_agent(agent) after constructing it."
            )

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)
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
    "Agent": AgentOpponentPolicy,
}
