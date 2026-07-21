"""Define different strategies for Opponents"""

from typing import Protocol, Optional, TYPE_CHECKING
from rsoccer_gym.Entities import Ball, Frame, Robot
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
                action[:, 2] *= -1.0  # v_theta
        return action

OPPONENT_POLICIES = {
    "Random": RandomOpponentPolicy,
    "Agent": AgentOpponentPolicy,
}
