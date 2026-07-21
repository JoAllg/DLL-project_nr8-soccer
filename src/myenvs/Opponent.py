"""Define different strategies for Opponents"""

from typing import Protocol, Optional, TYPE_CHECKING
from rsoccer_gym.Entities import Ball, Frame, Robot
import numpy as np
from agent import Agent

if TYPE_CHECKING: #avoid circular import
    from DynamicRobots import SSLDynamicRobots

class OpponentPolicy(Protocol):
    def act(self, env: "SSLDynamicRobots") -> np.ndarray: ...

class RandomOpponentPolicy:
    def act(self, env) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(env.n_robots_yellow, 5))

class AgentOpponentPolicy:
    def __init__(self, agent: Optional[Agent] = None):
        self.agent = agent

    def act(self, env) -> np.ndarray:
        obs = env._build_obs_for(env.frame, env.frame.robots_yellow,
                                  env.frame.robots_blue, mirror=True)
        if self.agent is None:
            return np.zeros([env.n_robots_yellow, 5])

        action, _, _, _ = self.agent.get_action_and_value(obs) #TODO: add deterministic flag
            # action here is flat for n_robots_yellow==1; reshape if you support more
        return action.reshape(env.n_robots_yellow, 5)
