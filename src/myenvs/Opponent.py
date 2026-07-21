"""Define different strategies for Opponents"""

from typing import Protocol
from rsoccer_gym.Entities import Ball, Frame, Robot
import numpy as np

class OpponentPolicy(Protocol):
    def act(self, frame: Frame, n_robots_yellow: int) -> np.ndarray:
        """Return shape (n_robots_yellow, 5) actions in [-1, 1]."""
        ...

class RandomOpponentPolicy:
    def act(self, frame: Frame, n_robots_yellow: int) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(n_robots_yellow, 5))

