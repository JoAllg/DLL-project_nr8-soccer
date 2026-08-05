import os
import sys

import gymnasium as gym
from config import load_config, Config

# add src/ to path so `import myenvs` resolves the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import myenvs


config = load_config("example_config.yml")

print(config.stages[0].environment)
env = gym.make(
    "SSLDynamicRobots-v0",
    render_mode="human",
    **config.stages[0].environment.model_dump(),
)
env.reset()

# Run simulation
for _ in range(100):
    action = env.action_space.sample()
    env.step(action)

env.close()
