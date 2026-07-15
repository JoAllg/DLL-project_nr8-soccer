from gymnasium.envs.registration import register


register(
    id="SSLSingleRobot-v0",
    entry_point="myenvs.SingleRobot:SSLSingleRobot",
    kwargs={"render_mode": None},   # default kwargs, can be overridden in gym.make
)
register(
    id="SSLDynamicRobots-v0",
    entry_point="myenvs.DynamicRobots:SSLDynamicRobots",
    kwargs={"render_mode": None},   # default kwargs, can be overridden in gym.make
)
