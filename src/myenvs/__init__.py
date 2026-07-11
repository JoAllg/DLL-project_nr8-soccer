from gymnasium.envs.registration import register


register(
    id="SSLSingleRobot-v0",
    entry_point="myenvs.SingleRobot:SSLSingleRobot",
    kwargs={"render_mode": None},   # default kwargs, can be overridden in gym.make
)
