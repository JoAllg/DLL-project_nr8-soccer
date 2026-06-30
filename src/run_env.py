import gymnasium as gym                                                                                                                             
import myenvs
env = gym.make('SSLSingleRobot-v0', render_mode="human")                                                                                                       
env.reset()
# Run simulation (you can add your agent logic here)                                                                                                
for _ in range(1000):                                                                                                                                
    action = env.action_space.sample()  # random actions
    env.step(action)                                                                                                                                

env.close()

