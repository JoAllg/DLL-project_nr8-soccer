import gymnasium as gym                                                                                                                             
import rsoccer_gym                                                                                                                                  
                                                                                                                                                    
env = gym.make('VSS-v0', render_mode="human")                                                                                                       
env.reset()
                                                                                                                                                    
# Run simulation (you can add your agent logic here)                                                                                                
for _ in range(100):                                                                                                                                
    action = env.action_space.sample()  # random actions
    env.step(action)                                                                                                                                

env.close()