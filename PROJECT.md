# Issues



# Milestones
1. PPO + single robot training (small network, ~100k parameters — also works on CPU) -> achieve score goal
   1. BwUniCluster scripts
   2. Environment and PPO/RPO adaptation
   3. Creating VSS environment with one Robot + Ball (init at random position), Rewards
   4. Add this environment to the PPO algorithm
   5. Dynamic Environment
   6. (automatic) HPO
   7. Reward shaping
   8. Add Transformer Encoder
2. Increasing team sizes (stepwise), adapting Rewards, improving learning
   - ~Parallelize environments (simulation) with random actions and step them/backpropagate in steps (policy gradient).~
   - Rewards that facilitate/shape cooperative teamwork
3. Optional Step
   - Evaluate cooperation and generalization to different team sizes (optional!)


