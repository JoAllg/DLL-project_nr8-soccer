# Issues
- PPO inputs all environments vectorized into the policy/ciritc model. Does this make sense for our rSoccer game and transformer model?
- for transformers the learning rate annealing should be replaced with Cosine Annealing with Warmup (without restarts)
- RPO is better than PPO https://docs.cleanrl.dev/rl-algorithms/rpo/#overview (in most cases, in 4 cases with worse results use rho_alpha=0.1)
- Adaption of PPO/RPO to rSoccer environment: Are Observations/States single values (i.e. robot positions) or pixel values (c.f. adaption of PPO to carRacing pixel environment https://wandb.ai/cleanrl/cleanrl.benchmark/runs/34pstq7f/code?nw=nwuser_scott)


# Milestones
1. PPO + single robot training (with small network 100k parameters - also works on CPU) -> achieve score goal
   1. BwUniCluster scripts (Joshua)
   2. Environment and PPO adoption (Joshua)
   3. Creating VSS environment with one Robot + Ball (init at random position), Rewards (Muskan)
   4. Add this environment to the PPO algorithm (Tobias)
   5. HPO, Reward shaping
   6. Add Transformer Encorder (Joshua)
2. Increasing Team sizes (stepwise), adapting Rewards, improving lea
   - ~Parallelize environments (simulation) with random actions and step them/backropagate in steps (policy gradient).~
   - Rewards that faciliate/shape cooperative teamwork (e.g. robots should not block each other -> penalty on low distance)
3. ?
	- Evaluate cooperation and generalization to different team sizes (optional!)
4. Poster presentation DIN A0
   - We will be there and talk about the content, our results and so on
   - half-selfexplainatory poster


## TODOs
- LLM usage declaration in the code on a module level or file level
- open about how we split the work, split as clean as possible. Track worktime?