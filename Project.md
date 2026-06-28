
- rSoccer VSS Environment https://github.com/robocin/rSoccer/blob/main/rsoccer_gym/vss/README.md 
	- 3vs3
	- extend to 5vs5 to evaluate cooperation and generalization
- A single transformer based policy trained by "self-play"
	- Start with one robot and the goal, then increase robots and opponent
	- train first against the a random Ornstein Uhlenbeck process-led opponent, 
	  in regular intervals update the opponent with checkpoints (or with a sample from multiple checkpoints) of the trained transformer policy (i.e. 80% current + 20% past sample)
	- input: observations for each player in a team; output their actions
		- For training with 3 players, should they be randomly assigned to the input vector? - no, the transformer is able to take different size of tokens (???)
		- Mirror observations/input for the opponent
- How to shape cooperative teamwork?
	- rewards?
		- robots should not block each other -> distance
- delayed learning
- Poster presentation Din A0
	- We will be there and talk about the content, our results and so on
	- half-selfexplainatory poster




- open ai spinning up (refresher of important RL concepts)
	- Proximal policy optimization (we will implement it)
		- "clean RL" for PPO ppo.py
		- 
- JAX numpy replacement with ai libraries on to of it

- no positional encoding
- Milestone
	1. PPO + single robot training (with small network 100k parameters - also works on CPU) -> achieve score goal
	2. Parallelize environments (simulation) with random actions and step them/backropagate in steps (policy gradient).
- LLM usage declaration in the code on a module level or file level
- open about how we split the work, split as clean as possible. Track worktime?
- 