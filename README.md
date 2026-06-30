# NR#8 Learning Cooperation in Soccer with Transformer-Based Policies (Julien) 

In soccer, players need to work together to play well. Transformers can be useful as policies as they can handle teams with different numbers of players. You will:
- Get familiar with rSoccer
- Implement a transformer-based team policy
- Train agents with deep reinforcement learning
- Evaluate cooperation and generalization to different team sizes (optional!)


## Installation

### Base install
**uv** (recommended):
```bash
uv sync
```

**pip**:
```bash
pip install .
```

### With optional extras

| Extra | Contents |
|---|---|
| `mujoco` | MuJoCo physics simulator + imageio |
| `dm-control` | DeepMind Control Suite (includes MuJoCo) |

**uv**:
```bash
uv sync --all-extras          # install all extras
```

**pip**:
```bash
pip install ".[mujoco,dm-control]"
```

# References
- PPO Algorithm is based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py / https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy