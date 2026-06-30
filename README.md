# NR#8 Learning Cooperation in Soccer with Transformer-Based Policies (Julien)

In soccer, players need to work together to play well. Transformers can be useful as policies as they can handle teams with different numbers of players. You will:

- Get familiar with rSoccer
- Implement a transformer-based team policy
- Train agents with deep reinforcement learning
- Evaluate cooperation and generalization to different team sizes (optional!)

## Installation
The suggested python environment manager is [uv](https://docs.astral.sh/uv/getting-started/installation/)

`rc-robosim` is build from source. For that the [ODE (Open Dynamics Engine)](https://ode.org/wiki/index.php?title=Manual) library is required. Install it via your system package manager (see the manual) or build it from source:

```bash
rm -rf /tmp/ode-build && \
git clone https://bitbucket.org/odedevs/ode.git /tmp/ode-build && \
mkdir -p /tmp/ode-build/build && \
cd /tmp/ode-build/build && \
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && \
make -j4 && \
make install
```

If ODE was installed via the above command, sync the uv environment with:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib64:$LD_LIBRARY_PATH
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DODE_INCLUDE_DIRS=$HOME/.local/include -DODE_LIBRARIES=$HOME/.local/lib64/libode.so" uv sync --all-extras
```

Otherwise (ODE available system-wide):

```bash
uv sync --all-extras
```

# References

- The PPO Algorithm is based on [https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) / [https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)

