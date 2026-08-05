[![Contributors][contributors-shield]][contributors-url]
[![Creative Commons License][license-shield]][license-url]

[contributors-shield]: https://img.shields.io/github/contributors/JoAllg/DLL-project_nr8-soccer.svg?style=for-the-badge
[contributors-url]: https://github.com/JoAllg/DLL-project_nr8-soccer/graphs/contributors
[license-shield]: https://img.shields.io/badge/Licence-CC_BY--NC_4.0-red?style=for-the-badge
[license-url]: https://creativecommons.org/licenses/by-nc/4.0/

# NR#8 Learning Cooperation in Soccer with Transformer-Based Policies (Julien)

This project explores reinforcement learning in the simulated rSoccer environment. We train a transformer-based policy to play soccer by learning to score goals, coordinate with teammates, and evaluate how the learned policy scales across different team sizes.

<img width="506" height="308" alt="image" src="https://github.com/user-attachments/assets/90d9657b-87ac-40e6-a1f1-0628ec7866cf" />

# Installation

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
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DODE_INCLUDE_DIRS=$HOME/.local/include -DODE_LIBRARIES=$HOME/.local/lib64/libode.so" uv sync
```

Otherwise (ODE available system-wide):

```bash
uv sync
```



## Usage
### Training:

Via uv you can run the training with

```bash
uv run src/ppo.py --track --capture-video
```

To use multiple GPUs do

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<NUM_GPUS> src/ppo.py --track --capture-video
```

### Simulating a model:
```bash
uv run  src/simulate.py --model-path=<path_to_model> --stage-name=<name_of_stage> --render
```

# References

- The PPO Algorithm is based on [https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) / [https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)



# Resources @Julien

rSoccer environment: [https://github.com/robocin/rSoccer](https://github.com/robocin/rSoccer)

Proximal Policy Optimization (PPO) paper: [https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)

Very good initial implementation (CleanRL): [https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)

Introduction into Deep RL: [https://spinningup.openai.com/en/latest/spinningup/rl_intro.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) (PPO is also explained there!)

For transformers, there are a lof of very good tutorials out there, but it depends on how much you already understand. In the end, you will probably be working with something of the PyTorch library: [https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html](https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html)
