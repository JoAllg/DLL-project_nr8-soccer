"""Run a saved agent checkpoint in the environment, without training.

Usage:
    python simulate.py --model-path runs/.../stage0_explore.cleanrl_model \
                       --config v3.yml --stage-name explore --episodes 10

    # live rendered window:
    python simulate.py --model-path ... --config v3.yml --render

    # video capture instead:
    python simulate.py --model-path ... --config v3.yml --capture-video --video-dir videos/eval
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro
import gymnasium as gym
from gymnasium.vector import AutoresetMode

import rsoccer_gym  # noqa: F401
import myenvs  # noqa: F401

from agent import Agent
from config import load_config


@dataclass
class SimArgs:
    model_path: Optional[str] = None
    """path to the .cleanrl_model checkpoint to load; if omitted, blue robots stay static"""
    config: str = "config.yml"
    """yaml config used during training (env args + architecture hyperparams), relative to configs/"""
    stage_name: Optional[str] = None
    """which stage's environment config to simulate in (defaults to the first stage)"""
    episodes: int = 10
    """number of episodes to run"""
    deterministic: bool = True
    """use the policy mean instead of sampling (recommended for evaluation)"""
    render: bool = False
    """open a live window and render each step as it happens"""
    fps: int = 60
    """playback speed cap when --render is set (env's own render clock)"""
    capture_video: bool = False
    """record video of the episodes instead of live rendering"""
    video_dir: str = "videos/eval"
    """directory to save videos to, if capture_video is set"""
    cuda: bool = True
    """use GPU if available"""
    seed: int = 1
    """environment seed"""


def build_env_fn(args, config, env_args):
    """Eval-time wrapper stack: like ppo.make_env minus NormalizeReward/TransformReward,
    so the reported return is the env's raw reward."""
    render_mode = "human" if args.render else ("rgb_array" if args.capture_video else None)

    def thunk():
        env = gym.make(config.env_id, render_mode=render_mode, **env_args)
        if args.render:
            # rSoccer throttles human rendering with clock.tick(metadata["render_fps"]);
            # copy the dict so the class-level metadata stays untouched
            env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": args.fps}
        if args.capture_video:
            env = gym.wrappers.RecordVideo(env, args.video_dir)
        if config.agent_type == "mlp":
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def build_agent(envs, config, path, device):
    agent = Agent(
        envs, config.rpo_alpha, agent_type=config.agent_type, d_model=config.d_model,
        n_layers=config.n_layers, n_heads=config.n_heads, ff_dim=config.ff_dim,
        dropout=config.dropout, critic_pooling=config.critic_pooling,
    ).to(device)
    agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    agent.eval()
    agent.requires_grad_(False)
    return agent


def main():
    args = tyro.cli(SimArgs)
    config = load_config(args.config)

    if args.render and args.capture_video:
        raise ValueError("--render and --capture-video are mutually exclusive "
                         "(render_mode='human' vs 'rgb_array' can't both drive the same env)")

    if args.stage_name is not None:
        stage = next((s for s in config.stages if s.name == args.stage_name), None)
        if stage is None:
            raise ValueError(f"no stage named '{args.stage_name}' in {args.config}")
    else:
        stage = config.stages[0]
    env_args = stage.environment.model_dump()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # single env, but Agent reads single_observation_space/single_action_space
    # and expects a batch dim, so keep the vector interface
    envs = gym.vector.SyncVectorEnv(
        [build_env_fn(args, config, env_args)],
        autoreset_mode=AutoresetMode.SAME_STEP,  # match training semantics
    )

    # no model_path -> blue robots stay static (zero action); useful to inspect opponents alone
    agent = None
    if args.model_path is not None:
        agent = build_agent(envs, config, args.model_path, device)

    if stage.environment.opponent_strategy == "Agent" and stage.environment.opponent_model:
        # SyncVectorEnv passes the module directly — nothing crosses a process pipe
        envs.call("set_opponent_agent",
                  build_agent(envs, config, stage.environment.opponent_model, torch.device("cpu")))

    zero_action = np.zeros_like(envs.action_space.sample())

    returns, goals, passes = [], [], []
    obs, _ = envs.reset(seed=args.seed)

    for ep in range(args.episodes):
        ep_return = 0.0
        done = False
        infos = {}
        while not done:
            if agent is None:
                action = zero_action
            else:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(
                        torch.Tensor(obs).to(device), deterministic=args.deterministic
                    )
                action = action.cpu().numpy()

            obs, reward, terminations, truncations, infos = envs.step(action)
            ep_return += float(reward[0])
            done = bool(terminations[0] or truncations[0])

        # SAME_STEP autoreset stashes the finished episode's info under final_info
        ep_info = infos.get("final_info", {})
        goals.append(float(ep_info.get("episode_goal_count", [0])[0]))
        passes.append(float(ep_info.get("episode_pass_count", [0])[0]))
        length = int(ep_info["episode"]["l"][0]) if "episode" in ep_info else -1
        returns.append(ep_return)
        print(f"episode {ep + 1}/{args.episodes}: return={ep_return:.2f} "
              f"length={length} goals={goals[-1]:.0f} passes={passes[-1]:.0f}")

    print(f"\nmean return over {args.episodes} episodes: "
          f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"goals/episode: {np.mean(goals):.2f}  passes/episode: {np.mean(passes):.2f}")

    envs.close()


if __name__ == "__main__":
    main()
