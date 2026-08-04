"""
Disclaimer: This script was written by code prompting with the Claude Code Sonnet & Opus 5.

Run a saved agent checkpoint in the environment, without training.

Usage:
    python simulate.py --model-path runs/.../stage0_explore.cleanrl_model \
                       --config v3.yml --stage-name explore --episodes 10

    # live rendered window (runs straight through; ESC quits, SPACE pauses, RIGHT skips):
    python simulate.py --model-path ... --config v3.yml --render

    # video capture instead:
    python simulate.py --model-path ... --config v3.yml --capture-video --video-dir videos/eval
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame
import torch
import tyro
import gymnasium as gym
from gymnasium.vector import AutoresetMode

import rsoccer_gym  # noqa: F401
import myenvs  # noqa: F401
from rsoccer_gym.Render import SSLRenderField

from agent import Agent
from config import load_config
from scene import open_window, present


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
    """open a live window and render each step as it happens (resizable, scales with the window)"""
    fullscreen: bool = False
    """with --render: open the window fullscreen instead of resizable"""
    scale: int = 300
    """render resolution in pixels per metre (rSoccer's own default is 100)"""
    supersample: int = 2
    """with --render: render this far above --scale and downsample, for antialiased edges"""
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


def poll_events():
    """Drain the pygame queue -> "quit" (ESC / close), "pause" (SPACE),
    "skip" (RIGHT arrow) or None."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if event.key == pygame.K_SPACE:
                return "pause"
            if event.key == pygame.K_RIGHT:
                return "skip"
    return None


def wait_for_key():
    """Block while paused until SPACE (resume) or ESC/close; keeps the window responsive."""
    while True:
        key = poll_events()
        if key is not None:
            return key
        pygame.time.wait(50)


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

    # px/m the field, robots and ball are all drawn from; supersampling only helps
    # the live window, which downsamples on present -- RecordVideo would just get
    # oversized frames.
    SSLRenderField._scale = args.scale * (args.supersample if args.render else 1)

    # single env, but Agent reads single_observation_space/single_action_space
    # and expects a batch dim, so keep the vector interface
    envs = gym.vector.SyncVectorEnv(
        [build_env_fn(args, config, env_args)],
        autoreset_mode=AutoresetMode.SAME_STEP,  # match training semantics
    )

    render_env = None
    if args.render:
        render_env = envs.envs[0].unwrapped
        open_window(render_env, args.fullscreen, caption="SSL Environment")
        print("[render] ESC / close window: quit   SPACE: pause/resume   RIGHT: skip episode")

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

    quit_requested = False
    for ep in range(args.episodes):
        ep_return = 0.0
        done = False
        skipped = False
        infos = {}
        while not done:
            if args.render:
                key = poll_events()
                if key == "pause":
                    print("  paused — SPACE to resume, RIGHT to skip, ESC to quit")
                    key = wait_for_key()
                if key == "quit":
                    quit_requested = True
                    break
                if key == "skip":
                    skipped = True
                    break

            if agent is None:
                action = zero_action
            else:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(
                        torch.Tensor(obs).to(device), deterministic=args.deterministic
                    )
                action = action.cpu().numpy()

            obs, reward, terminations, truncations, infos = envs.step(action)
            if args.render:
                present(render_env)  # env.step drew into the offscreen surface
            ep_return += float(reward[0])
            done = bool(terminations[0] or truncations[0])

        if quit_requested:
            print("quit requested — aborting the current episode")
            break

        if skipped:
            # cut short by hand, so no final_info and no comparable stats:
            # leave it out of the summary and start a fresh episode
            print(f"episode {ep + 1}/{args.episodes}: skipped (return so far {ep_return:.2f})")
            obs, _ = envs.reset()
            continue

        # SAME_STEP autoreset stashes the finished episode's info under final_info
        ep_info = infos.get("final_info", {})
        goals.append(float(ep_info.get("episode_goal_count", [0])[0]))
        passes.append(float(ep_info.get("episode_pass_count", [0])[0]))
        length = int(ep_info["episode"]["l"][0]) if "episode" in ep_info else -1
        returns.append(ep_return)
        print(f"episode {ep + 1}/{args.episodes}: return={ep_return:.2f} "
              f"length={length} goals={goals[-1]:.0f} passes={passes[-1]:.0f}")

    if not returns:
        print("\nno episode completed")
    else:
        print(f"\nmean return over {len(returns)} episodes: "
              f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"goals/episode: {np.mean(goals):.2f}  passes/episode: {np.mean(passes):.2f}")

    envs.close()


if __name__ == "__main__":
    main()
