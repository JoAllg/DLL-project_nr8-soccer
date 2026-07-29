# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import io
import os
import resource
import time
import warnings
import signal
from datetime import datetime

from typing import Optional, Annotated

# Pin the BLAS/OMP thread pools *before* numpy/torch load — OpenBLAS reads these at
# import time. AsyncVectorEnv workers (context="spawn") inherit them, so each of the
# num_envs worker processes runs single-threaded instead of sizing its pool to the
# core count. The trainer restores its own parallelism via torch.set_num_threads below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Each AsyncVectorEnv worker costs pipes plus a shared-memory fd per obs buffer; a
# 1024 soft limit runs out well before num_envs does. Raise to the hard limit.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < _hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pygame.pkgdata",
)
warnings.filterwarnings(
    "ignore",
    message=".*Overwriting existing videos.*",
    category=UserWarning,
    module="gymnasium.wrappers.rendering",
)

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tyro
from dataclasses import asdict, field, make_dataclass
from torch.utils.tensorboard.writer import SummaryWriter
import wandb

import utils
from agent import Agent
from scheduler.CosineWarmupScheduler import get_cosine_schedule_with_warmup

# import environments
import rsoccer_gym  # noqa: F401
import myenvs  # noqa: F401
from config import load_config, override_with_args, flatten_dict, Config

def _default(name: str):
    field_info = Config.model_fields[name]
    if field_info.default_factory is not None:
        return field_info.default_factory
    return field_info.default


def _build_args_fields(cli_fields: dict[str, str]):
    fields_spec = []
    for name, help_text in cli_fields.items():
        annotation = Config.model_fields[name].annotation
        annotated_type = Annotated[annotation, tyro.conf.arg(help=help_text)]
        fields_spec.append(
            (name, annotated_type, field(default_factory=(lambda n=name: _default(n))))
        )
    return fields_spec

CLI_FIELDS: dict[str, str] = {
    "exp_name": "the name of this experiment",
    "seed": "seed of the experiment",
    "torch_deterministic": "if toggled, `torch.backends.cudnn.deterministic=False`",
    "cuda": "if toggled, cuda will be enabled by default",
    "track": "if toggled, this experiment will be tracked with Weights and Biases",
    "wandb_project_name": "the wandb's project name",
    "wandb_entity": "the entity (team) of wandb's project",
    "capture_video": "whether to capture videos of the agent performances (check out `videos` folder)",
    "save_model": "whether to save model into the `runs/{run_name}` folder",
    
    # Algorithm specific arguments
    "env_id": "the id of the environment",
    "total_timesteps": "total timesteps of the experiments",
    "num_envs": "the number of parallel game environments (0 -> envs_per_cpu * available CPUs)",
    "envs_per_cpu": "multiplier used to derive num_envs from the available CPU count when num_envs is unset",
    "num_steps": "the number of steps to run in each environment per policy rollout (overrides every stage's `steps`)",
    "num_minibatches": "the number of mini-batches",
    "update_epochs": "the K epochs to update the policy",
    "learning_rate": "the learning rate of the optimizer",
    "anneal_lr": "Toggle the cosine-with-warmup learning rate schedule for policy and value networks",
    "warmup_ratio": "fraction of total optimizer steps used for linear LR warmup at the start of each cycle (total warmup = num_cycles * this)",
    "min_lr_ratio": "the LR floor, as a fraction of learning_rate, that the cosine schedule decays to",
    "num_cycles": "number of warmup+cosine-decay LR cycles across training (1 = single cycle, no restarts)",
    "cycle_decay": "peak-LR multiplier applied at each LR restart (0.5 halves the max LR every cycle); 1.0 = no decay",
    "weight_decay": "AdamW weight decay (applied to matrix weights only, see optimizer setup)",
    "gamma": "the discount factor gamma",
    "gae_lambda": "the lambda for the general advantage estimation",
    "norm_adv": "Toggles advantages normalization",
    "clip_coef": "the surrogate clipping coefficient",
    "clip_vloss": "Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    "ent_coef": "initial coefficient of the entropy bonus (annealed linearly to final_ent_coef)",
    "final_ent_coef": "final entropy coefficient after linear annealing from ent_coef over total_timesteps",
    "vf_coef": "coefficient of the value function",
    "max_grad_norm": "the maximum norm for the gradient clipping",
    "target_kl": "the target KL divergence threshold",
    "rpo_alpha": "the alpha parameter for RPO",

    # Agent architecture arguments
    "agent_type": "the actor/critic architecture: CleanRL MLP baseline or per-entity-token transformer",
    "d_model": "(transformer) the model/embedding dimension",
    "n_layers": "(transformer) the number of encoder layers",
    "n_heads": "(transformer) the number of attention heads (must divide d_model)",
    "ff_dim": "(transformer) the feedforward dimension inside encoder layers",
    "dropout": "(transformer) dropout inside encoder layers",
    "critic_pooling": "(transformer) how the critic pools entity tokens into a scalar value",

    # to be filled in runtime
    "config": "path to yaml file providing configuration training stages and environments",
    "stage_name": "which stage of the config file should be executed. None: execute all stages in Order",
    "rewards": "name of a reward_templates entry that overrides every run stage's reward weights (e.g. --rewards coop)",
    "load_model": "Path to a .cleanrl_model checkpoint to load before the first training stage.",
    "save_steps": "How often the model should be saved in between (0 -> only save at the end)",
}

Args = make_dataclass(
    "Args",
    _build_args_fields(CLI_FIELDS)
)


def get_explicit_args(args_cls, parsed_args) -> dict:
    defaults = tyro.cli(args_cls, args=[])
    parsed_dict = asdict(parsed_args)
    default_dict = asdict(defaults)
    explicit = {
        k: v for k, v in parsed_dict.items()
        if v != default_dict.get(k)
    }
    return explicit


def upload_model_artifact(model_path: str, stage_id: int, stage_name: str, global_step: int, is_final: bool):
    if wandb.run is None:
        return
    artifact = wandb.Artifact(
        name=f"model-stage{stage_id}-{stage_name}",
        type="model",
        metadata={
            "stage_id": stage_id,
            "stage_name": stage_name,
            "global_step": global_step,
            "final": is_final,
        },
    )
    artifact.add_file(model_path)
    aliases = ["latest"]
    if is_final:
        aliases.append("final")
    wandb.log_artifact(artifact, aliases=aliases)


def make_env(env_id, idx, capture_video, video_folder, gamma, flatten=True,
             environment_args: Optional[dict] = None):
    def thunk():
        # runs inside the worker process: keep env stepping (and any opponent-agent
        # inference) single-threaded so workers don't oversubscribe the cores
        torch.set_num_threads(1)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **(environment_args
                                                               or {}))
            env = gym.wrappers.RecordVideo(env, video_folder)
        else:
            env = gym.make(env_id, **(environment_args or {}))
        if flatten:
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))  # pyright: ignore[reportArgumentType, reportCallIssue]
        return env

    return thunk


def run_stage(
        stage,
        stage_id: int,
        envs,
        iterations,
        agent,
        optimizer,
        device,
        writer,
        is_main,
        is_distributed,
        local_rank,
        local_num_envs,
        local_batch_size,
        local_minibatch_size,
        stage_num_minibatches,
        global_step: int,
        start_time: float,) -> int:
    """Runs training loop of one stage and returns updated global_step"""

    print(f"steps: {stage.steps} , iterations: {iterations} ")

    agent.set_env(envs)

    def save_checkpoint(sig=None, frame=None):
        print(f"saving_model checkpoint...")
        model_path = f"runs/{run_name}/{config.exp_name}_stage{stage_id}_{stage.name}_steps_{global_step}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"[stage {stage_id} at steps {global_step}] model saved to {model_path}")
        if is_main and config.track:
            upload_model_artifact(model_path, stage_id, stage.name, global_step, is_final=False)

    signal.signal(signal.SIGTERM, save_checkpoint)
    signal.signal(signal.SIGINT, save_checkpoint)

    scheduler = None
    if config.anneal_lr:
        total_optimizer_steps = iterations * config.update_epochs * stage_num_minibatches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_optimizer_steps),
            num_training_steps=total_optimizer_steps,
            num_cycles=config.num_cycles,
            cycle_decay=config.cycle_decay,
            min_lr_ratio=config.min_lr_ratio,
        )

    obs = torch.zeros((stage.steps, local_num_envs) + envs.single_observation_space.shape).to(device)  # pyright: ignore[reportOperatorIssue]
    actions = torch.zeros((stage.steps, local_num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((stage.steps, local_num_envs)).to(device)
    rewards = torch.zeros((stage.steps, local_num_envs)).to(device)
    dones = torch.zeros((stage.steps, local_num_envs)).to(device)
    values = torch.zeros((stage.steps, local_num_envs)).to(device)

    # For wandb video upload
    # separate subfolder per stage: RecordVideo's episode_id restarts at 0 on every
    # new stage's env, so a shared folder would let stage N+1 overwrite stage N's
    # video files (same name) before log_new_videos ever notices the new content
    video_dir = f"videos/{run_name}/stage{stage_id}_{stage.name}"
    videos_seen: set[str] = set()
    videos_sizes: dict[str, int] = {}

    env_seed = config.seed + local_rank * local_num_envs
    next_obs, _ = envs.reset(seed=env_seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(local_num_envs).to(device)

    last_save_step = 0

    # throughput is measured per stage, not per run: episode length, robot count and
    # env count all change at a stage boundary, and a run-wide average turns that step
    # change into a slow decay that looks like a leak
    stage_start_time = time.time()
    stage_start_step = global_step

    for iteration in range(1, iterations + 1):
        t_iter_start = time.time()
        # entropy-coefficient annealing, after cleanrl ppo_trxl.py (init/final_ent_coef):
        # a decaying entropy bonus buys exploration early (finding ball/goal at all)
        # without keeping the policy noisy late in training
        frac = min(global_step / config.total_timesteps, 1.0)
        ent_coef = config.ent_coef + (config.final_ent_coef - config.ent_coef) * frac

        # [MUSKAN] Rollout accumulators for scale-invariant metrics
        rollout_passes = 0
        rollout_goals = 0
        rollout_episodes = 0

        for step in range(0, stage.steps):
            global_step += config.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Time-limit bootstrapping: GAE below cuts the value target at every
            # `done`, which is correct for a goal but wrong for the max_steps
            # cutoff — the episode was still going. Fold gamma*V(s_T) into the
            # reward for truncated envs so the tail value is not thrown away.
            # Under SAME_STEP autoreset `next_obs` is already the reset obs, so
            # s_T has to come from infos["final_obs"].
            if truncations.any():
                trunc_mask = torch.as_tensor(truncations, dtype=torch.bool, device=device)
                final_obs = np.stack(infos["final_obs"][truncations])
                with torch.no_grad():
                    final_values = agent.get_value(torch.Tensor(final_obs).to(device)).flatten()
                rewards[step][trunc_mask] += config.gamma * final_values

            #fixed logging (rank 0 only; its envs are an accepted approximation for the parallel runs)
            # SAME_STEP autoreset nests the terminating step's info — where
            # RecordEpisodeStatistics writes `episode` — under `final_info`
            episode_infos = infos.get("final_info", {})
            episode_infos = infos.get("final_info", {})
            if writer is not None and "episode" in episode_infos:
                for i, r in enumerate(episode_infos["episode"]["r"]):
                    if episode_infos["_episode"][i]:
                        print(f"global_step={global_step}, episodic_return={r:.2f}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", episode_infos["episode"]["l"][i], global_step)

                    # [MUSKAN] Accumulate pass/goal counts per completed episode
                    rollout_episodes += 1
                    if "episode_pass_count" in episode_infos and episode_infos["episode_pass_count"] is not None:
                        pc = episode_infos["episode_pass_count"][i]
                        gc = episode_infos["episode_goal_count"][i]
                        if pc is not None:
                            rollout_passes += int(pc)
                        if gc is not None:
                            rollout_goals += int(gc)

                    # [MUSKAN] Per-reward-term breakdown
                    if "episode_reward_breakdown" in episode_infos:
                        for name, arr in episode_infos["episode_reward_breakdown"].items():
                            if not name.startswith("_"):
                                writer.add_scalar(f"rewards/{name}", arr[i], global_step)

            # save checkpoint
            if is_main and config.save_steps > 0 and global_step - last_save_step >= config.save_steps:
                save_checkpoint()
                last_save_step = global_step

        t_rollout_end = time.time()
        # [MUSKAN] Log rollout-level metrics after each rollout — scale invariant
        if writer is not None and rollout_episodes > 0:
            writer.add_scalar("charts/passes_per_episode", rollout_passes / rollout_episodes, global_step)
            writer.add_scalar("charts/goals_per_episode", rollout_goals / rollout_episodes, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(stage.steps)):
                if t == stage.steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # pyright: ignore[reportOperatorIssue]
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        t_update_start = time.time()

        # Optimizing the policy and value network
        b_inds = np.arange(local_batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, local_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if is_distributed:
                    for param in agent.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if config.target_kl is not None:
                kl_stop = torch.tensor(float(approx_kl > config.target_kl), device=device)
                if is_distributed:
                    dist.all_reduce(kl_stop, op=dist.ReduceOp.MAX)
                if kl_stop.item():
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if is_main and writer is not None:
            writer.add_scalar("charts/stage_id", stage_id, global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            t_now = time.time()
            iter_seconds = max(t_now - t_iter_start, 1e-9)
            iter_steps = config.num_envs * stage.steps
            stage_sps = int((global_step - stage_start_step) / max(t_now - stage_start_time, 1e-9))

            print("SPS:", stage_sps)
            print("global_step:", global_step)
            print("stage_id:", stage_id)
            # mean since this stage began, so a stage boundary resets the baseline
            writer.add_scalar("charts/SPS", stage_sps, global_step)
            # this iteration only — the one that reacts immediately to a slowdown
            writer.add_scalar("charts/SPS_instant", int(iter_steps / iter_seconds), global_step)
            # Performance analysis metrics:
            writer.add_scalar("perf/rollout_seconds", t_rollout_end - t_iter_start, global_step)
            writer.add_scalar("perf/gae_seconds", t_update_start - t_rollout_end, global_step)
            writer.add_scalar("perf/update_seconds", t_now - t_update_start, global_step)
            writer.add_scalar("perf/iteration_seconds", iter_seconds, global_step)
            writer.add_scalar("perf/elapsed_hours", (t_now - start_time) / 3600, global_step)

        if is_main and config.track and config.capture_video:
            utils.log_new_videos(video_dir, videos_seen, videos_sizes, global_step)

    if config.save_model and is_main:
        print(f"saving_model...")
        model_path = f"runs/{run_name}/{config.exp_name}_stage{stage_id}_{stage.name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"[stage {stage_id}] model saved to {model_path}")
        if config.track:
            upload_model_artifact(model_path, stage_id, stage.name, global_step, is_final=True)

    if is_main and config.track and config.capture_video:
        utils.log_new_videos(video_dir, videos_seen, videos_sizes, global_step)

    return global_step


if __name__ == "__main__":
    args = tyro.cli(Args)
    explicit_args = get_explicit_args(Args, args)
    config = load_config(args.config)
    config = override_with_args(explicit_args, config)
    # rollout length: a stage's own `steps` wins, an explicit --num-steps beats both
    config.apply_num_steps(force="num_steps" in explicit_args)

    # default envs_per_cpu: 2 on GPU (cheap CPU-side env stepping vs. one shared GPU
    # for the forward/backward pass), 1 on CPU-only; an explicit --envs-per-cpu wins
    if "envs_per_cpu" not in explicit_args:
        config.envs_per_cpu = 2 if (config.cuda and torch.cuda.is_available()) else 1

    if config.rewards is not None:
        if config.rewards not in config.reward_templates:
            raise ValueError(
                f"--rewards '{config.rewards}' not found in reward_templates; "
                f"available: {sorted(config.reward_templates)}"
            )
        template = config.reward_templates[config.rewards]
        for stage in config.stages:
            stage.environment.rewards = dict(template)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = local_rank == 0

    # scale the env count to the machine unless it was given explicitly. Rounded down to
    # a multiple of world_size to keep the num_envs % world_size assert below satisfied.
    def envs_from_cpus(envs_per_cpu: int) -> int:
        cpus = utils.available_cpus()
        return max(world_size, (cpus * envs_per_cpu) // world_size * world_size)

    num_envs_explicit = config.num_envs > 0
    if not num_envs_explicit:
        config.num_envs = envs_from_cpus(config.envs_per_cpu)
        print(f"cpus available: {utils.available_cpus()} | num_envs: {config.num_envs} "
              f"(auto: envs_per_cpu {config.envs_per_cpu})")
    else:
        print(f"cpus available: {utils.available_cpus()} | num_envs: {config.num_envs} (explicit)")

    default_num_envs = config.num_envs

    def stage_env_count(s) -> int:
        """Per-stage env count: an explicit --num-envs wins, then the stage's own
        num_envs / envs_per_cpu, else the global default."""
        if num_envs_explicit:
            return config.num_envs
        if s.num_envs is not None:
            return s.num_envs
        if s.envs_per_cpu is not None:
            return envs_from_cpus(s.envs_per_cpu)
        return default_num_envs

    # preview of the per-stage batch shapes the loop below recomputes, so a bad
    # num_envs / steps / num_minibatches combination is visible before training starts
    for s in config.stages:
        s_mb = s.num_minibatches if s.num_minibatches is not None else config.num_minibatches
        s_envs = stage_env_count(s)
        s_batch = s_envs * s.steps
        print(f"  stage '{s.name}': num_envs={s_envs} steps={s.steps} batch_size={s_batch} "
              f"num_minibatches={s_mb} minibatch_size={s_batch // s_mb}")

    # NOTE: batch_size / minibatch_size / num_minibatches are computed per-stage,
    # inside the stage loop below, since each stage may have its own `steps` and
    # (optionally) its own `num_minibatches` override. There is no longer a
    # global pre-loop computation — it was dead code referencing a nonexistent
    # config.num_steps and was overwritten before first use anyway.

    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if config.stage_name:
        stage_ids = config.get_stages_from_name(config.stage_name)

    steps = sum(s.steps if s.steps is not None else config.num_envs for s in config.stages)
    config.total_timesteps = sum(stage.total_steps for stage in config.stages)

    writer = None
    if is_main:
        if config.track:
            os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
            wandb.init(
                project=config.wandb_project_name,
                entity=config.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=False,
                save_code=True,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            if config.capture_video:
                wandb.define_metric("media/video", step_metric="global_step")
        writer = SummaryWriter(f"runs/{run_name}")
        config_model = config.model_dump()
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join(
                f"|{key}|{value}|" for key, value in flatten_dict(config_model).items()
            )),
        )
        if is_main and config.track:
            wandb.config.update(config_model, allow_val_change=True)

    utils.set_seed(config.seed, config.torch_deterministic)

    stage_ids = config.get_stages_from_name(config.stage_name)

    agent = None
    agent_opponent = None
    optimizer = None
    device = None
    global_step = 0
    start_time = time.time()

    for stage_id in stage_ids:
        stage = config.stages[stage_id]
        env_args = stage.environment.model_dump()

        # a stage may run fewer/more envs than the rest (e.g. agent opponents cost
        # extra memory per worker); run_stage reads config.num_envs for global_step
        config.num_envs = stage_env_count(stage)

        # calculate iterations from total_steps per stage
        # total_steps = iterations * num_envs * stage.steps
        iterations = stage.total_steps // (config.num_envs * stage.steps)

        print(f"running stage: {stage_id} : {stage.name}")

        assert config.num_envs % world_size == 0, "num_envs must be divisible by world_size"
        local_num_envs = config.num_envs // world_size

        stage_num_minibatches = (
            stage.num_minibatches if stage.num_minibatches is not None else config.num_minibatches
        )

        local_batch_size = local_num_envs * stage.steps
        assert local_batch_size % stage_num_minibatches == 0, (
            f"stage '{stage.name}': local batch size ({local_batch_size}) must be divisible "
            f"by num_minibatches ({stage_num_minibatches})"
        )
        local_minibatch_size = local_batch_size // stage_num_minibatches
        config.batch_size = int(config.num_envs * stage.steps)
        config.minibatch_size = int(config.batch_size // stage_num_minibatches)

        utils.set_seed(config.seed, config.torch_deterministic)

        video_folder = f"videos/{run_name}/stage{stage_id}_{stage.name}"
        envs = gym.vector.AsyncVectorEnv(
            [
                make_env(config.env_id, i, config.capture_video and is_main, video_folder, config.gamma,
                          flatten=config.agent_type == "mlp", environment_args=env_args)
                for i in range(local_num_envs)
            ],
            context="spawn", # spawn new process each time otherwise deadlock at 2nd stage
            # gymnasium's NEXT_STEP default spends a whole extra step resetting,
            # which lands in the rollout as a fake transition (terminal obs, ignored
            # action, reward 0) that GAE then trains on. SAME_STEP resets in place
            # and hands back the real final obs via infos["final_obs"], restoring the
            # `dones[t] marks a reset obs` semantics this loop's GAE assumes.
            observation_mode='different',
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        if agent is None:
            if is_distributed:
                if not torch.cuda.is_available():
                    raise RuntimeError("Distributed launch (WORLD_SIZE>1) requires CUDA; run without torchrun instead.")
                torch.cuda.set_device(local_rank)
                dist.init_process_group("nccl")
                device = torch.device(f"cuda:{local_rank}")
            else:
                device, _ = utils.get_device(config.cuda)

            # undo the OMP_NUM_THREADS=1 pin for the trainer only (workers keep it).
            # Matters most CPU-only, where the update phase should use every core.
            torch.set_num_threads(utils.available_cpus())

            agent = Agent(
                envs, config.rpo_alpha, agent_type=config.agent_type, d_model=config.d_model,
                n_layers=config.n_layers, n_heads=config.n_heads, ff_dim=config.ff_dim,
                dropout=config.dropout, critic_pooling=config.critic_pooling,
            ).to(device)

            if config.load_model:
                print(f"loading model weights from: {config.load_model}")
                agent.load_state_dict(torch.load(config.load_model, map_location=device))

            no_decay = {"actor_logstd", "critic.pool_query"}
            decay_params = [p for n, p in agent.named_parameters() if p.ndim >= 2 and n not in no_decay]
            other_params = [p for n, p in agent.named_parameters() if p.ndim < 2 or n in no_decay]
            optimizer = optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": config.weight_decay},
                    {"params": other_params, "weight_decay": 0.0},
                ],
                lr=config.learning_rate, eps=1e-5,
            )

            if is_distributed:
                torch.manual_seed(config.seed + local_rank)
                np.random.seed(config.seed + local_rank)

        if agent_opponent is None and stage.environment.opponent_strategy == "Agent" and stage.environment.opponent_model:
            # stays on CPU: it is pickled to every worker and only ever run there
            # (AgentOpponentPolicy.act). On device it would pickle CUDA tensors and give
            # each of the num_envs workers its own CUDA context.
            agent_opponent = Agent(
                envs, config.rpo_alpha, agent_type=config.agent_type, d_model=config.d_model,
                n_layers=config.n_layers, n_heads=config.n_heads, ff_dim=config.ff_dim,
                dropout=config.dropout, critic_pooling=config.critic_pooling,
            )
            agent_opponent.load_state_dict(
                torch.load(stage.environment.opponent_model, map_location="cpu", weights_only=True)
            )
            agent_opponent.eval()
            agent_opponent.requires_grad_(False)
            # serialize to bytes before crossing the worker pipes: pickling a live
            # nn.Module makes torch hand over every parameter storage as its own fd
            # (file_descriptor sharing), which blows past the fd limit at high num_envs.
            buf = io.BytesIO()
            torch.save(agent_opponent, buf)
            envs.call("set_opponent_agent", buf.getvalue())
            

        



            

        global_step = run_stage(
            stage, stage_id, envs, iterations, agent, optimizer, device, writer,
            is_main, is_distributed, local_rank, local_num_envs,
            local_batch_size, local_minibatch_size, stage_num_minibatches,
            global_step, start_time,
        )

        envs.close()

    if writer is not None:
        writer.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
