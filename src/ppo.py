# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import time
from datetime import datetime

# Auto-select headless rendering backend if no display is available
if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    if os.path.exists("/dev/nvidia0"):
        os.environ["MUJOCO_GL"] = "egl"   # GPU offscreen rendering
    else:
        os.environ["MUJOCO_GL"] = "osmesa" # CPU software rendering

from dataclasses import dataclass
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from agent import Agent
from scheduler.CosineWarmupScheduler import get_cosine_schedule_with_warmup

# import environments
import rsoccer_gym  # noqa: F401
import myenvs  # noqa: F401

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "SSLSingleRobot-v0"
    """the id of the environment"""
    total_timesteps: int = 800000000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = True
    """Toggle the cosine-with-warmup learning rate schedule for policy and value networks"""
    warmup_ratio: float = 0.1
    """fraction of total optimizer steps used for linear LR warmup at the start of each cycle (total warmup = num_cycles * this)"""
    min_lr_ratio: float = 1e-8
    """the LR floor, as a fraction of learning_rate, that the cosine schedule decays to"""
    num_cycles: int = 1
    """number of warmup+cosine-decay LR cycles across training (1 = single cycle, no restarts)"""
    cycle_decay: float = 0.5
    """peak-LR multiplier applied at each LR restart (0.5 halves the max LR every cycle); 1.0 = no decay"""
    weight_decay: float = 0.01
    """AdamW weight decay (applied to matrix weights only, see optimizer setup)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """initial coefficient of the entropy bonus (annealed linearly to final_ent_coef)"""
    # entropy-coefficient annealing, after cleanrl ppo_trxl.py (init/final_ent_coef):
    # a decaying entropy bonus buys exploration early (finding ball/goal at all)
    # without keeping the policy noisy late in training
    final_ent_coef: float = 0.0
    """final entropy coefficient after linear annealing from ent_coef over total_timesteps"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5 # Best values between 0.5 to 0.1
    """the alpha parameter for RPO"""

    # Agent architecture arguments
    agent_type: Literal["mlp", "transformer"] = "mlp"
    """the actor/critic architecture: CleanRL MLP baseline or per-entity-token transformer"""
    d_model: int = 256
    """(transformer) the model/embedding dimension"""
    n_layers: int = 4
    """(transformer) the number of encoder layers"""
    n_heads: int = 8
    """(transformer) the number of attention heads (must divide d_model)"""
    ff_dim: int = 512
    """(transformer) the feedforward dimension inside encoder layers"""
    dropout: float = 0.0
    """(transformer) dropout inside encoder layers"""
    critic_pooling: Literal["mean", "max", "attention"] = "mean"
    """(transformer) how the critic pools entity tokens into a scalar value"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, flatten=True):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # flatten only for the MLP (and dm_control's Dict obs); the transformer
        # agent skips it — flattening would destroy the per-entity structure it
        # re-parses into tokens
        if flatten:
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # no NormalizeObservation/clip here: rSoccer envs (VSS-v0) already normalize
        # observations themselves, so a running-stats wrapper on top would rescale
        # an already-bounded signal against a moving mean/variance for no benefit
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.track:
        # silence DeprecationWarnings from wandb's Sentry telemetry (its own crash
        # reporting only; unrelated to our logging)
        os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # off: wandb's gym integration patches RecordVideo.close to read
            # `self.enabled`, which gymnasium 1.x lacks, crashing on env close
            monitor_gym=False,
            save_code=True,
        )
        if args.capture_video:
            # sync_tensorboard owns the wandb step, so give videos an explicit
            # x-axis via a custom step metric (see utils.log_new_videos)
            wandb.define_metric("media/video_step")
            wandb.define_metric("media/video", step_metric="media/video_step")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    utils.set_seed(args.seed, args.torch_deterministic)

    # env setup
    # AsyncVectorEnv forks worker subprocesses (default multiprocessing start
    # method on Linux). Construct it before utils.get_device() touches CUDA -
    # forking a process with an already-initialized CUDA context is unsafe and
    # can deadlock a worker later (e.g. on its first subprocess spawn for video
    # encoding), rather than failing immediately.
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, args.gamma,
                flatten=args.agent_type == "mlp",
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print("envs.single_action_space.shape:", envs.single_action_space.shape)
    print("envs.single_observation_space.shape:", envs.single_observation_space.shape)

    device, _ = utils.get_device(cuda)

    agent = Agent(
        envs,
        args.rpo_alpha,
        agent_type=args.agent_type,
        env_id=args.env_id,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        critic_pooling=args.critic_pooling,
    ).to(device)
    # AdamW instead of Adam, after cleanrl ppo_trxl.py: decoupled weight
    # decay is the standard transformer regularizer. Decay only matrix weights — biases,
    # LayerNorms, actor_logstd and the PMA pool_query are excluded (decaying actor_logstd
    # toward 0 would fight the learned exploration schedule).
    no_decay = {"actor_logstd", "critic.pool_query"}
    decay_params = [p for n, p in agent.named_parameters() if p.ndim >= 2 and n not in no_decay]
    other_params = [p for n, p in agent.named_parameters() if p.ndim < 2 or n in no_decay]
    optimizer = optim.AdamW(  # pyright: ignore[reportPrivateImportUsage]
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": other_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        eps=1e-5,
    )
    # cosine-with-warmup instead of CleanRL's linear anneal: linear warmup
    # protects the fresh transformer from large early Adam steps, then cosine
    # decays onto an LR floor (see CosineWarmupScheduler.py)
    scheduler = None
    if args.anneal_lr:
        total_optimizer_steps = args.num_iterations * args.update_epochs * args.num_minibatches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * total_optimizer_steps),
            num_training_steps=total_optimizer_steps,
            num_cycles=args.num_cycles,
            cycle_decay=args.cycle_decay,
            min_lr_ratio=args.min_lr_ratio,
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # For wandb video upload
    video_dir = f"videos/{run_name}"
    videos_seen: set[str] = set()
    videos_sizes: dict[str, int] = {}

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # entropy-coefficient annealing, after cleanrl ppo_trxl.py (init/final_ent_coef):
        # a decaying entropy bonus buys exploration early (finding ball/goal at all)
        # without keeping the policy noisy late in training
        frac = min(global_step / args.total_timesteps, 1.0)
        ent_coef = args.ent_coef + (args.final_ent_coef - args.ent_coef) * frac

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            #fixed logging
            if "episode" in infos:
                for i, r in enumerate(infos["episode"]["r"]):
                    if infos["_episode"][i]:
                        print(f"global_step={global_step}, episodic_return={r:.2f}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.track and args.capture_video:
            utils.log_new_videos(video_dir, videos_seen, videos_sizes, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = utils.evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            agent_type=args.agent_type,
            device=device,
            gamma=args.gamma,
            rpo_alpha=args.rpo_alpha,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            critic_pooling=args.critic_pooling,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()

    if args.track and args.capture_video:
        # final flush: two polls to catch the last video when its creation is done (needs a stable-size pass)
        utils.log_new_videos(video_dir, videos_seen, videos_sizes, global_step)
        utils.log_new_videos(video_dir, videos_seen, videos_sizes, global_step)

    writer.close()
