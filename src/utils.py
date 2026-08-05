import os
import glob
import torch
import random
import numpy as np
import gc
import gymnasium as gym


# Diclaimer log_new_videos: Generated with Claude to fix wandb missing videos from other stages
def log_new_videos(video_dir, seen, sizes, step):
    """Upload freshly-recorded RecordVideo files to W&B, tagged with `step`.

    RecordVideo (on the idx==0 env) runs inside an AsyncVectorEnv subprocess, so we
    cannot upload from its close callback — wandb.run only lives in the main process.
    Instead we poll `video_dir` from the training loop. moviepy writes the .mp4
    progressively, so a file is only uploaded once its byte size is stable across two
    polls (guarding against half-written files); `sizes` carries the previous poll's
    sizes and `seen` the filenames already uploaded.

    `step` is logged as the `global_step` field (the video's step_metric, defined
    at wandb.init as `wandb.define_metric("media/video", step_metric="global_step")`)
    rather than passed as wandb.log(step=...), which sync_tensorboard ignores.
    """
    import wandb

    if not os.path.isdir(video_dir):
        return
    for path in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
        if path in seen:
            continue
        try:
            cur = os.path.getsize(path)
        except OSError:
            continue
        if sizes.get(path) == cur and cur > 0:
            wandb.log(
                {"media/video": wandb.Video(path, format="mp4"), "global_step": step}
            )
            seen.add(path)
            sizes.pop(path, None)
        else:
            sizes[path] = cur


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.

    Args:
        seed (int): The seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(deterministic)


def available_cpus() -> int:
    """CPUs this process may actually run on.

    Uses the scheduler affinity mask rather than os.cpu_count(), so a Slurm cgroup
    (or taskset) is honoured instead of reporting the whole node's core count.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1

    nnodes = int(os.environ.get("SLURM_NNODES", 1))
    if nnodes > 1:
        print(
            f"WARNING: Slurm allocation spans {nnodes} nodes but only this node's "
            f"{cpus} CPUs are usable — request --nodes=1 --ntasks=1 --cpus-per-task=N"
        )
    return cpus


def get_device(cuda: bool):
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    elif torch.backends.mps.is_available() and cuda:
        device = torch.device("mps")
        gc.collect()
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")

    # Dataloader variables
    pin_memory = device.type == "cuda"  # Speeds up transfering dataset from CPU to GPU
    # num_workers = X

    print(f"Using device: {device}")

    return device, pin_memory  # , num_cuda_devices, num_workers


# Modified from cleanlr_utils/evals/ppo_eval.py
def evaluate(
    model_path,
    make_env,
    env_id,
    eval_episodes,
    run_name,
    Model,
    agent_type="mlp",
    device=torch.device("cpu"),
    capture_video=True,
    gamma=0.99,
    rpo_alpha=0.5,
    d_model=64,
    n_layers=2,
    n_heads=4,
    ff_dim=256,
    dropout=0.0,
    critic_pooling="mean",
):
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id, 0, capture_video, run_name, gamma, flatten=agent_type == "mlp"
            )
        ]
    )
    agent = Model(
        envs,
        rpo_alpha,
        agent_type=agent_type,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        critic_pooling=critic_pooling,
    ).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "episode" in infos:
            for i, r in enumerate(infos["episode"]["r"]):
                if infos["_episode"][i]:
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={r:.2f}"
                    )
                    episodic_returns.append(r)
        obs = next_obs

    envs.close()
    return episodic_returns
