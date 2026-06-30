# HPC Scripts — BwUniCluster 3.0

Deploy and Slurm helper scripts for [BwUniCluster 3.0](https://wiki.bwhpc.de/e/BwUniCluster3.0).

## Setup

1. Copy `.env.example` → `.env` and fill in your credentials (`HPC_USER`, `HPC_IDENTITY`, etc.)
2. Edit `.env.remote` with your remote workspace/repo settings
3. Place your `deploy_ed25519` GitHub deploy key in this folder
4. Run `./deploy.sh`

`deploy.sh` sets up the local SSH config, copies scripts and the deploy key to the cluster, installs `uv`, allocates a Lustre workspace, and clones/updates the repository.

## SSH Commands

After deployment, the following SSH aliases are available:

| Command | Description |
|---|---|
| `ssh hpc` | Login shell on the login node |
| `ssh hpc.sinfo` | Show available resources and your queued jobs |
| `ssh hpc.salloc` | Interactively reserve compute nodes (runs inside tmux; stop via `scancel`, reattach via `tmux attach -t <session>`, or use `ssh hpc.login` to log into the running node) |
| `ssh hpc.sbatch` | Submit a batch job running JOB_CMD |
| `ssh hpc.srun` | Run the JOB_CMD on an allocated node |
| `ssh hpc.scancel` | Cancel a job |
| `ssh hpc.scontrol` | Show details of a job |
| `ssh hpc.login` | Open a shell on a running/allocated node (only one connection at the same time is possible) |
| `ssh hpc.tunnel` | Start a VS Code / Cursor remote tunnel on a running job (authenticates via GitHub, names the machine `codetunnel` or `cursortunnel`). Runs in tmux session. |

## Links

- https://wiki.bwhpc.de/e/BwUniCluster3.0
- https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture
- https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs
