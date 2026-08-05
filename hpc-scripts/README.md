# HPC Scripts — BwUniCluster 3.0

Deploy and Slurm helper scripts for [BwUniCluster 3.0](https://wiki.bwhpc.de/e/BwUniCluster3.0).

## Setup

1. Copy `.env.example` → `.env` and fill in your credentials (`HPC_USER`, `HPC_IDENTITY`, `JUMPHOST_USER`, `JUMPHOST_HOST`, `JUMPHOST_IDENTITY`, etc. — identity vars accept a single path or a bash array of paths)
2. Edit `.env.remote` with your remote workspace/repo settings
3. Place your `deploy_ed25519` GitHub deploy key in this folder
4. Run `./deploy.sh`

`deploy.sh` sets up the local SSH config, copies scripts and the deploy key to the cluster, installs `uv`, allocates a Lustre workspace, and clones/updates the repository.

## Run archiving (git identity & deploy key)

`hpc.srun` and `hpc.sbatch` automatically commit and push each run to a
`runs/<date>+<node>` branch. Two one-time prerequisites:

1. **Git identity** on the cluster (otherwise the commit fails):

   ```bash
   ssh hpc
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```

   (Or set it per-repo inside `$REPO_DIR`.)

2. **Write access for the deploy key** — The GitHub deploy key, need Write Access, otherwise the `git push` to the run branch is rejected.

## SSH Commands

After deployment, the following SSH aliases are available:

| Command | Description |
|---|---|
| `ssh hpc` | Login shell on the login node |
| `ssh hpc.sinfo` | Show available resources and your queued jobs |
| `ssh hpc.salloc` | Interactively reserve compute nodes (runs inside tmux; stop via `scancel`, reattach via `tmux attach -t <session>`, or use `ssh hpc.login` to log into the running node) |
| `ssh hpc.sbatch` | Submit a batch job. Prompts for a command and branch, fills the `run.job.template` Slurm script, and submits it. The job checks out the branch, `git pull`s, `uv sync`s, runs the command, then **automatically commits and pushes a `runs/<name>+<date>` branch** with the filled job script and all outputs. |
| `ssh hpc.srun` | Run a command on an already-allocated node. Same flow as `hpc.sbatch`.|
| `ssh hpc.scancel` | Cancel a job |
| `ssh hpc.scontrol` | Show details of a job |
| `ssh hpc.logs` | Pick one of your recent jobs and view its log — pages with `less` if finished, follows live with `less +F` if still running |
| `ssh hpc.login` | Open a shell on a running/allocated node |
| `ssh hpc.tunnel` | Start a VS Code / Cursor remote tunnel on a running job (authenticates via GitHub, names the machine `codetunnel` or `cursortunnel`). Runs in tmux session. |

## Links

- https://wiki.bwhpc.de/e/BwUniCluster3.0
- https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture
- https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs


## Disclaimer:
These scripts were written with the help of code prompting.