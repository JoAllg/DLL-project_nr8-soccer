#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
# Runs on login node via ProxyCommand. Resolves job ID/name to a compute node
# and proxies the SSH connection via nc. All diagnostics go to stderr.
# Based on https://docs.rc.fas.harvard.edu/kb/vscode-remote-development-via-ssh-or-tunnel/#articleTOC_4
set -euo pipefail

JOB_ID="${1:-auto}"

if [[ "$JOB_ID" == "auto" || -z "$JOB_ID" ]]; then
    # Auto-select newest running job
    JOBID=$(squeue -u "$USER" -t RUNNING -h -S -V -o "%i" | head -1)
    if [[ -z "$JOBID" ]]; then
        echo "No running jobs found." >&2
        exit 1
    fi
    echo "Auto-selected job $JOBID" >&2
elif [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    # Numeric — treat as job ID, verify it exists
    JOBID="$JOB_ID"
    if ! squeue -j "$JOBID" -u "$USER" -t RUNNING -h -o "%i" | grep -q .; then
        echo "Job $JOBID not found or not running." >&2
        exit 1
    fi
else
    # Non-numeric — treat as job name
    JOBID=$(squeue -u "$USER" -t RUNNING --name="$JOB_ID" -h -S -V -o "%i" | head -1)
    if [[ -z "$JOBID" ]]; then
        echo "No running job named '$JOB_ID' found." >&2
        exit 1
    fi
    echo "Resolved job name '$JOB_ID' to job $JOBID" >&2
fi

NODELIST=$(squeue -j "$JOBID" -h -o "%N")
NODE=$(scontrol show hostnames "$NODELIST" | head -1)
echo "Connecting to node $NODE (job $JOBID)..." >&2
exec srun --jobid="$JOBID" --overlap --unbuffered nc localhost 22
