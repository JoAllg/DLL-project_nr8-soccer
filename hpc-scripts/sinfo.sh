#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

echo "### Available resources"
sinfo_t_idle

echo ""
echo "### Your jobs"
squeue -u "$USER" -l
squeue -u "$USER" --start
