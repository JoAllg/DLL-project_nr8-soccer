#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

cd "$REPO_DIR" || { echo "Repository not found at $REPO_DIR"; exec bash -i; }

# Make sure $REPO_DIR/.venv exists and is populated — the job's own worktree
# symlinks to it instead of building its own (see run.job.template), so this is
# what guarantees a real target is there the first time hpc.srun is ever used.
echo "### Syncing uv environment..."
uv sync --all-extras

# Show running jobs
echo "### Your running jobs:"
JOBS=($(squeue -u "$USER" -t RUNNING -h -o "%i"))
if [ ${#JOBS[@]} -eq 0 ]; then
    echo "No running jobs found. Reserve resources first: ssh hpc.salloc"
    exec bash -i
fi

mapfile -t JOB_LINES < <(squeue -u "$USER" -t RUNNING -h -o "%i|%P|%j|%T|%M|%R")
printf "  #)  %-10s %-15s %-20s %-8s %-10s %s\n" "JOBID" "PARTITION" "NAME" "STATE" "TIME" "NODELIST"
for i in "${!JOB_LINES[@]}"; do
    IFS='|' read -r id part name state time nodelist <<< "${JOB_LINES[$i]}"
    printf "  %d)  %-10s %-15s %-20s %-8s %-10s %s\n" $((i+1)) "$id" "$part" "$name" "$state" "$time" "$nodelist"
done

if [ ${#JOBS[@]} -eq 1 ]; then
    JOBID=${JOBS[0]}
    echo ""
    echo "Using job $JOBID"
else
    echo ""
    read -p "Select job [1]: " SEL
    SEL=${SEL:-1}
    JOBID=${JOBS[$((SEL-1))]}
fi

# Gather the run inputs (the branch checkout and the run itself happen on the node).
# The launcher (uv run python vs torchrun) is picked at runtime in run.job.template
# from SLURM_GPUS_ON_NODE. JOB_FILE/JOB_ARGS come from .env.remote; JOB_ARGS always
# applies, the prompted extra arguments are appended after it (last-one-wins,
# so extras override base args; booleans via their --no-* form).
export JOB_FILE=${JOB_FILE:-src/ppo.py}
export JOB_ARGS=${JOB_ARGS:-}
echo ""
echo "Run: <uv run python | torchrun if >1 GPU> $JOB_FILE $JOB_ARGS"
read -p "Extra arguments []: " EXTRA_ARGS
export EXTRA_ARGS

read -p "Branch [${BRANCH:-main}]: " INPUT_BRANCH
export BRANCH=${INPUT_BRANCH:-${BRANCH:-main}}

# Fill the job template. The #SBATCH header is ignored by srun (the allocation is
# already fixed), so those directives are left blank; the body is what runs.
export PARTITION="" JOBNAME="" TIME="" RESOURCE_DIRECTIVES="" MAIL_DIRECTIVES=""
export REPO_DIR LOG_DIR
RUN_SCRIPT="$REPO_DIR/run.sbatch"
envsubst '${PARTITION} ${JOBNAME} ${TIME} ${REPO_DIR} ${LOG_DIR} ${RESOURCE_DIRECTIVES} ${MAIL_DIRECTIVES} ${BRANCH} ${JOB_FILE} ${JOB_ARGS} ${EXTRA_ARGS}' \
    < "$SCRIPTS_DIR/run.job.template" > "$RUN_SCRIPT"

echo ""
echo "### Starting run on job $JOBID..."
srun --jobid="$JOBID" --pty bash "$RUN_SCRIPT"
