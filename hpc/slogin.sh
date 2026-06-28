#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

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

echo ""
echo "### Logging into job $JOBID..."
exec srun --jobid="$JOBID" --chdir="$REPO_DIR" --pty bash -l
