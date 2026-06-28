#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

echo "### Your jobs:"
JOBS=($(squeue -u "$USER" -h -o "%i"))
if [ ${#JOBS[@]} -eq 0 ]; then
    echo "No jobs found."
    exec bash -i
fi

mapfile -t JOB_LINES < <(squeue -u "$USER" -h -o "%i|%P|%j|%T|%M|%R")
printf "  #)  %-10s %-15s %-20s %-8s %-10s %s\n" "JOBID" "PARTITION" "NAME" "STATE" "TIME" "NODELIST"
for i in "${!JOB_LINES[@]}"; do
    IFS='|' read -r id part name state time nodelist <<< "${JOB_LINES[$i]}"
    printf "  %d)  %-10s %-15s %-20s %-8s %-10s %s\n" $((i+1)) "$id" "$part" "$name" "$state" "$time" "$nodelist"
done

echo ""
read -p "Select job to cancel: " SEL
JOBID=${JOBS[$((SEL-1))]}

echo ""
echo "Cancelling job $JOBID..."
scancel "$JOBID"

echo ""
echo "### Remaining jobs:"
squeue -u "$USER" -l
