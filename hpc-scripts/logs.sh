#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

echo "### Your recent jobs:"
mapfile -t JOB_LINES < <(sacct -u "$USER" -X -n -P -S "$(date -d '30 days ago' +%Y-%m-%d)" \
    -o JobID,JobName,Partition,State,Start,NodeList | tac | head -n 15)

if [ ${#JOB_LINES[@]} -eq 0 ]; then
    echo "No jobs found."
    exec bash -i
fi

JOBS=()
printf "  #)  %-10s %-20s %-12s %-12s %-19s %s\n" "JOBID" "NAME" "PARTITION" "STATE" "START" "NODELIST"
for i in "${!JOB_LINES[@]}"; do
    IFS='|' read -r id name part state start nodelist <<< "${JOB_LINES[$i]}"
    JOBS+=("$id")
    printf "  %d)  %-10s %-20s %-12s %-12s %-19s %s\n" $((i+1)) "$id" "$name" "$part" "$state" "$start" "$nodelist"
done

echo ""
read -p "Select job [1]: " SEL
SEL=${SEL:-1}
JOBID=${JOBS[$((SEL-1))]}

STATE=$(sacct -j "$JOBID" -X -n -o State | awk '{print $1}')
LOGFILE="$LOG_DIR/slurm-$JOBID.out"

case "$STATE" in
    RUNNING|COMPLETING)
        if [ -f "$LOGFILE" ]; then
            echo "### Following live log for job $JOBID (Ctrl+C to stop following, F to resume, q to quit)..."
            less +F "$LOGFILE"
        else
            echo "Log file not found yet at $LOGFILE"
        fi
        ;;
    PENDING)
        echo "Job $JOBID has not started yet — no log available."
        ;;
    *)
        if [ -f "$LOGFILE" ]; then
            less "$LOGFILE"
        else
            echo "No log captured for job $JOBID at $LOGFILE."
            echo "(Only jobs submitted via hpc.sbatch write a log file; interactive"
            echo " hpc.srun/hpc.salloc/hpc.login sessions print straight to the terminal.)"
        fi
        ;;
esac
