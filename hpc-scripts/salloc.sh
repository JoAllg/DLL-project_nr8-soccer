#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

cd "$REPO_DIR" || { echo "Repository not found at $REPO_DIR"; exec bash -i; }

# Partition metadata: "cpu_flag|max_time|description"
declare -A PART_INFO
PART_INFO[dev_cpu]="true|00:30:00|AMD EPYC 9454, 96 cores, 384 GiB RAM, 30min max"
PART_INFO[cpu]="true|72:00:00|AMD EPYC 9454, 96 cores, 384 GiB RAM"
PART_INFO[highmem]="true|72:00:00|AMD EPYC 9454, 96 cores, 2304 GiB RAM"
PART_INFO[dev_gpu_h100]="false|00:30:00|H100 94GiB, max 4 GPUs, 30min max"
PART_INFO[gpu_h100_short]="false|00:30:00|H100 94GiB, max 4 GPUs, 30min max"
PART_INFO[gpu_h100]="false|72:00:00|H100 94GiB, max 4 GPUs"
PART_INFO[gpu_mi300]="false|72:00:00|AMD MI300A, 128GiB HBM3, max 4 GPUs"
PART_INFO[dev_cpu_il]="true|00:30:00|Intel Ice Lake, 64 cores, 256 GiB RAM, 30min max"
PART_INFO[cpu_il]="true|72:00:00|Intel Ice Lake, 64 cores, 256 GiB RAM"
PART_INFO[dev_gpu_a100_il]="false|00:30:00|A100 80GiB Ice Lake, max 4 GPUs, 30min max"
PART_INFO[gpu_a100_il]="false|48:00:00|A100 80GiB Ice Lake, max 4 GPUs"
PART_INFO[gpu_h100_il]="false|48:00:00|H100 80GiB Ice Lake, max 4 GPUs"
PART_INFO[gpu_a100_short]="false|00:30:00|A100 40GiB, max 4 GPUs, 30min max"

# Parse sinfo_t_idle for partitions with idle nodes
AVAIL_PARTS=()
declare -A IDLE_COUNTS

while IFS= read -r line; do
    part=$(awk '{print $2}' <<< "$line")
    count=$(awk '{print $4}' <<< "$line")
    if [ "$count" -gt 0 ]; then
        AVAIL_PARTS+=("$part")
        IDLE_COUNTS[$part]=$count
    fi
done < <(sinfo_t_idle)

# Sort: CPU normal, CPU short, CPU dev, GPU normal, GPU short, GPU dev
SORTED_PARTS=()
for pass in 1 2 3 4 5 6; do
    for part in "${AVAIL_PARTS[@]}"; do
        is_cpu=false; is_gpu=false; is_short=false; is_dev=false
        [[ "$part" == *cpu* || "$part" == *mem* ]] && is_cpu=true
        [[ "$part" == *gpu* ]] && is_gpu=true
        [[ "$part" == *short* ]] && is_short=true
        [[ "$part" == *dev* ]] && is_dev=true
        case $pass in
            1) $is_cpu && ! $is_short && ! $is_dev && SORTED_PARTS+=("$part") ;;
            2) $is_cpu && $is_short && SORTED_PARTS+=("$part") ;;
            3) $is_cpu && $is_dev && SORTED_PARTS+=("$part") ;;
            4) $is_gpu && ! $is_short && ! $is_dev && SORTED_PARTS+=("$part") ;;
            5) $is_gpu && $is_short && SORTED_PARTS+=("$part") ;;
            6) $is_gpu && $is_dev && SORTED_PARTS+=("$part") ;;
        esac
    done
done
AVAIL_PARTS=("${SORTED_PARTS[@]}")

if [ ${#AVAIL_PARTS[@]} -eq 0 ]; then
    echo "No partitions with idle nodes available."
    exec bash -i
fi

echo "### Available partitions:"
for i in "${!AVAIL_PARTS[@]}"; do
    part="${AVAIL_PARTS[$i]}"
    count="${IDLE_COUNTS[$part]}"
    info="${PART_INFO[$part]}"
    if [ -n "$info" ]; then
        IFS='|' read -r _cpu _time desc <<< "$info"
        printf "  %2d)  %-18s %2d idle  (%s)\n" $((i+1)) "$part" "$count" "$desc"
    else
        printf "  %2d)  %-18s %2d idle\n" $((i+1)) "$part" "$count"
    fi
done

echo ""
read -p "Select partition [1]: " SEL
SEL=${SEL:-1}
PARTITION="${AVAIL_PARTS[$((SEL-1))]}"

# Look up metadata for selected partition
info="${PART_INFO[$PARTITION]}"
if [ -n "$info" ]; then
    IFS='|' read -r IS_CPU MAX_TIME _desc <<< "$info"
else
    IS_CPU=false
    MAX_TIME="72:00:00"
fi

if [ "$IS_CPU" = "true" ]; then
    CPUS=1
    read -p "CPUs [$CPUS]: " INPUT_CPUS
    CPUS=${INPUT_CPUS:-$CPUS}
else
    GPUS=1
    read -p "GPUs [$GPUS]: " INPUT_GPUS
    GPUS=${INPUT_GPUS:-$GPUS}
fi

MEM_FLAG=""
if [ "$PARTITION" = "highmem" ]; then
    MEM_DEFAULT="380001mb"
    read -p "Memory [$MEM_DEFAULT]: " INPUT_MEM
    MEM=${INPUT_MEM:-$MEM_DEFAULT}
    MEM_FLAG="--mem=$MEM"
fi

read -p "Time [$MAX_TIME]: " INPUT_TIME
TIME=${INPUT_TIME:-$MAX_TIME}

read -p "Job name [none]: " INPUT_JOBNAME
JOBNAME=${INPUT_JOBNAME:-none}

echo ""
if [ "$IS_CPU" = "true" ]; then
    echo "### Requesting: $PARTITION, ${CPUS} CPU(s), $TIME, job=$JOBNAME"
    SALLOC_CMD="salloc -p \"$PARTITION\" --ntasks=\"$CPUS\" $MEM_FLAG --time=\"$TIME\" --job-name=\"$JOBNAME\" --chdir=\"$REPO_DIR\""
else
    echo "### Requesting: $PARTITION, ${GPUS} GPU(s), $TIME, job=$JOBNAME"
    SALLOC_CMD="salloc -p \"$PARTITION\" --gres=gpu:\"$GPUS\" --time=\"$TIME\" --job-name=\"$JOBNAME\" --chdir=\"$REPO_DIR\""
fi

# The salloc job runs inside a tmux session so it survives SSH disconnects.
# To stop the allocation, either use `scancel <jobid>` or reattach to the
# tmux session and exit the shell.
SESSION="salloc-${PARTITION}-$$"
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "$SALLOC_CMD" Enter
echo "### tmux session: $SESSION"
echo "### Reattach with: tmux attach -t $SESSION"
tmux attach -t "$SESSION"
