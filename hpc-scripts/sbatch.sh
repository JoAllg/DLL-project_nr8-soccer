#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

cd "$REPO_DIR" || { echo "Repository not found at $REPO_DIR"; exec bash -i; }

# Update code and environment on login node (has internet access)
echo "### Updating repository..."
git pull

echo "### Syncing uv environment..."
uv sync

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

# Parse sinfo_t_idle for all partitions
ALL_PARTS=()
declare -A IDLE_COUNTS

while IFS= read -r line; do
    part=$(awk '{print $2}' <<< "$line")
    count=$(awk '{print $4}' <<< "$line")
    [ -z "$part" ] && continue
    ALL_PARTS+=("$part")
    IDLE_COUNTS[$part]=$count
done < <(sinfo_t_idle)

# Sort: CPU normal, CPU short, CPU dev, GPU normal, GPU short, GPU dev
SORTED_PARTS=()
for pass in 1 2 3 4 5 6; do
    for part in "${ALL_PARTS[@]}"; do
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
ALL_PARTS=("${SORTED_PARTS[@]}")

if [ ${#ALL_PARTS[@]} -eq 0 ]; then
    echo "No partitions found."
    exec bash -i
fi

# Probe busy partitions for estimated availability
declare -A EST_TIMES
BUSY_PARTS=()
for part in "${ALL_PARTS[@]}"; do
    if [ "${IDLE_COUNTS[$part]}" -eq 0 ] 2>/dev/null; then
        BUSY_PARTS+=("$part")
    fi
done

if [ ${#BUSY_PARTS[@]} -gt 0 ]; then
    echo ""
    echo "Checking availability for busy partitions..."
    for part in "${BUSY_PARTS[@]}"; do
        info="${PART_INFO[$part]}"
        is_cpu="false"
        if [ -n "$info" ]; then
            IFS='|' read -r is_cpu _ _ <<< "$info"
        fi
        if [ "$is_cpu" = "true" ]; then
            mem_flag=""
            [[ "$part" == "highmem" ]] && mem_flag="--mem=380001mb"
            result=$(sbatch -p "$part" --ntasks=1 $mem_flag --time=00:10:00 --wrap="true" --test-only 2>&1)
        else
            result=$(sbatch -p "$part" --gres=gpu:1 --time=00:10:00 --wrap="true" --test-only 2>&1)
        fi
        ts=$(grep -oP 'to start at \K\S+' <<< "$result")
        if [ -n "$ts" ]; then
            EST_TIMES[$part]=$(date -d "$ts" '+%b %d %H:%M' 2>/dev/null || echo "$ts")
        else
            EST_TIMES[$part]="busy"
        fi
    done
fi

echo ""
echo "### Partitions:"
for i in "${!ALL_PARTS[@]}"; do
    part="${ALL_PARTS[$i]}"
    count="${IDLE_COUNTS[$part]}"
    info="${PART_INFO[$part]}"
    if [ -n "$info" ]; then
        IFS='|' read -r _cpu _time desc <<< "$info"
        desc_str="  ($desc)"
    else
        desc_str=""
    fi
    if [ "$count" -gt 0 ] 2>/dev/null; then
        status=$(printf "%2d idle" "$count")
    elif [ -n "${EST_TIMES[$part]}" ]; then
        status="~ ${EST_TIMES[$part]}"
    else
        status=" 0 idle"
    fi
    printf "  %2d)  %-18s %-16s%s\n" $((i+1)) "$part" "$status" "$desc_str"
done

echo ""
read -p "Select partition [1]: " SEL
SEL=${SEL:-1}
PARTITION="${ALL_PARTS[$((SEL-1))]}"

# Check if selected partition is unavailable
IS_UNAVAILABLE=false
for p in "${BUSY_PARTS[@]}"; do
    [[ "$p" == "$PARTITION" ]] && IS_UNAVAILABLE=true && break
done

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

read -p "Job name [sbatch]: " INPUT_JOBNAME
JOBNAME=${INPUT_JOBNAME:-sbatch}

# The launcher (uv run python vs torchrun) is picked at runtime in run.job.template
# from SLURM_GPUS_ON_NODE. JOB_FILE/JOB_ARGS come from .env.remote; JOB_ARGS always
# applies, the prompted extra arguments are appended after it (last-one-wins,
# so extras override base args; booleans via their --no-* form).
export JOB_FILE=${JOB_FILE:-src/ppo.py}
export JOB_ARGS=${JOB_ARGS:-}
echo "Run: <uv run python | torchrun if >1 GPU> $JOB_FILE $JOB_ARGS"
read -p "Extra arguments []: " EXTRA_ARGS
export EXTRA_ARGS

read -p "Branch [${BRANCH:-main}]: " INPUT_BRANCH
export BRANCH=${INPUT_BRANCH:-${BRANCH:-main}}

# Build the #SBATCH resource directives for the template
if [ "$IS_CPU" = "true" ]; then
    RESOURCE_DIRECTIVES="#SBATCH --ntasks=$CPUS"
    [ -n "$MEM_FLAG" ] && RESOURCE_DIRECTIVES="$RESOURCE_DIRECTIVES"$'\n'"#SBATCH $MEM_FLAG"
else
    RESOURCE_DIRECTIVES="#SBATCH --gres=gpu:$GPUS"
fi
export RESOURCE_DIRECTIVES

# Build mail directives for unavailable partitions
MAIL_DIRECTIVES=""
if $IS_UNAVAILABLE; then
    MAIL_DIRECTIVES="#SBATCH --mail-type=BEGIN"
    [ -n "$WORKSPACE_EMAIL" ] && MAIL_DIRECTIVES="$MAIL_DIRECTIVES"$'\n'"#SBATCH --mail-user=$WORKSPACE_EMAIL"
fi
export MAIL_DIRECTIVES

# Fill the job template and submit it
export PARTITION JOBNAME TIME REPO_DIR LOG_DIR
RUN_SCRIPT="$REPO_DIR/run.sbatch"
envsubst '${PARTITION} ${JOBNAME} ${TIME} ${REPO_DIR} ${LOG_DIR} ${RESOURCE_DIRECTIVES} ${MAIL_DIRECTIVES} ${BRANCH} ${JOB_FILE} ${JOB_ARGS} ${EXTRA_ARGS}' \
    < "$SCRIPTS_DIR/run.job.template" > "$RUN_SCRIPT"

echo ""
if [ "$IS_CPU" = "true" ]; then
    echo "### Submitting: $PARTITION, ${CPUS} CPU(s), $TIME, job=$JOBNAME"
else
    echo "### Submitting: $PARTITION, ${GPUS} GPU(s), $TIME, job=$JOBNAME"
fi
sbatch "$RUN_SCRIPT"

if $IS_UNAVAILABLE; then
    echo "### Note: Partition busy — you will be emailed at $WORKSPACE_EMAIL when the job starts."
fi

echo ""
echo "### Your jobs:"
squeue -u "$USER" -l
