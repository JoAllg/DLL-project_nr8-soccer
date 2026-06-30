#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
source "$HOME/hpc-scripts/.env.remote" 2>/dev/null

# Editor selection
echo "### Select editor:"
echo "  1)  VSCode"
echo "  2)  Cursor"
echo ""
read -p "Select editor [1]: " EDITOR_SEL
EDITOR_SEL=${EDITOR_SEL:-1}

case "$EDITOR_SEL" in
    1)
        BINARY_NAME="code"
        DOWNLOAD_URL="https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
        EDITOR_LABEL="VSCode"
        TUNNEL_NAME="codetunnel"
        ;;
    2)
        BINARY_NAME="cursor"
        DOWNLOAD_URL="https://api2.cursor.sh/updates/download-latest?os=cli-alpine-x64"
        EDITOR_LABEL="Cursor"
        TUNNEL_NAME="cursortunnel"
        ;;
    *)
        echo "Invalid selection."
        exec bash -i
        ;;
esac

# Download CLI if missing
mkdir -p "$HOME/.local/bin"
if [ ! -x "$HOME/.local/bin/$BINARY_NAME" ]; then
    echo ""
    echo "### Downloading $EDITOR_LABEL CLI..."
    TMPFILE=$(mktemp /tmp/${BINARY_NAME}-cli.XXXXXX.tar.gz)
    curl -Lk "$DOWNLOAD_URL" -o "$TMPFILE"
    tar -xzf "$TMPFILE" -C "$HOME/.local/bin/"
    rm -f "$TMPFILE"
    echo "  Installed to $HOME/.local/bin/$BINARY_NAME"
fi

# Show running jobs
echo ""
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

# Authenticate with GitHub
echo ""
echo "### Authenticating with GitHub..."
srun --jobid="$JOBID" --pty env CURSOR_CLI_DISABLE_KEYCHAIN_ENCRYPT=1 \
    "$HOME/.local/bin/$BINARY_NAME" tunnel user login --provider github

# Start tunnel in tmux
echo ""
echo "### Starting $EDITOR_LABEL tunnel on job $JOBID..."
SESSION="tunnel-${BINARY_NAME}-$$"
TUNNEL_CMD="srun --jobid=\"$JOBID\" --pty env $HOME/.local/bin/$BINARY_NAME tunnel --accept-server-license-terms --name $TUNNEL_NAME"
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "$TUNNEL_CMD" Enter
echo "### tmux session: $SESSION"
echo "### Reattach with: tmux attach -t $SESSION"
tmux attach -t "$SESSION"
