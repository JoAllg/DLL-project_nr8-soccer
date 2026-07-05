#!/bin/bash
# =========================
# bwHPC deployment and Slurm scripts (2026)
# Joshua Allgeier <allgeier@informatik.uni-freiburg.de>
# =========================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/.env"

HPC="hpc"

###############################################################################
# 1. Generate and install SSH config
###############################################################################
echo "=== Setting up SSH config ==="

HPC_CONFIG="$HOME/.ssh/hpc_config"

# Prints "  IdentityFile <path>" for each key in $1, which may be a plain
# string (single key) or a bash array (multiple keys).
identity_lines() {
    local var_name="$1"
    if declare -p "$var_name" 2>/dev/null | grep -q '^declare -a'; then
        local -n arr_ref="$var_name"
        for id in "${arr_ref[@]}"; do
            printf '  IdentityFile %s\n' "$id"
        done
    else
        printf '  IdentityFile %s\n' "${!var_name}"
    fi
}

HPC_IDENTITY_FILE=$(mktemp)
JUMPHOST_IDENTITY_FILE=$(mktemp)
trap 'rm -f "$HPC_IDENTITY_FILE" "$JUMPHOST_IDENTITY_FILE"' EXIT
identity_lines HPC_IDENTITY > "$HPC_IDENTITY_FILE"
identity_lines JUMPHOST_IDENTITY > "$JUMPHOST_IDENTITY_FILE"

sed -e "s|{{HPC_HOST}}|${HPC_HOST}|g" \
    -e "s|{{HPC_USER}}|${HPC_USER}|g" \
    -e "s|{{JUMPHOST_HOST}}|${JUMPHOST_HOST}|g" \
    -e "s|{{JUMPHOST_USER}}|${JUMPHOST_USER}|g" \
    -e "s|{{REMOTE_SCRIPTS_DIR}}|${REMOTE_SCRIPTS_DIR}|g" \
    -e "/{{HPC_IDENTITY_LINES}}/r ${HPC_IDENTITY_FILE}" \
    -e "/{{HPC_IDENTITY_LINES}}/d" \
    -e "/{{JUMPHOST_IDENTITY_LINES}}/r ${JUMPHOST_IDENTITY_FILE}" \
    -e "/{{JUMPHOST_IDENTITY_LINES}}/d" \
    "$SCRIPT_DIR/hpc_config.template" > "$HPC_CONFIG"

# cp "$SCRIPT_DIR/connect-hpc-node.sh" "$HOME/.ssh/connect-hpc-node.sh"
# chmod +x "$HOME/.ssh/connect-hpc-node.sh"

INCLUDE_LINE="Include ${HPC_CONFIG}"
SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
if ! grep -qF "$INCLUDE_LINE" "$SSH_CONFIG" 2>/dev/null; then
    if [ -f "$SSH_CONFIG" ]; then
        { echo "$INCLUDE_LINE"; echo ""; cat "$SSH_CONFIG"; } > "$SSH_CONFIG.tmp" \
            && mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"
    else
        echo "$INCLUDE_LINE" > "$SSH_CONFIG"
    fi
    echo "  Added hpc_config Include to ~/.ssh/config"
else
    echo "  hpc_config already included in ~/.ssh/config"
fi

# Retire any running control master so the new config applies to future
# connections. Unlike -O exit, -O stop keeps existing sessions alive; the
# old master exits once the last of them closes.
if ssh -O stop "$HPC" 2>/dev/null; then
    echo "  Control master retired; next connection uses the new config"
fi

###############################################################################
# 2. Copy scripts and deploy key to remote
###############################################################################
echo ""
echo "=== Deploying scripts to HPC ==="

retry() {
    local retries=3
    for i in $(seq 1 $retries); do
        "$@" && return 0
        echo "  Attempt $i/$retries failed, retrying in 3s..."
        sleep 3
    done
    "$@"
}

ssh "$HPC" "mkdir -p ${REMOTE_SCRIPTS_DIR} ~/.ssh"

retry rsync -avhz --delete --chmod=F755 --exclude='deploy.sh' --include='run.job.template' --exclude='*.template' --include='*.sh' --include='.env.remote' --exclude='*' \
    "$SCRIPT_DIR/" "$HPC:${REMOTE_SCRIPTS_DIR}/"

retry rsync -av --chmod=F600 "$SCRIPT_DIR/deploy_ed25519" "$HPC:~/.ssh/deploy_ed25519"
echo "  Scripts and deploy key deployed"

###############################################################################
# 3. Set up GitHub SSH access on remote via deploy key
###############################################################################
echo ""
echo "=== Setting up GitHub deploy key ==="

ssh "$HPC" bash <<'REMOTE_SSH'
MARKER="# HPC deploy key for GitHub"
if ! grep -qF "$MARKER" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config <<INNER

$MARKER
Host github.com
  IdentityFile ~/.ssh/deploy_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
INNER
    echo "  GitHub deploy key config added"
else
    echo "  GitHub deploy key already configured"
fi
REMOTE_SSH

###############################################################################
# 4. Install/update uv
###############################################################################
echo ""
echo "=== Installing/updating uv ==="

ssh "$HPC" '
    if command -v uv &>/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
        uv self update
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
'

###############################################################################
# 5. Set up remote .bashrc
###############################################################################
echo ""
echo "=== Setting up remote .bashrc ==="

ssh "$HPC" "sed -i '/source.*hpc-scripts.*\.env\.remote/d' ~/.bashrc \
    && echo '[ -f ${REMOTE_SCRIPTS_DIR}/.env.remote ] && source ${REMOTE_SCRIPTS_DIR}/.env.remote' >> ~/.bashrc"

ssh "$HPC" "sed -i '/uv generate-shell-completion/d' ~/.bashrc \
    && echo 'command -v uv &>/dev/null && eval \"\$(uv generate-shell-completion bash)\"' >> ~/.bashrc"
echo "  .bashrc configured"

###############################################################################
# 6. Set up workspace
###############################################################################
echo ""
echo "=== Setting up workspace ==="

ssh "$HPC" "
    source ${REMOTE_SCRIPTS_DIR}/.env.remote
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    if ws_find \$WORKSPACE_NAME &>/dev/null; then
        REMAINING=\$(ws_list \$WORKSPACE_NAME 2>/dev/null | grep -oP '\\d+(?= day)' | head -1)
        if [ \"\${REMAINING:-0}\" -lt 7 ]; then
            ws_extend \$WORKSPACE_NAME \$WORKSPACE_DAYS
            echo \"  Workspace extended (was \${REMAINING:-0} days remaining)\"
        else
            echo \"  Workspace has \$REMAINING days remaining, skipping extension\"
        fi
        EXTENSIONS=\$(ws_list \$WORKSPACE_NAME 2>/dev/null | awk -F': *' '/available extensions/ {print \$NF}')
        if [ \"\$EXTENSIONS\" = \"0\" ]; then
            echo \"  WARNING: no workspace extensions left - it cannot be extended past its current expiry\"
        fi
    else
        echo '  Allocating workspace...'
        ws_allocate \$WORKSPACE_NAME \$WORKSPACE_DAYS
        ws_send_ical \$WORKSPACE_NAME \$WORKSPACE_EMAIL
        ws_register workspaces
    fi
    ln -sfn \$(ws_find \$WORKSPACE_NAME) \$HOME/\$WORKSPACE_NAME
    mkdir -p \$LOG_DIR
"

###############################################################################
# 7. Clone or update repository
###############################################################################
echo ""
echo "=== Setting up repository ==="

ssh "$HPC" "
    source ${REMOTE_SCRIPTS_DIR}/.env.remote
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    WS=\$(ws_find \$WORKSPACE_NAME)
    if [ -d \"\$WS/\$REPO_NAME\" ]; then
        echo '  Repository exists, pulling latest...'
        cd \"\$WS/\$REPO_NAME\" && git checkout \${BRANCH:-main} && git pull
    else
        echo '  Cloning repository...'
        git clone --template= --branch \${BRANCH:-main} \$REPO_URL \"\$WS/\$REPO_NAME\"
    fi
"

###############################################################################
# 8. Install ODE
###############################################################################
echo ""
echo "=== Installing ODE ==="

ssh "$HPC" '[ -f "$HOME/.local/lib64/libode.so" ] && echo "  ODE already installed, skipping" || bash '"${REMOTE_SCRIPTS_DIR}/install-ode.sh"

###############################################################################
# 9. Install Python dependencies
###############################################################################
echo ""
echo "=== Installing Python dependencies ==="

ssh "$HPC" "
    source ${REMOTE_SCRIPTS_DIR}/.env.remote
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    export LD_LIBRARY_PATH=\"\$HOME/.local/lib64:\$LD_LIBRARY_PATH\"
    CMAKE_ARGS=\"-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DODE_INCLUDE_DIRS=\$HOME/.local/include -DODE_LIBRARIES=\$HOME/.local/lib64/libode.so\" \
        uv sync --all-extras --project \"\$REPO_DIR\"
"

echo ""
echo "=== Deployment complete ==="
echo "  ssh hpc            - Login to HPC"
echo "  ssh hpc.sinfo      - Show resources & your jobs"
echo "  ssh hpc.salloc     - Reserve compute nodes (interactive)"
echo "  ssh hpc.sbatch     - Submit batch job"
echo "  ssh hpc.srun       - Run training on a running node"
echo "  ssh hpc.scancel    - Cancel a job"
echo "  ssh hpc.scontrol   - Show job details"
echo "  ssh hpc.login     - Login to a running node"
# echo "  ssh hpc.node       - SSH to compute node (newest job)"
# echo "  ssh hpc.node.JOBID - SSH to compute node (specific job)"
echo "  ssh hpc.tunnel     - Start VSCode/Cursor tunnel"
