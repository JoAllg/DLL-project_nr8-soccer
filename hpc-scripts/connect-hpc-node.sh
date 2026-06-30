#!/bin/bash
# Runs locally as ProxyCommand. Parses SSH alias to extract job identifier,
# then invokes the remote node_proxy.sh through the login node.
# Usage (as ProxyCommand): connect-hpc-node.sh %n
#   ssh hpc.node          → auto-select newest running job
#   ssh hpc.node.12345    → connect to job 12345
#   ssh hpc.node.vscode   → connect to job named "vscode"

ALIAS="$1"
JOB_ID="${ALIAS#hpc.node}"
JOB_ID="${JOB_ID#.}"
JOB_ID="${JOB_ID:-auto}"

exec ssh -T -q -e none -o ControlPath=none hpc "~/hpc-scripts/node_proxy.sh $JOB_ID"
