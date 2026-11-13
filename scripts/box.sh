#!/usr/bin/env bash
# Run a command on the box inside the project venv and work tree.
set -euo pipefail
cfg="${RINGTRACE_SSH_CONFIG:?set RINGTRACE_SSH_CONFIG to the ssh config of the compute box}"
host="${RINGTRACE_HOST:-compute-box}"
box_dir="${RINGTRACE_BOX_DIR:-ringtrace}"
ssh -n -F "$cfg" "$host" "cd ~/work/$box_dir 2>/dev/null || cd ~; source ~/venv/bin/activate 2>/dev/null; export PYTHONPATH=~/work/$box_dir/src; $*" 2>&1 | grep -v "Welcome to\|Have fun"
