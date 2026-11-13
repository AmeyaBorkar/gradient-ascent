#!/usr/bin/env bash
# Run a command on the box inside ~/work/<dir> with the project venv and a small thread budget.
#   scripts/box_in.sh <dir> <command...>      (RINGTRACE_THREADS defaults to 2 here)
set -euo pipefail
dir="$1"; shift
cfg="${RINGTRACE_SSH_CONFIG:?set RINGTRACE_SSH_CONFIG to the ssh config of the compute box}"
host="${RINGTRACE_HOST:-compute-box}"
threads="${RINGTRACE_THREADS:-2}"
ssh -n -F "$cfg" "$host" "cd ~/work/$dir && source ~/venv/bin/activate && export PYTHONPATH=~/work/$dir/src RINGTRACE_THREADS=$threads POLARS_MAX_THREADS=$threads OMP_NUM_THREADS=$threads; $*" 2>&1 | grep -v "Welcome to\|Have fun"
