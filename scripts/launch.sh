#!/usr/bin/env bash
# Start a long job on the box in a detached tmux session, from its own snapshot of the code
# (~/work/jobs/<name>), logging to ~/logs/<name>.log.     scripts/launch.sh <name> <command...>
set -euo pipefail
name="$1"; shift
cfg="${RINGTRACE_SSH_CONFIG:?set RINGTRACE_SSH_CONFIG to the ssh config of the compute box}"
host="${RINGTRACE_HOST:-compute-box}"
box_dir="${RINGTRACE_BOX_DIR:-ringtrace}"
snap="~/work/jobs/$name"
cmd="cd $snap && source ~/venv/bin/activate && export PYTHONPATH=$snap/src && $* > ~/logs/${name}.log 2>&1"
ssh -n -F "$cfg" "$host" "rm -rf $snap && mkdir -p ~/work/jobs && cp -r ~/work/$box_dir $snap && tmux kill-session -t '$name' 2>/dev/null; tmux new-session -d -s '$name' \"$cmd\" && echo launched $name from $snap" 2>&1 | grep -v "Welcome to\|Have fun"
