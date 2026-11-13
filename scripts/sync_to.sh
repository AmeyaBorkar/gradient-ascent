#!/usr/bin/env bash
# Copy the local working tree (src, tests, recipes; uncommitted edits included) to ~/work/<dir> on
# the box, without touching anyone else's copy.   scripts/sync_to.sh <dir>
set -euo pipefail
dir="$1"
here="$(cd "$(dirname "$0")/.." && pwd)"
cfg="${RINGTRACE_SSH_CONFIG:?set RINGTRACE_SSH_CONFIG to the ssh config of the compute box}"
host="${RINGTRACE_HOST:-compute-box}"
cd "$here"
tar -c --exclude=__pycache__ src tests recipes pyproject.toml | ssh -F "$cfg" "$host" "rm -rf ~/work/$dir && mkdir -p ~/work/$dir && tar -x -C ~/work/$dir" 2>&1 | grep -v "Welcome to\|Have fun" || true
echo "synced to ~/work/$dir"
