#!/usr/bin/env bash
# Copy the committed tree (HEAD) plus uncommitted edits of tracked files to ~/work/$box_dir on the box.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cfg="${RINGTRACE_SSH_CONFIG:?set RINGTRACE_SSH_CONFIG to the ssh config of the compute box}"
host="${RINGTRACE_HOST:-compute-box}"
box_dir="${RINGTRACE_BOX_DIR:-ringtrace}"
cd "$here"
stash=$(git stash create || true)
git archive "${stash:-HEAD}" | ssh -F "$cfg" "$host" "rm -rf ~/work/$box_dir && mkdir -p ~/work/$box_dir && tar -x -C ~/work/$box_dir" 2>&1 | grep -v "Welcome to\|Have fun" || true
echo "shipped ${stash:-$(git rev-parse --short HEAD)}"
