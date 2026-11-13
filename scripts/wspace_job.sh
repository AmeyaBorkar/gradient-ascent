#!/usr/bin/env bash
# One weight-space job on the box. The normal feature cache is built once under a lock, so two
# jobs on one machine never write the same cache file; then the contexts are fitted.
# Each step is a context (fitted) or <subcommand>:<context>, run in the order given.
#     scripts/wspace_job.sh <python> <step> [<step> ...]
set -euo pipefail
py="$1"; shift
cache="${RINGTRACE_CACHE:-$HOME/cache/ringtrace}"
mkdir -p "$cache"
(
  flock 9
  if [ ! -f "$cache/.wspace-features" ]; then
    "$py" -c "from ringtrace import run; run.feature_matrix(run.load_recipe('recipes/tabm.json'))"
    touch "$cache/.wspace-features"
  fi
) 9>"$cache/.wspace-lock"
for step in "$@"; do
  case "$step" in
    *:*) "$py" -u -m ringtrace.wspace "${step%%:*}" "${step#*:}" ;;
    *) "$py" -u -m ringtrace.wspace fit "$step" ;;
  esac
done
