#!/usr/bin/env bash
# Leave-one-family-out ablations of a recipe, N at a time, on the box (run there, inside tmux):
#   bash scripts/ablate.sh recipes/champion.json 3 6 marks bursts drains ...
# Each run uses THREADS CPU threads and the one GPU; logs go to ~/logs/ablate-<family>.log.
set -euo pipefail
recipe="$1"; parallel="$2"; threads="$3"; shift 3
printf '%s\n' "$@" | xargs -P "$parallel" -I{} bash -c \
  "RINGTRACE_THREADS=$threads python -m ringtrace.run $recipe --no-test --without {} > ~/logs/ablate-{}.log 2>&1; echo finished {}"
