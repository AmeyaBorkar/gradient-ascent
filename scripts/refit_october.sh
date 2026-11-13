#!/usr/bin/env bash
# Refit October predictions for each member recipe, retrying after GPU out-of-memory:
#   bash scripts/refit_october.sh champion clean-event clean-l2
set -euo pipefail
for r in "$@"; do
  bash scripts/retry_gpu.sh 3 python -m ringtrace.refit october "recipes/$r.json"
done
