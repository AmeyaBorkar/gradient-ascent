#!/usr/bin/env bash
# Refit lgbm and catboost for October (one GPU fit at a time), then the refit and hedge blends:
#   bash scripts/refit_oct_blends.sh
set -euo pipefail
bash scripts/retry_gpu.sh 3 python -m ringtrace.refit october recipes/lgbm.json
bash scripts/retry_gpu.sh 3 python -m ringtrace.refit october recipes/catboost.json --gpu-ram-part 0.45
for k in refit hedge; do
  python -m ringtrace.blend "recipes/$k-blend.json"
done
