#!/usr/bin/env bash
# Rebuild the submission from a clean checkout and an empty cache, on the box:
#   RINGTRACE_THREADS=4 bash scripts/build_submission.sh [recipes/blend.json] [--final]
# 1. builds every feature family and freezes the cleaned labels (via the champion build path);
# 2. runs each member recipe of the blend under its run name (retrying on GPU out-of-memory);
# 3. blends, decodes, scores and writes the validated submission (--final also copies it to
#    $RINGTRACE_OUT/submission.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
blend="${1:-recipes/blend.json}"; shift || true
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
start=$(date +%s)
python - <<'PY'
from ringtrace import run, silver, store, teacher

led = store.ledger()
for fam in run.load_recipe("recipes/champion.json")["families"]:
    store.family(fam)
if silver.load(led, "label") is None:
    teacher.main([])
PY
python -m ringtrace.blend "$blend" --members | while read -r recipe name; do
  echo "== member $name ($recipe) $(date -Is)"
  bash scripts/retry_gpu.sh 20 python -m ringtrace.run "$recipe" --name "$name" < /dev/null
done
python -m ringtrace.blend "$blend" "$@"
echo "built in $(( $(date +%s) - start )) s"
