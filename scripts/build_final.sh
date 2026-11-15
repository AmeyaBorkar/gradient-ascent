#!/usr/bin/env bash
# Rebuild an uploaded file from a clean checkout and an empty cache:
#   RINGTRACE_THREADS=8 bash scripts/build_final.sh recipes/bait-edge.json <out dir> [--against <csv>]
# 1. builds every feature family and freezes the cleaned labels;
# 2. runs the member recipe behind each run the final recipe reads (retrying on GPU out-of-memory),
#    skipping runs that already have their arrays;
# 3. `python -m ringtrace.final build`: blend, decode, rules, validation against the sample
#    submission, sha256 (and, with --against, the rows that differ from another file).
set -euo pipefail
cd "$(dirname "$0")/.."
final="${1:?usage: build_final.sh <final recipe> <out dir> [--against <csv>]}"
dest="${2:?usage: build_final.sh <final recipe> <out dir> [--against <csv>]}"
shift 2
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
# the member recipe behind each run name a final recipe reads
python - "$final" <<'PY' | while read -r recipe name; do
import json
import sys

RECIPE_OF = {
    "blend-champion-m1": "recipes/champion.json",
    "lgbm": "recipes/lgbm.json",
    "catboost": "recipes/catboost.json",
    "clean-event": "recipes/clean-event.json",
    "clean-l2": "recipes/clean-l2.json",
    "final-tabm": "recipes/tabm.json",
}
runs = [m["run"] for m in json.load(open(sys.argv[1]))["members"]]
unknown = [r for r in runs if r not in RECIPE_OF]
if unknown:
    sys.exit(f"no member recipe known for runs {unknown}")
for r in runs:
    print(RECIPE_OF[r], r)
PY
  recipe=${recipe%$'\r'}; name=${name%$'\r'}   # Python on Windows ends lines with \r\n

  if [[ -f "${RINGTRACE_OUT:-$HOME/outputs}/runs/$name/test.npy" ]]; then
    echo "== member $name: saved arrays found, kept"
    continue
  fi
  echo "== member $name ($recipe) $(date -Is)"
  bash scripts/retry_gpu.sh 20 python -m ringtrace.run "$recipe" --name "$name" < /dev/null
done
python -m ringtrace.final build "$final" --out "$dest" "$@"
echo "built in $(( $(date +%s) - start )) s"
