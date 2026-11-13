#!/usr/bin/env bash
# Score an in-context member run against the champion and inside the blend (run on the box):
#   scripts/icl_score.sh <member-run>
# Needs the five blend members' runs and recipes/<member-run>-blend{6,3}.json.
set -euo pipefail
run="$1"
echo "== member alone: paired event bootstrap against the champion"
python -m ringtrace.compare blend-champion-m1 "$run"
echo "== the five-way blend"
python -m ringtrace.blend recipes/blend.json
for tag in blend6 blend3; do
  echo "== six-way blend $tag"
  python -m ringtrace.blend "recipes/$run-$tag.json"
  python -m ringtrace.compare blend "$run-$tag"
done
