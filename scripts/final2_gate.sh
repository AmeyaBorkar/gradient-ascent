#!/usr/bin/env bash
# Joint gate of the final file: the bait rules and the three XGBoost members trained on the
# device counter at October's density, together on the best recipe. From saved arrays and cells
# only (no refits).
#   bash scripts/final2_gate.sh [prod|half0|cells|held|build|all|tabm]
# Inputs under $RINGTRACE_OUT: the member runs (runs/<run>/, with runs/<run>-at-half0/ for the
# month folds predicted at October's density) and the October-like cells
# (density/cells-honest, density/cells-held) of `python -m ringtrace.density cells`.
# Reports: $RINGTRACE_OUT/final2/gates/*.json; files: $RINGTRACE_OUT/final2/build*/, counted
# against the best file ($RINGTRACE_BEST, default $RINGTRACE_OUT/best/blend-tabmF-bq.csv).
set -euo pipefail
what="${1:-all}"
out="${RINGTRACE_OUT:-$HOME/outputs}"
D=recipes/density
F=recipes/final2
N="$out/density/cells-honest"
H="$out/density/cells-held"
G="$out/final2/gates"
E="$out/final2/no-cells"
mkdir -p "$G" "$E"

lomo() { python -m ringtrace.final table "$@" --cells "$E"; }
half() { python -m ringtrace.density table --counter half0 --suffix=-at-half0 "$@"; }
cellgate() { python -m ringtrace.density cellgate "$@"; }

if [[ "$what" == prod || "$what" == all ]]; then
  # month folds, production counter: each against the best recipe, then the density members
  # against the bait recipe (the second component on top of the first)
  lomo recipes/final-half.json recipes/bait-combo.json "$D/final-half-chalf3b.json" \
    recipes/final2.json --report "$G/prod-vs-best.json"
  lomo recipes/bait-combo.json recipes/final2.json --report "$G/prod-vs-bait.json"
fi

if [[ "$what" == half0 || "$what" == all ]]; then
  # month folds with each held-out month at October's density (two halves, own passes, draw 0);
  # the base is the best recipe with the refitted XGBoost members (final-half-cpu3)
  half "$D/final-half-cpu3.json" "$F/cpu3-bait.json" "$D/final-half-chalf3b.json" \
    recipes/final2.json --report "$G/half0-vs-cpu3.json"
  half "$F/cpu3-bait.json" recipes/final2.json --report "$G/half0-vs-cpu3-bait.json"
fi

if [[ "$what" == cells || "$what" == all ]]; then
  # October-like cells, September's counter rebuilt at October's density (s13-s15, set r0)
  cellgate "$D/final-half-cpu3.json:$N:honest" "recipes/final2.json:$N:honest" \
    --report "$G/cells-honest-joint.json"
  cellgate "$D/final-half-cpu3.json:$N:honest" "$F/cpu3-bait.json:$N:honest" \
    --report "$G/cells-honest-bait.json"
  cellgate "$F/cpu3-bait.json:$N:honest" "recipes/final2.json:$N:honest" \
    --report "$G/cells-honest-half3-on-bait.json"
fi

if [[ "$what" == held || "$what" == all ]]; then
  # the same under the held counter (the old cells), for the record
  cellgate "$D/final-half-cpu3.json:$H:held" "recipes/final2.json:$H:held" \
    --report "$G/cells-held-joint.json"
  cellgate "$D/final-half-cpu3.json:$H:held" "$F/cpu3-bait.json:$H:held" \
    --report "$G/cells-held-bait.json"
fi

if [[ "$what" == tabm ]]; then
  # follow-up, once TabM trained at October's density has its cells and month folds
  # (runs density-tabm, density-tabm-half0; cells tabm-4060, tabm-half0): both arms on one GPU
  cellgate "$F/bait-half3-t4060.json:$N:honest" "$F/bait-half3t.json:$N:honest" \
    --seeds "${TABM_SEEDS:-13,14,15}" --report "$G/cells-honest-tabm.json"
  half "$F/bait-half3-t4060.json" "$F/bait-half3t.json" --report "$G/half0-tabm.json"
  lomo "$F/bait-half3-t4060.json" "$F/bait-half3t.json" --report "$G/prod-tabm.json"
fi

if [[ "$what" == build || "$what" == all ]]; then
  # the best file and the bait file must rebuild to their recorded sha256 under this code; the
  # final file is built twice, each in a fresh process and directory, and must match itself
  best="${RINGTRACE_BEST:-$out/best/blend-tabmF-bq.csv}"
  for d in build-a build-b; do
    python -m ringtrace.final build recipes/final-half.json recipes/bait-combo.json \
      "$D/final-half-chalf3b.json" recipes/final2.json --out "$out/final2/$d" --against "$best"
  done
  for f in final-half bait-combo final-half-chalf3b final2-bait-half3; do
    sha256sum "$out/final2/build-a/$f.csv" "$out/final2/build-b/$f.csv"
  done
  for f in "$best" "$out/final2/build-a/bait-combo.csv" "$out/final2/build-a/final-half-chalf3b.csv" \
    "$out/final2/build-a/final2-bait-half3.csv"; do
    python -m ringtrace.audit ceiling --submission "$f"
  done
fi
