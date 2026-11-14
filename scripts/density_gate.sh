#!/usr/bin/env bash
# Every gate of the density study, from saved cells and month-fold runs (no refits).
#   bash scripts/density_gate.sh [step1|rule|nopos3|half|half2|mix|all]
# Inputs: cells from `python -m ringtrace.density cells ...` under $RINGTRACE_OUT/density/,
# month-fold runs `density-<member>[-nopos|-half<d>]` from `python -m ringtrace.density lomo`.
# The LightGBM and CatBoost members keep the arrays of the best file in the month-fold tables
# (both arms share them); their September cells are the CPU refits.
# Reports: $RINGTRACE_OUT/density/gates/*.json.
set -euo pipefail
what="${1:-all}"
out="${RINGTRACE_OUT:-$HOME/outputs}"
R=recipes/density
H="$out/density/cells-held"
N="$out/density/cells-honest"
G="$out/density/gates"
E="$out/density/no-cells"
mkdir -p "$G" "$E"

# month-fold runs whose held-out months carry their standalone counter
for run in "$out"/runs/density-*; do
  [[ -f "$run/oof_alone.npy" ]] || continue
  alone="$run-alone"
  mkdir -p "$alone"
  ln -sf "$run/oof_alone.npy" "$alone/oof.npy"
  ln -sf "$run/test.npy" "$alone/test.npy"
done

cellgate() { python -m ringtrace.density cellgate "$@"; }
lomo() { python -m ringtrace.final table "$@" --cells "$E"; }
alone() { python -m ringtrace.density table "$@"; }

if [[ "$what" == step1 || "$what" == all ]]; then
  # how much the held counter overstates the best recipe: same models, September's counter only
  cellgate "$R/final-half-cpu.json:$H:held" "$R/final-half-cpu.json:$N:honest" \
    --report "$G/step1-cells.json"
  # the three refitted XGBoost members against those of the best file (month folds)
  lomo recipes/final-half.json "$R/final-half-cpu3.json" --report "$G/step1-cpu-vs-best.json"
fi

if [[ "$what" == rule || "$what" == all ]]; then
  # the tree blend alone under either counter, and the anchored template rule's gain under each
  cellgate "$R/blend-cpu.json:$H:held" "$R/blend-cpu.json:$N:honest" \
    --report "$G/blend-cells.json"
  for c in held honest; do
    d="$H"
    [[ "$c" == honest ]] && d="$N"
    cellgate "$R/final-half-cpu-norule.json:$d:$c" "$R/final-half-cpu.json:$d:$c" \
      --report "$G/rule-cells-$c.json"
    # the TabM share of the fraud mass (1/3 against 1/2) under each counter
    cellgate "$R/final-third-cpu.json:$d:$c" "$R/final-half-cpu.json:$d:$c"       --report "$G/tabm-share-cells-$c.json"
  done
fi

if [[ "$what" == nopos3 || "$what" == all ]]; then
  # (a) no position column in the three XGBoost members
  for c in honest held; do
    d="$N"
    [[ "$c" == held ]] && d="$H"
    cellgate "$R/final-half-cpu3.json:$d:$c" "$R/final-half-nopos3.json:$d:$c" \
      --report "$G/nopos3-cells-$c.json"
  done
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-nopos.json:$N:honest" \
    --report "$G/nopos-champion-cells-honest.json"
  lomo "$R/final-half-cpu3.json" "$R/final-half-nopos3.json" --report "$G/nopos3-lomo.json"
  alone "$R/final-half-cpu3-alone.json" "$R/final-half-nopos3-alone.json" \
    --report "$G/nopos3-lomo-alone.json"
fi

if [[ "$what" == half || "$what" == all ]]; then
  # (b) the champion trained on a counter at October's density (draw 0)
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-half0.json:$N:honest" \
    --report "$G/half0-champion-cells-honest.json"
  cellgate "$R/final-half-cpu3.json:$N:honest" "$R/final-half-chalf0.json:$N:honest" \
    --report "$G/half0-blend-cells-honest.json"
  lomo "$R/m-champion.json" "$R/m-champion-half0.json" "$R/m-champion-nopos.json" \
    --report "$G/champion-lomo.json"
  lomo "$R/final-half-cpu3.json" "$R/final-half-chalf0.json" --report "$G/half0-blend-lomo.json"
  alone "$R/m-champion-alone.json" "$R/m-champion-half0-alone.json" \
    "$R/m-champion-nopos-alone.json" --report "$G/champion-lomo-alone.json"
  alone "$R/final-half-cpu3-alone.json" "$R/final-half-chalf0-alone.json" \
    --report "$G/half0-blend-lomo-alone.json"
fi

if [[ "$what" == half2 || "$what" == all ]]; then
  # (b) with both draws
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-half.json:$N:honest" \
    --report "$G/half-champion-cells-honest.json"
  cellgate "$R/final-half-cpu3.json:$N:honest" "$R/final-half-chalf.json:$N:honest" \
    --report "$G/half-blend-cells-honest.json"
  lomo "$R/final-half-cpu3.json" "$R/final-half-chalf.json" --report "$G/half-blend-lomo.json"
  alone "$R/m-champion-alone.json" "$R/m-champion-half-alone.json" \
    --report "$G/champion-half-lomo-alone.json"
fi

if [[ "$what" == mix || "$what" == all ]]; then
  cellgate "$R/m-champion.json:$H:held" "$R/m-champion.json:$N:honest" \
    --seeds 23,24,25 --report "$G/mix-champion-held-vs-honest.json"
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-nopos.json:$N:honest" \
    --seeds 23,24,25 --report "$G/mix-champion-nopos-honest.json"
fi
