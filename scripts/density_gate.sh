#!/usr/bin/env bash
# Every gate of the density study, from saved cells and month-fold runs (no refits).
#   bash scripts/density_gate.sh [step1|rule|nopos3|half|half3|half2|all]
# Inputs: cells from `python -m ringtrace.density cells ...` under $RINGTRACE_OUT/density/,
# month-fold runs `density-<member>[-nopos|-half<d>]` from `python -m ringtrace.density lomo`.
# The LightGBM, CatBoost and TabM members keep the arrays of the best file in the month-fold
# tables (both arms share them); their September cells are refits.
# Month folds are scored three ways: the production counter (the held-out month on the shared
# full-density pass), the standalone counter (the held-out month as a pass of its own, full
# density) and the counter at October's density (the held-out month in two halves, each a pass
# of its own; draw 0). Reports: $RINGTRACE_OUT/density/gates/*.json.
set -euo pipefail
what="${1:-all}"
out="${RINGTRACE_OUT:-$HOME/outputs}"
R=recipes/density
H="$out/density/cells-held"
N="$out/density/cells-honest"
G="$out/density/gates"
E="$out/density/no-cells"
mkdir -p "$G" "$E"

# month-fold runs whose held-out months carry their standalone counter (<run>-alone) or their
# counter at October's density (<run>-at-half0; "-half0" alone names a training counter)
for run in "$out"/runs/density-*; do
  for kind in alone half0; do
    [[ -f "$run/oof_$kind.npy" ]] || continue
    d="$run-$kind"
    [[ "$kind" == half0 ]] && d="$run-at-half0"
    mkdir -p "$d"
    ln -sf "$run/oof_$kind.npy" "$d/oof.npy"
    ln -sf "$run/test.npy" "$d/test.npy"
  done
done

cellgate() { python -m ringtrace.density cellgate "$@"; }
lomo() { python -m ringtrace.final table "$@" --cells "$E"; }
alone() { python -m ringtrace.density table "$@"; }
half() { python -m ringtrace.density table --counter half0 --suffix -at-half0 "$@"; }

if [[ "$what" == step1 || "$what" == all ]]; then
  # how much the held counter overstates the best recipe: same models, September's counter only
  cellgate "$R/final-half-cpu.json:$H:held" "$R/final-half-cpu.json:$N:honest" \
    --report "$G/step1-cells.json"
  cellgate "$R/m-champion.json:$H:held" "$R/m-champion.json:$N:honest" \
    --report "$G/step1-champion-cells.json"
  # the three refitted XGBoost members against those of the best file (month folds)
  lomo recipes/final-half.json "$R/final-half-cpu3.json" --report "$G/step1-cpu3-vs-best.json"
  # the champion's month folds under each counter (base scores of each table)
  lomo "$R/m-champion.json" "$R/m-champion-nopos.json" --report "$G/champion-lomo.json"
  alone "$R/m-champion-alone.json" "$R/m-champion-nopos-alone.json" \
    --report "$G/champion-lomo-alone.json"
  half "$R/m-champion.json" "$R/m-champion-nopos.json" --report "$G/champion-lomo-half0.json"
fi

if [[ "$what" == rule || "$what" == all ]]; then
  # the anchored template rule's gain and the TabM share (1/3 against 1/2) under each counter
  for c in held honest; do
    d="$H"
    [[ "$c" == honest ]] && d="$N"
    cellgate "$R/final-half-cpu-norule.json:$d:$c" "$R/final-half-cpu.json:$d:$c" \
      --report "$G/rule-cells-$c.json"
    cellgate "$R/final-third-cpu.json:$d:$c" "$R/final-half-cpu.json:$d:$c" \
      --report "$G/tabm-share-cells-$c.json"
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
  half "$R/final-half-cpu3.json" "$R/final-half-nopos3.json" --report "$G/nopos3-lomo-half0.json"
fi

if [[ "$what" == half || "$what" == all ]]; then
  # (b) the champion trained on a counter at October's density (draw 0)
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-half0.json:$N:honest" \
    --report "$G/half0-champion-cells-honest.json"
  cellgate "$R/final-half-cpu3.json:$N:honest" "$R/final-half-chalf0.json:$N:honest" \
    --report "$G/half0-blend-cells-honest.json"
  lomo "$R/m-champion.json" "$R/m-champion-half0.json" --report "$G/half0-champion-lomo.json"
  lomo "$R/final-half-cpu3.json" "$R/final-half-chalf0.json" --report "$G/half0-blend-lomo.json"
  alone "$R/m-champion-alone.json" "$R/m-champion-half0-alone.json" \
    --report "$G/half0-champion-lomo-alone.json"
  alone "$R/final-half-cpu3-alone.json" "$R/final-half-chalf0-alone.json" \
    --report "$G/half0-blend-lomo-alone.json"
  half "$R/m-champion.json" "$R/m-champion-half0.json" \
    --report "$G/half0-champion-lomo-half0.json"
  half "$R/final-half-cpu3.json" "$R/final-half-chalf0.json" \
    --report "$G/half0-blend-lomo-half0.json"
fi

if [[ "$what" == half3 || "$what" == all ]]; then
  # (b) all three XGBoost members trained on the counter at October's density (draw 0)
  for c in honest held; do
    d="$N"
    [[ "$c" == held ]] && d="$H"
    cellgate "$R/final-half-cpu3.json:$d:$c" "$R/final-half-chalf3.json:$d:$c" \
      --report "$G/half3-blend-cells-$c.json"
  done
  lomo "$R/final-half-cpu3.json" "$R/final-half-chalf3.json" --report "$G/half3-blend-lomo.json"
  alone "$R/final-half-cpu3-alone.json" "$R/final-half-chalf3-alone.json" \
    --report "$G/half3-blend-lomo-alone.json"
  half "$R/final-half-cpu3.json" "$R/final-half-chalf3.json" \
    --report "$G/half3-blend-lomo-half0.json"
fi

if [[ "$what" == half2 || "$what" == all ]]; then
  # (b) with both draws of the training counter
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-half.json:$N:honest" \
    --report "$G/half-champion-cells-honest.json"
  cellgate "$R/final-half-cpu3.json:$N:honest" "$R/final-half-chalf.json:$N:honest" \
    --report "$G/half-blend-cells-honest.json"
  lomo "$R/final-half-cpu3.json" "$R/final-half-chalf.json" --report "$G/half-blend-lomo.json"
  half "$R/final-half-cpu3.json" "$R/final-half-chalf.json" \
    --report "$G/half-blend-lomo-half0.json"
fi
