#!/usr/bin/env bash
# Every gate of the density study, from saved cells and month-fold runs (no refits).
#   bash scripts/density_gate.sh [step1|rule|nopos|half|lomo|all]
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
both() {
  # cellgate <a> <b> <report stem> under the rebuilt counter and under the held one
  cellgate "$1:$N:honest" "$2:$N:honest" --report "$G/$3-cells-honest.json"
  cellgate "$1:$H:held" "$2:$H:held" --report "$G/$3-cells-held.json"
}

if [[ "$what" == step1 || "$what" == all ]]; then
  # how much the held counter overstates: same models, only September's counter differs
  cellgate "$R/m-champion.json:$H:held" "$R/m-champion.json:$N:honest" \
    --report "$G/step1-champion-cells.json"
  cellgate "$R/final-half-cpu.json:$H:held" "$R/final-half-cpu.json:$N:honest" \
    --report "$G/step1-cells.json"
fi

if [[ "$what" == rule || "$what" == all ]]; then
  # the anchored template rule's gain and the TabM share (1/3 against 1/2) under each counter
  both "$R/final-half-cpu-norule.json" "$R/final-half-cpu.json" rule
  both "$R/final-third-cpu.json" "$R/final-half-cpu.json" tabm-share
fi

if [[ "$what" == nopos || "$what" == all ]]; then
  # (a) the champion without the absolute position column
  both "$R/m-champion.json" "$R/m-champion-nopos.json" nopos-champion
fi

if [[ "$what" == half || "$what" == all ]]; then
  # (b) members trained on the counter at October's density (draw 0): the champion alone, then
  # in the best recipe the champion only (chalf0) and all three XGBoost members (chalf3)
  both "$R/m-champion.json" "$R/m-champion-half0.json" half0-champion
  both "$R/final-half-cpu3.json" "$R/final-half-chalf0.json" half0-blend
  both "$R/final-half-cpu3.json" "$R/final-half-chalf3.json" half3-blend
fi

if [[ "$what" == lomo || "$what" == all ]]; then
  # month folds: the champion under each counter, then the candidates over the best file's
  # arrays (final-half-r1 is the best recipe with the refitted champion)
  lomo "$R/m-champion.json" "$R/m-champion-half0.json" --report "$G/champion-lomo.json"
  alone "$R/m-champion-alone.json" "$R/m-champion-half0-alone.json" \
    --report "$G/champion-lomo-alone.json"
  half "$R/m-champion.json" "$R/m-champion-half0.json" --report "$G/champion-lomo-half0.json"
  lomo recipes/final-half.json "$R/final-half-r1.json" "$R/final-half-chalf0b.json" \
    "$R/final-half-chalf3b.json" --report "$G/half-best-lomo.json"
  half "$R/final-half-r1.json" "$R/final-half-chalf0b.json" --report "$G/half-best-lomo-half0.json"
fi
