#!/usr/bin/env bash
# Every gate of the density study, from saved cells and month-fold runs (no refits).
#   bash scripts/density_gate.sh [step1|nopos|half|mix|all]
# Inputs: cells from `python -m ringtrace.density cells ...` under $RINGTRACE_OUT/density/,
# month-fold runs `density-<member>[-nopos|-half<d>]` from `python -m ringtrace.density lomo`.
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

if [[ "$what" == step1 || "$what" == all ]]; then
  # how much the held counter overstates the best recipe: same models, September's counter only
  cellgate "$R/final-half-cpu.json:$H:held" "$R/final-half-cpu.json:$N:honest" \
    --report "$G/step1-cells.json"
  # the refitted members against the members of the best file (held counter, GPU fits)
  python -m ringtrace.final table recipes/final-half.json "$R/final-half-cpu.json" \
    --cells "$E" --report "$G/step1-cpu-vs-best.json"
fi

if [[ "$what" == rule || "$what" == all ]]; then
  # the tree blend alone under either counter, and the anchored template rule's gain under each
  cellgate "$R/blend-cpu.json:$H:held" "$R/blend-cpu.json:$N:honest" \
    --report "$G/blend-cells.json"
  for c in held honest; do
    d="$H"; [[ "$c" == honest ]] && d="$N"
    cellgate "$R/final-half-cpu-norule.json:$d:$c" "$R/final-half-cpu.json:$d:$c" \
      --report "$G/rule-cells-$c.json"
  done
fi

if [[ "$what" == nopos3 || "$what" == all ]]; then
  # no position column in the three XGBoost members only
  cellgate "$R/final-half-cpu.json:$N:honest" "$R/final-half-nopos3.json:$N:honest" \
    --report "$G/nopos3-cells-honest.json"
  python -m ringtrace.final table "$R/final-half-cpu.json" "$R/final-half-nopos3.json" \
    --cells "$E" --report "$G/nopos3-lomo.json"
fi

if [[ "$what" == nopos || "$what" == all ]]; then
  cellgate "$R/final-half-cpu.json:$N:honest" "$R/final-half-nopos.json:$N:honest" \
    --report "$G/nopos-cells-honest.json"
  cellgate "$R/final-half-cpu.json:$H:held" "$R/final-half-nopos.json:$H:held" \
    --report "$G/nopos-cells-held.json"
  python -m ringtrace.final table "$R/final-half-cpu.json" "$R/final-half-nopos.json" \
    --cells "$E" --report "$G/nopos-lomo.json"
  python -m ringtrace.final table recipes/final-half.json "$R/final-half-nopos.json" \
    --cells "$E" --report "$G/nopos-lomo-vs-best.json"
  python -m ringtrace.density table "$R/final-half-cpu-alone.json" \
    "$R/final-half-nopos-alone.json" --report "$G/nopos-lomo-alone.json"
fi

if [[ "$what" == half || "$what" == all ]]; then
  for c in m-champion-half0 m-champion-half; do
    cellgate "$R/m-champion.json:$N:honest" "$R/$c.json:$N:honest" \
      --report "$G/$c-cells-honest.json"
  done
  python -m ringtrace.final table "$R/m-champion.json" "$R/m-champion-half0.json" \
    "$R/m-champion-half.json" "$R/m-champion-nopos.json" --cells "$E" \
    --report "$G/champion-lomo.json"
  python -m ringtrace.density table "$R/m-champion-alone.json" "$R/m-champion-half0-alone.json" \
    "$R/m-champion-half-alone.json" "$R/m-champion-nopos-alone.json" \
    --report "$G/champion-lomo-alone.json"
fi

if [[ "$what" == mix || "$what" == all ]]; then
  cellgate "$R/m-champion.json:$H:held" "$R/m-champion.json:$N:honest" \
    --seeds 23,24,25 --report "$G/mix-champion-held-vs-honest.json"
  cellgate "$R/m-champion.json:$N:honest" "$R/m-champion-nopos.json:$N:honest" \
    --seeds 23,24,25 --report "$G/mix-champion-nopos-honest.json"
fi
