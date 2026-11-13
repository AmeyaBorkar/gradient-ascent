#!/usr/bin/env bash
# October-like gate of rule candidates on one blend of cells:
#   scripts/squeeze_gate.sh <cells, e.g. champion or champion+lgbm+...> <rules> [<rules> ...]
# Each <rules> is a comma-separated rule list replacing the base's for the challenger.
set -euo pipefail
cells="$1"; shift
for r in "$@"; do
  python -u -m ringtrace.octlike score "$cells" "$cells" --rules-b "$r" --tag "_${r//,/_}"
done
