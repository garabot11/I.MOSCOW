#!/bin/bash
# Generate sets (sequentially) from solution/synth/configs/<set>.yaml into /home/gag/i.mos/synthetic/<set>.
#   solution/synth/tools/gen_sets.sh val_hesai_param val_falcon_param ...
HERE=$(dirname "$(readlink -f "$0")")
SYNTH=$(dirname "$HERE")
PY=${IMOS_SYNTH_PY:-$HOME/miniconda3/envs/imos-synth/bin/python}
ROOT=${IMOS_SYNTHETIC:-/home/gag/i.mos/synthetic}
LOGS=${IMOS_SYNTH_LOGS:-$ROOT/_logs}
mkdir -p "$LOGS"
cd "$SYNTH" || exit 1
for s in "$@"; do
  rm -rf "${ROOT:?}/$s"
  env -u PYTHONPATH "$PY" -m synth.generate --config "configs/$s.yaml" --out "$ROOT/$s" > "$LOGS/$s.log" 2>&1
  echo "$s exit $? $(tail -1 "$LOGS/$s.log")"
done
