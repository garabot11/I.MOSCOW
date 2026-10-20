#!/bin/bash
# Smoke test of every generator path (few frames each) + consistency check, into a scratch directory.
#   solution/synth/tools/smoke_all.sh /tmp/smoke_out
HERE=$(dirname "$(readlink -f "$0")")
SYNTH=$(dirname "$HERE")
PY=${IMOS_SYNTH_PY:-$HOME/miniconda3/envs/imos-synth/bin/python}
OUT=${1:?output directory}
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$SYNTH" || exit 1
run() {  # set, scenes, frames
  env -u PYTHONPATH "$PY" -m synth.generate --config "configs/$1.yaml" --out "$OUT/$1" --only "$2" --max-frames "$3" > "$OUT/$1.log" 2>&1
  echo "$1: generate exit $? | $(env -u PYTHONPATH "$PY" "$HERE/verify_set.py" "$OUT/$1" | tail -1)"
}
run val_hesai_insert 1,2 8
run val_hesai_param 2,3 8
run val_falcon_param 3 6
run val_dust 0,8 5
