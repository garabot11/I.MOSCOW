#!/bin/bash
# Consistency (host) + readability (metro-lidar:final) checks for one or more sets under /home/gag/i.mos/synthetic.
#   solution/synth/tools/check_sets.sh val_hesai_param val_falcon_param
HERE=$(dirname "$(readlink -f "$0")")
PY=${IMOS_SYNTH_PY:-$HOME/miniconda3/envs/imos-synth/bin/python}
ROOT=${IMOS_SYNTHETIC:-/home/gag/i.mos/synthetic}
for s in "$@"; do
  echo "== $s"
  env -u PYTHONPATH "$PY" "$HERE/verify_set.py" "$ROOT/$s" --json "$ROOT/_checks/${s}_verify.json" | tail -1
  "$HERE/container_check.sh" "$ROOT/$s" "$ROOT/_checks/$s" 2>&1 | tail -14
done
