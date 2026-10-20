#!/bin/bash
# verify_set.py on several sets in parallel, then container_check.sh on each (sequentially).
#   solution/synth/tools/verify_many.sh train_hesai_insert train_hesai_param ...
HERE=$(dirname "$(readlink -f "$0")")
PY=${IMOS_SYNTH_PY:-$HOME/miniconda3/envs/imos-synth/bin/python}
ROOT=${IMOS_SYNTHETIC:-/home/gag/i.mos/synthetic}
mkdir -p "$ROOT/_checks"
for s in "$@"; do
  env -u PYTHONPATH "$PY" "$HERE/verify_set.py" "$ROOT/$s" --json "$ROOT/_checks/${s}_verify.json" > "$ROOT/_checks/${s}_verify.txt" 2>&1 &
done
wait
for s in "$@"; do echo "== $s: $(tail -1 "$ROOT/_checks/${s}_verify.txt")"; done
for s in "$@"; do
  echo "== $s container"
  "$HERE/container_check.sh" "$ROOT/$s" "$ROOT/_checks/$s" > "$ROOT/_checks/${s}_container.txt" 2>&1
  python3 - "$ROOT/_checks/${s}_container.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
ok = sum(1 for r in rows if r["complete"] and r["bag_info_messages"] == r["offline_read"])
frames = sum(r["offline_read"] for r in rows)
play = [l.strip() for l in open(sys.argv[1]) if l.startswith("play exit") or l.strip().startswith("average rate")]
print(f"scenes {len(rows)}, offline complete and == bag info: {ok}, frames {frames}; {play[:2]}")
PY
done
