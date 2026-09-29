#!/bin/bash
# Development helper (host, Jazzy): run the offline detector over all bags with a stride.
# ALLDONE only if every run is complete (check_offline_result.py), otherwise FAILED and exit 1.
STRIDE=${1:-4}; TAG=${2:-dev}
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/home/gag/i.mos/solution/src/metro_obstacle_detector:$PYTHONPATH
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/home/gag/i.mos/solution/results/$TAG; mkdir -p "$OUT"; rm -f "$OUT/ALLDONE" "$OUT/FAILED"
declare -A PIDS
for bag in doubleT_obstacle doubleT_platform roundT_doubleT roundT_pressureGate_roundT roundT_squareT_pressureGate_squareT squareT_platform_squareT_switch; do
  /usr/bin/python3 -m metro_obstacle_detector.offline --bag /home/gag/i.mos/dataset/for_hackathon/$bag --stride "$STRIDE" \
     --out "$OUT/${bag}.jsonl" > "$OUT/${bag}.log" 2>&1 &
  PIDS[$bag]=$!
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 2; done
done
FAILED=()
for bag in "${!PIDS[@]}"; do
  wait "${PIDS[$bag]}"; rc=$?
  python3 "$HERE/check_offline_result.py" "$OUT" "$bag" "$rc" || FAILED+=("$bag")
done
if [ ${#FAILED[@]} -gt 0 ]; then printf '%s\n' "${FAILED[@]}" > "$OUT/FAILED"; exit 1; fi
echo ALLDONE > "$OUT/ALLDONE"
