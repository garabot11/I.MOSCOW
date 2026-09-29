#!/bin/bash
# Offline evaluation of every bag in <data_root> inside the Humble container (stride 1 by default).
# usage: eval_all_container.sh <data_root_on_host> <results_dir_on_host> [stride] [image] [parallel]
# env OFFLINE_ARGS: extra arguments of the offline tool, e.g. OFFLINE_ARGS="--set max_analysed_points=180000"
# Relative paths are allowed. Exit code 0 and <results>/ALLDONE only if every bag was processed
# completely (container exit 0, JSONL + summary present, summary "complete", one JSONL line per
# processed frame); otherwise <results>/FAILED lists the failures and the exit code is 1.
set -u
DATA=${1:?data root}; RES=${2:?results dir}; STRIDE=${3:-1}; IMG=${4:-metro-lidar:final}; PAR=${5:-3}
DATA=$(cd "$DATA" && pwd -P) || { echo "no such data directory: $1" >&2; exit 2; }
mkdir -p "$RES" && RES=$(cd "$RES" && pwd -P) || { echo "cannot create results directory: $2" >&2; exit 2; }
rm -f "$RES/ALLDONE" "$RES/FAILED"
declare -A PIDS
for bagdir in "$DATA"/*/; do
  bag=$(basename "$bagdir")
  [ -f "$bagdir/metadata.yaml" ] || continue
  docker run --rm -v "$DATA":/data:ro -v "$RES":/results "$IMG" \
    ros2 run metro_obstacle_detector offline --bag "/data/$bag" --stride "$STRIDE" --out "/results/$bag.jsonl" --quiet ${OFFLINE_ARGS:-} > "$RES/$bag.log" 2>&1 &
  PIDS[$bag]=$!
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 2; done
done
[ ${#PIDS[@]} -gt 0 ] || { echo "no rosbag2 directories (with metadata.yaml) in $DATA" >&2; exit 2; }
FAILED=()
for bag in "${!PIDS[@]}"; do
  wait "${PIDS[$bag]}"; rc=$?
  python3 "$(dirname "$0")/check_offline_result.py" "$RES" "$bag" "$rc" || FAILED+=("$bag")
done
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '%s\n' "${FAILED[@]}" > "$RES/FAILED"
  echo "FAILED: ${FAILED[*]} (see $RES/<bag>.log)" >&2
  exit 1
fi
echo ALLDONE > "$RES/ALLDONE"
echo "all ${#PIDS[@]} bags complete: $RES"
