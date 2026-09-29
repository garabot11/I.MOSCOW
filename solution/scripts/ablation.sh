#!/bin/bash
# Experiment matrix: run the offline evaluator with parameter overrides over all bags (container).
# usage: ablation.sh <data_root> <results_root> [stride] [image]
# Relative paths are allowed. <results_root>/ALLDONE and exit code 0 only if every run of every
# variant is complete (see check_offline_result.py); otherwise <results_root>/FAILED, exit code 1.
set -u
DATA=${1:?}; ROOT=${2:?}; STRIDE=${3:-2}; IMG=${4:-metro-lidar:final}
DATA=$(cd "$DATA" && pwd -P) || { echo "no such data directory: $1" >&2; exit 2; }
mkdir -p "$ROOT" && ROOT=$(cd "$ROOT" && pwd -P) || { echo "cannot create results directory: $2" >&2; exit 2; }
CHECK="$(cd "$(dirname "$0")" && pwd)/check_offline_result.py"
rm -f "$ROOT/ALLDONE" "$ROOT/FAILED"
declare -A VARIANTS
VARIANTS[full]=""
VARIANTS[no_temporal]="--no-temporal"
VARIANTS[box_only]="--set cluster_max_extent_d=1000 --set above_min_points=100000 --set attach_ratio=1e9 --set shape_min_height_ratio=0 --set grazing_planarity_max=0 --set low_flat_max_height_m=0 --set edge_sliver_max_width_m=0 --set report_beyond_support_m=1000 --set cluster_min_extent_h_far=0"
VARIANTS[fixed_thresholds]="--set cluster_eps_per_m=0 --set voxel_per_m=0 --set bottom_margin_per_m=0 --set bottom_margin_beyond_per_m=0 --set top_margin_per_m=0 --set top_margin_beyond_per_m=0 --set lateral_shrink_per_m=0 --set lateral_curv_factor=0 --set min_points_far=5 --set attach_radius_per_m=0"
VARIANTS[straight_corridor]="--set curve_max_abs_a=0 --set curve_min_radius_m=1e12"
VARIANTS[no_voxel]="--set voxel_base_m=0.001 --set voxel_per_m=0"
FAILED=()
for name in full no_temporal box_only fixed_thresholds straight_corridor no_voxel; do
  RES="$ROOT/$name"; mkdir -p "$RES"
  declare -A PIDS=()
  for bagdir in "$DATA"/*/; do
    bag=$(basename "$bagdir"); [ -f "$bagdir/metadata.yaml" ] || continue
    docker run --rm -v "$DATA":/data:ro -v "$RES":/results "$IMG" \
      ros2 run metro_obstacle_detector offline --bag "/data/$bag" --stride "$STRIDE" --quiet --out "/results/$bag.jsonl" ${VARIANTS[$name]} > "$RES/$bag.log" 2>&1 &
    PIDS[$bag]=$!
    while [ "$(jobs -r | wc -l)" -ge 3 ]; do sleep 2; done
  done
  [ ${#PIDS[@]} -gt 0 ] || { echo "no rosbag2 directories (with metadata.yaml) in $DATA" >&2; exit 2; }
  for bag in "${!PIDS[@]}"; do
    wait "${PIDS[$bag]}"; rc=$?
    python3 "$CHECK" "$RES" "$bag" "$rc" || FAILED+=("$name/$bag")
  done
  echo "variant $name done"
done
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '%s\n' "${FAILED[@]}" > "$ROOT/FAILED"
  echo "FAILED: ${FAILED[*]}" >&2
  exit 1
fi
echo ALLDONE > "$ROOT/ALLDONE"
