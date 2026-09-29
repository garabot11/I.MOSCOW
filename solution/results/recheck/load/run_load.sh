#!/bin/bash
# Load probe: whole-bag stream (paused start + resume, reliable, depth 1) with the node pinned to NODE_CPU
# (empty = no pinning) and extra node parameters; psutil samples of the node, coverage vs the bag.
# usage: run_load.sh <bag_dir> <out_dir> [node_cpu] [extra ros args...]
BAG=$(cd "$1" && pwd -P); OUT=$(mkdir -p "$2" && cd "$2" && pwd -P); CPU=${3:-}; shift 3; EXTRA="$*"
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=""; [ -n "$CPU" ] && PIN="taskset -c $CPU"
{ date -Is; uptime; top -bn1 | sed -n '8,12p'; } > "$OUT/host_load_before.txt"
docker run --rm --network host --shm-size=1g -e ROS_DOMAIN_ID=81 -e ROS_LOCALHOST_ONLY=1 \
  -v "$BAG":/data/bag:ro -v "$OUT":/results -v "$HERE":/probe:ro ${IMG:-metro-lidar:final} bash -c "
source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; set -m
python3 /probe/sample_node.py /results/samples.jsonl & SAMP=\$!
$PIN /workspace/install/lib/metro_obstacle_detector/detector_node --ros-args -p log_path:=/results/stream.jsonl $EXTRA > /results/node.log 2>&1 & NODE=\$!
sleep 4
ros2 bag play /data/bag --rate 1.0 --start-paused --read-ahead-queue-size 20 --disable-keyboard-controls --wait-for-all-acked 5000 > /results/play.log 2>&1 & PLAYER=\$!
sleep 3; ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume > /dev/null
wait \$PLAYER; sleep 2
kill -INT \$NODE; wait \$NODE; sleep 1; kill \$SAMP
ros2 run metro_obstacle_detector bag_tools coverage --bag /data/bag --name $(basename "$BAG") --log /results/stream.jsonl --out /results/coverage.json > /dev/null
"
{ date -Is; uptime; } > "$OUT/host_load_after.txt"
python3 - "$OUT" <<'PY'
import json, sys, statistics as st
o = sys.argv[1]
c = json.load(open(o + "/coverage.json")); s = [json.loads(l) for l in open(o + "/samples.jsonl")]
act = [x["cpu_percent"] for x in s if x["cpu_percent"] > 5]
r = {"bag": c["bag"], "sent": c["sent"], "processed": c["processed"], "not_processed": c["not_processed"],
     "processing_ms": c["processing_ms"], "inter_arrival_s": c["inter_arrival_s"],
     "node_cpu_s_total": s[-1]["cpu_s"] if s else None,
     "node_cpu_ms_per_processed_frame": round(1000 * s[-1]["cpu_s"] / c["processed"], 1) if s and c["processed"] else None,
     "node_cpu_percent_median_active": st.median(act) if act else None, "rss_mb_max": max(x["rss_mb"] for x in s) if s else None,
     "affinity": s[-1]["affinity"] if s else None, "load_before": open(o + "/host_load_before.txt").read().splitlines()[1]}
json.dump(r, open(o + "/summary.json", "w"), indent=1)
print(json.dumps(r))
PY
