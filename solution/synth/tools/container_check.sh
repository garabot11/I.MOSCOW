#!/bin/bash
# Readability proof of a synthetic set inside the submitted image (read-only mounts):
#   ros2 bag info + the project's offline tool on every scene, ros2 bag play + ros2 topic hz on the first one.
#   solution/synth/tools/container_check.sh /home/gag/i.mos/synthetic/val_hesai_insert /tmp/out_dir [image]
set -euo pipefail
SET_DIR=$(readlink -f "$1")
mkdir -p "$2"
OUT_DIR=$(readlink -f "$2")
IMAGE=${3:-metro-lidar:final}
mkdir -p "$OUT_DIR"
docker run --rm -v "$SET_DIR":/set:ro -v "$OUT_DIR":/out "$IMAGE" bash -lc '
set -e
first=""
for s in /set/*/; do
  n=$(basename "$s")
  [ -z "$first" ] && first="$s"
  ros2 bag info "$s/bag" > "/out/${n}_info.txt" 2>&1
  extra=""
  if grep -q "falcon_prime" "$s/scene.json"; then extra="--set forward_axis=+x --set up_axis=+z"; fi
  ros2 run metro_obstacle_detector offline --bag "$s/bag" --out "/out/${n}.jsonl" --quiet $extra > "/out/${n}_offline.txt" 2>&1 || echo "offline exit $? for $n"
  msgs=$(grep -E "^Messages:" "/out/${n}_info.txt" | awk "{print \$2}")
  python3 - "$n" "/out/${n}_summary.json" "$msgs" <<PY
import json, sys
n, path, msgs = sys.argv[1], sys.argv[2], sys.argv[3]
s = json.load(open(path))
print(json.dumps({"scene": n, "bag_info_messages": int(msgs), "offline_read": s["messages_read"], "complete": s["complete"],
                  "status_counts": s["status_counts"]}))
PY
done
topic=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(\"$first/bag/metadata.yaml\"))[\"rosbag2_bagfile_information\"][\"topics_with_message_count\"][0][\"topic_metadata\"][\"name\"])")
( timeout 25 ros2 topic hz "$topic" --window 20 > /out/play_hz.txt 2>&1 & )
sleep 3
ros2 bag play "$first/bag" > /out/play.txt 2>&1
echo "play exit $?"
sleep 2
tail -4 /out/play_hz.txt
'
