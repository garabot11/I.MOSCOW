#!/bin/bash
# Live demo in ONE Humble container with RViz2 (X11): detector node + RViz + bag playback.
# usage: run_demo.sh <bag_dir_on_host> [input_topic|auto] [rate] [results_dir_on_host] [image]
# Relative paths are allowed. input_topic "auto" (default) = the PointCloud2 topic of the bag. The RViz fixed frame is the real header.frame_id of the first cloud
# of the bag (read inside the container with `bag_tools info --frame-id`); FIXED_FRAME=<frame>
# overrides it. The X11 cookie is passed through a private Xauthority file (no `xhost +`).
set -e
BAG=${1:?bag directory}; TOPIC=${2:-auto}; RATE=${3:-1.0}; RES=${4:-results/demo}; IMG=${5:-metro-lidar:final}
BAG=$(cd "$BAG" && pwd -P)
mkdir -p "$RES"; RES=$(cd "$RES" && pwd -P)
if [ "$TOPIC" = auto ]; then
  TOPIC=$(docker run --rm -v "$BAG":/data/bag:ro "$IMG" \
    ros2 run metro_obstacle_detector bag_tools info --bag /data/bag --cloud-topic | tail -n 1)
fi
if [ -z "${FIXED_FRAME:-}" ]; then
  FIXED_FRAME=$(docker run --rm -v "$BAG":/data/bag:ro "$IMG" \
    ros2 run metro_obstacle_detector bag_tools info --bag /data/bag --topic "$TOPIC" --frame-id | tail -n 1)
fi
[ -n "$FIXED_FRAME" ] || { echo "cannot read header.frame_id from $BAG; set FIXED_FRAME=<frame>" >&2; exit 1; }
echo "bag: $BAG  topic: $TOPIC  fixed frame: $FIXED_FRAME  results: $RES"
IMOS_XAUTH="$(mktemp /tmp/imos-xauth.XXXXXX)"
trap 'rm -f -- "$IMOS_XAUTH"' EXIT
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$IMOS_XAUTH" nmerge -
chmod 600 "$IMOS_XAUTH"; test -s "$IMOS_XAUTH"
docker run --rm --name metro-lidar-demo --network host --shm-size=1g \
  -e ROS_DOMAIN_ID=78 -e ROS_LOCALHOST_ONLY=1 \
  -e DISPLAY -e XAUTHORITY=/tmp/imos.xauth -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_X11_NO_MITSHM=1 \
  -v "$IMOS_XAUTH":/tmp/imos.xauth:ro -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$BAG":/data/bag:ro -v "$RES":/results "$IMG" bash -c "
source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash
ros2 launch metro_obstacle_detector detector.launch.py input_topic:=$TOPIC rviz:=true fixed_frame:=$FIXED_FRAME rviz_geometry:=${RVIZ_GEOMETRY:-1500x1000+20+40} log_path:=/results/demo.jsonl > /results/node.log 2>&1 &
sleep 6
ros2 run metro_obstacle_detector status_monitor --no-color --tail-file /results/status_tail.txt --tail-lines 10 > /results/status_monitor.log 2>&1 &
echo \"bag: $(basename "$BAG")   topic: $TOPIC   playback rate: $RATE   (ros2 bag play inside the Humble container)\" > /results/demo_info.txt
ros2 bag play /data/bag --rate $RATE --read-ahead-queue-size 20 --disable-keyboard-controls --wait-for-all-acked 5000 ${PLAY_ARGS:-}
sleep 2
" || true
