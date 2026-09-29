#!/bin/bash
# Resource probe: same protocol as scripts/run_stream_test.sh plus a psutil sampler (CPU%, RSS) of node and player.
BAG=$(cd "$1" && pwd -P); TOPIC=$2; OUT=$(mkdir -p "$3" && cd "$3" && pwd -P); HERE=$(cd "$(dirname "$0")" && pwd)
docker run --rm --network host --shm-size=1g -e ROS_DOMAIN_ID=79 -e ROS_LOCALHOST_ONLY=1 \
  -v "$BAG":/data/bag:ro -v "$OUT":/results -v "$HERE":/probe:ro metro-lidar:final bash -c "
source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; set -m
python3 /probe/sample_node.py /results/samples.jsonl & SAMP=\$!
ros2 launch metro_obstacle_detector detector.launch.py input_topic:=$TOPIC log_path:=/results/stream.jsonl > /results/node.log 2>&1 & NODE=\$!
sleep 4
ros2 bag play /data/bag --rate 1.0 --start-paused --read-ahead-queue-size 20 --disable-keyboard-controls --wait-for-all-acked 5000 > /results/play.log 2>&1 & PLAYER=\$!
sleep 3; ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume > /dev/null
wait \$PLAYER; sleep 2
kill -INT \$NODE; wait \$NODE; kill \$SAMP
nproc > /results/nproc.txt
"
