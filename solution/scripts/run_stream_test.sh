#!/bin/bash
# Stream test in one Humble container: detector node + an independent status observer +
# `ros2 bag play --rate 1.0` of the WHOLE bag, then a coverage report (`bag_tools coverage`):
#   sent      = messages of the bag (the player publishes every one of them when it exits with 0),
#   processed = records of the node's JSONL log, published = statuses seen by the observer,
# all matched by header stamp, never by counters.
# usage: run_stream_test.sh <bag_dir_on_host> <results_dir_on_host> [input_topic|auto] [max_seconds] [image]
#   input_topic  default "auto": the node finds the PointCloud2 topic itself (input_topic:=auto)
#   max_seconds  optional safety limit for the playback (default 0 = none); if it is hit the player
#                exit code is 124, the report says so and the script fails.
# env: QOS=reliable|best_effort (default reliable: the recorded profile and the node default),
#      QOS_DEPTH (subscription depth, default 1 = freshest frame), DOMAIN (ROS_DOMAIN_ID, default 78)
# Relative paths are allowed. Output in <results>: stream.jsonl, observed_status.jsonl,
# coverage.json, node.log, play.log, observer.log. Exit code = player exit code.
set -u
BAG=${1:?bag directory}; RES=${2:?results directory}; TOPIC=${3:-auto}; SECS=${4:-0}; IMG=${5:-metro-lidar:final}
BAG=$(cd "$BAG" && pwd -P) || { echo "no such bag directory: $1" >&2; exit 2; }
mkdir -p "$RES" && RES=$(cd "$RES" && pwd -P) || { echo "cannot create results directory: $2" >&2; exit 2; }
QOS=${QOS:-reliable}; DEPTH=${QOS_DEPTH:-1}
LIMIT=""; [ "$SECS" -gt 0 ] && LIMIT="timeout ${SECS}s"
echo "stream test: bag $BAG, topic $TOPIC, QoS $QOS depth $DEPTH, results $RES"
docker run --rm --network host --shm-size=1g -e ROS_DOMAIN_ID="${DOMAIN:-78}" -e ROS_LOCALHOST_ONLY=1 \
  -v "$BAG":/data/bag:ro -v "$RES":/results "$IMG" bash -c "
source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash
set -m   # job control: background jobs keep SIGINT (a non-interactive shell would start them with SIGINT ignored)
stop() { kill -INT \$1 2>/dev/null; for i in \$(seq 1 30); do kill -0 \$1 2>/dev/null || break; sleep 0.5; done
         kill -0 \$1 2>/dev/null && { echo \"pid \$1 ignored SIGINT, sending SIGTERM\" >&2; kill -TERM \$1; }; wait \$1 2>/dev/null; }
rm -f /results/stream.jsonl /results/observed_status.jsonl /results/coverage.json
python3 -m metro_obstacle_detector.status_monitor --quiet --jsonl /results/observed_status.jsonl > /results/observer.log 2>&1 &
OBS=\$!
ros2 launch metro_obstacle_detector detector.launch.py input_topic:=$TOPIC qos_reliability:=$QOS qos_depth:=$DEPTH log_path:=/results/stream.jsonl > /results/node.log 2>&1 &
NODE=\$!
# ready = the node publishes UNKNOWN for the missing input (after stale_timeout_s) and the observer sees it
for i in \$(seq 1 60); do [ -s /results/observed_status.jsonl ] && break; sleep 0.5; done
[ -s /results/observed_status.jsonl ] || echo 'WARNING: no status before playback' >&2
# The player starts paused and is resumed once its read-ahead queue is full and discovery is done:
# started unpaused, Humble's player runs its clock while it fills the queue and then sends the first
# frames in a burst, which a depth-1 subscriber cannot keep (a player artefact, not a detector loss).
$LIMIT ros2 bag play /data/bag --rate 1.0 --start-paused --read-ahead-queue-size 20 --disable-keyboard-controls --wait-for-all-acked 5000 > /results/play.log 2>&1 &
PLAYER=\$!
sleep 3
ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume > /results/resume.log 2>&1 || echo 'WARNING: resume service call failed' >&2
wait \$PLAYER; PLAY=\$?
sleep 3   # let the last frames finish; the node then reports UNKNOWN for the stopped input
stop \$NODE
stop \$OBS
ros2 run metro_obstacle_detector bag_tools coverage --bag /data/bag --name $(basename "$BAG") --topic $TOPIC --log /results/stream.jsonl \
  --observed /results/observed_status.jsonl --player-exit \$PLAY --out /results/coverage.json
[ \$PLAY -eq 0 ] || echo \"player exit code \$PLAY (124 = stopped by the max_seconds limit): not a whole-bag test\" >&2
exit \$PLAY
"
