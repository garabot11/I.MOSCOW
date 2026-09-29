#!/bin/bash
# Open a terminal window on the host X session that shows the status monitor of the running demo container.
# usage: status_window.sh [container_name]
NAME=${1:-metro-lidar-demo}
until docker ps --format '{{.Names}}' | grep -q "^$NAME$"; do sleep 1; done
sleep 3
CMD="docker exec -it $NAME bash -c 'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; echo \"== metro obstacle detector: /obstacles/status (playback: \$(cat /results/demo_info.txt)) ==\"; ros2 run metro_obstacle_detector status_monitor'"
if command -v gnome-terminal > /dev/null; then
  gnome-terminal --geometry=150x22+20+1080 --title="metro obstacle detector - /obstacles/status" -- bash -c "$CMD; sleep 5" &
elif command -v xterm > /dev/null; then
  xterm -geometry 150x22+20+1080 -T "metro obstacle detector - /obstacles/status" -e bash -c "$CMD; sleep 5" &
fi
