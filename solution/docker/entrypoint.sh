#!/bin/bash
# Container entrypoint: source ROS 2 Humble and the built workspace, then run the command.
set -e
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
exec "$@"
