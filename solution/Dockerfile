# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage "env": Ubuntu 22.04 (Jammy) + ROS 2 Humble + all runtime dependencies.
# Stage "solution": copies the sources, builds the ROS 2 workspace and sets
# an entrypoint that sources Humble and the built workspace.
# ---------------------------------------------------------------------------
FROM ros:humble-ros-base-jammy AS env
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# NumPy/SciPy work on small matrices here; BLAS thread pools only add contention
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros2bag \
    ros-humble-rosbag2-transport \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosbag2-py \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py \
    ros-humble-std-msgs \
    ros-humble-visualization-msgs \
    ros-humble-rviz2 \
    python3-numpy \
    python3-scipy \
    python3-yaml \
    python3-matplotlib \
    python3-pytest \
    python3-psutil \
    sqlite3 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" imos \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
       --create-home --shell /bin/bash imos \
    && mkdir -p /workspace /results \
    && chown -R imos:imos /workspace /results
WORKDIR /workspace
CMD ["bash"]

# ---------------------------------------------------------------------------
FROM env AS solution
# The entrypoint sources ROS for the main process only. Shells started with
# `docker exec` source it here: login shells (`bash -lc '...'`) via profile.d,
# interactive shells (`docker exec -it ... bash`) via bash.bashrc. Commands
# without a shell: `docker exec <c> /workspace/entrypoint.sh ros2 ...`.
RUN printf '%s\n' \
      'if [ -n "${BASH_VERSION:-}" ] && [ -z "${METRO_ROS_SOURCED:-}" ]; then' \
      '  source /opt/ros/humble/setup.bash' \
      '  [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' \
      '  export METRO_ROS_SOURCED=1' \
      'fi' > /etc/profile.d/metro_ros.sh \
    && echo 'source /etc/profile.d/metro_ros.sh' >> /etc/bash.bashrc
USER imos
WORKDIR /workspace
COPY --chown=imos:imos src/ /workspace/src/
COPY --chown=imos:imos config/ /workspace/config/
COPY --chown=imos:imos scripts/ /workspace/scripts/
COPY --chown=imos:imos tests/ /workspace/tests/
COPY --chown=imos:imos docker/entrypoint.sh /workspace/entrypoint.sh
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install \
    && rm -rf build log
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
