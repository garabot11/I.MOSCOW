# ---------------------------------------------------------------------------
# Stage "env": Ubuntu 22.04 (Jammy) + ROS 2 Humble + all runtime dependencies,
#   incl. PyTorch 2.11.0+cu128 with its own CUDA 12.8 libraries (kernels for
#   sm_75/80/86/90/100/120: the RTX 5060 sm_120 natively, the RTX 4070 Ti SUPER
#   sm_89 through the sm_86 cubins (CUDA binary compatibility within major 8);
#   host driver >= 570) that also
#   runs on the CPU, and Mesa for software OpenGL (RViz without a GPU).
# Stage "solution": copies the sources, builds the ROS 2 workspace, puts the
#   `metro` command on PATH and sets an entrypoint that sources Humble and the
#   built workspace.
# Builds on a machine without a GPU (nothing here needs the driver).
#   docker build -t metro-lidar:gpu-dev solution
# ---------------------------------------------------------------------------
FROM ros:humble-ros-base-jammy AS env
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# NumPy/SciPy work on small matrices here; BLAS thread pools only add contention.
# PyTorch would also default to OMP_NUM_THREADS intra-op threads: the runtime
# (metro_obstacle_detector.device) sets torch.set_num_threads() explicitly.
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
# Graphics: glxinfo (mesa-utils) and xdpyinfo (x11-utils) for the self-check,
# glvnd + Mesa (software GL fallback: LIBGL_ALWAYS_SOFTWARE=1). With
# `--gpus all` the NVIDIA Container Toolkit mounts the driver's GL/EGL/CUDA
# libraries (NVIDIA_DRIVER_CAPABILITIES=all below includes "graphics").
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    mesa-utils \
    x11-utils \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libglx-mesa0 libegl-mesa0 \
    ros-humble-tf2-ros \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
# PyTorch CUDA 12.8 build (pip wheel, ships its own CUDA/cuDNN libraries; only
# the host driver is needed). It does not depend on NumPy, so Humble's
# python3-numpy 1.21 stays the one in use (checked: torch.from_numpy works).
# Exact versions of torch and of every dependency: docker/requirements-torch.txt. No BuildKit
# features are used, so the classic builder (docker.io without buildx) builds this file as well.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
COPY docker/requirements-torch.txt /tmp/requirements-torch.txt
RUN pip3 install --no-cache-dir --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple -r /tmp/requirements-torch.txt \
    && pip3 check && rm -f /tmp/requirements-torch.txt \
    && python3 -c "import torch, numpy; assert torch.from_numpy(numpy.zeros(3)).sum().item() == 0; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__)"
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
# without a shell: `docker exec <c> /workspace/entrypoint.sh ros2 ...`; the
# `metro` command sources ROS itself.
RUN printf '%s\n' \
      'if [ -n "${BASH_VERSION:-}" ] && [ -z "${METRO_ROS_SOURCED:-}" ]; then' \
      '  source /opt/ros/humble/setup.bash' \
      '  [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' \
      '  export METRO_ROS_SOURCED=1' \
      'fi' > /etc/profile.d/metro_ros.sh \
    && echo 'source /etc/profile.d/metro_ros.sh' >> /etc/bash.bashrc
COPY scripts/metro /usr/local/bin/metro
RUN chmod 755 /usr/local/bin/metro
USER imos
WORKDIR /workspace
COPY --chown=imos:imos src/ /workspace/src/
COPY --chown=imos:imos config/ /workspace/config/
COPY --chown=imos:imos scripts/ /workspace/scripts/
COPY --chown=imos:imos tests/ /workspace/tests/
COPY --chown=imos:imos models/ /workspace/models/
# the synthetic-data generator: imported by tests/test_synth_*.py (and usable in the container)
COPY --chown=imos:imos synth/ /workspace/synth/
COPY --chown=imos:imos docker/ /workspace/docker/
COPY --chown=imos:imos docker/entrypoint.sh /workspace/entrypoint.sh
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install \
    && rm -rf build log
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
