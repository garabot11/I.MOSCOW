#!/bin/bash
# Host launcher of the metro LiDAR obstacle detector (team IRS-MIREA).
#
#   ./run.sh [--data DIR] [--results DIR] [--cpu] [--no-gui] [--udp] [--topic T] [--image IMG] [--cpuset-cpus LIST] [--detach]
#        starts the persistent container "metro-lidar" (GPU when available, otherwise the emergency
#        CPU mode with an explanation), runs the self-check, the detector and RViz, then shows the
#        live console (Ctrl-C leaves the console, the detector keeps running)
#   ./run.sh play <bag|name> [rate] [ros2 bag play args]   play a bag (SPACE pause, UP/DOWN rate)
#   ./run.sh bags | status | doctor [--full] | console | banner | shell
#   ./run.sh stop                                          stop and remove the container
#
# --data     host directory with the rosbag2 directories (any depth; mounted read-only at /data;
#            repeatable: several go to /data/<name>), default: $METRO_DATA, else ../dataset next to this script
# --results  host directory for logs and results (mounted at /results), default ~/metro-lidar-results
# --cpu      force the CPU mode;  --no-gui  no RViz;  --udp  Fast DDS over UDP only (needed when the bag
#            is played OUTSIDE the container, e.g. by `ros2 bag play` on the host)
# --topic    PointCloud2 topic of the detector for a player outside the container (default auto: the only
#            cloud topic on the ROS domain, or /lidar_points, where ./run.sh play publishes, among several)
set -u
NAME=${METRO_CONTAINER:-metro-lidar}
LABEL=org.irs-mirea.metro-lidar=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
IMG=${METRO_IMAGE:-metro-lidar:gpu-dev}

if [ -t 1 ]; then B=$'\033[1m'; R=$'\033[1;31m'; G=$'\033[32m'; Y=$'\033[33m'; N=$'\033[0m'; else B= R= G= Y= N=; fi
say()  { printf '%s\n' "${B}[run.sh]${N} $*"; }
warn() { printf '%s\n' "${Y}[run.sh] $*${N}"; }
die()  { printf '%s\n' "${R}[run.sh] $*${N}" >&2; exit 1; }

TTY=(-i); [ -t 0 ] && [ -t 1 ] && TTY=(-it)

need_docker() {
  command -v docker > /dev/null 2>&1 || die "docker is not installed: https://docs.docker.com/engine/install/ubuntu/"
  docker info > /dev/null 2>&1 || die "cannot talk to the Docker daemon: sudo systemctl start docker; and add yourself to the docker group: sudo usermod -aG docker \$USER (then log in again)"
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" = true ]; }
ours()    { [ "$(docker inspect -f '{{index .Config.Labels "org.irs-mirea.metro-lidar"}}' "$NAME" 2> /dev/null)" = 1 ]; }

in_container() {  # run a metro command in the running container
  running || die "the container is not running: start it with ./run.sh"
  docker exec "${TTY[@]}" "$NAME" metro "$@"
}

# ---------------------------------------------------------------------------------------- GPU decision
GPU_ARGS=(); GPU_CODE=""; GPU_DETAIL=""; HOST_DRIVER=""
decide_gpu() {
  local cpu=$1 out rt
  if [ "$cpu" = 1 ]; then GPU_CODE=cpu_forced; return; fi
  # test hook of docker/failure_matrix.sh: behave as on a host without the NVIDIA Container Toolkit
  if [ "${METRO_TEST_NO_TOOLKIT:-0}" = 1 ]; then GPU_CODE=no_toolkit; GPU_DETAIL="docker runtimes: runc (simulated)"; return; fi
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    GPU_CODE=no_host_gpu; GPU_DETAIL="nvidia-smi not found on the host"; return
  fi
  if ! out=$(timeout 20 nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>&1) || [ -z "$out" ]; then
    GPU_CODE=no_host_gpu; GPU_DETAIL="nvidia-smi: $(printf '%s' "$out" | head -n 1)"; return
  fi
  HOST_DRIVER=$(printf '%s' "$out" | head -n 1 | cut -d, -f1 | tr -d ' ')
  say "host GPU: $(printf '%s' "$out" | head -n 1 | cut -d, -f2- | sed 's/^ *//'), driver $HOST_DRIVER"
  if [ "${HOST_DRIVER%%.*}" -lt 570 ] 2> /dev/null; then
    GPU_CODE=old_driver; GPU_DETAIL="host driver $HOST_DRIVER"
  fi
  # names only ({{json .Runtimes}} is a multi-KB status blob on Docker >= 29)
  rt=$(docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2> /dev/null | cut -c1-160)
  if [[ "$rt" != *nvidia* ]] && ! command -v nvidia-container-runtime-hook > /dev/null 2>&1 && ! command -v nvidia-ctk > /dev/null 2>&1; then
    GPU_CODE=no_toolkit; GPU_DETAIL="docker runtimes: ${rt% }"; return
  fi
  # the toolkit is there: prove that a container really gets the GPU
  if ! out=$(timeout 90 docker run --rm --gpus all --entrypoint true "$IMG" 2>&1); then
    GPU_CODE=no_toolkit; GPU_DETAIL="docker run --gpus all failed: $(printf '%s' "$out" | tail -n 1 | cut -c1-200)"; return
  fi
  GPU_ARGS=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all)
}

# ---------------------------------------------------------------------------------------- start
cmd_start() {
  local datas=() results="${METRO_RESULTS:-$HOME/metro-lidar-results}" cpu=0 gui=1 udp=0 cpuset="" detach=0 topic=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --data) datas+=("${2:?--data DIR}"); shift ;;
      --results) results=${2:?--results DIR}; shift ;;
      --cpu) cpu=1 ;;
      --no-gui|--no-rviz) gui=0 ;;
      --udp) udp=1 ;;
      --topic) topic=${2:?--topic T}; shift ;;
      --image) IMG=${2:?--image IMG}; shift ;;
      --cpuset-cpus) cpuset=${2:?--cpuset-cpus LIST}; shift ;;
      --detach) detach=1 ;;
      -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
      *) die "unknown option $1 (./run.sh --help)" ;;
    esac
    shift
  done
  need_docker
  if running; then
    ours || die "a container named $NAME exists and was not started by run.sh: docker rm -f $NAME"
    say "the container $NAME is already running (options are kept; ./run.sh stop to change them)"
    docker exec "${TTY[@]}" "$NAME" metro start
    exit 0
  fi
  if docker inspect "$NAME" > /dev/null 2>&1; then
    ours || die "a container named $NAME exists and was not started by run.sh: docker rm -f $NAME"
    docker rm -f "$NAME" > /dev/null
  fi
  docker image inspect "$IMG" > /dev/null 2>&1 || die "image $IMG not found. Load the archive: gunzip -c metro-lidar.tar.gz | docker load   (or build it: docker build -t $IMG $HERE)"
  if [ ${#datas[@]} -eq 0 ]; then
    if [ -n "${METRO_DATA:-}" ]; then datas=("$METRO_DATA"); elif [ -d "$HERE/../dataset" ]; then datas=("$HERE/../dataset"); fi
  fi
  local mounts=() d data=""
  for d in "${datas[@]}"; do
    [ -d "$d" ] || die "data directory $d does not exist (./run.sh --data <directory with rosbag2 directories>)"
    d=$(cd "$d" && pwd -P)
    if [ ${#datas[@]} -eq 1 ]; then mounts+=(-v "$d":/data:ro); data="$d -> /data"
    else mounts+=(-v "$d":/data/"$(basename "$d")":ro); data="$data $d -> /data/$(basename "$d")"; fi
  done
  [ ${#datas[@]} -eq 0 ] && warn "no --data given: no bags mounted (./run.sh --data <dir>)"
  mkdir -p "$results/logs" 2> /dev/null || die "cannot create $results (./run.sh --results <writable dir>)"
  results=$(cd "$results" && pwd -P); mounts+=(-v "$results":/results)
  local user=()
  [ "$(id -u)" != 1000 ] && user=(--user "$(id -u):$(id -g)" -e HOME=/tmp/metro-home)

  decide_gpu $cpu
  local env=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" -e METRO_DATA=/data -e METRO_RESULTS_DIR=/results
             -e CUDA_CACHE_PATH=/results/.nv/ComputeCache -e CUDA_CACHE_MAXSIZE=2147483648)
  [ -n "${RMW_IMPLEMENTATION:-}" ] && env+=(-e "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION")
  [ -n "${RVIZ_GEOMETRY:-}" ] && env+=(-e "RVIZ_GEOMETRY=$RVIZ_GEOMETRY")    # RViz window WxH+X+Y (e.g. for recordings)
  [ $udp = 1 ] && env+=(-e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_udp_only.xml)
  [ $cpu = 1 ] && env+=(-e METRO_DEVICE=cpu)
  [ -n "$GPU_CODE" ] && env+=(-e "METRO_GPU_REASON_CODE=$GPU_CODE" -e "METRO_GPU_REASON_DETAIL=$GPU_DETAIL")
  [ -n "$HOST_DRIVER" ] && env+=(-e "METRO_HOST_DRIVER=$HOST_DRIVER")
  if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    say "${G}GPU mode${N}: the container gets --gpus all"
  elif [ "$GPU_CODE" = cpu_forced ]; then
    warn "CPU mode forced by --cpu"
  else
    warn "GPU NOT available ($GPU_CODE: $GPU_DETAIL): emergency CPU mode; the banner below says how to fix it"
  fi

  local x11=() xauth_file=""
  if [ $gui = 1 ] && [ -n "${DISPLAY:-}" ]; then
    # private cookie file (mktemp, mode 600) in the user's runtime dir; removed by ./run.sh stop
    xauth_file=$(mktemp "${XDG_RUNTIME_DIR:-/tmp}/metro-lidar.xauth.XXXXXX") || die "cannot create a temporary file for the X cookie"
    if command -v xauth > /dev/null 2>&1; then
      xauth nlist "$DISPLAY" 2> /dev/null | sed -e 's/^..../ffff/' | xauth -f "$xauth_file" nmerge - 2> /dev/null
    fi
    x11=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/metro.xauth -e QT_X11_NO_MITSHM=1
         -v "$xauth_file":/tmp/metro.xauth:ro --label "org.irs-mirea.metro-lidar.xauth=$xauth_file")
    if [ -d /tmp/.X11-unix ]; then x11+=(-v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    else warn "/tmp/.X11-unix does not exist on the host: only a TCP display (DISPLAY=host:N) can work"; fi
    [ -s "$xauth_file" ] || warn "no X cookie for $DISPLAY (xauth missing?): RViz may not connect; then: xhost +si:localuser:\$(id -un)"
  elif [ $gui = 1 ]; then
    warn "no DISPLAY on the host: starting without RViz (status in the terminal)"
    gui=0
  fi
  local extra=()
  [ -n "$cpuset" ] && extra+=(--cpuset-cpus "$cpuset")

  say "starting container $NAME from $IMG (data, read-only: ${data:-none}; results: $results -> /results)"
  if ! docker run -d --name "$NAME" --label "$LABEL" --network host --shm-size=1g --init \
        "${user[@]}" "${GPU_ARGS[@]}" "${env[@]}" "${x11[@]}" "${mounts[@]}" "${extra[@]}" \
        "$IMG" sleep infinity > /dev/null 2> "$results/logs/docker_run.err"; then
    [ -n "$xauth_file" ] && rm -f "$xauth_file"
    die "docker run failed: $(tail -n 1 "$results/logs/docker_run.err")"
  fi
  local sargs=()
  [ $gui = 0 ] && sargs+=(--no-rviz)
  [ $cpu = 1 ] && sargs+=(--cpu)
  [ $detach = 1 ] && sargs+=(--no-console)
  [ -n "$topic" ] && sargs+=(--topic "$topic")
  docker exec "${TTY[@]}" "$NAME" metro start "${sargs[@]}"
  say "next steps:"
  say "  ./run.sh play <bag> [rate]   play a bag (./run.sh bags lists them); another play switches the bag"
  say "  ./run.sh status | console | doctor | stop"
}

cmd_stop() {
  need_docker
  if ! docker inspect "$NAME" > /dev/null 2>&1; then say "no container $NAME"; exit 0; fi
  ours || die "the container $NAME was not started by run.sh; not touching it"
  local xf
  xf=$(docker inspect -f '{{index .Config.Labels "org.irs-mirea.metro-lidar.xauth"}}' "$NAME" 2> /dev/null)
  running && docker exec "$NAME" metro stop > /dev/null 2>&1
  docker rm -f "$NAME" > /dev/null && say "container $NAME stopped and removed"
  case "$xf" in */metro-lidar.xauth.*) rm -f -- "$xf" ;; esac
}

cmd=${1:-start}
case "$cmd" in
  --*|start) [ "$cmd" = start ] && shift; cmd_start "$@" ;;
  play|bags|status|doctor|console|banner) shift; need_docker; in_container "$cmd" "$@" ;;
  shell) need_docker; running || die "not running: ./run.sh"; docker exec "${TTY[@]}" "$NAME" bash ;;
  stop) cmd_stop ;;
  -h|--help|help) sed -n '2,18p' "$0" ;;
  *) die "unknown command $cmd (./run.sh --help)" ;;
esac
