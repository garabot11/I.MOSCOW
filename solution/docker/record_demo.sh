#!/bin/bash
# Demo video for the judges (host, X11 desktop session, ffmpeg): RViz on the GPU + burnt-in captions
# and status lines, driven through run.sh / metro exactly as an operator would.
#
#   solution/docker/record_demo.sh [--data DATASET_DIR] [--out DIR] [--image IMG] [--name NAME]
#
# Scenes (all at rate 1.0, i.e. real time):
#   1 for_hackathon/doubleT_obstacle: obstacle at ~56 m; SPACE pause and DOWN slow-down sent through a
#     pseudo-terminal like real key presses
#   2 extra/cloud_with_fake_obj (switch bag without restart): inserted wall from ~76 m, 0.3 m cubes, low plate
#   3-5 org_synth_markers/synthetic_{platform,switch,tunnels}_cloud_with_markers: person from ~119 m, cube on
#     the rail, hanging cable, large cube and the beam across the rails from ~74 m
#   6 falcon/falcon_test_second_dark from t = 262 s: dark person from ~154 m (the Falcon axes are detected
#     automatically, the RViz view turns the cloud so that it faces the same way as the Hesai clouds)
# Output: <out>/<name>.mp4 (1920x1080, H.264) and <out>/<name>.txt (timestamps of what is shown).
# The RViz window is placed at the top-left corner and only its area is captured: keep other windows
# off it while recording (about 4.5 minutes).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA=/home/gag/i.mos/dataset
OUT=/home/gag/i.mos/work/video
IMG=${METRO_IMAGE:-metro-lidar:gpu-dev}
NAME=metro_demo_$(date +%Y%m%d_%H%M)
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift ;; --out) OUT=$2; shift ;; --image) IMG=$2; shift ;; --name) NAME=$2; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
[ -n "${DISPLAY:-}" ] || { echo "needs an X11 desktop session (DISPLAY)"; exit 1; }
command -v ffmpeg > /dev/null || { echo "needs ffmpeg on the host: sudo apt-get install -y ffmpeg"; exit 1; }
export METRO_CONTAINER=metro-lidar-demo METRO_IMAGE=$IMG RVIZ_GEOMETRY=1920x1080+0+0
C=$METRO_CONTAINER
RUNSH="$HERE/run.sh"
mkdir -p "$OUT"; RES="$OUT/${NAME}_results"; mkdir -p "$RES/logs"
TXT="$OUT/$NAME.txt"; CAP="$RES/caption.txt"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
T0=0
caption() {  # caption <text>: shown in the video and logged with the video time
  printf '%s\n' "$*" > "$CAP.tmp"; mv "$CAP.tmp" "$CAP"
  [ "$T0" != 0 ] && printf '%s  %s\n' "$(date -u -d @$(( $(date +%s) - T0 )) +%M:%S)" "$*" >> "$TXT"
}
keys_play() {  # keys_play <script of "sleep N / key K"> -- metro play args : play in a pty, feed keys
  local spec=$1; shift 2
  { for step in $spec; do
      case "$step" in
        s*) sleep "${step#s}" ;;
        SPACE) printf ' ' ;;
        DOWN) printf '\033[B' ;;
        UP) printf '\033[A' ;;
        INT) printf '\003' ;;
      esac
    done; sleep 2; } | script -qfc "docker exec -it $C metro play $*" /dev/null > "$RES/play_$(date +%s).log" 2>&1
}
PP=""
play_bg() {  # play_bg <log name> <metro play args...>: another play switches the bag, like an operator would
  ( "$RUNSH" play "${@:2}" > "$RES/play_$1.log" 2>&1 ) &
  PP=$!
}
stop_play() {
  docker exec "$C" bash -c 'kill -INT "$(cat /tmp/metro-run/player.pid)"' 2> /dev/null; wait $PP
}

printf 'metro obstacle detector - starting ...\n' > "$RES/logs/status_tail.txt"
caption "IRS-MIREA: обнаружение препятствий в тоннеле метро (ROS 2 Humble, GPU)"
"$RUNSH" stop > /dev/null 2>&1
"$RUNSH" --data "$DATA" --results "$RES" --detach > "$RES/start.log" 2>&1
grep -q "detector started" "$RES/start.log" || { echo "start failed, see $RES/start.log"; exit 1; }
START_LINES=$(grep -E "GPU mode|D3 on|detector started" "$RES/start.log" | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^\[[a-z.]*\] //' | head -n 3)

# the RViz window: capture exactly its area
WIN=""
for _ in $(seq 1 30); do
  # the RViz client window (title "... - RViz", class rviz2), not the window manager's frame around it
  WIN=$(xwininfo -root -tree 2> /dev/null | grep -E '"[^"]* - RViz": \("rviz2" "rviz2"\)' | awk '{print $1}' | head -n 1)
  [ -n "$WIN" ] && break; sleep 1
done
[ -n "$WIN" ] || { echo "RViz window not found (no display/GL?)"; "$RUNSH" stop; exit 1; }
X=$(xwininfo -id "$WIN" | awk '/Absolute upper-left X/{print $4}'); Y=$(xwininfo -id "$WIN" | awk '/Absolute upper-left Y/{print $4}')
W=$(xwininfo -id "$WIN" | awk '/Width/{print $2}'); H=$(xwininfo -id "$WIN" | awk '/Height/{print $2}')
SW=$(xdpyinfo -display "$DISPLAY" | awk '/dimensions/{split($2, a, "x"); print a[1]}')
SH=$(xdpyinfo -display "$DISPLAY" | awk '/dimensions/{split($2, a, "x"); print a[2]}')
[ $((X + W)) -gt "$SW" ] && W=$((SW - X)); [ $((Y + H)) -gt "$SH" ] && H=$((SH - Y))   # stay inside the screen
W=$(( W / 2 * 2 )); H=$(( H / 2 * 2 ))
echo "RViz window $WIN at ${X},${Y} size ${W}x${H}"
# capture only once the window is mapped and drawn: before that the area shows the terminal underneath
for _ in $(seq 1 30); do xwininfo -id "$WIN" | grep -q 'Map State: IsViewable' && break; sleep 1; done
sleep 8
# the last 6 status lines only (the monitor's tail file keeps 20)
SHORT="$RES/status_short.txt"; printf '%s\n' "$START_LINES" > "$SHORT"
( while sleep 0.5; do [ -s "$RES/logs/status_tail.txt" ] && tail -n 6 "$RES/logs/status_tail.txt" | cut -c1-150 > "$SHORT.tmp" && mv "$SHORT.tmp" "$SHORT"; done ) &
TP=$!
ffmpeg -loglevel error -y -f x11grab -framerate 20 -video_size "${W}x${H}" -i "$DISPLAY+$X,$Y" \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile=$FONT:textfile=$CAP:reload=1:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=10:x=20:y=20,drawtext=fontfile=$FONT:textfile=$SHORT:reload=1:fontsize=19:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=8:x=20:y=75" \
  -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p "$OUT/$NAME.mp4" < /dev/null &
FF=$!
T0=$(date +%s)
: > "$TXT"
printf '%s\n' "video $NAME.mp4 - what is shown (mm:ss)" >> "$TXT"
caption "./run.sh --data <датасет>: режим GPU, Д3 на GPU, RViz на GPU"
printf '%s\n' "$START_LINES" > "$RES/logs/status_tail.txt"; printf '%s\n' "$START_LINES" > "$SHORT"
sleep 6

caption "1/6 doubleT_obstacle: объект в габарите на ~56 м; путь проверен до N м (строки статуса ниже)"
keys_play "s9 SPACE s5 SPACE s1 DOWN s0.3 DOWN s0.3 DOWN s0.3 DOWN s0.3 DOWN s18" -- for_hackathon/doubleT_obstacle 1.0 &
KP=$!
sleep 9; caption "1/6 ПРОБЕЛ: пауза проигрывания (клавиши в терминале play)"
sleep 5; caption "1/6 ПРОБЕЛ: продолжение; ↓ x5: скорость 1.0 -> 0.5 (-0.1 за нажатие)"
wait $KP

caption "2/6 смена записи без перезапуска: объекты, вставленные организаторами (cloud_with_fake_obj)"
play_bg fake extra/cloud_with_fake_obj 1.0 --start-offset 8
sleep 5;  caption "2/6 стена 2x2 м поперёк пути: сообщается с ~76 м и сопровождается до 3 м"
sleep 22; caption "2/6 кубы 0,3 м на пути: сообщаются с 37-52 м"
sleep 34; caption "2/6 низкая плита на пути: сообщается с ~91 м"
sleep 9; stop_play

caption "3/6 синтетика организаторов с разметкой (platform): человек сообщается с ~119 м"
play_bg platform org_synth_markers/synthetic_platform_cloud_with_markers 1.0 --start-offset 11
sleep 28; stop_play

caption "4/6 синтетика организаторов (switch): куб на рельсе с ~29 м, затем подвесной кабель с ~50 м"
play_bg switch org_synth_markers/synthetic_switch_cloud_with_markers 1.0 --start-offset 30
sleep 13; stop_play

caption "5/6 синтетика организаторов (tunnels): большой куб с ~60 м, балка поперёк рельсов с ~74 м"
play_bg tunnels org_synth_markers/synthetic_tunnels_cloud_with_markers 1.0 --start-offset 37
sleep 14; stop_play

# Falcon Prime: forward = +x (detected by the node); turn its frame so the RViz view matches Hesai (-y)
docker exec "$C" bash -c 'source /etc/profile.d/metro_ros.sh; mkdir -p /tmp/metro-run; [ -f /tmp/metro-run/tf_seyond.pid ] && kill "$(cat /tmp/metro-run/tf_seyond.pid)" 2> /dev/null;
  setsid /opt/ros/humble/lib/tf2_ros/static_transform_publisher --yaw -1.5708 --frame-id hesai_lidar --child-frame-id seyond > /results/logs/tf_seyond_demo.log 2>&1 < /dev/null & echo $! > /tmp/metro-run/tf_seyond.pid'
caption "6/6 второй лидар Seyond Falcon Prime: оси определяются автоматически"
play_bg falcon falcon/falcon_test_second_dark 1.0 --start-offset 262
sleep 10; caption "6/6 Falcon Prime: человек в тёмной одежде далеко впереди - сообщается с ~154 м"
sleep 40; caption "6/6 Falcon Prime: человек приближается; расстояние - в строке статуса"
sleep 12; stop_play
caption "IRS-MIREA - Telegram: IRS-MIREA"
sleep 4
kill -INT $FF; wait $FF; kill $TP 2> /dev/null
"$RUNSH" stop > /dev/null 2>&1
echo "recorded $OUT/$NAME.mp4 ($(du -h "$OUT/$NAME.mp4" | cut -f1)); timestamps: $TXT"
