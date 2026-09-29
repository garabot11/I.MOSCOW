#!/bin/bash
# Record the live demo (RViz + terminal) from the X11 screen with ffmpeg while run_demo.sh plays a bag.
# usage: record_video.sh <bag_dir> <input_topic> <seconds> <out.mp4> [rate] [image]
BAG=${1:?bag directory}; TOPIC=${2:-/lidar_points}; SECS=${3:-60}; OUT=${4:-video/demo.mp4}; RATE=${5:-1.0}; IMG=${6:-metro-lidar:final}
HERE=$(cd "$(dirname "$0")" && pwd)
BAG=$(cd "$BAG" && pwd -P) || exit 2
mkdir -p "$(dirname "$OUT")"
SIZE=$(xdpyinfo -display "$DISPLAY" | awk '/dimensions/{print $2}')
RES="$(cd "$HERE/.." && pwd)/results/demo_video"; mkdir -p "$RES"
printf 'metro obstacle detector - waiting for /obstacles/status ...\n' > "$RES/status_tail.txt"
printf 'bag: %s   topic: %s   playback rate: %s\n' "$(basename "$BAG")" "$TOPIC" "$RATE" > "$RES/demo_info.txt"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
# the status monitor's last lines and the playback info are burnt into the recording (drawtext reloads the files)
ffmpeg -loglevel error -y -f x11grab -framerate 15 -video_size "$SIZE" -i "$DISPLAY" -t "$SECS" \
  -vf "scale=1920:-2,drawtext=fontfile=$FONT:textfile=$RES/demo_info.txt:reload=1:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=8:x=20:y=20,drawtext=fontfile=$FONT:textfile=$RES/status_tail.txt:reload=1:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=8:x=20:y=h-th-20" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$OUT" &
FF=$!
sleep 1
timeout "$((SECS - 2))" "$HERE/run_demo.sh" "$BAG" "$TOPIC" "$RATE" "$RES" "$IMG"
wait $FF
docker stop metro-lidar-demo > /dev/null 2>&1 || true
echo "recorded $OUT"
