#!/bin/bash
# Failure matrix of the launch path (host).  Every case goes through ./run.sh and the metro CLI, the
# console output is captured and checked for Python tracebacks.  Uses its own container name
# (metro-lidar-matrix), never touches other containers.
#
#   solution/docker/failure_matrix.sh [--data DATASET_DIR] [--test-bags DIR] [--out DIR] [--image IMG] [--quick]
#
# --test-bags: output of docker/make_test_bags.py (two topics, MCAP, truncated, no metadata, empty dir);
#              created on the fly when missing (from <data>/for_hackathon/doubleT_platform).
# Result: <out>/matrix.tsv (case, verdict, tracebacks, note) + one log per case.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
DATA=/home/gag/i.mos/dataset
TB=/home/gag/i.mos/synthetic/infra_test
OUT=$HOME/metro-lidar-matrix/$(date +%Y%m%d_%H%M%S)
IMG=${METRO_IMAGE:-metro-lidar:gpu-dev}
QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift ;; --test-bags) TB=$2; shift ;; --out) OUT=$2; shift ;; --image) IMG=$2; shift ;;
    --quick) QUICK=1 ;; *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
export METRO_CONTAINER=metro-lidar-matrix METRO_IMAGE=$IMG
RUNSH="$HERE/run.sh"
mkdir -p "$OUT"
TSV="$OUT/matrix.tsv"
printf 'case\tverdict\ttracebacks\tnote\n' > "$TSV"

if [ ! -d "$TB/two_topics_sqlite" ]; then
  mkdir -p "$TB"
  docker run --rm -v "$DATA":/data:ro -v "$TB":/out "$IMG" \
    python3 /workspace/docker/make_test_bags.py --src /data/for_hackathon/doubleT_platform --out /out --frames 30 > "$OUT/make_test_bags.log" 2>&1
fi

RES=""
fresh() {  # fresh() <name> <run.sh options...>: stop, new results dir, start detached
  local name=$1; shift
  "$RUNSH" stop > /dev/null 2>&1
  RES="$OUT/res_$name"; mkdir -p "$RES"
  "$RUNSH" --data "$DATA" --data "$TB" --results "$RES" --detach "$@" > "$OUT/$name.start.log" 2>&1 < /dev/null
  warmup "$name"
}
warmup() {
  local name=$1
  # warm-up: the first frame after a start sets up cuDNN for D3 (~15 s on the GPU); a short bag absorbs it
  # so that the cases below test playback, not the warm-up. Its frames are excluded from the statistics.
  docker exec "$METRO_CONTAINER" metro play infra_test/single_mcap 1.0 > "$OUT/$name.warmup.log" 2>&1
  local w=0
  until [ -s "$RES/logs/detector.jsonl" ] || [ $w -ge 60 ]; do sleep 1; w=$((w + 1)); done
  sleep 3
  SKIP=$(wc -l < "$RES/logs/detector.jsonl" 2> /dev/null || echo 0)
}
SKIP=0
ex() { docker exec "$METRO_CONTAINER" "$@"; }
tb() { cat "$@" 2> /dev/null | grep -c -E '^Traceback|^\s+File ".*", line [0-9]+' ; }
record() {  # record <case> <verdict> <note> <logs...>
  local c=$1 v=$2 n=$3; shift 3
  local t; t=$(tb "$@")
  [ "$t" != 0 ] && v=FAIL
  printf '%s\t%s\t%s\t%s\n' "$c" "$v" "$t" "$n" | tee -a "$TSV"
}
frames() { python3 - "$1" "$SKIP" <<'EOF'
import json, sys
rows = []
try:
    for ln in open(sys.argv[1]):
        try: rows.append(json.loads(ln))
        except Exception: pass
except Exception: pass
rows = rows[int(sys.argv[2]):]  # without the warm-up frames
ms = sorted(r.get("processing_ms", 0) for r in rows)
p95 = ms[int(0.95 * (len(ms) - 1))] if ms else 0
print("%d frames, %d resets, frames %s, p95 %.1f ms" % (len(rows), sum(1 for r in rows if r.get("reset")),
      sorted({r.get("frame_id") for r in rows}), p95))
EOF
}
mode_of() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("mode"), d.get("code"), d.get("device_name") or "")' "$1" 2> /dev/null; }

# 1. GPU (toolkit installed) -------------------------------------------------------------------
fresh gpu --no-gui
ex metro doctor --no-color > "$OUT/gpu.doctor.log" 2>&1
m=$(mode_of "$RES/logs/device.json")
ex metro play for_hackathon/doubleT_platform 2.0 > "$OUT/gpu.play.log" 2>&1
sleep 2
record gpu "$([ "${m%% *}" = gpu ] && echo PASS || echo CHECK)" "mode: $m; $(frames "$RES/logs/detector.jsonl")" \
  "$OUT/gpu.start.log" "$OUT/gpu.doctor.log" "$OUT/gpu.play.log"

# 2. no GPU: host without the toolkit (simulated in run.sh) -------------------------------------
METRO_TEST_NO_TOOLKIT=1 fresh no_toolkit --no-gui
m=$(mode_of "$RES/logs/device.json")
ok=CHECK
grep -q "Contact team IRS-MIREA on Telegram: IRS-MIREA" "$OUT/no_toolkit.start.log" && [ -f "$RES/ВНИМАНИЕ_GPU.txt" ] && \
  grep -q "nvidia-ctk runtime configure" "$OUT/no_toolkit.start.log" && [ "${m%% *}" = cpu_emergency ] && ok=PASS
ex metro play for_hackathon/roundT_doubleT 2.0 > "$OUT/no_toolkit.play.log" 2>&1
record no_gpu_toolkit_missing "$ok" "mode: $m; banner+fix+file; $(frames "$RES/logs/detector.jsonl")" \
  "$OUT/no_toolkit.start.log" "$OUT/no_toolkit.play.log"

# 3. no GPU: container started by hand without --gpus -------------------------------------------
docker run --rm -e METRO_RESULTS_DIR=/tmp/r "$IMG" bash -lc 'mkdir -p /tmp/r; metro doctor --no-bags --no-color; metro banner; ls /tmp/r' > "$OUT/no_gpus_flag.log" 2>&1
ok=CHECK; grep -q "no_gpus_flag\|without GPU access" "$OUT/no_gpus_flag.log" && grep -q "ВНИМАНИЕ_GPU.txt" "$OUT/no_gpus_flag.log" && ok=PASS
record no_gpu_plain_docker_run "$ok" "reason: $(grep -o 'started without GPU access[^)]*)' "$OUT/no_gpus_flag.log" | head -n 1)" "$OUT/no_gpus_flag.log"

# 4. --cpu forced --------------------------------------------------------------------------------
fresh cpu_forced --no-gui --cpu
m=$(mode_of "$RES/logs/device.json")
record cpu_forced "$([ "${m%% *}" = cpu_forced ] && [ ! -f "$RES/ВНИМАНИЕ_GPU.txt" ] && echo PASS || echo CHECK)" "mode: $m" "$OUT/cpu_forced.start.log"

# 5. no DISPLAY (GUI requested) ------------------------------------------------------------------
"$RUNSH" stop > /dev/null 2>&1
RES="$OUT/res_no_display"; mkdir -p "$RES"
env -u DISPLAY "$RUNSH" --data "$DATA" --data "$TB" --results "$RES" --detach > "$OUT/no_display.start.log" 2>&1 < /dev/null
warmup no_display
ok=CHECK; grep -q "no DISPLAY" "$OUT/no_display.start.log" && grep -q "detector started" "$OUT/no_display.start.log" && ok=PASS
record no_display "$ok" "$(grep -o 'RViz: [^,]*' "$OUT/no_display.start.log" | head -n 1)" "$OUT/no_display.start.log"

# 6. bag missing / corrupt -----------------------------------------------------------------------
for b in does_not_exist infra_test/truncated_sqlite infra_test/no_metadata infra_test/empty_dir; do
  n=bag_$(basename "$b")
  ex metro play "$b" > "$OUT/$n.log" 2>&1
  ok=CHECK; grep -q -E "not found|cannot be played|not a rosbag2|cannot" "$OUT/$n.log" && ok=PASS
  record "$n" "$ok" "$(grep -v '^$' "$OUT/$n.log" | tail -n 1 | cut -c1-160)" "$OUT/$n.log"
done
ex metro bags > "$OUT/bags.log" 2>&1
record bags_listing "$(grep -q BROKEN "$OUT/bags.log" && grep -q falcon "$OUT/bags.log" && echo PASS || echo CHECK)" \
  "$(grep -c -E '^(dataset|infra_test)/' "$OUT/bags.log") bags listed" "$OUT/bags.log"

# 7. two PointCloud2 topics, MCAP ----------------------------------------------------------------
for b in two_topics_sqlite two_topics_mcap single_mcap; do
  before=$(wc -l < "$RES/logs/detector.jsonl" 2> /dev/null || echo 0)
  ex metro play "infra_test/$b" 1.0 > "$OUT/$b.log" 2>&1
  sleep 2
  after=$(wc -l < "$RES/logs/detector.jsonl" 2> /dev/null || echo 0)
  ok=CHECK; [ $((after - before)) -ge 20 ] && grep -q "playback finished" "$OUT/$b.log" && ok=PASS
  record "$b" "$ok" "$((after - before)) of 30 frames processed; $(grep -o 'playing.*' "$OUT/$b.log" | head -n 1 | cut -c1-120)" "$OUT/$b.log"
done

# 8. switching bags mid-run (incl. another frame_id, 921k cloud, Falcon) -------------------------
fresh switch --no-gui
( ex metro play for_hackathon/doubleT_platform 1.0 > "$OUT/switch.1.log" 2>&1 ) &
sleep 8
( ex metro play for_hackathon/doubleT_obstacle 1.0 > "$OUT/switch.2.log" 2>&1 ) &
sleep 8
( ex metro play falcon/falcon_test_second_dark 1.0 > "$OUT/switch.3.log" 2>&1 ) &
sleep 8
ex metro play for_hackathon/roundT_doubleT 1.0 > "$OUT/switch.4.log" 2>&1
wait
ex metro status > "$OUT/switch.status.log" 2>&1
note=$(frames "$RES/logs/detector.jsonl")
ok=CHECK; grep -q "stopping the previous playback" "$OUT/switch.2.log" && grep -q "playback finished" "$OUT/switch.4.log" && ok=PASS
record switch_bags "$ok" "$note" "$OUT"/switch.*.log "$OUT/switch.start.log"

# 9. container limited to 8 CPUs ------------------------------------------------------------------
fresh cpus8 --no-gui --cpuset-cpus 0-7
ex metro play for_hackathon/squareT_platform_squareT_switch 1.0 > "$OUT/cpus8.play.log" 2>&1 &
sleep "$([ $QUICK = 1 ] && echo 20 || echo 60)"
ex metro stop > /dev/null 2>&1; wait
ex metro doctor --no-bags --no-color > "$OUT/cpus8.doctor.log" 2>&1
ok=CHECK; grep -q "8 CPUs available" "$OUT/cpus8.doctor.log" && ok=PASS
record cpus8 "$ok" "$(grep -o '8 CPUs available[^;]*' "$OUT/cpus8.doctor.log" | head -n 1); torch threads $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("torch_threads"))' "$RES/logs/device.json" 2> /dev/null); $(frames "$RES/logs/detector.jsonl")" \
  "$OUT/cpus8.start.log" "$OUT/cpus8.play.log" "$OUT/cpus8.doctor.log"

# 10. artificial GPU failure: GPU hidden at start, run-time failure -> hot switch -------------------
fresh gpu_fail --no-gui
ex bash -lc 'CUDA_VISIBLE_DEVICES= metro doctor --no-bags --no-color' > "$OUT/gpu_hidden.log" 2>&1
record gpu_hidden_cuda_visible_devices "$(grep -q 'cpu_emergency' "$OUT/gpu_hidden.log" && echo PASS || echo CHECK)" \
  "$(grep -o 'cpu_emergency: [^(]*' "$OUT/gpu_hidden.log" | head -n 1 | cut -c1-140)" "$OUT/gpu_hidden.log"
ex bash -lc 'python3 -c "
import time
from metro_obstacle_detector.device import DeviceManager
dm = DeviceManager(results_dir=\"/tmp/hs\")
print(\"before:\", dm.mode, dm.device)
dm.report_failure(RuntimeError(\"CUDA error: an illegal memory access was encountered (simulated)\"))
print(\"after:\", dm.mode, dm.device, \"|\", dm.reason)
print(dm.maybe_repeat_banner(time.monotonic()) is not None)
import os; print(os.listdir(\"/tmp/hs\"))
"' > "$OUT/gpu_runtime_failure.log" 2>&1
record gpu_runtime_failure_hot_switch "$(grep -q 'after: cpu_emergency cpu' "$OUT/gpu_runtime_failure.log" && echo PASS || echo CHECK)" \
  "$(grep -E '^(before|after)' "$OUT/gpu_runtime_failure.log" | tr '\n' ' ' | cut -c1-200)" "$OUT/gpu_runtime_failure.log"

"$RUNSH" stop > /dev/null 2>&1
echo "matrix: $TSV"
