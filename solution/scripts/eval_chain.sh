#!/bin/bash
# Full evaluation chain: every recording we have, one offline JSONL per bag, then chain_summary.py.
#
# usage: eval_chain.sh <tag> [mode]          mode = gpu (default) | cpu
# env:   IMG=metro-lidar:gpu-dev   image to run
#        SRC=<dir>                 package dir mounted over the installed metro_obstacle_detector
#                                  (e.g. <worktree>/solution/src/metro_obstacle_detector/metro_obstacle_detector)
#        MODELS=<dir>              mounted over /workspace/models (D3 weights)
#        OFFLINE_ARGS="..."        extra offline arguments (e.g. "--no-d3" or "--set key=value"); gpu mode adds
#                                  "--set d3_max_ms=100000" (D3 held on) unless D3_HOLD=0 or d3_max_ms is given
#        CHAIN_GROUPS="real falcon extra newdata orgsynth synth"   subset to run (orgmarkers: the organisers' marker bags, opt-in)
#        SYNTH_SETS="val_fake_style"   synthetic sets of the synth group (default: the seven v1 sets); EXTRA_SYNTH adds more
#        PAR=4                     containers in parallel inside this chain
#        REF=<tag>                 reference chain for the frame-level diff (default v1-<mode>)
#        RESUME=1                  keep the recordings of an interrupted run of the same tag that pass
#                                  check_offline_result.py (complete bag, one JSONL record per frame); run the rest
# Results: /home/gag/i.mos/work/eval/<tag>/<group>/... ; summary.md + summary.json next to them.
# At most two chains run at the same time on the machine (slots under work/eval/.slots).
set -u
TAG=${1:?tag}; MODE=${2:-gpu}
ROOT=/home/gag/i.mos; DS=$ROOT/dataset; SYN=$ROOT/synthetic
OUT=$ROOT/work/eval/$TAG; IMG=${IMG:-metro-lidar:gpu-dev}; PAR=${PAR:-4}
CHAIN_GROUPS=${CHAIN_GROUPS:-real falcon extra newdata orgsynth synth}
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=/workspace/install/lib/python3.10/site-packages/metro_obstacle_detector
mkdir -p "$OUT" "$ROOT/work/eval/.slots"; rm -f "$OUT/ALLDONE" "$OUT/FAILED"

MOUNTS=(-v "$DS":/data:ro -v "$SYN":/syn:ro -v "$OUT":/results)
[ -n "${SRC:-}" ] && MOUNTS+=(-v "$SRC":"$PKG":ro)
[ -n "${MODELS:-}" ] && MOUNTS+=(-v "$MODELS":/workspace/models:ro)
DEV=(--device cpu); GPU=()
[ "$MODE" = gpu ] && { DEV=(--device gpu); GPU=(--gpus all); }
# GPU chains keep D3 on (as in v1-gpu): on the shared machine GPU contention makes three D3 calls exceed
# d3_max_ms and the run silently continues without D3 (not comparable with the reference). D3_HOLD=0 disables.
if [ "$MODE" = gpu ] && [ "${D3_HOLD:-1}" = 1 ] && [[ "${OFFLINE_ARGS:-}" != *d3_max_ms* ]]; then
  OFFLINE_ARGS="${OFFLINE_ARGS:-} --set d3_max_ms=100000"
fi

# Wait for one of two machine-wide slots (other chains of other builders).
exec 9>/dev/null
while :; do
  for s in 0 1; do
    exec 9>"$ROOT/work/eval/.slots/$s"
    flock -n 9 && break 2
  done
  sleep 20
done
TEE=(tee); [ "${RESUME:-0}" = 1 ] && TEE=(tee -a)
echo "$(date '+%F %T') $TAG start (slot held, mode $MODE, image $IMG, src ${SRC:-image}${RESUME:+, resume})" | "${TEE[@]}" "$OUT/chain.log"

declare -A PIDS
run() {  # run <relative-out-path> <bag-path-in-container>
  local rel=$1 bag=$2 name; name=$(echo "$TAG-$rel" | tr '/' '-')
  mkdir -p "$OUT/$(dirname "$rel")"
  if [ "${RESUME:-0}" = 1 ] && python3 "$HERE/check_offline_result.py" "$OUT/$(dirname "$rel")" "$(basename "$rel")" 0 >/dev/null 2>&1; then
    return
  fi
  docker run --rm --name "eval-$name" "${GPU[@]}" "${MOUNTS[@]}" "$IMG" \
    ros2 run metro_obstacle_detector offline --bag "$bag" --out "/results/$rel.jsonl" --quiet \
    "${DEV[@]}" ${OFFLINE_ARGS:-} > "$OUT/$rel.log" 2>&1 &
  PIDS[$rel]=$!
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 2; done
}

for g in $CHAIN_GROUPS; do
  case $g in
    real)     for b in "$DS"/for_hackathon/*/; do run "real/$(basename "$b")" "/data/for_hackathon/$(basename "$b")"; done ;;
    falcon)   run falcon/falcon_test_second_dark /data/falcon/falcon_test_second_dark ;;
    extra)    run extra/cloud_with_fake_obj /data/extra/cloud_with_fake_obj ;;
    newdata)  run newdata/new_data /data/new_data_raw/new_data ;;
    orgsynth) for b in "$DS"/org_synth/*/; do run "orgsynth/$(basename "$b")" "/data/org_synth/$(basename "$b")"; done ;;
    orgmarkers) for b in "$DS"/org_synth_markers/*/; do
                  run "orgmarkers/$(basename "$b")" "/data/org_synth_markers/$(basename "$b")"; done ;;
    synth)    for s in ${SYNTH_SETS:-val_hesai_insert val_hesai_insert_tunnel val_hesai_param val_hesai_curve val_falcon_param \
                       val_dust val_hesai_far_dark} ${EXTRA_SYNTH:-}; do
                for sc in "$SYN/$s"/*/; do
                  [ -d "$sc/bag" ] && run "synth/$s/$(basename "$sc")" "/syn/$s/$(basename "$sc")/bag"
                done
              done ;;
    *) echo "unknown group $g" >&2 ;;
  esac
done

FAILED=()
for rel in "${!PIDS[@]}"; do
  wait "${PIDS[$rel]}"; rc=$?
  python3 "$HERE/check_offline_result.py" "$OUT/$(dirname "$rel")" "$(basename "$rel")" "$rc" >/dev/null 2>&1 || FAILED+=("$rel")
done
REF=${REF:-v1-$MODE}
python3 "$HERE/chain_summary.py" "$OUT" --ref "$ROOT/work/eval/$REF" > "$OUT/summary.md" 2>"$OUT/summary.err"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '%s\n' "${FAILED[@]}" > "$OUT/FAILED"; echo "$(date '+%F %T') FAILED: ${FAILED[*]}" | tee -a "$OUT/chain.log"; exit 1
fi
echo "$(date '+%F %T') ALLDONE" | tee -a "$OUT/chain.log"; echo ALLDONE > "$OUT/ALLDONE"
