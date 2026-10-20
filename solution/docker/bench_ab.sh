#!/bin/bash
# A/B timing of two code versions on the same frames, alternating A and B (shared machine: compare runs of
# the same minute).  A = the package installed in the image, B = SRC mounted over it.
#
# usage: bench_ab.sh <out.jsonl> <bag-in-/data> <start> <n> [gpu|cpu] [rounds]
# env:   IMG=metro-lidar:gpu-dev  SRC=<package dir of version B>  SRC_A=<package dir of version A, optional>
#        CPUS=0-7  ENV_B="METRO_PARALLEL=0 ..." (extra environment of B)
set -u
OUT=${1:?out.jsonl}; BAG=${2:?bag}; START=${3:?start}; N=${4:?n}; MODE=${5:-gpu}; ROUNDS=${6:-2}
IMG=${IMG:-metro-lidar:gpu-dev}; CPUS=${CPUS:-0-7}
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=/workspace/install/lib/python3.10/site-packages/metro_obstacle_detector
DS=/home/gag/i.mos/dataset
GPU=(); [ "$MODE" = gpu ] && GPU=(--gpus all)
run() {  # run <label> [mount args...]
  local label=$1; shift
  docker run --rm --name "infra-bench-$label-$$" --cpuset-cpus "$CPUS" "${GPU[@]}" -v "$DS":/data:ro -v "$HERE":/bench:ro "$@" "$IMG" \
    python3 /bench/bench_frames.py --bag "/data/$BAG" --start "$START" --n "$N" --device "$MODE" --label "$label" 2>/dev/null \
    | grep '^{' >> "$OUT"
}
for r in $(seq 1 "$ROUNDS"); do
  A=(); [ -n "${SRC_A:-}" ] && A=(-v "$SRC_A":"$PKG":ro)
  run A "${A[@]}"
  B=(-v "${SRC:?SRC}":"$PKG":ro)
  for kv in ${ENV_B:-}; do B+=(-e "$kv"); done
  run B "${B[@]}"
done
