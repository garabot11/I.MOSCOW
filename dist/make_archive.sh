#!/bin/bash
# Make the delivery archive of the Docker image: metro-lidar-final.tar.gz + SHA-256 + 95 MB parts
# (GitHub rejects files > 100 MB).
#
#   dist/make_archive.sh [IMAGE] [--out DIR] [--part-size 95M] [--force]
#
#   IMAGE        image to archive (default: metro-lidar:gpu-dev); its tag is kept inside the archive,
#                `docker load` restores exactly this tag
#   --out DIR    output directory (default: the directory of this script)
#   --force      replace existing metro-lidar-final.tar.gz* files in DIR
#
# Writes into DIR:
#   metro-lidar-final.tar.gz              docker save IMAGE | gzip (reproducible gzip header: -n)
#   metro-lidar-final.tar.gz.sha256       SHA-256 of the archive
#   metro-lidar-final.tar.gz.part-NN      the archive cut into parts of --part-size
#   metro-lidar-final.parts.sha256        SHA-256 of every part
#   metro-lidar-final.image.txt           image tag, ID, size, date (what the archive contains)
# and checks that the joined parts reproduce the archive.
#
# On the receiving side (from DIR):
#   sha256sum -c metro-lidar-final.parts.sha256                      # parts intact
#   cat metro-lidar-final.tar.gz.part-* > metro-lidar-final.tar.gz   # join (parts sort by name)
#   sha256sum -c metro-lidar-final.tar.gz.sha256                     # archive intact
#   gunzip -c metro-lidar-final.tar.gz | docker load                 # restores IMAGE
set -euo pipefail
IMG=metro-lidar:gpu-dev
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PART=95M
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=${2:?--out DIR}; shift ;;
    --part-size) PART=${2:?--part-size SIZE}; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 1 ;;
    *) IMG=$1 ;;
  esac
  shift
done
NAME=metro-lidar-final.tar.gz
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P)
cd "$OUT"

docker image inspect "$IMG" > /dev/null 2>&1 || { echo "image $IMG not found (docker images metro-lidar)" >&2; exit 1; }
if ls "$NAME"* > /dev/null 2>&1 && [ $FORCE = 0 ]; then
  echo "$OUT already has $NAME* files (an earlier archive); re-run with --force to replace them" >&2
  exit 1
fi
size=$(docker image inspect -f '{{.Size}}' "$IMG")
need=$(( size / 2 + 2 * 1024 * 1024 * 1024 ))   # archive ≈ 35-50 % of the image + parts of the same size
free=$(( $(df -Pk "$OUT" | awk 'NR==2 {print $4}') * 1024 ))
if [ "$free" -lt $(( need * 2 )) ]; then
  echo "not enough free space in $OUT: $((free / 2**30)) GiB free, about $((need * 2 / 2**30)) GiB needed" >&2
  exit 1
fi

rm -f "$NAME" "$NAME".sha256 "$NAME".part-* metro-lidar-final.parts.sha256 metro-lidar-final.image.txt
id=$(docker image inspect -f '{{.Id}}' "$IMG")
echo "saving $IMG ($id, $((size / 2**20)) MiB) -> $OUT/$NAME ..."
gz="gzip -n -6"
command -v pigz > /dev/null 2>&1 && gz="pigz -n -6"
docker save "$IMG" | $gz > "$NAME.tmp"
mv "$NAME.tmp" "$NAME"
sha256sum "$NAME" > "$NAME.sha256"
split -b "$PART" -d -a 2 "$NAME" "$NAME.part-"
sha256sum "$NAME".part-* > metro-lidar-final.parts.sha256
joined=$(cat "$NAME".part-* | sha256sum | cut -d' ' -f1)
[ "$joined" = "$(cut -d' ' -f1 "$NAME.sha256")" ] || { echo "joined parts differ from the archive" >&2; exit 1; }
nparts=$(ls "$NAME".part-* | wc -l)
{
  echo "image: $IMG"
  echo "id: $id"
  echo "image_size_bytes: $size"
  echo "archive: $NAME $(stat -c %s "$NAME") bytes, sha256 $(cut -d' ' -f1 "$NAME.sha256")"
  echo "parts: $nparts x $PART (metro-lidar-final.parts.sha256)"
  echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > metro-lidar-final.image.txt
cat metro-lidar-final.image.txt
echo "check on the receiving side: sha256sum -c metro-lidar-final.parts.sha256 && cat $NAME.part-* > $NAME && sha256sum -c $NAME.sha256 && gunzip -c $NAME | docker load"
