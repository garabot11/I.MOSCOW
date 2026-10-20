"""Small bags for the failure matrix (run inside the image; the dataset stays read-only).

    python3 /workspace/docker/make_test_bags.py --src /data/for_hackathon/doubleT_platform --out /synthetic/infra_test [--frames 30]

Writes (raw CDR copied, nothing re-serialised):
  two_topics_sqlite/   the first N clouds on /lidar_points and again on /lidar_points_copy (SQLite3)
  two_topics_mcap/     the same as MCAP
  single_mcap/         the first N clouds on /lidar_points as MCAP
  truncated_sqlite/    single-topic SQLite3 bag cut to a third of its size (corrupt)
  no_metadata/         SQLite3 bag without metadata.yaml
  empty_dir/           a directory without a bag
"""
import argparse
import os
import shutil
import sys

import rosbag2_py


def read_first(src, topic, n):
    storage = "sqlite3"
    try:
        import yaml
        with open(os.path.join(src, "metadata.yaml")) as fh:
            storage = yaml.safe_load(fh)["rosbag2_bagfile_information"]["storage_identifier"]
    except Exception:
        pass
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=src, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while r.has_next() and len(out) < n:
        name, raw, t = r.read_next()
        out.append((bytes(raw), t))
    return out


def write(path, storage, topics, msgs):
    if os.path.exists(path):
        shutil.rmtree(path)
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name in topics:
        w.create_topic(rosbag2_py.TopicMetadata(name=name, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))
    for raw, t in msgs:
        for name in topics:
            w.write(name, raw, t)
    del w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--topic", default="/lidar_points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()
    msgs = read_first(a.src, a.topic, a.frames)
    if not msgs:
        print("no messages read from", a.src)
        return 1
    os.makedirs(a.out, exist_ok=True)
    two = [a.topic, a.topic + "_copy"]
    write(os.path.join(a.out, "two_topics_sqlite"), "sqlite3", two, msgs)
    write(os.path.join(a.out, "two_topics_mcap"), "mcap", two, msgs)
    write(os.path.join(a.out, "single_mcap"), "mcap", [a.topic], msgs)
    tr = os.path.join(a.out, "truncated_sqlite")
    write(tr, "sqlite3", [a.topic], msgs)
    db = [f for f in os.listdir(tr) if f.endswith(".db3")][0]
    p = os.path.join(tr, db)
    size = os.path.getsize(p)
    with open(p, "r+b") as fh:
        fh.truncate(size // 3)
    nm = os.path.join(a.out, "no_metadata")
    write(nm, "sqlite3", [a.topic], msgs[:5])
    os.remove(os.path.join(nm, "metadata.yaml"))
    os.makedirs(os.path.join(a.out, "empty_dir"), exist_ok=True)
    print("wrote %d frames per bag to %s" % (len(msgs), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
