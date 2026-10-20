"""Read-only random access to the real development bags (rosbag2 SQLite3, CDR PointCloud2).

No ROS needed: the SQLite file is opened read-only (``mode=ro``) and every message is decoded with
``metro_obstacle_detector.cloud_io.parse_pointcloud2_cdr`` (same parser as the node).  Frames are
indexed 0..N-1 in bag (timestamp, id) order, identical to ``offline.py`` / ``annotations/events.yaml``.
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(_HERE, "..", "..", "src", "metro_obstacle_detector"))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct  # noqa: E402

DATASET_DIR = os.environ.get("IMOS_DATASET", "/home/gag/i.mos/dataset/for_hackathon")


@dataclass
class RealFrame:
    index: int
    bag_time_ns: int
    stamp_ns: int
    frame_id: str
    topic: str
    cloud: np.ndarray          # structured array (x,y,z,intensity,ring,timestamp), own copy
    fields: list
    point_step: int


class RealBag:
    """Random access to the PointCloud2 messages of one rosbag2 SQLite3 directory."""

    def __init__(self, bag: str):
        bag_dir = bag if os.path.isdir(bag) else os.path.join(DATASET_DIR, bag)
        dbs = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
        if not dbs:
            raise FileNotFoundError(f"no .db3 in {bag_dir}")
        self.name = os.path.basename(os.path.normpath(bag_dir))
        self.path = bag_dir
        # split bags (e.g. new_data: 221 files) are read as one sequence ordered by (timestamp, file, id)
        self.dbs = [sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False) for p in dbs]
        self.db = self.dbs[0]
        rows_all = []
        topic = None
        for k, db in enumerate(self.dbs):
            topics = db.execute("select id, name, type from topics").fetchall()
            clouds = [(i, n) for i, n, t in topics if t == "sensor_msgs/msg/PointCloud2"]
            if len(clouds) != 1:
                raise RuntimeError(f"{dbs[k]}: expected one PointCloud2 topic, got {clouds}")
            tid, tname = clouds[0]
            if topic is not None and tname != topic:
                raise RuntimeError(f"{bag_dir}: topic differs between files ({topic} vs {tname})")
            topic = tname
            if k == 0:
                self.topic_id = tid
            for mid, ts in db.execute("select id, timestamp from messages where topic_id=?", (tid,)).fetchall():
                rows_all.append((int(ts), k, int(mid)))
        rows_all.sort()
        self.topic = topic
        self.file_idx = [r[1] for r in rows_all]
        self.msg_ids = [r[2] for r in rows_all]
        self.bag_times = [r[0] for r in rows_all]

    def __len__(self) -> int:
        return len(self.msg_ids)

    def raw(self, i: int) -> bytes:
        return self.dbs[self.file_idx[i]].execute("select data from messages where id=?", (self.msg_ids[i],)).fetchone()[0]

    def frame(self, i: int) -> RealFrame:
        msg = parse_pointcloud2_cdr(self.raw(i))
        st = np.array(pointcloud2_to_struct(msg))  # own copy
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        return RealFrame(i, self.bag_times[i], stamp, msg.header.frame_id, self.topic, st, list(msg.fields),
                         int(msg.point_step))

    def frames(self, idx: Optional[List[int]] = None):
        for i in (range(len(self)) if idx is None else idx):
            yield self.frame(i)


def xyz_of(cloud: np.ndarray) -> np.ndarray:
    return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)
