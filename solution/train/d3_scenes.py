"""Interface B reader: ``<root>/<set>/<scene_id>/{bag/, scene.json, labels.jsonl, slot_labels.npz}``.

Bags are read with the pure-Python ``rosbags`` library (sqlite3 and MCAP), clouds are decoded with
the project's ``cloud_io`` (no ROS installation needed on the training host).
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import _paths  # noqa: F401  (sys.path for metro_obstacle_detector)
from metro_obstacle_detector import cloud_io

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"


@dataclass
class Scene:
    set_name: str
    scene_id: str
    path: str
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.set_name}/{self.scene_id}"

    @property
    def sensor(self) -> str:
        return str(self.meta.get("sensor", "hesai128"))

    @property
    def bag_dir(self) -> str:
        if self.meta.get("source") == "real_negative":
            return self.path                      # a real rosbag2 directory itself
        return os.path.join(self.path, "bag")

    @property
    def labels_path(self) -> str:
        return os.path.join(self.path, "labels.jsonl")

    @property
    def slot_labels_path(self) -> str:
        return os.path.join(self.path, "slot_labels.npz")

    def missing(self) -> List[str]:
        if self.meta.get("source") == "real_negative":
            return [] if os.path.isdir(self.bag_dir) else [self.bag_dir]
        return [p for p in (self.bag_dir, self.labels_path, self.slot_labels_path) if not os.path.exists(p)]


def discover(root: str, patterns: Sequence[str]) -> List[Scene]:
    """All scenes of the sets whose directory name matches one of ``patterns`` (fnmatch), sorted."""
    out: List[Scene] = []
    if not os.path.isdir(root):
        return out
    for set_name in sorted(os.listdir(root)):
        set_dir = os.path.join(root, set_name)
        if not os.path.isdir(set_dir) or not any(fnmatch.fnmatch(set_name, p) for p in patterns):
            continue
        for sid in sorted(os.listdir(set_dir)):
            sdir = os.path.join(set_dir, sid)
            sj = os.path.join(sdir, "scene.json")
            if os.path.isfile(sj):
                with open(sj, encoding="utf-8") as fh:
                    meta = json.load(fh)
                out.append(Scene(set_name, sid, sdir, meta))
    return out


def real_negative_scenes(spec: str) -> List[Scene]:
    """Scenes of REAL frames used as pure negatives (every pixel background) -- only for bags annotated as
    free of obstacles.  ``spec`` = comma-separated ``/path/to/bag[:start:stop]`` (frame range, default all)."""
    out: List[Scene] = []
    for item in [x for x in spec.split(",") if x.strip()]:
        parts = item.strip().split(":")
        bag = parts[0].rstrip("/")
        start = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        stop = int(parts[2]) if len(parts) > 2 and parts[2] else None
        name = os.path.basename(bag)
        out.append(Scene("real_negatives", f"{name}_{start}_{stop if stop is not None else 'end'}", bag,
                         {"source": "real_negative", "real_bag": name, "frames": [start, stop], "topic": None,
                          "sensor": "auto", "generator_version": "real"}))
    return out


def is_real_negative(scene: Scene) -> bool:
    return scene.meta.get("source") == "real_negative"


def read_labels(scene: Scene) -> List[dict]:
    rows = []
    with open(scene.labels_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def read_slot_labels(scene: Scene) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """frame -> (slots sorted ascending int64, obj_ids int64)."""
    z = np.load(scene.slot_labels_path)
    frame = np.asarray(z["frame"], np.int64)
    slot = np.asarray(z["slot"], np.int64)
    obj = np.asarray(z["obj_id"], np.int64)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if len(frame) == 0:
        return out
    o = np.lexsort((slot, frame))
    frame, slot, obj = frame[o], slot[o], obj[o]
    cuts = np.flatnonzero(np.diff(frame)) + 1
    for fs, ss, oo in zip(np.split(frame, cuts), np.split(slot, cuts), np.split(obj, cuts)):
        out[int(fs[0])] = (ss, oo)
    return out


def lookup_obj(slots: np.ndarray, frame_slots: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Object id per slot (-1 = none / slot < 0)."""
    out = np.full(slots.shape, -1, np.int64)
    if frame_slots is None or len(frame_slots[0]) == 0:
        return out
    ss, oo = frame_slots
    flat = slots.ravel()
    pos = np.searchsorted(ss, flat)
    pos_c = np.minimum(pos, len(ss) - 1)
    hit = (flat >= 0) & (ss[pos_c] == flat)
    res = out.ravel()
    res[hit] = oo[pos_c[hit]]
    return res.reshape(slots.shape)


def _stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def iter_messages(bag_dir: str, topic: Optional[str] = None) -> Iterator[Tuple[int, int, object]]:
    """(index, header stamp ns, parsed PointCloud2) for every cloud of ``topic``, in bag order.

    The message is ``cloud_io.parse_pointcloud2_cdr`` output -- what the node feeds to
    ``DetectionPipeline.build_frame`` -- and stays valid after the next iteration (it owns its bytes).
    """
    from rosbags.rosbag2 import Reader

    with Reader(bag_dir) as r:
        conns = [c for c in r.connections if c.msgtype == POINTCLOUD2]
        if topic:
            sel = [c for c in conns if c.topic == topic]
            conns = sel or conns
        if not conns:
            raise ValueError(f"no PointCloud2 topic in {bag_dir}")
        topics = sorted({c.topic for c in conns})
        if len(topics) > 1:
            raise ValueError(f"several PointCloud2 topics in {bag_dir}: {topics}; pass the scene's topic")
        for i, (_, _, raw) in enumerate(r.messages(connections=conns)):
            msg = cloud_io.parse_pointcloud2_cdr(bytes(raw))
            yield i, _stamp_ns(msg), msg


def iter_clouds(bag_dir: str, topic: Optional[str] = None) -> Iterator[Tuple[int, int, np.ndarray]]:
    """(index, header stamp ns, structured cloud array in message order) for every cloud of ``topic``."""
    for i, stamp, msg in iter_messages(bag_dir, topic):
        yield i, stamp, np.array(cloud_io.pointcloud2_to_struct(msg), copy=True)


def sqlite_message(bag_dir: str, topic: str, index: int):
    """The ``index``-th message (by timestamp) of ``topic`` of a sqlite3 rosbag2, fetched by row id
    without scanning the bag (read-only, immutable).  Returns (stamp ns, parsed PointCloud2)."""
    import glob
    import sqlite3

    dbs = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not dbs:
        raise FileNotFoundError(f"no .db3 in {bag_dir}")
    con = sqlite3.connect(f"file:{dbs[0]}?immutable=1", uri=True)
    try:
        row = con.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
        if row is None:
            raise ValueError(f"topic {topic} not in {dbs[0]}")
        mid = con.execute("SELECT id FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET ?",
                          (row[0], int(index))).fetchone()
        if mid is None:
            raise IndexError(f"{topic} has no message #{index}")
        raw = con.execute("SELECT data FROM messages WHERE id = ?", (mid[0],)).fetchone()[0]
    finally:
        con.close()
    msg = cloud_io.parse_pointcloud2_cdr(bytes(raw))
    return _stamp_ns(msg), msg
