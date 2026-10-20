"""rosbag2 writing (host side, no ROS) and label files of interface B.

PointCloud2 messages are serialised here (plain CDR, little endian; identical layout to what
``cloud_io.parse_pointcloud2_cdr`` reads) and written as MCAP with zstd chunks (``mcap`` library) or as
SQLite3 (``rosbags``).  ``metadata.yaml`` is always written in the rosbag2 version-5 layout that Humble
writes and reads itself (``rosbags`` alone writes version 8/9).  Readability is proven with ``ros2 bag info``/``play`` and the project's
``offline`` tool inside ``metro-lidar:final`` (see README).
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from typing import Dict, List, Optional, Sequence

import numpy as np

PF = {"i1": 1, "u1": 2, "i2": 3, "u2": 4, "i4": 5, "u4": 6, "f4": 7, "f8": 8}

# --------------------------------------------------------------------------- point layouts
HESAI_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
                        "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26})
# Falcon Prime as recorded by the Seyond ROS driver (/iv_points, frame "seyond", point_step 48)
FALCON_DTYPE = np.dtype({"names": ["x", "y", "z", "timestamp", "intensity", "flags", "elongation", "scan_id",
                                   "scan_idx", "is_2nd_return"],
                         "formats": ["<f4", "<f4", "<f4", "<f8", "<f4", "u1", "u1", "<u2", "<u2", "u1"],
                         "offsets": [0, 4, 8, 16, 24, 28, 29, 30, 32, 34], "itemsize": 48})


def _align(buf: bytearray, n: int) -> None:
    pad = (-(len(buf) - 4)) % n
    buf.extend(b"\x00" * pad)


def _string(buf: bytearray, s: str) -> None:
    _align(buf, 4)
    b = s.encode("utf-8") + b"\x00"
    buf.extend(struct.pack("<I", len(b)))
    buf.extend(b)


def serialize_pointcloud2(points: np.ndarray, stamp_ns: int, frame_id: str, height: int = 1,
                          is_dense: bool = False) -> bytes:
    """CDR (little endian) sensor_msgs/msg/PointCloud2 for a structured array (fields at its own offsets)."""
    dt = points.dtype
    n = len(points)
    width = n // height
    buf = bytearray(b"\x00\x01\x00\x00")
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    buf.extend(struct.pack("<iI", sec, nsec))
    _string(buf, frame_id)
    _align(buf, 4)
    buf.extend(struct.pack("<II", height, width))
    names = list(dt.names)
    buf.extend(struct.pack("<I", len(names)))
    for name in names:
        fdt, off = dt.fields[name][0], dt.fields[name][1]
        _string(buf, name)
        _align(buf, 4)
        buf.extend(struct.pack("<I", off))
        buf.extend(struct.pack("<B", PF[fdt.base.str.lstrip("<>|=")]))
        _align(buf, 4)
        buf.extend(struct.pack("<I", 1))
    buf.extend(struct.pack("<B", 0))  # is_bigendian
    _align(buf, 4)
    buf.extend(struct.pack("<II", dt.itemsize, dt.itemsize * width))
    raw = np.ascontiguousarray(points).view(np.uint8).tobytes() if n else b""
    buf.extend(struct.pack("<I", len(raw)))
    buf.extend(raw)
    buf.extend(struct.pack("<B", 1 if is_dense else 0))
    buf.extend(b"\x00" * ((-len(buf)) % 4))  # Fast-CDR pads the message to a multiple of 4 bytes
    return bytes(buf)


QOS_SENSOR = ("- history: 3\n  depth: 0\n  reliability: 1\n  durability: 2\n  deadline:\n    sec: 9223372036\n"
              "    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n"
              "  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n"
              "  avoid_ros_namespace_conventions: false")


PC2_MSGDEF = 'std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: sensor_msgs/PointField\nuint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\nuint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\nstring name\nuint32 offset\nuint8 datatype\nuint32 count\n'


class BagWriter:
    """Write one PointCloud2 topic into a rosbag2 directory.

    ``storage="mcap"`` (default): written with the ``mcap`` library, zstd chunk compression (decoded by the MCAP
    storage plugin itself, no rosbag2 compression plugin needed), ros2 profile, ros2msg schema.
    ``storage="sqlite3"``: written with ``rosbags`` (uncompressed).
    """

    def __init__(self, path: str, topic: str, storage: str = "mcap", compression: str = "zstd"):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.path = path
        self.topic = topic
        self.storage = storage
        self.count = 0
        self.t_first = None
        self.t_last = None
        name = os.path.basename(os.path.normpath(path))
        if storage == "mcap":
            from mcap.writer import CompressionType, Writer as McapWriter

            self._fh = open(os.path.join(path, f"{name}_0.mcap"), "wb")
            comp = {"zstd": CompressionType.ZSTD, "lz4": CompressionType.LZ4, "none": CompressionType.NONE}[compression]
            self._w = McapWriter(self._fh, chunk_size=4 * 1024 * 1024, compression=comp)
            self._w.start(profile="ros2", library="imos-synth")
            sid = self._w.register_schema(name="sensor_msgs/msg/PointCloud2", encoding="ros2msg",
                                          data=PC2_MSGDEF.encode())
            self._cid = self._w.register_channel(topic=topic, message_encoding="cdr", schema_id=sid,
                                                 metadata={"offered_qos_profiles": QOS_SENSOR})
        else:
            from rosbags.rosbag2 import StoragePlugin, Writer
            from rosbags.typesys import Stores, get_typestore

            os.rmdir(path)
            self._fh = None
            self._w = Writer(path, version=8, storage_plugin=StoragePlugin.SQLITE3)
            self._w.open()
            self._conn = self._w.add_connection(topic, "sensor_msgs/msg/PointCloud2",
                                                typestore=get_typestore(Stores.ROS2_HUMBLE))

    def write(self, raw: bytes, bag_time_ns: int) -> None:
        t = int(bag_time_ns)
        if self.storage == "mcap":
            self._w.add_message(channel_id=self._cid, log_time=t, data=raw, publish_time=t, sequence=self.count)
        else:
            self._w.write(self._conn, t, raw)
        self.count += 1
        self.t_first = t if self.t_first is None else self.t_first
        self.t_last = t

    def close(self) -> None:
        if self.storage == "mcap":
            self._w.finish()
            self._fh.close()
        else:
            self._w.close()
        finalize_for_humble(self.path, self.topic, self.storage, self.count, self.t_first or 0, self.t_last or 0)


def finalize_for_humble(path: str, topic: str, storage: str, count: int, t_first: int, t_last: int) -> None:
    """Rewrite metadata.yaml in the rosbag2 version-5 layout understood by Humble (and newer releases)."""
    files = sorted(f for f in os.listdir(path) if f.endswith(".mcap") or f.endswith(".db3"))
    if len(files) != 1:
        raise RuntimeError(f"expected one storage file in {path}: {files}")
    dur = max(0, int(t_last) - int(t_first))
    qos = QOS_SENSOR.replace("\n", "\\n").replace('"', '\\"')
    text = (
        "rosbag2_bagfile_information:\n"
        "  version: 5\n"
        f"  storage_identifier: {storage}\n"
        f"  duration:\n    nanoseconds: {dur}\n"
        f"  starting_time:\n    nanoseconds_since_epoch: {int(t_first)}\n"
        f"  message_count: {count}\n"
        "  topics_with_message_count:\n"
        "    - topic_metadata:\n"
        f"        name: {topic}\n"
        "        type: sensor_msgs/msg/PointCloud2\n"
        "        serialization_format: cdr\n"
        f"        offered_qos_profiles: \"{qos}\"\n"
        f"      message_count: {count}\n"
        "  compression_format: \"\"\n"
        "  compression_mode: \"\"\n"
        "  relative_file_paths:\n"
        f"    - {files[0]}\n"
        "  files:\n"
        f"    - path: {files[0]}\n"
        f"      starting_time:\n        nanoseconds_since_epoch: {int(t_first)}\n"
        f"      duration:\n        nanoseconds: {dur}\n"
        f"      message_count: {count}\n"
    )
    with open(os.path.join(path, "metadata.yaml"), "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- labels
class LabelWriter:
    """labels.jsonl (one line per frame) and slot_labels.npz (sparse per-slot truth)."""

    def __init__(self, scene_dir: str):
        self.dir = scene_dir
        os.makedirs(scene_dir, exist_ok=True)
        self._fh = open(os.path.join(scene_dir, "labels.jsonl"), "w", encoding="utf-8")
        self._frames: List[np.ndarray] = []
        self._slots: List[np.ndarray] = []
        self._ids: List[np.ndarray] = []

    def frame(self, i: int, stamp_ns: int, objects: Sequence[dict], sight_limit_m: Optional[float],
              slots: np.ndarray, obj_ids: np.ndarray, extra: Optional[dict] = None) -> None:
        rec = {"frame": int(i), "stamp_ns": int(stamp_ns),
               "sight_limit_m": None if sight_limit_m is None else round(float(sight_limit_m), 2),
               "objects": [_round_obj(o) for o in objects]}
        if extra:
            rec.update(extra)
        self._fh.write(json.dumps(rec) + "\n")
        if len(slots):
            self._frames.append(np.full(len(slots), i, dtype=np.int32))
            self._slots.append(np.asarray(slots, dtype=np.int32))
            self._ids.append(np.asarray(obj_ids, dtype=np.int32))

    def close(self) -> None:
        self._fh.close()
        cat = (lambda a: np.concatenate(a) if a else np.zeros(0, dtype=np.int32))
        np.savez_compressed(os.path.join(self.dir, "slot_labels.npz"), frame=cat(self._frames), slot=cat(self._slots),
                            obj_id=cat(self._ids))


def _round_obj(o: dict) -> dict:
    out = {}
    for k, v in o.items():
        if isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 3)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            out[k] = [round(float(x), 3) for x in v]
        else:
            out[k] = v
    return out


def write_scene_json(scene_dir: str, info: Dict) -> None:
    with open(os.path.join(scene_dir, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
