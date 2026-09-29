"""Sequential (offline) evaluation of a rosbag2 directory.

Reads every PointCloud2 message in bag order with rosbag2_py (no DDS, no
frame loss), runs the same per-frame pipeline as the ROS node
(``pipeline.DetectionPipeline``: time-jump check and reset before the
geometry, detector core, temporal filter) and writes one JSON line per frame
plus a summary.  The summary compares the number of messages read with the
bag metadata (``complete``); the exit code is 3 when the bag was not read to
the end.  Usage::

    python3 -m metro_obstacle_detector.offline --bag /data/roundT_doubleT \
        --out /results/roundT_doubleT.jsonl [--config detector.yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Iterator, Optional, Tuple

import numpy as np

from .cloud_io import CloudFormatError, cloud_to_xyz, parse_pointcloud2_cdr
from .config import DetectorConfig
from .pipeline import DetectionPipeline, header_stamp_ns


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def open_reader(bag_dir: str, topic: Optional[str] = None):
    """Open a rosbag2 directory; return (reader, PointCloud2 topics to read, message count of them in metadata)."""
    import rosbag2_py

    storage_id = "sqlite3"
    try:
        import yaml
        with open(os.path.join(bag_dir, "metadata.yaml"), encoding="utf-8") as fh:
            storage_id = yaml.safe_load(fh)["rosbag2_bagfile_information"]["storage_identifier"] or storage_id
    except Exception:
        pass
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [n for n, ty in types.items() if ty == "sensor_msgs/msg/PointCloud2" and (topic is None or n == topic)]
    if not wanted:
        raise RuntimeError(f"no PointCloud2 topic in {bag_dir} (topics: {types})")
    expected = None
    try:
        expected = sum(int(t.message_count) for t in reader.get_metadata().topics_with_message_count
                       if t.topic_metadata.name in wanted)
    except Exception:  # pragma: no cover - metadata API differences
        pass
    return reader, wanted, expected


def iter_pointclouds(bag_dir: str, topic: Optional[str] = None, reader_info=None) -> Iterator[Tuple[int, str, object, int]]:
    """Yield (index, topic, PointCloud2 message, bag_time_ns) in recording order.

    The message is decoded by ``parse_pointcloud2_cdr`` like in the node (identical values to
    ``deserialize_message``, without building a Python message).  A message that cannot be
    parsed is yielded as a ``CloudFormatError`` instance and reported as UNKNOWN.
    """
    import rosbag2_py

    reader, wanted, _ = reader_info or open_reader(bag_dir, topic)
    if hasattr(rosbag2_py, "StorageFilter"):
        try:
            reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
        except Exception:  # pragma: no cover - optional API
            pass
    i = 0
    while reader.has_next():
        name, raw, t = reader.read_next()
        if name not in wanted:
            continue
        try:
            msg = parse_pointcloud2_cdr(raw)
        except CloudFormatError as exc:
            msg = exc
        yield i, name, msg, int(t)
        i += 1


def run_bag(bag_dir: str, out_path: Optional[str], cfg: DetectorConfig, max_frames: Optional[int] = None,
            stride: int = 1, topic: Optional[str] = None, verbose: bool = True, no_temporal: bool = False):
    pipeline = DetectionPipeline(cfg)
    reader_info = open_reader(bag_dir, topic)
    messages_expected = reader_info[2]
    messages_read = 0
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    counts = {"clear": 0, "detected": 0, "unknown": 0}
    raw_counts = {"clear": 0, "detected": 0, "unknown": 0}
    total_ms, read_ms = [], []
    first_bag_t = None
    n_done = 0
    t_start = time.perf_counter()
    try:
        t_read = time.perf_counter()
        for i, name, msg, bag_t in iter_pointclouds(bag_dir, topic, reader_info):
            messages_read = i + 1
            read_ms.append((time.perf_counter() - t_read) * 1e3)
            if first_bag_t is None:
                first_bag_t = bag_t
            if i % stride:
                t_read = time.perf_counter()
                continue
            t0 = time.perf_counter()
            stamp_ns = None if isinstance(msg, CloudFormatError) else header_stamp_ns(msg)
            try:
                if isinstance(msg, CloudFormatError):
                    raise msg
                xyz, _inten, info = cloud_to_xyz(msg, intensity=False)
            except CloudFormatError as exc:
                xyz, info = None, None
                res, dec = pipeline.step_invalid(f"format error: {exc}", stamp_ns)
            if xyz is not None:
                res, dec = pipeline.step(xyz, stamp_ns)
            status = res.status if no_temporal else dec.status
            distance = (res.nearest.distance_m if res.nearest else None) if no_temporal else dec.distance_m
            proc_ms = (time.perf_counter() - t0) * 1e3
            total_ms.append(proc_ms)
            counts[status] += 1
            raw_counts[res.status] += 1
            rec = {
                "bag": os.path.basename(os.path.normpath(bag_dir)),
                "topic": name,
                "frame_index": i,
                "bag_time_ns": bag_t,
                "relative_time_s": round((bag_t - first_bag_t) / 1e9, 4),
                "header_time_ns": stamp_ns,
                "frame_id": info.frame_id if info else (None if stamp_ns is None else str(msg.header.frame_id)),
                "status": status,
                "distance_m": None if distance is None else round(float(distance), 3),
                "forward_m": None if dec.forward_m is None else round(float(dec.forward_m), 3),
                "reset": dec.reset,
                "processing_ms": round(proc_ms, 2),
                "read_ms": round(read_ms[-1], 2),
                "detector": res.to_dict(),
            }
            if fh:
                fh.write(json.dumps(rec, default=_json_default) + "\n")
            if verbose:
                near = res.nearest
                extra = ""
                if near is not None:
                    extra = f" nearest raw: {near.distance_m:6.1f} m (d={near.forward_m:.1f} l={near.lateral_offset_from_track:+.2f} h={near.height_above_rail:.2f} n={near.n_points} conf={near.confidence:.2f})"
                rej = f" rej={len(res.rejected)}" if res.rejected else ""
                curve = res.curve
                cs = f" curve a={curve.a:+.4f} b={curve.b:+.2e} l0={curve.l0:+.2f} n={curve.n_used}" if curve else ""
                rl = f" rail z0={res.rail.z0:.2f} g={res.rail.grade:+.4f}" if res.rail and res.rail.ok else ""
                print(f"[{rec['bag']}] #{i:4d} t={rec['relative_time_s']:7.2f}s {status:8s} dist={'-' if distance is None else f'{distance:6.1f}'} box={res.n_box:5d}{rej}{extra}{cs}{rl} {proc_ms:6.1f}ms", flush=True)
            n_done += 1
            if max_frames and n_done >= max_frames:
                break
            t_read = time.perf_counter()
    finally:
        if fh:
            fh.close()
    wall = time.perf_counter() - t_start
    arr = np.array(total_ms) if total_ms else np.zeros(1)
    frames_expected = None if messages_expected is None else -(-messages_expected // stride)
    summary = {
        "bag": os.path.basename(os.path.normpath(bag_dir)),
        "messages_in_metadata": messages_expected,
        "messages_read": messages_read,
        "stride": stride,
        "frames_expected": frames_expected,
        "frames_processed": n_done,
        # every message of the bag was read and every stride-th one processed
        "complete": bool(not max_frames and messages_expected is not None and messages_read == messages_expected
                         and n_done == frames_expected),
        "status_counts": counts,
        "status_counts_raw": raw_counts,
        "processing_ms": {"median": round(float(np.median(arr)), 2), "p95": round(float(np.percentile(arr, 95)), 2),
                          "p99": round(float(np.percentile(arr, 99)), 2), "max": round(float(arr.max()), 2), "mean": round(float(arr.mean()), 2)},
        "read_ms_median": round(float(np.median(read_ms)), 2) if read_ms else None,
        "wall_s": round(wall, 2),
        "frames_per_s_including_io": round(n_done / wall, 2) if wall > 0 else None,
        "config": cfg.to_dict(),
    }
    if out_path:
        with open(os.path.splitext(out_path)[0] + "_summary.json", "w", encoding="utf-8") as sh:
            json.dump(summary, sh, indent=1, default=_json_default)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline obstacle detection over a rosbag2 directory")
    p.add_argument("--bag", required=True, help="path to a rosbag2 directory (metadata.yaml + .db3)")
    p.add_argument("--out", help="output JSONL path (a *_summary.json is written next to it)")
    p.add_argument("--config", help="detector YAML (flat mapping or ROS parameter file)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a parameter, e.g. --set confirm_hits=1")
    p.add_argument("--topic", help="restrict to this PointCloud2 topic")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-temporal", action="store_true", help="report raw single-frame decisions")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    cfg = DetectorConfig.from_yaml(args.config) if args.config else DetectorConfig()
    if args.set:
        data = cfg.to_dict()
        for kv in args.set:
            k, v = kv.split("=", 1)
            data[k] = json.loads(v) if v[:1] in "[{tfn0123456789-." else v
        cfg = DetectorConfig.from_dict(data)
    summary = run_bag(args.bag, args.out, cfg, args.max_frames, args.stride, args.topic, not args.quiet, args.no_temporal)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    if not args.max_frames and not summary["complete"]:
        print(f"ERROR: incomplete run: read {summary['messages_read']} of {summary['messages_in_metadata']} messages, "
              f"processed {summary['frames_processed']} of {summary['frames_expected']} frames", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
