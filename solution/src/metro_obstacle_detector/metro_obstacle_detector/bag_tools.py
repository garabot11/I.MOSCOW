"""Bag inspection and stream-coverage accounting (no detector involved).

    ros2 run metro_obstacle_detector bag_tools info --bag /data/bag [--topic T] [--frame-id | --cloud-topic]
    ros2 run metro_obstacle_detector bag_tools coverage --bag /data/bag --log stream.jsonl \
        [--observed observed_status.jsonl] [--player-exit 0] [--out coverage.json]

``info`` prints the PointCloud2 topics, message counts and the real
``header.frame_id`` of the first cloud (``--frame-id``: only that, for scripts).

``coverage`` is the independent reference for streaming tests: the frames that
were *sent* are the messages of the bag (``ros2 bag play`` publishes every one
of them when it exits with 0), the frames that were *processed* are the
records of the node's JSONL log and the results that were *published* are the
statuses seen by a separate subscriber (``status_monitor --jsonl``).  All three
are matched by the header stamp, never by counters.

Only the header of each message is decoded (CDR), so a whole bag is scanned in
seconds without deserialising the clouds.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"


def cdr_header(raw: bytes) -> Tuple[int, str]:
    """(stamp_ns, frame_id) of a CDR-serialised message whose first field is std_msgs/Header."""
    if len(raw) < 16:
        raise ValueError("message too short for a header")
    endian = "<" if raw[1] & 1 else ">"  # encapsulation 0x0001 CDR_LE / 0x0000 CDR_BE
    sec, nsec, n = struct.unpack_from(endian + "iII", raw, 4)
    frame_id = raw[16:16 + max(n - 1, 0)].decode("utf-8", "replace")
    return sec * 1_000_000_000 + nsec, frame_id


def _open(bag_dir: str):
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
    return reader, storage_id


def bag_topics(bag_dir: str) -> Dict[str, dict]:
    reader, _ = _open(bag_dir)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    counts = {t.topic_metadata.name: int(t.message_count) for t in reader.get_metadata().topics_with_message_count}
    return {n: {"type": ty, "messages": counts.get(n)} for n, ty in types.items()}


def pick_cloud_topic(topics: Dict[str, dict], topic: Optional[str]) -> str:
    clouds = [n for n, t in topics.items() if t["type"] == POINTCLOUD2]
    if topic and topic in clouds:
        return topic
    if topic and len(clouds) != 1:
        raise SystemExit(f"topic {topic} is not a PointCloud2 topic of the bag (PointCloud2 topics: {clouds})")
    if topic:  # remapped at playback: the bag has exactly one cloud topic
        print(f"note: {topic} not in the bag, using its only PointCloud2 topic {clouds[0]}", file=sys.stderr)
    if not clouds:
        raise SystemExit(f"no PointCloud2 topic in the bag (topics: {sorted(topics)})")
    return clouds[0]


def read_headers(bag_dir: str, topic: str, limit: Optional[int] = None) -> List[dict]:
    """Header stamp and frame of every message of ``topic`` in recording order."""
    import rosbag2_py

    reader, _ = _open(bag_dir)
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    except Exception:  # pragma: no cover
        pass
    out = []
    while reader.has_next():
        name, raw, t = reader.read_next()
        if name != topic:
            continue
        stamp, frame = cdr_header(bytes(raw))
        out.append({"index": len(out), "bag_time_ns": int(t), "header_time_ns": stamp, "frame_id": frame})
        if limit and len(out) >= limit:
            break
    return out


def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"_corrupt": line[:80]})
    return rows


def _runs(idx: List[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def _stats(values: List[float]) -> Optional[dict]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None

    def q(p):
        return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]
    return {"median": round(q(0.5), 4), "p95": round(q(0.95), 4), "max": round(vals[-1], 4)}


def coverage(bag_dir: str, log_path: str, topic: Optional[str] = None, observed_path: Optional[str] = None,
             player_exit: Optional[int] = None, name: Optional[str] = None) -> dict:
    topic = pick_cloud_topic(bag_topics(bag_dir), topic)
    sent = read_headers(bag_dir, topic)
    by_stamp = {r["header_time_ns"]: r["index"] for r in sent}
    n = len(sent)
    rows = _load_jsonl(log_path) if os.path.exists(log_path) else []
    processed_idx, unmatched, dup = [], 0, 0
    seen = set()
    for r in rows:
        i = by_stamp.get(r.get("header_time_ns"))
        if i is None:
            unmatched += 1
            continue
        if i in seen:
            dup += 1
            continue
        seen.add(i)
        processed_idx.append(i)
    missing = [i for i in range(n) if i not in seen]
    first, last = (min(seen), max(seen)) if seen else (None, None)
    rep = {
        "bag": name or os.path.basename(os.path.normpath(bag_dir)),
        "topic": topic,
        "reference": "sent = every message of the bag (ros2 bag play publishes all of them when it exits with 0); "
                     "processed = node log records; published = statuses seen by a separate subscriber; matched by header stamp",
        "player_exit_code": player_exit,
        "sent": n,
        "processed": len(seen),
        "processed_fraction": round(len(seen) / n, 4) if n else None,
        "not_processed": len(missing),
        "not_processed_indices": missing,
        "not_processed_runs": _runs(missing),
        "not_processed_at_start": sum(1 for i in missing if first is None or i < first),
        "not_processed_at_end": sum(1 for i in missing if last is not None and i > last),
        "not_processed_inside": sum(1 for i in missing if first is not None and first < i < last),
        "log_records": len(rows),
        "log_records_not_in_bag": unmatched,
        "log_records_duplicate": dup,
        "status_counts": {},
        "processing_ms": _stats([r.get("processing_ms") for r in rows if "processing_ms" in r]),
        "inter_arrival_s": _stats([r.get("inter_arrival_s") for r in rows if r.get("inter_arrival_s") is not None]),
    }
    for r in rows:
        if r.get("header_time_ns") in by_stamp:
            rep["status_counts"][r.get("status")] = rep["status_counts"].get(r.get("status"), 0) + 1
    if observed_path:
        obs = _load_jsonl(observed_path) if os.path.exists(observed_path) else []
        pub = {by_stamp[o["header_time_ns"]] for o in obs if o.get("header_time_ns") in by_stamp}
        first_rx = min((o["rx_monotonic_s"] for o in obs if o.get("header_time_ns") in by_stamp), default=None)
        last_rx = max((o["rx_monotonic_s"] for o in obs if o.get("header_time_ns") in by_stamp), default=None)
        unknown = [o for o in obs if o.get("status") == 0 and o.get("header_time_ns") not in by_stamp]
        rep["observer"] = {
            "status_messages": len(obs),
            "published_results_for_bag_frames": len(pub),
            "processed_but_not_observed": len(seen - pub),
            "unknown_before_first_input": sum(1 for o in unknown if first_rx is None or o["rx_monotonic_s"] < first_rx),
            "unknown_after_input_stopped": sum(1 for o in unknown if last_rx is not None and o["rx_monotonic_s"] > last_rx),
            "first_unknown_diagnostics": unknown[0].get("diagnostics") if unknown else None,
        }
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info", help="topics, counts and the real frame_id of the first cloud")
    a.add_argument("--bag", required=True)
    a.add_argument("--topic")
    a.add_argument("--frame-id", action="store_true", help="print only header.frame_id of the first cloud")
    a.add_argument("--cloud-topic", action="store_true", help="print only the name of the PointCloud2 topic")
    c = sub.add_parser("coverage", help="sent (bag) vs processed (node log) vs published (observer) frames")
    c.add_argument("--bag", required=True)
    c.add_argument("--name", help="bag name for the report (default: directory name)")
    c.add_argument("--topic")
    c.add_argument("--log", required=True, help="JSONL log of the detector node (log_path)")
    c.add_argument("--observed", help="JSONL of status_monitor --jsonl")
    c.add_argument("--player-exit", type=int)
    c.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "info":
        topics = bag_topics(args.bag)
        topic = pick_cloud_topic(topics, None if args.topic == "auto" else args.topic)
        if args.cloud_topic:
            print(topic)
            return 0
        first = read_headers(args.bag, topic, limit=1)
        if args.frame_id:
            if not first:
                print(f"no message on {topic}", file=sys.stderr)
                return 1
            print(first[0]["frame_id"])
            return 0
        print(json.dumps({"bag": os.path.basename(os.path.normpath(args.bag)), "topics": topics, "cloud_topic": topic,
                          "frame_id": first[0]["frame_id"] if first else None}, indent=1))
        return 0
    rep = coverage(args.bag, args.log, None if args.topic == "auto" else args.topic, args.observed, args.player_exit, args.name)
    text = json.dumps(rep, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    short = {k: rep[k] for k in ("bag", "sent", "processed", "not_processed", "not_processed_runs", "status_counts")}
    if "observer" in rep:
        short["observer"] = rep["observer"]
    print(json.dumps(short))
    return 0


if __name__ == "__main__":
    sys.exit(main())
