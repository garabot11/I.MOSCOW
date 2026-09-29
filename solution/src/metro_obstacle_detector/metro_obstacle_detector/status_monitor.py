"""Console monitor: one line per ObstacleStatus message (for demos and quick checks).

    ros2 run metro_obstacle_detector status_monitor [--topic /obstacles/status] [--jsonl observed.jsonl]

``--jsonl`` records every received status (status, header stamp, frame index,
distance, receive time) -- an observer independent of the detector's own log,
used by ``scripts/run_stream_test.sh``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from metro_obstacle_msgs.msg import ObstacleStatus

NAMES = {ObstacleStatus.STATUS_UNKNOWN: "UNKNOWN ", ObstacleStatus.STATUS_CLEAR: "CLEAR   ", ObstacleStatus.STATUS_DETECTED: "OBSTACLE"}
COLORS = {ObstacleStatus.STATUS_UNKNOWN: "\033[33m", ObstacleStatus.STATUS_CLEAR: "\033[32m", ObstacleStatus.STATUS_DETECTED: "\033[1;41;97m"}


class Monitor(Node):
    def __init__(self, topic: str, color: bool, tail_file: str = "", tail_lines: int = 12, jsonl: str = "", quiet: bool = False):
        super().__init__("obstacle_status_monitor")
        self.color = color
        self.quiet = quiet
        self.t0 = time.monotonic()
        self.n = 0
        self.tail_file, self.tail_lines, self.tail = tail_file, tail_lines, []
        self.jsonl = open(jsonl, "w", encoding="utf-8", buffering=1) if jsonl else None
        self.create_subscription(ObstacleStatus, topic, self.cb, 10)
        print(f"listening on {topic} ...", flush=True)

    def cb(self, msg: ObstacleStatus) -> None:
        self.n += 1
        if self.jsonl:
            self.jsonl.write(json.dumps({
                "rx_monotonic_s": round(time.monotonic(), 4), "status": int(msg.status),
                "header_time_ns": int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
                "frame_id": msg.header.frame_id, "frame_index": int(msg.frame_index),
                "distance_m": None if math.isnan(msg.distance_m) else round(float(msg.distance_m), 3),
                "diagnostics": msg.diagnostics}) + "\n")
        if self.quiet:
            return
        name = NAMES.get(msg.status, str(msg.status))
        dist = f"{msg.distance_m:6.1f} m" if not math.isnan(msg.distance_m) else "   --   "
        c0, c1 = (COLORS.get(msg.status, ""), "\033[0m") if self.color else ("", "")
        extra = ""
        if msg.obstacles:
            o = min(msg.obstacles, key=lambda o: o.distance_m)
            extra = f" | nearest object: {o.distance_m:.1f} m, {o.size_lateral_m:.1f}x{o.size_height_m:.1f} m, {o.num_points} pts, conf {o.confidence:.2f}" + ("" if o.confirmed else " (unconfirmed)")
        diag = f" | {msg.diagnostics}" if msg.diagnostics else ""
        line = f"t+{time.monotonic() - self.t0:6.1f}s frame {msg.frame_index:5d} | {name} | distance {dist} | proc {msg.processing_ms:5.1f} ms{extra}{diag}"
        print(line.replace(name, c0 + name + c1, 1), flush=True)
        if self.tail_file:
            self.tail = (self.tail + [line])[-self.tail_lines:]
            tmp = self.tail_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.tail) + "\n")
            os.replace(tmp, self.tail_file)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/obstacles/status")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--tail-file", default="", help="also keep the last N lines in this file (for video overlays)")
    ap.add_argument("--tail-lines", type=int, default=12)
    ap.add_argument("--jsonl", default="", help="record every received status as one JSON line")
    ap.add_argument("--quiet", action="store_true", help="do not print (with --jsonl)")
    args, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = Monitor(args.topic, not args.no_color and sys.stdout.isatty(), args.tail_file, args.tail_lines, args.jsonl, args.quiet)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node.jsonl:
                node.jsonl.close()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
