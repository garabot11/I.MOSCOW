"""ROS 2 node: subscribes to PointCloud2, runs the detector, publishes status/markers.

Topics (defaults):
  in : ``input_topic`` (sensor_msgs/PointCloud2, reliable, depth 1 = freshest frame; default ``auto``)
  out: /obstacles/status   (metro_obstacle_msgs/ObstacleStatus)
       /obstacles/markers  (visualization_msgs/MarkerArray, in the cloud frame)
       /obstacles/candidates (sensor_msgs/PointCloud2, returns inside the clearance box)

Time handling: the result carries the input ``header.stamp`` (sensor clock) so
that it can be matched with the input; processing time is measured with the
monotonic clock and never derived from header stamps.  When no cloud arrives
for ``stale_timeout_s`` -- also right after start, before the first cloud --
the node publishes STATUS_UNKNOWN periodically.

Input topic: ``input_topic:=auto`` subscribes to the only PointCloud2 topic that
is published in the ROS graph (its own ``/obstacles/candidates`` excluded) as soon
as it appears; with several candidates it subscribes to none and says so in the
UNKNOWN diagnostic.  In auto mode a best-effort-only publisher is matched with a
best-effort subscription.

Decoding: by default the subscription is raw (``raw_decode``) and the CDR buffer
is parsed by ``cloud_io.parse_pointcloud2_cdr`` without building a Python message
(identical values, less work in the node's only thread).

Indices: ``frame_index`` (message and JSONL log) counts the clouds received by
this node instance, starting at 1.  It is *not* the index of the cloud in a bag
(frames replaced in the depth-1 queue or lost in transport are skipped); match
online results with offline results / annotations by ``header_time_ns``.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from metro_obstacle_msgs.msg import Obstacle, ObstacleStatus

from .cloud_io import CloudFormatError, cloud_to_xyz, parse_pointcloud2_cdr
from .config import DetectorConfig
from .detector import FrameResult
from .markers import build_markers
from .pipeline import DetectionPipeline, header_stamp_ns

STATUS_CODE = {"unknown": ObstacleStatus.STATUS_UNKNOWN, "clear": ObstacleStatus.STATUS_CLEAR, "detected": ObstacleStatus.STATUS_DETECTED}


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


class DetectorNode(Node):
    def __init__(self):
        super().__init__("metro_obstacle_detector")
        self.declare_parameter("input_topic", "auto")
        self.declare_parameter("config_file", "")
        self.declare_parameter("qos_reliability", "reliable")
        self.declare_parameter("qos_depth", 1)
        self.declare_parameter("log_path", "")
        self.declare_parameter("status_topic", "/obstacles/status")
        self.declare_parameter("marker_topic", "/obstacles/markers")
        self.declare_parameter("candidates_topic", "/obstacles/candidates")
        self.declare_parameter("stale_check_period_s", 0.5)
        self.declare_parameter("raw_decode", True)
        cfg_path = self.get_parameter("config_file").value
        cfg = DetectorConfig.from_yaml(cfg_path) if cfg_path else DetectorConfig()
        # expose every detector parameter as a ROS parameter (defaults from the YAML)
        data = cfg.to_dict()
        for key, val in data.items():
            if isinstance(val, (list, tuple)):
                val = [float(v) for v in val]
            self.declare_parameter(key, val)
            data[key] = self.get_parameter(key).value
        self.cfg = DetectorConfig.from_dict(data)
        self.pipeline = DetectionPipeline(self.cfg)
        self.R = self.pipeline.detector.R
        rel = str(self.get_parameter("qos_reliability").value).lower()
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter("qos_depth").value),
            reliability=ReliabilityPolicy.RELIABLE if rel == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.raw_decode = bool(self.get_parameter("raw_decode").value)
        self.pub_status = self.create_publisher(ObstacleStatus, self.get_parameter("status_topic").value, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.get_parameter("marker_topic").value, 10)
        self.pub_cands = self.create_publisher(PointCloud2, self.get_parameter("candidates_topic").value, 5)
        topic = str(self.get_parameter("input_topic").value).strip()
        self.auto_topic = topic.lower() == "auto"
        self.auto_note = "waiting for a PointCloud2 topic (input_topic:=auto)"
        self.input_topic = topic
        self.sub = None
        if not self.auto_topic:
            self._subscribe(topic, self.qos)
        self.frame_index = 0  # clouds received by this node instance (1-based in messages / log)
        self.start_mono = time.monotonic()
        self.last_rx_mono: Optional[float] = None
        self.last_stamp_ns: Optional[int] = None
        self.last_frame_id = ""
        self.stale_announced = False
        self.proc_ms: list = []
        self.log_fh = None
        log_path = self.get_parameter("log_path").value
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self.log_fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered: complete records on kill
        self.create_timer(float(self.get_parameter("stale_check_period_s").value), self.on_timer)
        self.get_logger().info(
            f"input {topic} ({rel}, depth {self.qos.depth}, {'raw CDR' if self.raw_decode else 'rclpy'} decoding); "
            f"axes forward={self.cfg.forward_axis} up={self.cfg.up_axis}; config={cfg_path or 'defaults'}; log={log_path or 'none'}")

    def _subscribe(self, topic: str, qos: QoSProfile) -> None:
        self.input_topic = topic
        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, qos, raw=self.raw_decode)

    def _auto_discover(self) -> None:
        """input_topic:=auto: subscribe to the only PointCloud2 topic in the graph (own outputs excluded)."""
        own = {self.pub_cands.topic_name}
        clouds = sorted(n for n, types in self.get_topic_names_and_types()
                        if "sensor_msgs/msg/PointCloud2" in types and n not in own)
        if not clouds:
            self.auto_note = "waiting for a PointCloud2 topic (input_topic:=auto)"
            return
        if len(clouds) > 1:
            self.auto_note = "input_topic:=auto found several PointCloud2 topics %s - set input_topic" % clouds
            return
        topic, qos = clouds[0], self.qos
        infos = self.get_publishers_info_by_topic(topic)
        if (infos and qos.reliability == ReliabilityPolicy.RELIABLE
                and all(i.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for i in infos)):
            qos = QoSProfile(history=qos.history, depth=qos.depth, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=qos.durability)
        self._subscribe(topic, qos)
        self.get_logger().info("input_topic:=auto -> subscribed to %s (%s)" % (
            topic, "best_effort" if qos.reliability == ReliabilityPolicy.BEST_EFFORT else "reliable"))

    # ------------------------------------------------------------------
    def on_cloud(self, msg) -> None:
        t0 = time.perf_counter()
        now = time.monotonic()
        inter = (now - self.last_rx_mono) if self.last_rx_mono is not None else None
        self.last_rx_mono = now
        self.stale_announced = False
        diag = ""
        xyz = None
        try:
            if self.raw_decode:
                msg = parse_pointcloud2_cdr(msg)
            stamp_ns = header_stamp_ns(msg)
            self.last_frame_id = str(msg.header.frame_id)
            xyz, _inten, _info = cloud_to_xyz(msg, intensity=False)
        except CloudFormatError as exc:
            stamp_ns = header_stamp_ns(msg) if not isinstance(msg, (bytes, bytearray)) else None
            res, dec = self.pipeline.step_invalid(f"format error: {exc}", stamp_ns)
            diag = res.reason
            self.get_logger().warn(f"cloud rejected: {exc}")
        header = Header()
        if stamp_ns is not None:
            header.stamp.sec, header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        else:
            header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.last_frame_id
        if xyz is not None:
            # time-jump check and reset happen inside, before the geometry of this frame
            res, dec = self.pipeline.step(xyz, stamp_ns)
        if dec.reset:
            diag = (diag + "; " if diag else "") + "state reset before this frame (time jump / gap)"
        self.last_stamp_ns = stamp_ns
        proc_ms = (time.perf_counter() - t0) * 1e3
        self.proc_ms.append(proc_ms)
        if len(self.proc_ms) > 1000:
            self.proc_ms = self.proc_ms[-1000:]
        self.frame_index += 1
        self.publish(header, res, dec.status, dec.distance_m, dec.forward_m, proc_ms, diag or res.reason)
        if self.log_fh:
            rec = {
                "frame_index": self.frame_index, "header_time_ns": stamp_ns, "frame_id": header.frame_id,
                "rx_monotonic_s": round(now, 4), "inter_arrival_s": None if inter is None else round(inter, 4),
                "status": dec.status, "distance_m": dec.distance_m, "forward_m": dec.forward_m, "reset": dec.reset,
                "processing_ms": round(proc_ms, 2), "detector": res.to_dict(),
            }
            self.log_fh.write(json.dumps(rec, default=_json_default) + "\n")
        if self.frame_index % 50 == 0:
            arr = np.array(self.proc_ms)
            self.get_logger().info(
                f"frames={self.frame_index} status={dec.status} dist={dec.distance_m} "
                f"proc ms median={np.median(arr):.1f} p95={np.percentile(arr, 95):.1f} max={arr.max():.1f}")

    # ------------------------------------------------------------------
    def on_timer(self) -> None:
        """UNKNOWN while no cloud arrives: before the first one (wrong topic / QoS, sensor down) and when the input stops."""
        if self.sub is None:
            self._auto_discover()
        now = time.monotonic()
        never = self.last_rx_mono is None
        if now - (self.start_mono if never else self.last_rx_mono) <= self.cfg.stale_timeout_s:
            return
        if never and self.sub is None:
            diag = "no input: %s (%.1f s since start)" % (self.auto_note, now - self.start_mono)
        elif never:
            diag = "no input: no cloud received on %s since start (%.1f s)" % (self.input_topic, now - self.start_mono)
        else:
            diag = "input stale: no cloud for > %.1f s" % self.cfg.stale_timeout_s
        if not self.stale_announced:
            self.get_logger().warn(diag + ": publishing UNKNOWN")
            self.stale_announced = True
            self.pipeline.reset()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.last_frame_id
        res = FrameResult(status="unknown", reason="no input" if never else "input stale")
        self.publish(header, res, "unknown", None, None, 0.0, diag)

    # ------------------------------------------------------------------
    def publish(self, header, res: FrameResult, status: str, distance, forward, proc_ms: float, diag: str) -> None:
        out = ObstacleStatus()
        out.header = header
        out.status = STATUS_CODE[status]
        out.distance_m = float(distance) if distance is not None else math.nan
        out.forward_m = float(forward) if forward is not None else math.nan
        out.frame_index = int(self.frame_index)
        out.num_candidates = len(res.candidates)
        out.processing_ms = float(proc_ms)
        out.rail_level_m = float(res.rail.z0) if (res.rail is not None and res.rail.ok) else math.nan
        out.track_offset_m = float(res.curve.l0) if (res.curve is not None and res.curve.ok) else math.nan
        out.diagnostics = diag
        for c in res.candidates:
            o = Obstacle()
            o.track_id = int(max(c.track_id, 0))
            o.confirmed = bool(c.confirmed)
            o.distance_m = float(c.distance_m)
            o.forward_m = float(c.forward_m)
            o.lateral_offset_m = float(c.lateral_offset_from_track)
            o.height_above_rail_m = float(c.height_above_rail)
            o.size_forward_m, o.size_lateral_m, o.size_height_m = [float(v) for v in c.extent]
            o.num_points = int(c.n_points)
            o.confidence = float(c.confidence)
            o.center.x, o.center.y, o.center.z = [float(v) for v in c.center_sensor]
            out.obstacles.append(o)
        self.pub_status.publish(out)
        if not header.frame_id:  # no cloud seen yet: markers without a frame cannot be displayed
            return
        self.pub_markers.publish(build_markers(header, res, status, distance, self.cfg, self.R))
        if self.cfg.debug_cloud and res.box_points_sensor is not None and len(res.box_points_sensor):
            pts = res.box_points_sensor.astype(np.float32)
            self.pub_cands.publish(point_cloud2.create_cloud_xyz32(header, pts.tolist()))

    def close_log(self) -> None:
        if self.log_fh:
            self.log_fh.close()
            self.log_fh = None

    def destroy_node(self):
        self.close_log()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # a second SIGINT (e.g. Ctrl-C forwarded by ros2 launch) must not abort the shutdown with a traceback
        try:
            node.close_log()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
