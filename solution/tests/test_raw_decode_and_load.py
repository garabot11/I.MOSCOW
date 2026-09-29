"""Raw CDR decoding of PointCloud2, optional point decimation, input_topic:=auto and MCAP bags
(corrections of ПРОВЕРКА_ПО_PDF_ПОСЛЕ_ИСПРАВЛЕНИЙ.md)."""
import json
import os
import struct
import time

import numpy as np
import pytest

from metro_obstacle_detector.cloud_io import CloudFormatError, cloud_to_xyz, parse_pointcloud2_cdr
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from synthetic import make_pointcloud2, tunnel_cloud


def _xyz(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 10, (n, 3)).astype(np.float32)
    xyz[::7] = 0.0
    return xyz


def _cdr(msg, big=False):
    """Hand-written XCDR1 serialisation of a PointCloud2 (to test the big-endian encapsulation too)."""
    e = ">" if big else "<"
    out = bytearray(b"\x00\x00\x00\x00" if big else b"\x00\x01\x00\x00")

    def align(n):
        out.extend(b"\x00" * ((-(len(out) - 4)) % n))

    def u32(v):
        align(4)
        out.extend(struct.pack(e + "I", v))

    def i32(v):
        align(4)
        out.extend(struct.pack(e + "i", v))

    def string(s):
        b = s.encode() + b"\x00"
        u32(len(b))
        out.extend(b)

    i32(msg.header.stamp.sec)
    u32(msg.header.stamp.nanosec)
    string(msg.header.frame_id)
    u32(msg.height)
    u32(msg.width)
    u32(len(msg.fields))
    for f in msg.fields:
        string(f.name)
        u32(f.offset)
        out.append(f.datatype)
        u32(f.count)
    out.append(1 if msg.is_bigendian else 0)
    u32(msg.point_step)
    u32(msg.row_step)
    u32(len(msg.data))
    out.extend(bytes(msg.data))
    out.append(1 if msg.is_dense else 0)
    return bytes(out)


@pytest.mark.parametrize("kw", [{}, {"bigendian": True}, {"height": 4, "row_pad": 8}, {"extra_fields": False}])
def test_raw_parser_equals_rclpy_deserialisation(kw):
    pytest.importorskip("sensor_msgs.msg")
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import PointCloud2

    msg = make_pointcloud2(_xyz(400, 3), **kw)
    raw = bytes(serialize_message(msg))
    ref = deserialize_message(raw, PointCloud2)
    # the hand-written serialiser is valid CDR (rclpy reads it back identically), so its big-endian
    # variant is a meaningful test; bytes are not compared because Fast-CDR leaves padding bytes undefined
    mine = deserialize_message(_cdr(msg), PointCloud2)
    np.testing.assert_array_equal(cloud_to_xyz(mine, intensity=False)[0], cloud_to_xyz(ref, intensity=False)[0])
    assert (mine.header, mine.fields, mine.point_step, mine.row_step) == (ref.header, ref.fields, ref.point_step, ref.row_step)
    for buf in (raw, _cdr(msg, big=True)):
        got = parse_pointcloud2_cdr(buf)
        x1, _, i1 = cloud_to_xyz(ref, intensity=False)
        x2, _, i2 = cloud_to_xyz(got, intensity=False)
        np.testing.assert_array_equal(x1, x2)
        assert i1 == i2
        assert [(f.name, f.offset, f.datatype, f.count) for f in ref.fields] == \
            [(f.name, f.offset, f.datatype, f.count) for f in got.fields]


def test_raw_parser_rejects_truncated_and_foreign_buffers():
    raw = _cdr(make_pointcloud2(_xyz(50, 4)))
    for bad in (raw[:3], raw[:40], raw[:-30], b"\x00\x03\x00\x00" + raw[4:]):
        with pytest.raises(CloudFormatError):
            cloud_to_xyz(parse_pointcloud2_cdr(bad), intensity=False)


def test_decimation_is_off_by_default_and_deterministic():
    cloud = tunnel_cloud(obstacle=(40.0, 0.2, 0.5, 1.7))
    full = ObstacleDetector(DetectorConfig()).process(cloud)
    cap = full.n_roi // 3
    cfg = DetectorConfig.from_dict(dict(DetectorConfig().to_dict(), max_analysed_points=cap))
    a = ObstacleDetector(cfg).process(cloud)
    b = ObstacleDetector(cfg).process(cloud)
    assert DetectorConfig().max_analysed_points == 0
    assert a.n_roi <= cap < full.n_roi
    assert a.to_dict()["candidates"] == b.to_dict()["candidates"]  # no randomness
    assert a.status == "detected"  # a near person survives keeping every 3rd point


def test_mcap_bag_gives_the_same_offline_result(tmp_path):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from metro_obstacle_detector.offline import run_bag

    clouds = [tunnel_cloud(obstacle=(40.0, 0.2, 0.5, 1.7)), tunnel_cloud(obstacle=(39.9, 0.2, 0.5, 1.7))]
    rows = {}
    for storage in ("sqlite3", "mcap"):
        uri = str(tmp_path / storage)
        w = rosbag2_py.SequentialWriter()
        try:
            w.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
        except RuntimeError:
            pytest.skip(f"rosbag2 storage plugin {storage} not installed")
        w.create_topic(rosbag2_py.TopicMetadata(name="/points", type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))
        for k, xyz in enumerate(clouds):
            m = make_pointcloud2(xyz)
            m.header.stamp.nanosec = 100_000_000 * k
            w.write("/points", serialize_message(m), 1_000_000_000 + k * 100_000_000)
        del w
        out = str(tmp_path / f"{storage}.jsonl")
        assert run_bag(uri, out, DetectorConfig(), verbose=False)["complete"]
        rows[storage] = [{k: v for k, v in json.loads(line).items() if k not in ("bag", "processing_ms", "read_ms")}
                         | {"detector": {k: v for k, v in json.loads(line)["detector"].items() if k != "timings_ms"}}
                         for line in open(out)]
    assert rows["sqlite3"] == rows["mcap"]
    assert rows["mcap"][1]["status"] == "detected"


def test_node_auto_topic_finds_the_cloud_topic():
    pytest.importorskip("metro_obstacle_msgs.msg")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from metro_obstacle_msgs.msg import ObstacleStatus
    from metro_obstacle_detector.node import DetectorNode

    tag = f"a{os.getpid()}"
    rclpy.init(args=["--ros-args", "-p", "input_topic:=auto", "-p", f"status_topic:=/{tag}/status",
                     "-p", f"marker_topic:=/{tag}/markers", "-p", f"candidates_topic:=/{tag}/cand",
                     "-p", "stale_timeout_s:=0.3", "-p", "stale_check_period_s:=0.1"],
               domain_id=int(os.environ.get("TEST_ROS_DOMAIN_ID", "92")))
    try:
        node = DetectorNode()
        probe = rclpy.create_node(f"{tag}_probe")
        got = []
        probe.create_subscription(ObstacleStatus, f"/{tag}/status", got.append, 50)
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(probe)

        def spin_until(cond, timeout):
            end = time.monotonic() + timeout
            while time.monotonic() < end and not cond():
                ex.spin_once(timeout_sec=0.05)

        spin_until(lambda: len(got) >= 2, 5.0)
        assert got and "waiting for a PointCloud2 topic" in got[0].diagnostics
        assert node.sub is None  # its own candidates publisher is not taken as input
        pub = probe.create_publisher(PointCloud2, f"/{tag}/some_lidar", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        spin_until(lambda: node.sub is not None and pub.get_subscription_count() > 0, 10.0)
        assert node.input_topic == f"/{tag}/some_lidar"
        n0 = len(got)
        msg = make_pointcloud2(tunnel_cloud())
        pub.publish(msg)
        spin_until(lambda: any(m.frame_index == 1 and m.status != ObstacleStatus.STATUS_UNKNOWN for m in got[n0:]), 20.0)
        res = [m for m in got[n0:] if m.frame_index == 1 and m.status != ObstacleStatus.STATUS_UNKNOWN]
        assert res and res[0].header.stamp == msg.header.stamp and res[0].header.frame_id == "hesai_lidar"
        ex.shutdown()
        probe.destroy_node()
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_fastdds_udp_only_profile_is_valid():
    import xml.etree.ElementTree as ET

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ns = {"p": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    tree = ET.parse(os.path.join(root, "config", "fastdds_udp_only.xml"))
    part = tree.find("p:participant", ns)
    assert part.get("is_default_profile") == "true"
    assert part.find("p:rtps/p:useBuiltinTransports", ns).text == "false"
    ids = [t.text for t in part.findall("p:rtps/p:userTransports/p:transport_id", ns)]
    types = {d.find("p:transport_id", ns).text: d.find("p:type", ns).text
             for d in tree.findall("p:transport_descriptors/p:transport_descriptor", ns)}
    assert ids and all(types[i] == "UDPv4" for i in ids)
