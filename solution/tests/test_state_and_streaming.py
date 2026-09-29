"""Regression tests for state handling across time jumps, UNKNOWN without input and the
stream/batch accounting tools (see ПРОВЕРКА_СООТВЕТСТВИЯ_PDF.md, items 3-8)."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from metro_obstacle_detector.pipeline import DetectionPipeline
from synthetic import make_pointcloud2, tunnel_cloud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = 946_687_297_199_933_052


def _geometry(res):
    """Frame result without the timing fields and the track id / confirmation set by the temporal filter."""
    d = res.to_dict()
    out = {k: v for k, v in d.items() if "timing" not in k and not k.endswith("_ms")}
    out["candidates"] = [{k: v for k, v in c.items() if k not in ("track_id", "confirmed")} for c in out["candidates"]]
    return out


@pytest.fixture(scope="module")
def seq_clouds():
    # two "sequences": a curved tunnel (the previous bag) and a straight one with an obstacle
    return {
        "curve": tunnel_cloud(curvature=1.0 / (2 * 400.0), yaw=0.03),
        "straight": tunnel_cloud(obstacle=(45.0, 0.1, 0.6, 1.2)),
    }


def test_state_matters_for_first_frame(seq_clouds):
    """Precondition of the reset test: the previous curve does change the next result."""
    det = ObstacleDetector(DetectorConfig())
    det.process(seq_clouds["curve"])
    carried = det.process(seq_clouds["straight"])
    fresh = ObstacleDetector(DetectorConfig()).process(seq_clouds["straight"])
    assert _geometry(carried) != _geometry(fresh)


@pytest.mark.parametrize("jump_ns", [-10_000_000_000, 5_000_000_000])  # backwards (new bag / loop), long gap
def test_reset_happens_before_the_first_frame_after_a_jump(seq_clouds, jump_ns):
    p = DetectionPipeline(DetectorConfig())
    p.step(seq_clouds["curve"], T0)
    p.step(seq_clouds["curve"], T0 + 100_000_000)
    res, dec = p.step(seq_clouds["straight"], T0 + 100_000_000 + jump_ns)
    assert dec.reset
    fresh = ObstacleDetector(DetectorConfig()).process(seq_clouds["straight"])
    assert _geometry(res) == _geometry(fresh)  # no geometric state from the previous sequence
    assert dec.status == "clear"  # a single hit is not confirmed: no temporal state either


def test_no_reset_within_a_sequence(seq_clouds):
    p = DetectionPipeline(DetectorConfig())
    p.step(seq_clouds["straight"], T0)
    _, dec = p.step(seq_clouds["straight"], T0 + 100_000_000)
    assert not dec.reset and dec.status == "detected"


def test_invalid_frame_keeps_the_time_reference(seq_clouds):
    p = DetectionPipeline(DetectorConfig())
    p.step(seq_clouds["straight"], T0)
    res, dec = p.step_invalid("format error: test", T0 + 100_000_000)
    assert res.status == dec.status == "unknown"
    _, dec = p.step(seq_clouds["straight"], T0 - 3_000_000_000)  # the jump after the bad frame is still seen
    assert dec.reset


# ---------------------------------------------------------------- bag based (Humble container)
def _write_bag(path, frames, topic="/lidar_points"):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    w.create_topic(rosbag2_py.TopicMetadata(name=topic, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))
    for k, (xyz, stamp) in enumerate(frames):
        msg = make_pointcloud2(xyz)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        w.write(topic, serialize_message(msg), 1_000_000_000_000 + k * 100_000_000)  # bag time is monotonic
    del w


@pytest.fixture(scope="module")
def jump_bag(tmp_path_factory, seq_clouds):
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("sensor_msgs.msg")
    path = str(tmp_path_factory.mktemp("bags") / "jump")
    stamps = [T0, T0 + 100_000_000, T0 - 10_000_000_000, T0 - 9_900_000_000]
    clouds = [seq_clouds["curve"], seq_clouds["curve"], seq_clouds["straight"], seq_clouds["straight"]]
    _write_bag(path, list(zip(clouds, stamps)))
    return path, stamps, clouds


def test_offline_resets_before_processing_like_the_node(jump_bag, tmp_path):
    from metro_obstacle_detector.offline import _json_default, run_bag

    path, stamps, clouds = jump_bag
    out = str(tmp_path / "jump.jsonl")
    summary = run_bag(path, out, DetectorConfig(), verbose=False)
    assert summary["complete"] and summary["messages_read"] == 4 and summary["frames_processed"] == 4
    rows = [json.loads(line) for line in open(out)]
    assert [r["reset"] for r in rows] == [False, False, True, False]
    # the same sequence through the shared pipeline (what the node runs) gives the same records
    p = DetectionPipeline(DetectorConfig())
    for r, xyz, st in zip(rows, clouds, stamps):
        res, dec = p.step(xyz, st)
        assert (r["status"], r["reset"], r["header_time_ns"]) == (dec.status, dec.reset, st)
        assert {k: v for k, v in r["detector"].items() if "timing" not in k} == \
            {k: v for k, v in json.loads(json.dumps(res.to_dict(), default=_json_default)).items() if "timing" not in k}
    fresh = ObstacleDetector(DetectorConfig()).process(clouds[2])
    assert rows[2]["detector"]["n_box"] == fresh.n_box


def test_offline_cli_exit_code_and_stride(jump_bag, tmp_path):
    path, _, _ = jump_bag
    out = str(tmp_path / "s.jsonl")
    r = subprocess.run([sys.executable, "-m", "metro_obstacle_detector.offline", "--bag", path, "--out", out,
                        "--stride", "3", "--quiet"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = json.load(open(str(tmp_path / "s_summary.json")))
    assert (s["frames_expected"], s["frames_processed"], s["complete"]) == (2, 2, True)


def test_bag_tools_headers_and_coverage(jump_bag, tmp_path):
    from metro_obstacle_detector import bag_tools

    path, stamps, _ = jump_bag
    heads = bag_tools.read_headers(path, "/lidar_points")
    assert [h["header_time_ns"] for h in heads] == stamps
    assert {h["frame_id"] for h in heads} == {"hesai_lidar"}
    # a node log that missed the first and the last frame, plus one stale UNKNOWN before the input
    log = tmp_path / "stream.jsonl"
    log.write_text("".join(json.dumps({"header_time_ns": s, "status": "clear", "processing_ms": 10.0}) + "\n" for s in stamps[1:3]))
    obs = tmp_path / "obs.jsonl"
    obs.write_text(json.dumps({"rx_monotonic_s": 1.0, "status": 0, "header_time_ns": 5, "diagnostics": "no input"}) + "\n" +
                   "".join(json.dumps({"rx_monotonic_s": 2.0 + i, "status": 1, "header_time_ns": s}) + "\n" for i, s in enumerate(stamps[1:3])))
    rep = bag_tools.coverage(path, str(log), observed_path=str(obs), player_exit=0)
    assert (rep["sent"], rep["processed"], rep["not_processed_indices"]) == (4, 2, [0, 3])
    assert (rep["not_processed_at_start"], rep["not_processed_at_end"], rep["not_processed_inside"]) == (1, 1, 0)
    assert rep["observer"]["published_results_for_bag_frames"] == 2
    assert rep["observer"]["unknown_before_first_input"] == 1


def test_cdr_header_matches_serialized_message():
    pytest.importorskip("sensor_msgs.msg")
    from rclpy.serialization import serialize_message
    from metro_obstacle_detector.bag_tools import cdr_header

    msg = make_pointcloud2(np.zeros((10, 3), np.float32))
    msg.header.frame_id = "lidar_livox"
    stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    assert cdr_header(bytes(serialize_message(msg))) == (stamp, "lidar_livox")


# ---------------------------------------------------------------- ROS node (Humble container)
def test_node_reports_unknown_before_first_cloud_and_after_input_stops(seq_clouds):
    pytest.importorskip("metro_obstacle_msgs.msg")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from metro_obstacle_msgs.msg import ObstacleStatus
    from metro_obstacle_detector.node import DetectorNode

    tag = f"t{os.getpid()}"
    rclpy.init(args=["--ros-args", "-p", f"input_topic:=/{tag}/in", "-p", f"status_topic:=/{tag}/status",
                     "-p", f"marker_topic:=/{tag}/markers", "-p", f"candidates_topic:=/{tag}/cand",
                     "-p", "stale_timeout_s:=0.4", "-p", "stale_check_period_s:=0.1"],
               domain_id=int(os.environ.get("TEST_ROS_DOMAIN_ID", "91")))
    try:
        node = DetectorNode()
        probe = rclpy.create_node(f"{tag}_probe")
        got = []
        probe.create_subscription(ObstacleStatus, f"/{tag}/status", got.append, 50)
        pub = probe.create_publisher(PointCloud2, f"/{tag}/in", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(probe)

        def spin_until(cond, timeout):
            end = time.monotonic() + timeout
            while time.monotonic() < end and not cond():
                ex.spin_once(timeout_sec=0.05)

        # 1) no input at all: UNKNOWN with a diagnostic, not silence
        spin_until(lambda: len(got) >= 2, 5.0)
        assert got and all(m.status == ObstacleStatus.STATUS_UNKNOWN for m in got)
        assert "no input" in got[0].diagnostics and f"/{tag}/in" in got[0].diagnostics
        # 2) one cloud: a real result for it (frame_index 1 = first received cloud)
        spin_until(lambda: pub.get_subscription_count() > 0, 5.0)
        n_before = len(got)
        msg = make_pointcloud2(seq_clouds["straight"])
        pub.publish(msg)
        spin_until(lambda: any(m.frame_index == 1 and m.header.frame_id == "hesai_lidar" and m.status != 0 for m in got[n_before:]), 20.0)
        results = [m for m in got[n_before:] if m.frame_index == 1 and m.status != ObstacleStatus.STATUS_UNKNOWN]
        assert results and results[0].header.stamp == msg.header.stamp
        # 3) input stops: UNKNOWN again ("input stale")
        n_after = len(got)
        spin_until(lambda: any("stale" in m.diagnostics for m in got[n_after:]), 5.0)
        assert any(m.status == ObstacleStatus.STATUS_UNKNOWN and "stale" in m.diagnostics for m in got[n_after:])
        ex.shutdown()
        probe.destroy_node()
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


# ---------------------------------------------------------------- batch result checker (host script)
def _check(tmp_path, rc, summary, lines):
    (tmp_path / "b.jsonl").write_text("".join('{"frame_index": %d}\n' % i for i in range(lines)))
    if summary is not None:
        (tmp_path / "b_summary.json").write_text(json.dumps(summary))
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_offline_result.py"), str(tmp_path), "b", str(rc)],
                          capture_output=True, text=True).returncode


def test_batch_checker_rejects_failed_or_incomplete_runs(tmp_path):
    ok = {"complete": True, "frames_processed": 3, "messages_read": 3, "messages_in_metadata": 3, "frames_expected": 3}
    assert _check(tmp_path, 0, ok, 3) == 0
    assert _check(tmp_path, 125, ok, 3) != 0                       # docker failed
    assert _check(tmp_path, 0, dict(ok, complete=False), 3) != 0   # bag not read to the end
    assert _check(tmp_path, 0, ok, 2) != 0                         # truncated JSONL
    (tmp_path / "b_summary.json").unlink()
    assert _check(tmp_path, 0, None, 3) != 0                       # no summary
