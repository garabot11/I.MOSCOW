import numpy as np
import pytest

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from metro_obstacle_detector.temporal import TemporalFilter
from synthetic import tunnel_cloud


@pytest.fixture(scope="module")
def clouds():
    return {
        "empty": tunnel_cloud(),
        "person_40": tunnel_cloud(obstacle=(40.0, 0.2, 0.5, 1.7)),
        "box_80": tunnel_cloud(obstacle=(80.0, -0.4, 0.8, 1.3)),
        "curve_empty": tunnel_cloud(curvature=1.0 / (2 * 600.0), yaw=0.01),
        "curve_person_60": tunnel_cloud(curvature=1.0 / (2 * 600.0), yaw=0.01, obstacle=(60.0, 0.0, 0.5, 1.7)),
        "side_object": tunnel_cloud(obstacle=(30.0, 2.2, 0.4, 1.0)),  # next to the wall, outside the gauge
    }


def _run(det, xyz):
    return det.process(xyz)


def test_empty_tunnel_is_clear(clouds):
    res = _run(ObstacleDetector(DetectorConfig()), clouds["empty"])
    assert res.status == "clear", (res.status, res.reason, [c.to_dict() for c in res.candidates])
    assert res.rail is not None and res.rail.ok
    assert abs(res.rail.z0 + 1.3) < 0.15  # sensor 1.3 m above rail level
    assert abs(res.curve.l0) < 0.2


def test_person_detected_with_correct_distance(clouds):
    res = _run(ObstacleDetector(DetectorConfig()), clouds["person_40"])
    assert res.status == "detected"
    c = res.nearest
    assert abs(c.forward_m - 40.0) < 0.6
    assert abs(c.distance_m - 40.0) < 1.0
    assert c.extent[2] > 0.8  # most of the 1.7 m body is inside the box (legs below the box bottom are not)


def test_low_object_at_80m_detected(clouds):
    res = _run(ObstacleDetector(DetectorConfig()), clouds["box_80"])
    assert res.status == "detected"
    assert abs(res.nearest.forward_m - 80.0) < 1.0


def test_curved_tunnel_clear_and_detects(clouds):
    det = ObstacleDetector(DetectorConfig())
    res = _run(det, clouds["curve_empty"])
    assert res.status == "clear", [c.to_dict() for c in res.candidates]
    assert res.curve.b > 0  # curvature recovered with the right sign
    res2 = _run(det, clouds["curve_person_60"])
    assert res2.status == "detected"
    assert abs(res2.nearest.forward_m - 60.0) < 1.0


def test_object_outside_gauge_is_ignored(clouds):
    res = _run(ObstacleDetector(DetectorConfig()), clouds["side_object"])
    assert res.status == "clear"


def test_degenerate_inputs_are_unknown_not_clear():
    det = ObstacleDetector(DetectorConfig())
    assert det.process(np.zeros((0, 3), np.float32)).status == "unknown"
    assert det.process(np.zeros((5000, 3), np.float32)).status == "unknown"  # only placeholder returns
    nan = np.full((5000, 3), np.nan, np.float32)
    assert det.process(nan).status == "unknown"
    # a cloud with no track bed (all returns far above the sensor) -> rail level not observable -> unknown
    sky = np.column_stack([np.random.default_rng(1).normal(0, 3, 5000), -np.random.default_rng(2).uniform(5, 100, 5000),
                           np.random.default_rng(3).uniform(3, 6, 5000)]).astype(np.float32)
    res = det.process(sky)
    assert res.status == "unknown" and "rail" in res.reason


def test_temporal_confirmation_and_reset(clouds):
    cfg = DetectorConfig()
    det, tf = ObstacleDetector(cfg), TemporalFilter(cfg)
    t = 1_000_000_000
    d1 = tf.update(det.process(clouds["person_40"]), t)
    assert d1.status == "clear"  # first hit is not yet confirmed
    d2 = tf.update(det.process(clouds["person_40"]), t + 100_000_000)
    assert d2.status == "detected" and d2.distance_m is not None
    # time jump backwards (bag loop) resets the state: the next single hit is again unconfirmed
    d3 = tf.update(det.process(clouds["person_40"]), t - 5_000_000_000)
    assert d3.reset and d3.status == "clear"
    # unknown frames never become clear and do not carry a detection
    from metro_obstacle_detector.detector import FrameResult
    d4 = tf.update(FrameResult(status="unknown", reason="test"), t + 200_000_000)
    assert d4.status == "unknown"
