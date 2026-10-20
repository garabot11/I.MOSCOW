"""A wall-like face across the corridor (closed gate, train ahead, panel) is an obstacle, not structure."""
import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from synthetic import tunnel_cloud


def test_a_wide_tall_face_across_the_track_is_reported():
    # 2.8 m wide, 4.4 m tall box face at 60 m: it reaches above the clearance top like a wall would
    cloud = tunnel_cloud(obstacle=(60.0, 0.0, 2.8, 4.4))
    res = ObstacleDetector(DetectorConfig()).process(cloud)
    assert res.status == "detected", [c.rejected_reason for c in res.rejected][:5]
    assert any(abs(c.forward_m - 60.0) < 2.0 for c in res.candidates)


def test_a_narrow_column_at_the_edge_is_still_structure():
    cloud = tunnel_cloud(obstacle=(60.0, 1.3, 0.3, 4.4))
    res = ObstacleDetector(DetectorConfig()).process(cloud)
    assert not any(abs(c.forward_m - 60.0) < 2.0 for c in res.candidates)


def test_a_flat_sliver_near_the_box_top_is_structure():
    # a 1.2 m wide, 3 cm thick horizontal patch at 3.0 m above rail (a cable duct) 40 m ahead
    cloud = tunnel_cloud()
    n = 120
    rng = np.random.default_rng(1)
    patch = np.column_stack([rng.uniform(-0.6, 0.6, n), -40.0 - rng.uniform(0, 0.5, n), 3.0 - 1.3 + rng.uniform(0, 0.03, n)]).astype(np.float32)
    res = ObstacleDetector(DetectorConfig()).process(np.vstack([cloud, patch]))
    assert not any(abs(c.forward_m - 40.0) < 2.0 for c in res.candidates), [c.to_dict() for c in res.candidates]
    assert any("flat sliver" in c.rejected_reason for c in res.rejected if abs(c.forward_m - 40.0) < 2.0)


def test_a_flat_plate_low_on_the_track_is_still_an_obstacle():
    # the organisers' edited bag carries a 1.3 m wide, 1 cm thick plate 0.5 m above the rails: an obstacle
    cloud = tunnel_cloud()
    n = 160
    rng = np.random.default_rng(2)
    plate = np.column_stack([rng.uniform(-0.65, 0.65, n), -40.0 - rng.uniform(0, 0.3, n), 0.5 - 1.3 + rng.uniform(0, 0.01, n)]).astype(np.float32)
    res = ObstacleDetector(DetectorConfig()).process(np.vstack([cloud, plate]))
    assert any(abs(c.forward_m - 40.0) < 2.0 for c in res.candidates), [c.rejected_reason for c in res.rejected if abs(c.forward_m - 40.0) < 2.0]


def test_a_train_face_ahead_in_a_curve_is_reported():
    # R ~ 1.7 km curve, a 2.8 m wide face 70 m ahead: the in-box part is narrower than on a straight (the
    # corridor shrinks with curvature), the face must still count as an obstacle
    for curvature in (3.0e-4, 4.5e-4):
        cloud = tunnel_cloud(obstacle=(70.0, 0.0, 2.8, 4.4), curvature=curvature)
        res = ObstacleDetector(DetectorConfig()).process(cloud)
        assert any(abs(c.forward_m - 70.0) < 2.5 for c in res.candidates), (curvature, [(round(c.forward_m, 1), c.rejected_reason) for c in res.rejected if abs(c.forward_m - 70.0) < 2.5])
