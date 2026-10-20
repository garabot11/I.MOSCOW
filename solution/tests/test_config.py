import os

import numpy as np
import pytest

from metro_obstacle_detector.config import DetectorConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_yaml_roundtrip_matches_defaults():
    cfg = DetectorConfig.from_yaml(os.path.join(ROOT, "config", "detector.yaml"))
    assert cfg.to_dict() == DetectorConfig().to_dict()


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        DetectorConfig.from_dict({"d_min": 3.0, "no_such_parameter": 1})


def test_axis_matrix_right_handed():
    R = DetectorConfig().axis_matrix()
    d, l, h = R @ np.array([0.0, -1.0, 0.0]), R @ np.array([1.0, 0.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(d, [1, 0, 0]) and np.allclose(h, [0, 0, 1])
    assert np.allclose(np.cross(R[0], R[1]), R[2])
    assert np.allclose(R @ R.T, np.eye(3))


def test_other_axis_convention():
    cfg = DetectorConfig(forward_axis="+x", up_axis="+z")
    R = cfg.axis_matrix()
    assert np.allclose(R @ np.array([5.0, 0.0, 0.0]), [5, 0, 0])
