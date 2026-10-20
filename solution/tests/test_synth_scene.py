"""Synthetic scenes: label consistency (n_hits == slot labels), return model calibration, geometry helpers.
Runs without GPU (NumPy / Open3D ray casting on a short tunnel)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))

from synth.geometry import ray_capsule, ray_obb  # noqa: E402
from synth.objects import CLASSES, make_object  # noqa: E402
from synth.raycast import Caster  # noqa: E402
from synth.returns import ReturnModel  # noqa: E402


def _caster():
    """A caster fast enough for whole frames (Warp or Open3D); brute-force NumPy is only for small meshes."""
    c = Caster("auto")
    if c.backend == "numpy":
        pytest.skip("neither Warp (CUDA) nor Open3D available for whole-frame ray casting")
    return c


def test_return_model_calibration():
    for ret, r10 in ((ReturnModel.hesai(dropout=0.0), 200.0), (ReturnModel.falcon(dropout=0.0), 250.0)):
        assert abs(float(ret.p_detect(0.1, 1.0, r10)) - 0.9) < 1e-6
        assert float(ret.p_detect(0.1, 1.0, 50.0)) > 0.99
        assert float(ret.p_detect(0.5, 1.0, ret.max_range_m + 1)) == 0.0
    assert float(ReturnModel.hesai().p_detect(0.9, 1.0, 209.5)) == 0.0     # no Hesai return beyond 209 m
    ret = ReturnModel.hesai()
    r = ret.measure_range(np.random.default_rng(0), np.full(1000, 50.0))
    assert np.allclose(np.round(r / 0.004) * 0.004, r) and abs(r.std() - 0.02) < 0.004


def test_ray_box_and_capsule_against_meshes():
    from synth.objects import mesh_box, mesh_capsule

    rs = np.random.default_rng(2)
    d = rs.normal(size=(4000, 3))
    d[:, 0] = np.abs(d[:, 0]) * 8
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = Caster("numpy")
    V, F = mesh_box([10, 0.2, 0.1], [0.5, 0.7, 0.4])
    t_mesh, _, _ = c.cast(c.mesh(V, F), np.zeros(3), d)
    t_an = ray_obb(d, [10, 0.2, 0.1], np.eye(3), [0.5, 0.7, 0.4])
    assert np.array_equal(np.isfinite(t_mesh), np.isfinite(t_an))
    both = np.isfinite(t_an)
    assert np.allclose(t_mesh[both], t_an[both], atol=1e-6)
    V, F = mesh_capsule([10, -0.5, 0], [10, 0.5, 0.3], 0.3, n_seg=32, n_cap=8)
    t_mesh, _, _ = c.cast(c.mesh(V, F), np.zeros(3), d)
    t_an = ray_capsule(d, [10, -0.5, 0], [10, 0.5, 0.3], 0.3)
    agree = np.isfinite(t_mesh) == np.isfinite(t_an)
    assert agree.mean() > 0.99          # faceting changes only grazing rays


@pytest.mark.parametrize("cls", CLASSES)
def test_object_sizes(cls):
    o = make_object(cls, np.random.default_rng(4))
    h = o.extent[2]
    lim = {"person_standing": (1.45, 1.95), "person_crouching": (0.85, 1.05), "person_lying": (0.15, 0.4),
           "box": (0.2, 1.15), "debris": (0.05, 0.35)}[cls]
    assert lim[0] <= h <= lim[1], (cls, h)
    assert o.V[:, 2].min() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("sensor", ["hesai128", "falcon_prime"])
def test_parametric_labels_consistent(sensor):
    from synth.parametric import ParametricScene, plan_parametric_objects

    cfg = {"sensor": sensor, "layout": 2400, "speed_mps": [5.0, 5.0], "p_stationary": 0.0,
           "tunnel": {"kinds": ["rectangular"], "p_curve": 0.0, "p_platform": 0.0, "length_m": 120, "grade": False},
           "objects": {"n_objects": [2, 2], "classes": ["box", "person_standing"], "d_range_m": [12, 30],
                       "p_out_of_clearance": 0.0, "p_crossing": 0.0}}
    if sensor == "falcon_prime":
        from synth.parametric import FALCON_TABLE

        if not os.path.exists(FALCON_TABLE):
            pytest.skip("Falcon table missing")
    rng = np.random.default_rng(3)
    sc = ParametricScene(cfg, rng, _caster(), 2)
    plan = plan_parametric_objects(sc, rng, cfg["objects"], False)
    assert len(plan) == 2
    pts, slots, ids, labels, sight, _ = sc.render(0, 1_800_000_000_000_000_000, np.random.default_rng(5))
    assert sight > 30
    for lab in labels:
        assert lab["n_hits"] == int((ids == lab["id"]).sum())
        assert lab["n_hits"] > 0 and lab["visible"] and lab["is_obstacle"] and lab["in_clearance"]
        assert 10 < lab["distance_m"] < 32 and abs(lab["d_along_m"] - lab["distance_m"]) < 1.5
    xyz = np.stack([pts["x"][slots], pts["y"][slots], pts["z"][slots]], 1)
    assert (np.abs(xyz).sum(1) > 0).all()
    if sensor == "hesai128":
        from metro_obstacle_detector.beam_model import hesai128

        assert len(pts) == 307200 and hesai128(2400).fired_mask()[slots].all()
        assert ((np.abs(np.stack([pts["x"], pts["y"], pts["z"]], 1)).sum(1) > 0) <= hesai128(2400).fired_mask()).all()
    else:
        assert (np.abs(np.stack([pts["x"], pts["y"], pts["z"]], 1)).sum(1) > 0).all()   # no zero points


def test_dust_is_labelled_and_never_an_obstacle():
    from synth.dust import DustField
    from synth.parametric import ParametricScene

    cfg = {"sensor": "hesai128", "layout": 2400, "speed_mps": [0.0, 0.0], "p_stationary": 1.0,
           "tunnel": {"kinds": ["circular"], "p_curve": 0.0, "p_platform": 0.0, "length_m": 100, "grade": False}}
    rng = np.random.default_rng(8)
    sc = ParametricScene(cfg, rng, _caster(), 1)
    sc.dust = DustField.sample({"n_clouds": [1, 1], "d_range_m": [12, 12], "lateral_m": [0, 0], "height_m": [1.0, 1.0],
                                "radius_m": [1.0, 1.0], "n_puffs": [8, 9], "extinction_per_m": [0.5, 0.5],
                                "p_object_behind": 0.0}, rng, sc)
    pts, slots, ids, labels, _, _ = sc.render(0, 0, np.random.default_rng(1))
    dust = [l for l in labels if l["class"] == "dust"]
    assert len(dust) == 1 and not dust[0]["is_obstacle"]
    assert dust[0]["n_hits"] == int((ids == dust[0]["id"]).sum()) > 50
    r = np.linalg.norm(np.stack([pts["x"][slots], pts["y"][slots], pts["z"][slots]], 1), axis=1)
    assert 8 < np.median(r) < 16
    assert np.median(pts["intensity"][slots]) < 10
    # dual return: the near (odd) column is never behind the far (even) column of the same beam
    rr = np.linalg.norm(np.stack([pts["x"], pts["y"], pts["z"]], 1).astype(np.float64), axis=1).reshape(-1, 2, 128)
    both = (rr[:, 0] > 0) & (rr[:, 1] > 0)
    assert (rr[:, 1][both] <= rr[:, 0][both] + 1e-4).all()


def _box_scene(sensor, extra=None):
    from synth.parametric import ParametricScene

    cfg = {"sensor": sensor, "layout": 2400, "speed_mps": [0.0, 0.0], "p_stationary": 1.0,
           "tunnel": {"kinds": ["rectangular"], "p_curve": 0.0, "p_platform": 0.0, "length_m": 120, "grade": False}}
    cfg.update(extra or {})
    return ParametricScene(cfg, np.random.default_rng(6), _caster(), 1)


def test_inter_object_occlusion_label():
    sc = _box_scene("hesai128")
    rng = np.random.default_rng(0)
    sc.add_object(1, "box", rng, 20.0, 0.0, 0.3, 0.0, size=[0.8, 1.2, 1.2])     # in front
    sc.add_object(2, "box", rng, 25.0, 0.0, 0.3, 0.0, size=[0.5, 0.5, 0.5])     # hidden behind it
    _, slots, ids, labels, _, _ = sc.render(0, 0, np.random.default_rng(1))
    lab = {l["id"]: l for l in labels}
    assert lab[1]["occluded_by_objects_frac"] == 0.0 and lab[1]["visible"]
    assert lab[2]["occluded_by_objects_frac"] == 1.0 and lab[2]["n_hits"] == 0 and not lab[2]["visible"]


@pytest.mark.parametrize("mode", ["native", "upside_down"])
def test_center_sensor_follows_coordinate_mode(mode):
    from synth.parametric import FALCON_TABLE

    if not os.path.exists(FALCON_TABLE):
        pytest.skip("Falcon table missing")
    sc = _box_scene("falcon_prime", {"coordinate_mode": mode})
    sc.add_object(1, "box", np.random.default_rng(0), 15.0, 0.0, 0.4, 0.0, size=[0.6, 0.6, 0.6])
    pts, slots, ids, labels, _, _ = sc.render(0, 0, np.random.default_rng(1))
    mine = slots[ids == 1]
    c_pts = np.stack([pts["x"][mine], pts["y"][mine], pts["z"][mine]], 1).mean(0)
    assert np.linalg.norm(c_pts - np.array(labels[0]["center_sensor"])) < 0.5


def test_falcon_keep_no_return_option():
    from synth.parametric import FALCON_TABLE, ParametricScene

    if not os.path.exists(FALCON_TABLE):
        pytest.skip("Falcon table missing")
    cfg = {"sensor": "falcon_prime", "speed_mps": [0.0, 0.0], "p_stationary": 1.0, "keep_no_return": True,
           "tunnel": {"kinds": ["rectangular"], "p_curve": 0.0, "p_platform": 0.0, "length_m": 150, "grade": False}}
    sc = ParametricScene(cfg, np.random.default_rng(2), _caster(), 1)
    pts, *_ = sc.render(0, 0, np.random.default_rng(3))
    first = pts["is_2nd_return"] == 0
    assert first.sum() == sc.rig.tab.n_slots            # every table slot once as a first point
    zero = (np.stack([pts["x"], pts["y"], pts["z"]], 1) == 0).all(1)
    assert 0 < zero.mean() < 0.5 and not (zero & ~first).any()
