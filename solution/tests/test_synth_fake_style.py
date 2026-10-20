"""Organisers'-style insertion (synth/fake_style.py, work/specs/11_fake_style.md): grid, deletion rule, order,
in-place style, 16-byte message round trip.  NumPy ray caster only (runs without Warp / Open3D)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))

from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct  # noqa: E402
from synth.fake_style import (AZ_STEP_DEG, FAKE_DTYPE, OT128_EL_DEG, FakeObject, FakeStyleRenderer,  # noqa: E402
                              InPlaceRenderer, grid_directions, to_fake_layout)
from synth.geometry import ray_obb  # noqa: E402
from synth.objects import mesh_box  # noqa: E402
from synth.raycast import Caster  # noqa: E402
from synth.writer import HESAI_DTYPE, serialize_pointcloud2  # noqa: E402

CASTER = Caster("numpy")


def test_ot128_table_and_grid():
    t = OT128_EL_DEG
    assert len(t) == 128 and np.all(np.diff(t) > 0)
    assert t[0] == -25.0 and t[-1] == 14.4 and 1.9 in t and 2.0 in t
    assert np.allclose(np.diff(t[38:102]), 0.125)          # -6.0 ... 1.875
    g = grid_directions()
    assert g.shape == (128 * 2400, 3)
    az = np.degrees(np.arctan2(g[:, 0], -g[:, 1])).reshape(128, 2400)
    el = np.degrees(np.arcsin(g[:, 2])).reshape(128, 2400)
    assert np.allclose(el[:, 0], t[::-1])                    # row 0 = top
    k = np.arange(2400)
    assert np.allclose(az[0], np.where(k < 1200, k, k - 2400) * AZ_STEP_DEG, atol=1e-9)


def _base(n=128 * 40, r=50.0, seed=0):
    """Toy base cloud: rays spread in front of the sensor (forward -y), every third point a zero (no return)."""
    rs = np.random.default_rng(seed)
    az = np.radians(rs.uniform(-10, 10, n)); el = np.radians(rs.uniform(-5, 5, n))
    d = np.stack([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)], 1) * rs.uniform(5, r, n)[:, None]
    b = np.zeros(n, dtype=HESAI_DTYPE)
    b["x"], b["y"], b["z"] = d.T.astype(np.float32)
    b["intensity"] = rs.integers(1, 200, n)
    b["x"][::3] = b["y"][::3] = b["z"][::3] = 0.0
    b["intensity"][::3] = 0
    return b


def _box(centre, half):
    V, F = mesh_box(centre, half)
    return V, F


def test_fake_a_deletion_append_and_order():
    base = _base()
    V1, F1 = _box([0.0, -20.0, 0.0], [1.0, 0.05, 1.0])     # near wall-like box at 20 m
    V2, F2 = _box([0.5, -40.0, 0.5], [0.3, 0.3, 0.3])       # small box farther away
    res = FakeStyleRenderer(CASTER).render(base, [FakeObject(1, V1, F1), FakeObject(2, V2, F2)])
    xyz = np.stack([base["x"], base["y"], base["z"]], 1).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    u = np.where(r[:, None] > 0, xyz / np.maximum(r, 1e-9)[:, None], 0)
    t1 = ray_obb(u, [0.0, -20.0, 0.0], np.eye(3), [1.0, 0.05, 1.0])
    t2 = ray_obb(u, [0.5, -40.0, 0.5], np.eye(3), [0.3, 0.3, 0.3])
    expect_del = (r > 0) & ((t1 < r) | (t2 < r))
    assert np.array_equal(res.deleted, expect_del)           # exactly the returns behind an object
    assert not res.deleted[r == 0].any()                     # zeros are never deleted
    kept = base[~res.deleted]
    assert np.array_equal(res.points["x"][:res.n_base], kept["x"])   # order of the base preserved
    tail = res.points[res.n_base:]
    assert len(tail) > 0 and np.all(tail["intensity"] == 1.0)
    # farthest object first, each object row-major on the grid
    a1, b1 = res.obj_slices[1]; a2, b2 = res.obj_slices[2]
    assert a2 < a1 and b2 == a1 and b1 == len(res.points)
    g = grid_directions()
    for a, b in ((a1, b1), (a2, b2)):
        p = np.stack([res.points["x"][a:b], res.points["y"][a:b], res.points["z"][a:b]], 1).astype(np.float64)
        un = p / np.linalg.norm(p, axis=1, keepdims=True)
        idx = np.argmax(un @ g.T, axis=1)
        assert np.all(np.diff(idx) > 0)
        assert np.max(np.degrees(np.arccos(np.clip((un * g[idx]).sum(1), -1, 1)))) < 1e-3


def test_fake_layout_roundtrip():
    p = to_fake_layout(_base())
    assert p.dtype.itemsize == 16 and p.dtype == FAKE_DTYPE
    raw = serialize_pointcloud2(p, 946685786_933397055, "hesai_lidar")
    msg = parse_pointcloud2_cdr(raw)
    assert msg.point_step == 16 and msg.height == 1 and msg.width == len(p)
    back = np.array(pointcloud2_to_struct(msg))
    assert [f.name for f in msg.fields] == ["x", "y", "z", "intensity"]
    assert np.array_equal(back["x"], p["x"]) and np.array_equal(back["intensity"], p["intensity"])


def test_in_place_never_replaces_nearer_return():
    from metro_obstacle_detector.beam_model import hesai128

    bm = hesai128(2400)
    d = bm.directions().astype(np.float64)
    fired = bm.fired_mask()
    base = np.zeros(bm.n_slots, dtype=HESAI_DTYPE)
    base["ring"] = np.arange(bm.n_slots) % 128
    rng = np.random.default_rng(3)
    rr = np.where(fired & (rng.random(bm.n_slots) < 0.6), rng.uniform(5, 60, bm.n_slots), 0.0)
    rr = rr.reshape(-1, 2, 128); rr[:, 0] = rr[:, 1]; rr = rr.reshape(-1)     # single echo: both columns equal
    p = d * rr[:, None]
    base["x"], base["y"], base["z"] = p.T.astype(np.float32)
    base["intensity"] = np.where(rr > 0, 20, 0)
    V, F = _box([0.0, -30.0, 0.0], [1.0, 0.2, 1.0])
    out, src = InPlaceRenderer(CASTER).render(base, [FakeObject(5, V, F)], bm)
    assert len(out) == len(base) and np.array_equal(out["ring"], base["ring"])
    hit = src == 5
    assert hit.any()
    ro = np.linalg.norm(np.stack([out["x"], out["y"], out["z"]], 1).astype(np.float64), axis=1)
    assert np.all(out["intensity"][hit] == 1.0)
    assert not np.any(hit & (rr > 0) & (rr < ro - 1e-3))     # never replaces a nearer return
    assert np.any(hit & (rr == 0))                           # fills no-return slots
    pairs = hit.reshape(-1, 2, 128)
    assert np.array_equal(pairs[:, 0], pairs[:, 1])          # both columns of a firing
