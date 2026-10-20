"""Tests of the beam model (interface A): exact slot directions of the real sensor, no-return slots, I/O."""
import glob
import os
import sqlite3

import numpy as np
import pytest

from metro_obstacle_detector.beam_model import BeamModel, axis_matrix, hesai128
from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct

TOL_DEG = 0.02
DATASET_CANDIDATES = [os.environ.get("IMOS_DATASET", ""), "/home/gag/i.mos/dataset/for_hackathon", "/data"]
PROBED = ["roundT_doubleT", "doubleT_obstacle", "roundT_squareT_pressureGate_squareT"]


def _bag_db(name):
    for root in DATASET_CANDIDATES:
        if root:
            dbs = glob.glob(os.path.join(root, name, "*.db3"))
            if dbs:
                return dbs[0]
    return None


def _frames(name, idx):
    db = _bag_db(name)
    if db is None:
        pytest.skip(f"dataset bag {name} not available")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ids = [r[0] for r in con.execute("select id from messages order by timestamp, id")]
    out = []
    for i in idx:
        raw = con.execute("select data from messages where id=?", (ids[i % len(ids)],)).fetchone()[0]
        msg = parse_pointcloud2_cdr(raw)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        out.append((np.array(pointcloud2_to_struct(msg)), stamp))
    con.close()
    return out


def _valid(cloud):
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], 1)
    return ~(xyz == 0).all(1)


# --------------------------------------------------------------------------- real data
@pytest.mark.parametrize("bag", PROBED)
def test_estimated_model_reproduces_valid_returns(bag):
    fit = _frames(bag, [0, 20, 40, 60])
    check = _frames(bag, [1, 57, -1])
    one = BeamModel.estimate([fit[0][0]], stamps_ns=[fit[0][1]])   # directions: one cloud is enough
    many = BeamModel.estimate([c for c, _ in fit], stamps_ns=[s for _, s in fit])  # crop edges need a few
    for m in (one, many):
        assert m.n_rings == 128 and m.dual_return
        assert m.n_cols in (2400, 7200)
    for cloud, _ in check:
        err = one.angular_error_deg(cloud)
        assert len(err) > 50000
        assert err.max() < TOL_DEG, f"{bag}: max angular error {err.max():.4f} deg"
        # returns only appear in slots the model says are fired
        assert not (_valid(cloud) & ~many.fired_mask()).any()


@pytest.mark.parametrize("bag", PROBED)
def test_builtin_model_reproduces_valid_returns(bag):
    for cloud, stamp in _frames(bag, [0, 33]):
        m = hesai128(len(cloud) // 128)
        err = m.angular_error_deg(cloud)
        assert err.max() < TOL_DEG, f"{bag}: max angular error {err.max():.4f} deg"
        assert not (_valid(cloud) & ~m.fired_mask()).any()
        # per-point timestamp = header stamp + column time (within 30 us)
        dt = np.asarray(cloud["timestamp"], dtype=np.float64) - stamp * 1e-9 - m.slot_time_s()
        assert np.abs(dt).max() < 3e-5


def test_layouts_share_the_sensor():
    a, b = hesai128(2400), hesai128(7200)
    assert np.allclose(a.elevation_deg, b.elevation_deg)
    assert np.allclose(a.ring_az_offset_deg, b.ring_az_offset_deg)
    # the 120 deg message is firings 1200..2399 of the full turn (az of firing 0 = 60 deg)
    assert abs(a.col_az_deg[0] - 60.0) < 0.02 and abs(b.col_az_deg[2400] - 60.0) < 0.02
    # 921k wrap: columns around the rear stay continuous in direction
    d = b.directions_dlh().reshape(7200, 128, 3)[0::2, 64]
    steps = np.degrees(np.arctan2(np.linalg.norm(np.cross(d[1:], d[:-1]), axis=1), (d[1:] * d[:-1]).sum(1)))
    assert np.allclose(steps, 0.1 * np.cos(np.radians(b.elevation_deg[64])), atol=1e-3)
    az = b.col_az_deg
    assert az.max() <= 180.0 and az.min() > -180.0


def test_fired_mask_counts():
    a = hesai128(2400)
    m = a.fired_mask().reshape(2400, 128)
    assert m[:, 25:89].mean() > 0.8           # dense band, inside the +-50 deg point-azimuth crop
    assert m.sum() < 2400 * 128 * 0.8
    az = a.slot_az_deg()[a.fired_mask()]
    assert az.min() >= -50.01 and az.max() <= 50.01
    b = hesai128(7200)
    fb = b.fired_mask().reshape(3600, 2, 128)
    assert fb[:, :, 64].all() and fb[1::2, :, 0].all() and not fb[0::2, :, 0].any()


# --------------------------------------------------------------------------- self-contained
def _synthetic_cloud(m, rng_m=10.0, drop=0.4, seed=0):
    rs = np.random.default_rng(seed)
    dirs = m.directions().astype(np.float64)
    r = rng_m * (1 + rs.random(m.n_slots))
    keep = (rs.random(m.n_slots) > drop) & m.fired_mask()
    xyz = (dirs * r[:, None]).astype(np.float32)
    xyz[~keep] = 0
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                   "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"], "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26})
    c = np.zeros(m.n_slots, dtype=dt)
    c["x"], c["y"], c["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c["ring"] = np.arange(m.n_slots) % m.n_rings
    c["timestamp"] = 1000.0 + m.slot_time_s()
    return c


def test_directions_unit_and_slot_order():
    m = hesai128(2400)
    d = m.directions()
    assert d.shape == (2400 * 128, 3) and d.dtype == np.float32
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-6)
    s = m.slot_index(5, 7)
    assert s == 7 * 128 + 5 and m.ring_of_slot(s) == 5 and m.col_of_slot(s) == 7
    # azimuth convention: forward = -y, az positive towards +l = +x (l = h x d)
    R = axis_matrix("-y", "+z")
    assert np.allclose(R[1], [1, 0, 0])
    az, el = m.angles_of_points(np.array([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0]]))
    assert np.allclose(az, [0, 45]) and np.allclose(el, 0)


def test_estimate_recovers_synthetic_model_and_roundtrip(tmp_path):
    truth = hesai128(2400)
    clouds = [_synthetic_cloud(truth, seed=s) for s in range(2)]
    m = BeamModel.estimate(clouds)
    assert m.dual_return and m.n_cols == 2400
    assert np.abs(m.elevation_deg - truth.elevation_deg).max() < 1e-4
    assert np.abs(m.ring_az_offset_deg - truth.ring_az_offset_deg).max() < 1e-4
    assert np.abs(((m.col_az_deg - truth.col_az_deg + 180) % 360) - 180).max() < 1e-4
    assert np.array_equal(m.ring_fire_parity, truth.ring_fire_parity)
    assert m.az_window_deg is not None and abs(m.az_window_deg[0] + 50) < 0.05
    p = tmp_path / "bm.npz"
    m.save(p)
    m2 = BeamModel.load(p)
    assert np.array_equal(m2.directions(), m.directions())
    assert np.array_equal(m2.fired_mask(), m.fired_mask())
    assert m2.az_window_deg == m.az_window_deg and m2.meta == m.meta


def test_estimate_other_axes():
    truth = hesai128(7200, forward_axis="+x", up_axis="+z")
    m = BeamModel.estimate([_synthetic_cloud(truth, seed=3)], forward_axis="+x", up_axis="+z")
    assert m.az_window_deg is None
    err = m.angular_error_deg(_synthetic_cloud(truth, seed=4))
    assert err.max() < 1e-3


# --------------------------------------------------------------------------- Falcon Prime slot table
FALCON_BAG = os.environ.get("IMOS_FALCON_BAG", "/home/gag/i.mos/dataset/falcon/falcon_test_second_dark")
FALCON_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth", "data",
                            "falcon_prime_seyond.npz")


def _falcon_frames(idx):
    dbs = glob.glob(os.path.join(FALCON_BAG, "*.db3"))
    if not dbs or not os.path.exists(FALCON_TABLE):
        pytest.skip("Falcon bag or table not available")
    con = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
    tid = con.execute("select id from topics where name='/iv_points'").fetchone()[0]
    ids = [r[0] for r in con.execute("select id from messages where topic_id=? order by timestamp, id", (tid,))]
    out = []
    for i in idx:
        msg = parse_pointcloud2_cdr(con.execute("select data from messages where id=?", (ids[i],)).fetchone()[0])
        out.append(np.array(pointcloud2_to_struct(msg)))
    con.close()
    return out


def test_falcon_table_reproduces_held_out_frames():
    from metro_obstacle_detector.beam_model import FalconBeamModel
    m = FalconBeamModel.load(FALCON_TABLE)
    assert m.forward_axis == "+x" and m.up_axis == "+z" and m.n_slots > 90000
    for c in _falcon_frames([1500, 3500]):          # not fit frames
        first = c["is_2nd_return"] == 0
        j = m.slot_of_cloud(c[first])
        assert (j < 0).mean() < 1e-3                 # almost every return has a table slot
        err = m.angular_error_deg(c, corrected=True)
        assert np.percentile(err, 50) < 0.03 and np.percentile(err, 95) < 0.1
        nr = m.no_return_mask(c)
        assert 0.0 < nr.mean() < 0.06
        # time of the slots: before the header stamp, within one 100 ms scan
        assert m.time_s.min() > -0.101 and m.time_s.max() < 0.0


def test_falcon_table_roundtrip(tmp_path):
    from metro_obstacle_detector.beam_model import FalconBeamModel
    rs = np.random.default_rng(1)
    n = 500
    m = FalconBeamModel(scan_id=rs.integers(0, 38, n), scan_idx=np.arange(n), flags=rs.integers(0, 40, n),
                        az_deg=rs.uniform(-60, 60, n), el_deg=rs.uniform(-12, 13, n), time_s=-rs.uniform(0, 0.1, n),
                        seen_frac=rs.uniform(0, 1, n))
    m.save(tmp_path / "f.npz")
    m2 = FalconBeamModel.load(tmp_path / "f.npz")
    assert np.allclose(m2.az_deg, m.az_deg, atol=1e-9) and np.array_equal(m2.scan_id, m.scan_id)
    k = m.slot_index(m.scan_id[:10], m.scan_idx[:10], m.flags[:10])
    assert np.array_equal(k, np.arange(10))
    assert m.slot_index(37, 1999, 63) == -1
    d = m.directions()
    assert np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-6)
    # forward +x, up +z: az 0 / el 0 is +x
    az, el = m.angles_of_points(np.array([[1.0, 0, 0], [1.0, 1.0, 0]]))
    assert np.allclose(az, [0, 45]) and np.allclose(el, 0)
