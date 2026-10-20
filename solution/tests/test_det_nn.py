"""Д3 tests: input builder (interface D), stand-in Hesai builder, U-Net, export/load, det_nn with GPU->CPU fallback.

The nn_input tests need NumPy only; everything with torch is skipped when torch is not installed.
``_StubDeviceManager`` is a minimal stand-in for interface C (device.py), used only here.
"""
import math

import numpy as np
import pytest

from metro_obstacle_detector import nn_input as ni
from metro_obstacle_detector.nn_input import CropConfig, RangeImage

# Hesai-like ring elevations: 25 sparse rings above, the 64-ring dense band +1.97 .. -6.07, 39 below
EL_HESAI = np.concatenate([np.linspace(14.4, 2.45, 25), np.linspace(1.97, -6.07, 64), np.linspace(-6.16, -25.1, 39)])


def make_ri(H, W, el, az, organized=True, sensor="hesai128"):
    """Empty RangeImage that satisfies interface D (all returns missing)."""
    return RangeImage(
        rng=np.zeros((2, H, W), np.float32), intensity=np.zeros((2, H, W), np.float32),
        xyz=np.zeros((2, H, W, 3), np.float32), dlh=np.zeros((2, H, W, 3), np.float32),
        valid=np.zeros((2, H, W), bool), el_deg=np.asarray(el, np.float64), az_deg=np.asarray(az, np.float64),
        ring_of_row=(np.arange(H) if organized else np.full(H, -1)).astype(np.int32),
        slot_of_pixel=np.full((2, H, W), -1, np.int32), sensor=sensor, organized=organized)


def put(ri, r, c, rng, inten=20.0, k_far_rng=None, slot=None):
    """Set a return at pixel (r, c) consistent with the pixel's angles (analysis frame == sensor frame here)."""
    el, az = math.radians(ri.el_deg[r]), math.radians(ri.az_deg[c])
    for k, rr in ((0, rng), (1, rng if k_far_rng is None else k_far_rng)):
        dlh = np.array([rr * math.cos(el) * math.cos(az), rr * math.cos(el) * math.sin(az), rr * math.sin(el)], np.float32)
        ri.rng[k, r, c] = rr
        ri.intensity[k, r, c] = inten
        ri.dlh[k, r, c] = dlh
        ri.xyz[k, r, c] = dlh
        ri.valid[k, r, c] = True
        ri.slot_of_pixel[k, r, c] = slot if slot is not None else (r * ri.rng.shape[2] + c) * 2 + k


def hesai_ri(descending=True):
    """Organised Hesai-like image; az DESCENDING like range_image (column 0 = +l side) by default."""
    W = 1356
    az = 72.3 - 0.1 * np.arange(W) if descending else -63.3 + 0.1 * np.arange(W)
    return make_ri(128, W, EL_HESAI, az)


# ---------------------------------------------------------------------------------------------------- nn_input
@pytest.mark.parametrize("descending", [True, False])
def test_native_crop_hesai_rows_and_columns(descending):
    ri = hesai_ri(descending)
    cm = ni.compute_crop_map(ri, CropConfig())
    assert cm.native_rows and cm.native_cols and cm.shape == (64, 256)
    rows = cm.src_row[:, 0]
    assert rows[0] == 25 and rows[-1] == 88 and np.all(np.diff(rows) == 1)
    az = ri.az_deg[cm.src_col[0]]
    # target columns are descending like range_image: column 0 = +12.8 deg, 128 = 0, 255 = -12.7
    assert az[0] == pytest.approx(12.8, abs=1e-6) and az[128] == pytest.approx(0.0, abs=1e-6)
    assert az[-1] == pytest.approx(-12.7, abs=1e-6)
    assert np.allclose(az, CropConfig().target_az())
    if descending:   # same order as the source: a contiguous slice
        assert np.all(np.diff(cm.src_col[0]) == 1)
    asc = CropConfig(az_descending=False)
    assert np.allclose(ri.az_deg[ni.compute_crop_map(ri, asc).src_col[0]], asc.target_az())


def test_channels_order_and_scaling():
    ri = hesai_ri()
    cm = ni.compute_crop_map(ri, CropConfig())
    r, c = 40, int(np.argmin(np.abs(ri.az_deg - 1.0)))
    put(ri, r, c, 56.0, inten=51.0, k_far_rng=80.0)
    aux = {"h_above_rail": np.full((128, 1356), 0.8, np.float32),
           "lateral_from_centreline": np.full((128, 1356), -9.0, np.float32),
           "r_expected": np.full((128, 1356), 70.0, np.float32)}
    x, cm2 = ni.build_input([ri], [aux])
    assert x.shape == (21, 64, 256) and x.dtype == np.float32
    tr, tc = np.argwhere((cm.src_row == r) & (cm.src_col == c))[0]
    v = x[:7, tr, tc]
    assert v[0] == pytest.approx(56.0 / 200) and v[1] == pytest.approx(80.0 / 200)
    assert v[2] == pytest.approx(51.0 / 255) and v[3] == 1.0
    assert v[4] == pytest.approx(0.8 / 4) and v[5] == -1.0          # lateral clipped to -1
    assert v[6] == pytest.approx((56.0 - 70.0) / 70.0)
    # no return where a return is expected: residual -1; elsewhere aux channels 0
    other = (tr, tc + 5)
    assert x[3][other] == 0 and x[4][other] == 0 and x[5][other] == 0 and x[6][other] == -1.0
    # a single frame is repeated as the history
    assert np.array_equal(x[0:7], x[7:14]) and np.array_equal(x[0:7], x[14:21])


def test_history_current_first_and_builder_state():
    frames = []
    for i in range(4):
        ri = hesai_ri()
        put(ri, 50, 680, 10.0 + i)
        frames.append(ri)
    cm = ni.compute_crop_map(frames[0], CropConfig())
    tr, tc = np.argwhere((cm.src_row == 50) & (cm.src_col == 680))[0]
    x, _ = ni.build_input(frames)  # chronological, last = current
    assert [round(x[k * 7, tr, tc] * 200, 4) for k in range(3)] == [13.0, 12.0, 11.0]
    b = ni.NNInputBuilder()
    got = []
    for ri in frames:
        xb, _ = b.push(ri)
        got.append([round(xb[k * 7, tr, tc] * 200, 4) for k in range(3)])
    assert got == [[10, 10, 10], [11, 10, 10], [12, 11, 10], [13, 12, 11]]
    b.reset()
    xb, _ = b.push(frames[0])
    assert [round(xb[k * 7, tr, tc] * 200, 4) for k in range(3)] == [10, 10, 10]


@pytest.mark.parametrize("az_desc", [True, False])
def test_binned_image_H60_W900_resampled(az_desc):
    """Angle-binned image of an unorganised sensor (Falcon-like): organized=False, ring_of_row=-1."""
    H, W = 60, 900
    el = np.linspace(12.0, -13.0, H)              # 0.424 deg rows, coarser than the target 0.128 deg
    az = np.linspace(60.0, -60.0, W) if az_desc else np.linspace(-60.0, 60.0, W)   # 0.1335 deg columns
    ri = make_ri(H, W, el, az, organized=False, sensor="falcon_prime")
    r_obj = int(np.argmin(np.abs(el - (-2.0))))
    c_obj = int(np.argmin(np.abs(az - 3.0)))
    put(ri, r_obj, c_obj, 120.0, slot=777)
    for c in range(W):                               # a row of returns outside the window (must not appear)
        put(ri, int(np.argmin(np.abs(el - 8.0))), c, 30.0)
    cfg = CropConfig()
    x, cm = ni.build_input([ri], cfg=cfg)
    assert x.shape == (21, 64, 256)
    assert not cm.native_rows and not cm.native_cols
    ins = cm.inside
    assert ins.all()  # window -6.07..+1.97 x -12.8..+12.7 deg lies inside the binned field of view
    # every target pixel comes from a source pixel at most one cell away in angle
    tel, taz = cfg.target_el(), cfg.target_az()
    assert np.abs(el[cm.src_row] - tel[:, None]).max() <= max(cfg.el_step_deg, 25.0 / 59) + 1e-9
    assert np.abs(az[cm.src_col] - taz[None, :]).max() <= max(cfg.az_pitch_deg, 120.0 / 899) + 1e-9
    hit = x[0] > 0
    assert hit.any() and np.allclose(x[0][hit], 120.0 / 200)          # only the object, not the +8 deg row
    rr, cc = np.nonzero(hit)
    assert np.abs(tel[rr] - el[r_obj]).max() <= 0.5 * (25.0 / 59) + cfg.el_step_deg
    assert np.abs(taz[cc] - az[c_obj]).max() <= 0.5 * (120.0 / 899) + cfg.az_pitch_deg
    assert set(cm.src_row[hit]) == {r_obj} and set(cm.src_col[hit]) == {c_obj}


def test_binned_finer_than_target_keeps_nearest_return():
    H, W = 200, 2400                       # 0.05 deg columns -> two source columns per target column
    el = np.linspace(8.0, -12.0, H)
    az = -59.975 + 0.05 * np.arange(W)     # +-0.025 deg around the target column centres
    ri = make_ri(H, W, el, az, organized=False, sensor="falcon_prime")
    r = int(np.argmin(np.abs(el - (-1.0))))
    c = int(np.argmin(np.abs(az - 0.025)))
    put(ri, r, c - 1, 90.0)                # az -0.025
    put(ri, r, c, 45.0)                    # az +0.025: same target cell, nearer surface -> must win
    cm = ni.compute_crop_map(ri, CropConfig())
    assert not cm.native_cols
    x, _ = ni.build_input([ri])
    vals = x[0][x[0] > 0] * 200
    assert vals.size and np.allclose(vals, 45.0)


def test_binned_image_partially_outside_fov_is_zero():
    H, W = 60, 900
    el = np.linspace(0.0, -20.0, H)       # nothing above 0 deg
    az = np.linspace(0.0, 120.0, W)       # nothing left of 0 deg
    ri = make_ri(H, W, el, az, organized=False)
    ri.rng[:] = 50.0
    ri.valid[:] = True
    x, cm = ni.build_input([ri])
    tel, taz = CropConfig().target_el(), CropConfig().target_az()
    assert not cm.inside[tel > 0.5].any() and not cm.inside[:, taz < -0.2].any()
    assert (x[0][~cm.inside] == 0).all() and (x[0][cm.inside] > 0).all()


def test_aux_from_track_matches_d1_definition():
    ri = hesai_ri()
    put(ri, 40, 700, 30.0)

    class Rail:
        ok = True

        def __call__(self, d):
            return -1.3 + 0.01 * d

    class Curve:
        ok = True

        def __call__(self, d):
            return 0.2 + 0.0 * d

    aux = ni.aux_from_track(ri, Rail(), Curve())
    d, l, h = ri.dlh[0, 40, 700]
    assert aux["h_above_rail"][40, 700] == pytest.approx(h - (-1.3 + 0.01 * d), abs=1e-5)
    assert aux["lateral_from_centreline"][40, 700] == pytest.approx(l - 0.2, abs=1e-5)
    assert np.isnan(aux["h_above_rail"][40, 701]) and aux["r_expected"] is None


def _hesai_cloud(nf=1200, seed=0):
    """Organised Hesai-layout structured cloud with known geometry: ring elevations EL_HESAI, per-ring
    azimuth intercepts, pitch -0.1 deg per firing, outer rings firing every second firing, (0,0,0) slots,
    dual returns (even column = far, odd = near).  Sensor frame: forward = -y, up = +z, l = +x."""
    rng = np.random.default_rng(seed)
    R = 128
    icpt = (56.7 if nf < 3000 else 176.7) + rng.choice([0.0, 2.2, 4.4, 6.6, 11.0], size=R)
    dt = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                   "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"], "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26})
    st = np.zeros(nf * 2 * R, dtype=dt)
    f = np.arange(nf)[:, None]
    az = np.radians(icpt[None, :] - 0.1 * f)
    el = np.radians(EL_HESAI)[None, :] * np.ones((nf, 1))
    r_near = rng.uniform(5, 150, size=(nf, R))
    r_far = np.where(rng.random((nf, R)) < 0.05, r_near + rng.uniform(1, 20, size=(nf, R)), r_near)
    fire = np.ones((nf, R), bool)
    sparse = np.r_[np.arange(25), np.arange(89, 128)]
    fire[:, sparse] = ((f + sparse[None, :]) % 2 == 0)
    ret = fire & (rng.random((nf, R)) < 0.6)
    if nf < 3000:
        ret &= np.abs(np.degrees(np.arctan2(np.sin(az), np.cos(az)))) <= 50.0
    for k, rr in ((1, r_near), (0, r_far)):  # message column 2f+k: k=1 odd = near, k=0 even = far
        d = rr * np.cos(el) * np.cos(az)
        l = rr * np.cos(el) * np.sin(az)
        h = rr * np.sin(el)
        x, y, z = l, -d, h
        m = ret
        idx = ((2 * f + k) * R + np.arange(R)[None, :])
        st["x"][idx] = np.where(m, x, 0)
        st["y"][idx] = np.where(m, y, 0)
        st["z"][idx] = np.where(m, z, 0)
        st["intensity"][idx] = np.where(m, 7 + k, 0)
    st["ring"] = np.arange(len(st)) % R
    return st, np.degrees(np.arctan2(np.sin(az), np.cos(az))), ret


@pytest.mark.parametrize("nf", [1200, 3600])
def test_standin_hesai_builder_matches_interface(nf):
    st, az_true, ret = _hesai_cloud(nf)
    b = ni.HesaiStandinBuilder()
    ri = b.build(st)
    H, W = ri.rng.shape[1:]
    assert ri.rng.shape == (2, H, W) and ri.xyz.shape == (2, H, W, 3) and ri.dlh.shape == (2, H, W, 3)
    assert ri.valid.dtype == bool and ri.slot_of_pixel.dtype == np.int64 and ri.rng.dtype == np.float32
    assert H == 128 and ri.organized and ri.sensor == "hesai128"
    assert np.all(np.diff(ri.el_deg) < 0) and np.allclose(ri.el_deg, EL_HESAI, atol=1e-3)
    assert np.array_equal(ri.ring_of_row, np.arange(128))
    xyz = np.stack([st["x"], st["y"], st["z"]], 1)
    for k in range(2):
        m = ri.slot_of_pixel[k] >= 0
        slots = ri.slot_of_pixel[k][m]
        assert len(np.unique(slots)) == len(slots)
        assert np.array_equal(ri.xyz[k][m], xyz[slots])
        assert np.all((slots // 128) % 2 == (1 if k == 0 else 0))          # near = odd message column
        assert np.array_equal(ri.valid[k][m], ~(xyz[slots] == 0).all(1))
    assert len(np.unique(ri.slot_of_pixel[ri.slot_of_pixel >= 0])) == len(st)
    assert np.all(ri.rng[0][ri.valid[0]] <= ri.rng[1][ri.valid[0]] + 1e-3)    # near <= far
    v = ri.valid[0]
    az_meas = np.degrees(np.arctan2(ri.dlh[0][..., 1], ri.dlh[0][..., 0]))
    err = ((az_meas - ri.az_deg[None, :] + 180) % 360 - 180)[v]
    assert np.abs(err).max() <= 0.0501                                       # columns are true azimuth bins
    assert np.allclose(ri.dlh[0][..., 0], -ri.xyz[0][..., 1]) and np.allclose(ri.dlh[0][..., 1], ri.xyz[0][..., 0])
    assert np.all(np.diff(np.unwrap(np.radians(ri.az_deg))) < 0)            # descending like range_image
    cm = ni.compute_crop_map(ri, CropConfig())
    assert cm.native_rows and cm.native_cols
    assert cm.src_row[0, 0] == 25 and cm.src_row[-1, 0] == 88
    assert ri.az_deg[cm.src_col[0, 128]] == pytest.approx(0.0, abs=1e-6)
    ri2 = b.build(st)  # cached geometry gives the same image
    assert np.array_equal(ri2.slot_of_pixel, ri.slot_of_pixel)


def _msg_of(st, frame_id="hesai_lidar", stamp_ns=1_700_000_000_000_000_000):
    """parse_pointcloud2_cdr-like message object for a structured cloud (what the node builds frames from)."""
    from types import SimpleNamespace as NS

    names = st.dtype.names
    codes = {"<f4": 7, "<f8": 8, "<u2": 4, "<u1": 2}
    fields = [NS(name=n, offset=st.dtype.fields[n][1], datatype=codes[st.dtype.fields[n][0].str], count=1) for n in names]
    return NS(header=NS(stamp=NS(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9), frame_id=frame_id), height=1,
              width=len(st), fields=fields, is_bigendian=False, point_step=st.dtype.itemsize,
              row_step=st.dtype.itemsize * len(st), data=st.tobytes(), is_dense=False)


def test_range_image_module_and_fallback_give_the_same_crop():
    """range_image.build_range_image (reference) and the fallback builder agree on the default crop."""
    from metro_obstacle_detector.config import DetectorConfig
    from metro_obstacle_detector.frame import FrameBuilder
    from metro_obstacle_detector.range_image import build_range_image

    st, _, _ = _hesai_cloud(1200)
    frame = FrameBuilder(DetectorConfig(axes_mode="config")).build(_msg_of(st))
    ri = build_range_image(frame)
    assert ri.organized and ri.sensor == "hesai128" and np.all(np.diff(ri.az_deg) < 0)
    assert ni.range_image_builder_id(ri) == ni.RANGE_IMAGE_BUILDER_ID
    fb = ni.HesaiStandinBuilder().build(st)
    assert ni.range_image_builder_id(fb) == ni.STANDIN_BUILDER_ID
    cfg = CropConfig()
    cm_a, cm_b = ni.compute_crop_map(ri, cfg), ni.compute_crop_map(fb, cfg)
    assert cm_a.native_rows and cm_a.native_cols and cm_b.native_rows and cm_b.native_cols
    assert np.array_equal(ri.ring_of_row[cm_a.src_row[:, 0]], np.arange(25, 89))
    assert np.abs(ri.az_deg[cm_a.src_col[0]] - cfg.target_az()).max() <= 0.051   # grid phase of range_image
    sa, sb = cm_a.gather(ri.slot_of_pixel[0], -1), cm_b.gather(fb.slot_of_pixel[0], -1)
    agree = np.mean(sa == sb)
    assert agree > 0.9, agree       # the two de-shear roundings may differ by one column for some rings
    xa, _ = ni.build_input([ri], cfg=cfg)
    xb, _ = ni.build_input([fb], cfg=cfg)
    same = sa == sb
    assert np.allclose(xa[:4][:, same], xb[:4][:, same], atol=1e-6)


def test_standin_rejects_wrong_layout():
    st, _, _ = _hesai_cloud(1200)
    st = st.copy()
    st["ring"] = (np.arange(len(st)) // 2400) % 128   # column-fastest
    with pytest.raises(ValueError):
        ni.HesaiStandinBuilder().build(st)
    with pytest.raises(ValueError):
        ni.HesaiStandinBuilder().build(np.zeros((1000, 3), np.float32))


# ---------------------------------------------------------------------------------------------------- torch
try:
    import torch
    from metro_obstacle_detector import det_nn
    from metro_obstacle_detector.nn_model import RangeUNet, count_macs, count_parameters, export_d3
    _Module = torch.nn.Module
except ImportError:  # the runtime image without torch: only the NumPy tests above run
    torch = None
    _Module = object
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


class _StubDeviceManager:
    """Test-only stand-in for interface C (device.py)."""

    def __init__(self, device="cpu", mode=None):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mode = mode or ("gpu" if self.device.type == "cuda" else "cpu_forced")
        self.reason = ""
        self.failures = []

    def report_failure(self, exc):
        self.failures.append(exc)
        self.mode = "cpu_emergency"
        self.reason = str(exc)
        self.device = torch.device("cpu")


class _RuleNet(_Module):
    """Deterministic 'model': obstacle where 40 m < near range < 70 m, dust where 0 < range < 15 m."""

    def forward(self, x):
        r = x[:, 0:1] * 200.0
        obst = ((r > 40.0) & (r < 70.0)).float() * 12.0
        dust = ((r > 0.0) & (r < 15.0)).float() * 12.0
        return torch.cat([torch.zeros_like(obst) + 2.0, obst, dust], dim=1)


def _save_rule(path, meta=None):
    m = torch.jit.trace(_RuleNet(), torch.zeros(1, 21, 64, 256))
    import json
    torch.jit.save(m, str(path), _extra_files={"meta.json": json.dumps(meta or {"crop": CropConfig().to_dict()})})
    return str(path)


@needs_torch
def test_unet_shapes_parameters_and_macs():
    m = RangeUNet().eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 21, 64, 256))
    assert y.shape == (2, 3, 64, 256)
    n = count_parameters(m)
    assert 0.5e6 <= n <= 2.0e6
    assert 0.3e9 <= count_macs(m) <= 1.5e9
    with torch.no_grad():
        assert m(torch.zeros(1, 21, 64, 512)).shape == (1, 3, 64, 512)
    ops = {type(mod).__name__ for mod in m.modules()}
    assert ops <= {"RangeUNet", "Sequential", "Conv2d", "BatchNorm2d", "ReLU", "ModuleList", "_Up", "Identity"}


@needs_torch
def test_export_and_load_torchscript_and_state_fallback(tmp_path):
    torch.manual_seed(0)
    m = RangeUNet().eval()
    paths = export_d3(m, str(tmp_path / "d3_unet.pt"), {"crop": CropConfig().to_dict(), "smoke_test": True})
    x = torch.randn(1, 21, 64, 256)
    with torch.no_grad():
        ref = m(x)
    ts, meta, kind = det_nn.load_model_cpu(paths["torchscript"])
    assert kind == "torchscript" and meta["smoke_test"] is True and meta["model_config"]["widths"] == [24, 48, 96, 160]
    with torch.no_grad():
        assert torch.allclose(ts(x), ref, atol=1e-4)
    (tmp_path / "d3_unet.pt").write_bytes(b"not a torchscript archive")
    sd, meta2, kind2 = det_nn.load_model_cpu(paths["torchscript"])
    assert kind2 == "state_dict" and meta2["smoke_test"] is True
    with torch.no_grad():
        assert torch.allclose(sd(x), ref, atol=1e-4)


def _scene():
    ri = hesai_ri()
    c0 = int(np.argmin(np.abs(ri.az_deg - 0.0)))
    obj = [(r, c) for r in range(50, 56) for c in range(c0 - 3, c0 + 3)]  # 36 pixels at 55..56 m
    for i, (r, c) in enumerate(obj):
        put(ri, r, c, 55.0 + 0.02 * i, slot=10_000 + i)
    far = [(r, c) for r in range(50, 53) for c in range(c0 + 3, c0 + 5)]  # adjacent in the image, 65 m
    for i, (r, c) in enumerate(far):
        put(ri, r, c, 65.0, slot=20_000 + i)
    dust = [(r, c) for r in range(60, 63) for c in range(c0 - 40, c0 - 36)]
    for i, (r, c) in enumerate(dust):
        put(ri, r, c, 8.0, slot=30_000 + i)
    return ri, obj, far, dust


@needs_torch
def test_det_nn_candidates_geometry_split_and_dust(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    dm = _StubDeviceManager("cpu")
    det = det_nn.NNDetector(dm, path)
    ri, obj, far, dust = _scene()
    cands = det.infer([ri])
    obst = [c for c in cands if c.cls == "obstacle"]
    dc = [c for c in cands if c.cls == "dust"]
    assert len(obst) == 2 and len(dc) == 1                       # split along range: 55 m vs 65 m
    near_c = obst[0]
    assert near_c.detector == "d3" and near_c.n_support == len(obj)
    assert near_c.distance_m == pytest.approx(np.percentile([55.0 + 0.02 * i for i in range(len(obj))], 10), abs=1e-3)
    assert sorted(near_c.slots.tolist()) == list(range(10_000, 10_000 + len(obj)))
    pts = np.array([ri.dlh[0, r, c] for r, c in obj])
    assert near_c.d == pytest.approx(0.5 * (pts[:, 0].min() + pts[:, 0].max()), abs=1e-4)
    assert near_c.extent_dlh[1] == pytest.approx(pts[:, 1].max() - pts[:, 1].min(), abs=1e-4)
    assert near_c.h_above_rail is None and 0.99 < near_c.score <= 1.0      # unknown -> None (JSON-safe)
    assert obst[1].n_support == len(far) and obst[1].distance_m == pytest.approx(65.0, abs=1e-4)
    assert dc[0].n_support == len(dust) and dc[0].distance_m == pytest.approx(8.0)
    assert dm.failures == [] and det.device_name == "cpu"
    assert near_c.lateral_offset is None and near_c.info["n_crop_px"] == len(obj)
    import json
    json.dumps({"h": near_c.h_above_rail, "lat": near_c.lateral_offset, "info": near_c.info}, allow_nan=False)
    # with aux arrays (Д1 rail / centre line) the candidate carries h_above_rail and lateral_offset
    aux = {"h_above_rail": np.full(ri.rng.shape[1:], 0.6, np.float32),
           "lateral_from_centreline": np.full(ri.rng.shape[1:], -0.3, np.float32)}
    c2 = [c for c in det.infer([ri], [aux]) if c.cls == "obstacle"][0]
    assert c2.h_above_rail == pytest.approx(0.6) and c2.lateral_offset == pytest.approx(-0.3)
    assert type(c2).__module__.endswith("candidates") and det_nn.DET_CANDIDATE_SOURCE == "candidates"


@needs_torch
def test_det_nn_binned_image_no_double_count(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    det = det_nn.NNDetector(_StubDeviceManager("cpu"), path)
    H, W = 60, 900
    ri = make_ri(H, W, np.linspace(12.0, -13.0, H), np.linspace(-60.0, 60.0, W), organized=False, sensor="falcon_prime")
    r, c = int(np.argmin(np.abs(ri.el_deg + 2.0))), int(np.argmin(np.abs(ri.az_deg)))
    put(ri, r, c, 50.0, slot=5)
    put(ri, r, c + 1, 50.2, slot=6)
    cands = det.infer([ri])
    assert len(cands) == 1 and cands[0].n_support == 2 and sorted(cands[0].slots.tolist()) == [5, 6]


class _FakeCuda:
    type = "cuda"


class _Boom(_Module):
    def forward(self, x):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


@needs_torch
def test_gpu_failure_reruns_on_cpu_and_reports_once(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    dm = _StubDeviceManager("cpu")
    det = det_nn.NNDetector(dm, path)
    ref = det.infer([_scene()[0]])
    # pretend the detector runs on a GPU whose next kernel faults
    dm.device, dm.mode = _FakeCuda(), "gpu"
    det._gpu_model, det._gpu_dev, det.gpu_fp16 = _Boom(), torch.device("cpu"), False
    assert det.on_gpu
    got = det.infer([_scene()[0]])
    assert len(dm.failures) == 1 and "illegal memory access" in str(dm.failures[0])
    assert dm.mode == "cpu_emergency" and det.device_name == "cpu" and det.gpu_failures == 1
    assert [(c.cls, c.n_support, round(c.distance_m, 4)) for c in got] == [(c.cls, c.n_support, round(c.distance_m, 4)) for c in ref]
    det.infer([_scene()[0]])
    assert len(dm.failures) == 1  # sticky: the faulted GPU copy is gone


@needs_torch
def test_bug_on_both_devices_is_raised_without_emergency(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    dm = _StubDeviceManager("cpu")
    det = det_nn.NNDetector(dm, path)
    dm.device, dm.mode = _FakeCuda(), "gpu"
    det._gpu_model, det._gpu_dev, det.gpu_fp16 = _Boom(), torch.device("cpu"), False
    det._cpu_model = _Boom()
    with pytest.raises(RuntimeError):
        det.predict_proba(np.zeros((21, 64, 256), np.float32))
    assert dm.failures == [] and dm.mode == "gpu"


@needs_torch
def test_follows_device_manager_leaving_gpu_mode(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    dm = _StubDeviceManager("cpu")
    det = det_nn.NNDetector(dm, path)
    dm.device, dm.mode = _FakeCuda(), "gpu"
    det._gpu_model, det._gpu_dev = _Boom(), torch.device("cpu")
    dm.mode = "cpu_emergency"     # another component failed and the manager switched
    det.predict_proba(np.zeros((21, 64, 256), np.float32))
    assert dm.failures == [] and det.device_name == "cpu"


@needs_torch
def test_step_keeps_history(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    det = det_nn.NNDetector(_StubDeviceManager("cpu"), path)
    ri = _scene()[0]
    a = det.step(ri)
    b = det.step(ri)
    assert len(a) == len(b) == 3 and set(det.last_timings_ms) >= {"input", "forward", "post", "total"}
    det.reset()


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="no CUDA device")
@needs_torch
def test_real_gpu_matches_cpu(tmp_path):
    torch.manual_seed(0)
    m = RangeUNet().eval()
    paths = export_d3(m, str(tmp_path / "d3_unet.pt"), {"crop": CropConfig().to_dict()})
    dm = _StubDeviceManager("cuda:0")
    bench_flag = torch.backends.cudnn.benchmark
    det = det_nn.NNDetector(dm, paths["torchscript"])
    assert not det.on_gpu                     # the GPU copy is created lazily
    assert det.warmup() == "gpu" and det.on_gpu and dm.failures == []
    assert torch.backends.cudnn.benchmark == bench_flag   # process-wide cudnn flags untouched
    x = np.random.default_rng(0).random((21, 64, 256)).astype(np.float32)
    p_gpu = det.predict_proba(x)
    p_cpu = det._run_cpu(x[None])
    assert np.abs(p_gpu - p_cpu).max() < 5e-3  # fp16 on the GPU vs fp32 on the CPU


@needs_torch
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_copy_created_when_device_manager_turns_to_gpu_later(tmp_path):
    torch.manual_seed(0)
    paths = export_d3(RangeUNet().eval(), str(tmp_path / "d3_unet.pt"), {"crop": CropConfig().to_dict()})
    dm = _StubDeviceManager("cpu")                       # not probed yet
    det = det_nn.NNDetector(dm, paths["torchscript"])
    x = np.random.default_rng(1).random((21, 64, 256)).astype(np.float32)
    det.predict_proba(x)
    assert det.device_name == "cpu"
    dm.device, dm.mode = torch.device("cuda:0"), "gpu"   # probe finished afterwards
    det.predict_proba(x)
    assert det.device_name == "gpu" and dm.failures == []


@needs_torch
def test_lazy_gpu_creation_failure_reports_once_and_stays_on_cpu(tmp_path):
    path = _save_rule(tmp_path / "rule.pt")
    dm = _StubDeviceManager("cpu")
    det = det_nn.NNDetector(dm, path)
    dm.device, dm.mode = _FakeCuda(), "gpu"              # .to(<not a device>) fails while creating the copy
    p = det.predict_proba(np.zeros((21, 64, 256), np.float32))
    assert p.shape == (1, 3, 64, 256) and len(dm.failures) == 1 and det.device_name == "cpu"
    dm.mode = "gpu"
    det.predict_proba(np.zeros((21, 64, 256), np.float32))
    assert len(dm.failures) == 1                          # no second attempt after a failure


@needs_torch
def test_range_image_params_checked_against_meta(tmp_path):
    meta = {"crop": CropConfig().to_dict(), "range_image_builder": ni.RANGE_IMAGE_BUILDER_ID,
            "range_image_params": {"az_step_deg": 0.1, "binned_res_deg": [0.09, 0.08], "az_limit_deg": None}}
    path = _save_rule(tmp_path / "rule.pt", meta)
    det = det_nn.NNDetector(_StubDeviceManager("cpu"), path,
                            range_image_params={"az_step_deg": 0.1, "binned_res_deg": (0.09, 0.08), "az_limit_deg": None})
    det.infer([hesai_ri()])
    assert det.input_warnings == []                       # organised 0.1 deg image: as trained
    H, W = 60, 900
    binned = make_ri(H, W, np.linspace(12.0, -13.0, H), np.linspace(60.0, -60.0, W), organized=False)
    det.infer([binned])
    assert any("binned image column pitch" in w for w in det.input_warnings)
    assert any("binned image row pitch" in w for w in det.input_warnings)
    n = len(det.input_warnings)
    det.infer([binned])
    assert len(det.input_warnings) == n                   # checked once per image kind
    det2 = det_nn.NNDetector(_StubDeviceManager("cpu"), path,
                             range_image_params={"az_step_deg": 0.2, "binned_res_deg": (0.09, 0.08), "az_limit_deg": 40.0})
    det2.infer([hesai_ri()])
    w = " ".join(det2.input_warnings)
    assert "az_step_deg" in w and "az_limit_deg" not in w     # trained on full images: any sector that holds the crop is fine
    meta3 = dict(meta, range_image_params={"az_step_deg": 0.1, "binned_res_deg": [0.09, 0.08], "az_limit_deg": 30.0})
    path3 = _save_rule(tmp_path / "rule3.pt", meta3)
    det3 = det_nn.NNDetector(_StubDeviceManager("cpu"), path3,
                             range_image_params={"az_step_deg": 0.1, "binned_res_deg": (0.09, 0.08), "az_limit_deg": 40.0})
    det3.infer([hesai_ri()])
    assert any("az_limit_deg" in w for w in det3.input_warnings)   # trained on a sector: a different one is reported


@needs_torch
def test_infer_memoises_frames_of_a_sliding_window(tmp_path, monkeypatch):
    path = _save_rule(tmp_path / "rule.pt")
    det = det_nn.NNDetector(_StubDeviceManager("cpu"), path)
    calls = []
    real = ni.frame_channels
    monkeypatch.setattr(ni, "frame_channels", lambda *a, **k: calls.append(1) or real(*a, **k))
    frames = []
    for i in range(5):
        ri, *_ = _scene()
        put(ri, 40, 600, 50.0 + i)
        frames.append(ri)
    det.infer(frames[0:3])
    assert len(calls) == 3
    c1 = det.infer(frames[1:4])
    assert len(calls) == 4                                # only the new frame is computed
    det2 = det_nn.NNDetector(_StubDeviceManager("cpu"), path)
    for ri in frames[:4]:
        c2 = det2.step(ri)
    key = lambda cs: [(c.cls, c.n_support, round(c.distance_m, 4)) for c in cs]
    assert key(c1) == key(c2) and np.array_equal(det.last_probs, det2.last_probs)   # same as step()
