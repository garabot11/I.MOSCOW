"""Binned range image (unorganised clouds): the CPU and GPU implementations are identical to the reference.

``_binned_reference`` is the v1 implementation (code 57c8cdb) copied verbatim; every array of the new
image must be bit-identical to it (``np.array_equal`` plus the sign of every zero).

With ``METRO_VERIFY_BAG=<rosbag2 dir>`` (and optionally ``METRO_VERIFY_STRIDE``, ``METRO_VERIFY_MAX``) the
last test also compares every frame of a real recording (run inside the image, the dataset mounted).
"""
import os

import numpy as np
import pytest

from metro_obstacle_detector import range_image as RI
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.frame import FrameBuilder
from metro_obstacle_detector.range_image import RangeImage, _empty, build_range_image
from metro_obstacle_detector.sensor_profile import axis_rotation

from synthetic import make_pointcloud2, tunnel_cloud

try:
    import torch
    HAVE_CUDA = bool(torch.cuda.is_available())
except Exception:  # torch missing
    torch = None
    HAVE_CUDA = False


def _binned_reference(frame, R, res, az_limit):
    """v1 range_image._binned (57c8cdb), verbatim."""
    p = frame.xyz.astype(np.float64)
    r = np.sqrt((p ** 2).sum(axis=1))
    ok = np.isfinite(r) & (r > 0.5)
    idx = np.flatnonzero(ok)
    dlh = p[idx] @ R.T
    rr = r[idx]
    az = np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0]))
    el = np.degrees(np.arcsin(np.clip(dlh[:, 2] / rr, -1, 1)))
    if az_limit is not None:
        m = np.abs(az) <= az_limit
        idx, dlh, rr, az, el = idx[m], dlh[m], rr[m], az[m], el[m]
    if len(idx) == 0:
        rng, inten, xyz, slots = _empty(1, 1)
        return RangeImage(rng, inten, xyz, xyz.copy(), rng > 0, np.zeros(1, np.float32), np.zeros(1, np.float32),
                          np.full(1, -1), slots, frame.profile.sensor, False)
    az_step, el_step = res
    az_hi, el_hi = float(az.max()), float(el.max())
    W = int((az_hi - az.min()) / az_step) + 1
    H = int((el_hi - el.min()) / el_step) + 1
    ci = ((az_hi - az) / az_step).astype(np.int64)
    ri = ((el_hi - el) / el_step).astype(np.int64)
    flat = ri * W + ci
    rng, inten, xyz, slots = _empty(H, W)
    slot = frame.slot[idx] if frame.slot is not None else idx.astype(np.int64)
    pts = frame.xyz[idx]
    it = frame.intensity[idx] if frame.intensity is not None else np.zeros(len(idx), np.float32)
    order = np.lexsort((rr, flat))          # one sort by (cell, range): group starts = nearest, group ends = farthest
    f = flat[order]
    change = f[1:] != f[:-1]
    first = np.concatenate([[True], change])
    last = np.concatenate([change, [True]])
    for layer, s in ((0, order[first]), (1, order[last])):
        rows, cs = ri[s], ci[s]
        rng[layer, rows, cs] = rr[s]
        xyz[layer, rows, cs] = pts[s]
        inten[layer, rows, cs] = it[s]
        slots[layer, rows, cs] = slot[s]
    dlh_img = xyz @ R.T.astype(np.float32)
    return RangeImage(rng=rng, intensity=inten, xyz=xyz, dlh=dlh_img, valid=rng > 0,
                      el_deg=(el_hi - el_step * np.arange(H)).astype(np.float32),
                      az_deg=(az_hi - az_step * np.arange(W)).astype(np.float32), ring_of_row=np.full(H, -1, np.int64),
                      slot_of_pixel=slots, sensor=frame.profile.sensor, organized=False)


FIELDS = ("rng", "intensity", "xyz", "dlh", "valid", "el_deg", "az_deg", "ring_of_row", "slot_of_pixel")


def assert_same_image(a, b, where=""):
    assert a.organized == b.organized and a.sensor == b.sensor, where
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert x.shape == y.shape and x.dtype == y.dtype, (where, f, x.shape, y.shape, x.dtype, y.dtype)
        assert np.array_equal(x, y), (where, f)
        if x.dtype.kind == "f":
            assert np.array_equal(np.signbit(x), np.signbit(y)), (where, f, "sign of zero")


def unorganised_frame(seed=0, n_dup=2000, n_junk=500, intensity=True, axes=("-y", "+z")):
    """A tunnel cloud without organisation, with exact duplicates (ties), ranges quantised to 4 mm,
    returns behind the sensor, NaN/zero/near points - everything the binning has to order or drop."""
    rng = np.random.default_rng(seed)
    xyz = tunnel_cloud(length=150.0, rng=np.random.default_rng(seed + 1), zero_fraction=0.0)
    xyz = xyz[(xyz != 0).any(axis=1)].astype(np.float64)
    # quantise the range to 4 mm like the real sensor (many equal ranges in a cell)
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / r * (np.round(r / 0.004) * 0.004)
    dup = xyz[rng.integers(0, len(xyz), n_dup)]
    back = xyz[rng.integers(0, len(xyz), n_junk)] * np.array([1.0, -1.0, 1.0])
    junk = np.zeros((n_junk, 3))
    junk[: n_junk // 3] = np.nan
    third = n_junk // 3
    junk[third: 2 * third] = rng.normal(0, 0.1, (third, 3))      # r < 0.5
    xyz = np.vstack([xyz, dup, back, junk]).astype(np.float32)
    xyz = xyz[rng.permutation(len(xyz))]
    cfg = DetectorConfig(forward_axis=axes[0], up_axis=axes[1])
    if axes != ("-y", "+z"):
        R0 = DetectorConfig().axis_matrix()
        R1 = cfg.axis_matrix()
        xyz = (xyz.astype(np.float64) @ R0.T @ R1).astype(np.float32)    # same scene in the other convention
    msg = make_pointcloud2(xyz, drop=("ring",) if intensity else ("ring", "intensity"))
    fr = FrameBuilder(cfg).build(msg)
    assert not fr.profile.organized
    return fr


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("az_limit", [25.0, None, 60.0])
def test_cpu_binning_is_identical_to_the_reference(seed, az_limit):
    fr = unorganised_frame(seed)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    for res in ((0.09, 0.08), (0.2, 0.2)):
        ref = _binned_reference(fr, R, res, az_limit)
        new = RI._binned(fr, R, res, az_limit)
        assert_same_image(ref, new, (seed, az_limit, res))


def test_cpu_binning_without_intensity_and_other_axes():
    fr = unorganised_frame(3, intensity=False)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), RI._binned(fr, R, (0.09, 0.08), 25.0))
    fr = unorganised_frame(4, axes=("+x", "+z"))
    R = axis_rotation("+x", "+z")
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), RI._binned(fr, R, (0.09, 0.08), 25.0))


def test_cpu_binning_with_a_general_rotation():
    """A pitched sensor (Falcon auto-detected axes): R is no signed permutation, dlh is the BLAS product."""
    fr = unorganised_frame(6)
    pitch = np.radians(2.0)
    tilt = np.array([[np.cos(pitch), 0.0, -np.sin(pitch)], [0.0, 1.0, 0.0], [np.sin(pitch), 0.0, np.cos(pitch)]])
    R = tilt @ axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    assert RI._rotation_parts(R)[0] is None
    for az_limit in (25.0, None):
        assert_same_image(_binned_reference(fr, R, (0.09, 0.08), az_limit), RI._binned(fr, R, (0.09, 0.08), az_limit), az_limit)


def test_cpu_binning_of_an_empty_sector():
    fr = unorganised_frame(5)
    fr.xyz[:] = np.array([0.0, 10.0, 0.0], np.float32)          # everything behind the sensor
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), RI._binned(fr, R, (0.09, 0.08), 25.0))


def test_dlh_image_matches_the_blas_product_bit_for_bit():
    rng = np.random.default_rng(7)
    xyz = rng.normal(0, 30, (2, 50, 70, 3)).astype(np.float32)
    xyz[rng.random((2, 50, 70)) < 0.4] = 0.0
    xyz[0, 0, 0] = (-0.0, 0.0, -0.0)
    for axes in (("-y", "+z"), ("+x", "+z"), ("+z", "+x"), ("-x", "-z")):
        R = axis_rotation(*axes)
        ref = xyz @ R.T.astype(np.float32)
        out = RI._dlh_image(xyz, R)
        assert np.array_equal(ref, out) and np.array_equal(np.signbit(ref), np.signbit(out)), axes


@pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_binning_is_identical_to_the_reference(seed):
    dev = torch.device("cuda:0")
    fr = unorganised_frame(seed)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    before = dict(RI.BIN_STATS)
    for az_limit in (25.0, None):
        ref = _binned_reference(fr, R, (0.09, 0.08), az_limit)
        new = RI._binned(fr, R, (0.09, 0.08), az_limit, device=dev)
        assert_same_image(ref, new, (seed, az_limit))
    assert RI.BIN_STATS["gpu"] + RI.BIN_STATS["gpu_guard"] - before["gpu"] - before["gpu_guard"] == 2
    assert RI.BIN_STATS["gpu_error"] == before["gpu_error"]


@pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")
def test_gpu_guard_falls_back_to_the_cpu(monkeypatch):
    dev = torch.device("cuda:0")
    fr = unorganised_frame(1)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    monkeypatch.setattr(RI, "_GUARD_Q", 0.6)                 # every point is "near an edge"
    before = RI.BIN_STATS["gpu_guard"]
    new = RI._binned(fr, R, (0.09, 0.08), 25.0, device=dev)
    assert RI.BIN_STATS["gpu_guard"] == before + 1
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), new)


def test_device_error_falls_back_to_the_cpu_and_reports(monkeypatch):
    fr = unorganised_frame(2)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)

    def boom(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(RI, "_plan_torch", boom)
    seen = []
    new = RI._binned(fr, R, (0.09, 0.08), 25.0, device="cuda:0", on_device_error=seen.append)
    assert len(seen) == 1 and "illegal memory access" in str(seen[0])
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), new)


def test_build_range_image_passes_the_device_only_to_binned_clouds():
    fr = unorganised_frame(0)
    a = build_range_image(fr, az_limit_deg=25.0)
    R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
    assert_same_image(_binned_reference(fr, R, (0.09, 0.08), 25.0), a)


@pytest.mark.skipif(not os.environ.get("METRO_VERIFY_BAG"), reason="set METRO_VERIFY_BAG=<rosbag2 dir> to run")
def test_real_recording_every_frame_identical():
    """Every frame of a real recording: CPU and (if available) GPU image == reference."""
    from metro_obstacle_detector.offline import iter_pointclouds

    bag = os.environ["METRO_VERIFY_BAG"]
    stride = int(os.environ.get("METRO_VERIFY_STRIDE", "1"))
    max_n = int(os.environ.get("METRO_VERIFY_MAX", "0"))
    dev = torch.device("cuda:0") if HAVE_CUDA else None
    fb = FrameBuilder(DetectorConfig())
    n = n_binned = 0
    before = dict(RI.BIN_STATS)
    for i, _name, msg, _t in iter_pointclouds(bag):
        if i % stride:
            continue
        fr = fb.build(msg)
        n += 1
        if fr.profile.organized:
            continue
        n_binned += 1
        R = axis_rotation(fr.profile.forward_axis, fr.profile.up_axis)
        ref = _binned_reference(fr, R, (0.09, 0.08), 25.0)
        assert_same_image(ref, RI._binned(fr, R, (0.09, 0.08), 25.0), ("cpu", i))
        if dev is not None:
            assert_same_image(ref, RI._binned(fr, R, (0.09, 0.08), 25.0, device=dev), ("gpu", i))
        if max_n and n >= max_n:
            break
    stats = {k: RI.BIN_STATS[k] - before[k] for k in before}
    print("VERIFY %s frames=%d binned=%d stats=%s" % (bag, n, n_binned, stats))
    assert n_binned > 0
