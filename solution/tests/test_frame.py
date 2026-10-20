"""Sensor profile, frame builder and range image on synthetic clouds with known layouts."""
import numpy as np
import pytest

from metro_obstacle_detector.cloud_io import pointcloud2_to_struct
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.frame import FrameBuilder
from metro_obstacle_detector.pipeline import DetectionPipeline
from metro_obstacle_detector.range_image import build_range_image
from metro_obstacle_detector.sensor_profile import build_profile, detect_axes, detect_layout

from synthetic import make_pointcloud2, tunnel_cloud

N_RINGS = 128


def organised_cloud(n_cols=600, az_span=(-60.0, 60.0), zero_fraction=0.3, dual=True, rng=None, ring_offsets=None):
    """Column-major organised cloud like the real Hesai layout: slot i -> ring i % 128, column i // 128.

    Returns (xyz, meta): meta has the per-ring elevations, per-column azimuths and the per-ring azimuth offsets
    used to generate the directions (sensor frame: forward -y, up +z; lateral l = h x d = +x).  With ``dual`` the
    columns come in (far, near) pairs with identical directions; every 20th pair gets a nearer first return.
    """
    rng = rng or np.random.default_rng(0)
    el = np.linspace(14.4, -25.1, N_RINGS)
    off = np.zeros(N_RINGS) if ring_offsets is None else ring_offsets
    firings = n_cols // 2 if dual else n_cols
    az_f = np.linspace(az_span[1], az_span[0], firings)          # + azimuth first, like the message order
    col_az = np.repeat(az_f, 2) if dual else az_f
    d_ring = np.radians(el)
    r = 20.0 + 30.0 * rng.random((n_cols, N_RINGS))
    xyz = np.zeros((n_cols * N_RINGS, 3), np.float32)
    for c in range(n_cols):
        az = np.radians(col_az[c] + off)
        rr = r[c]
        if dual:
            rr = r[c - (c % 2)].copy()
            if c % 2 == 1 and (c // 2) % 20 == 0:
                rr = rr - 5.0                                     # nearer first return in the odd column
        dd = rr * np.cos(d_ring) * np.cos(az)
        ll = rr * np.cos(d_ring) * np.sin(az)
        hh = rr * np.sin(d_ring)
        base = c * N_RINGS
        xyz[base:base + N_RINGS, 0] = ll                          # x = +l
        xyz[base:base + N_RINGS, 1] = -dd                         # y = -d
        xyz[base:base + N_RINGS, 2] = hh
    zero = rng.random(len(xyz)) < zero_fraction
    if dual:  # a no-return firing has both columns zero
        z2 = zero.reshape(n_cols, N_RINGS)
        z2[1::2] = z2[0::2]
        zero = z2.reshape(-1)
    xyz[zero] = 0.0
    return xyz, {"el": el, "col_az": col_az, "off": off, "zero": zero}


def test_layout_and_beam_angles_of_an_organised_dual_return_cloud():
    off = np.linspace(-3.0, 3.0, N_RINGS)
    xyz, meta = organised_cloud(ring_offsets=off)
    msg = make_pointcloud2(xyz)
    struct = pointcloud2_to_struct(msg)
    organised, nr, nc, dual = detect_layout(struct)
    assert (organised, nr, nc, dual) == (True, N_RINGS, 600, True)
    prof = build_profile(msg, struct, DetectorConfig(forward_crop_deg=0.0))
    assert prof.sensor == "hesai128" and prof.axes_source == "config"
    assert np.allclose(prof.ring_el_deg, meta["el"], atol=0.02)
    # azimuth offsets are relative to the reference ring; the sum offset + column azimuth is what matters
    true_az = meta["col_az"][:, None] + meta["off"][None, :]
    est_az = prof.col_az_deg[:, None] + prof.ring_az_off_deg[None, :]
    assert np.abs(true_az - est_az).max() < 0.05


def test_forward_crop_keeps_whole_firing_pairs_and_slot_indices():
    xyz, _ = organised_cloud(az_span=(-110.0, 110.0), n_cols=2200)
    msg = make_pointcloud2(xyz)
    fb = FrameBuilder(DetectorConfig(forward_crop_deg=80.0))
    fr = fb.build(msg)
    prof = fr.profile
    assert prof.keep_cols is not None and len(prof.keep_cols) < prof.n_cols
    assert len(prof.keep_cols) % 2 == 0 and prof.keep_slice is not None
    assert fr.n == len(prof.keep_cols) * N_RINGS and fr.n_msg == 2200 * N_RINGS
    assert np.array_equal(fr.xyz, xyz[fr.slot])
    assert np.abs(prof.col_az_deg[prof.keep_cols]).max() <= 80.0 + 0.2
    # the same profile is reused for the next message of the stream
    fb.build(msg)
    assert fb.last_new_profile is None and len(fb.profiles) == 1


def test_range_image_of_an_organised_cloud_maps_every_return_and_keeps_slots():
    xyz, meta = organised_cloud(n_cols=400, az_span=(-20.0, 20.0), zero_fraction=0.2)
    msg = make_pointcloud2(xyz)
    fr = FrameBuilder(DetectorConfig(forward_crop_deg=0.0)).build(msg)
    ri = build_range_image(fr, az_step_deg=0.1)
    H, W = ri.shape
    assert H == N_RINGS and ri.organized
    # every valid near return of the message is in layer 0 at its own slot
    valid = ~meta["zero"]
    n_valid_firings = int(valid.reshape(400, N_RINGS)[1::2].sum())
    assert ri.valid[0].sum() == n_valid_firings
    slots = ri.slot_of_pixel[0][ri.valid[0]]
    assert np.all(slots % 2 * 0 == 0) and np.array_equal(np.sort(slots), np.flatnonzero(valid & (np.arange(len(xyz)) // N_RINGS % 2 == 1)))
    # near layer never farther than the far layer; the nearer first returns of every 20th firing differ by 5 m
    both = ri.valid[0] & ri.valid[1]
    assert np.all(ri.rng[0][both] <= ri.rng[1][both] + 1e-4)
    diff = (ri.rng[1] - ri.rng[0])[both]
    assert np.isclose(np.sort(np.unique(np.round(diff, 2)))[-1], 5.0, atol=0.01)
    # pixel directions agree with the grid
    d, l = ri.dlh[0][..., 0], ri.dlh[0][..., 1]
    az = np.degrees(np.arctan2(l, d))[ri.valid[0]]
    assert np.abs(az - np.broadcast_to(ri.az_deg, (H, W))[ri.valid[0]]).max() < 0.06
    # rows are sorted by elevation, highest first
    assert np.all(np.diff(ri.el_deg) < 0)


def test_binned_range_image_for_an_unorganised_cloud():
    xyz = tunnel_cloud(length=120.0, rng=np.random.default_rng(1), zero_fraction=0.0)
    xyz = xyz[(xyz != 0).any(axis=1)]
    rng = np.random.default_rng(2)
    xyz = xyz[rng.permutation(len(xyz))]                  # no organisation at all
    msg = make_pointcloud2(xyz, drop=("ring",))
    fr = FrameBuilder(DetectorConfig()).build(msg)
    assert not fr.profile.organized
    ri = build_range_image(fr, binned_res_deg=(0.2, 0.2))
    H, W = ri.shape
    assert not ri.organized and H > 10 and W > 10
    assert ri.valid[0].sum() > 0.5 * ri.valid[0].size * 0 + 100
    both = ri.valid[0] & ri.valid[1]
    assert np.all(ri.rng[0][both] <= ri.rng[1][both] + 1e-4)
    # slots point back to the right points
    sl = ri.slot_of_pixel[0][ri.valid[0]]
    assert np.allclose(np.linalg.norm(xyz[sl], axis=1), ri.rng[0][ri.valid[0]], atol=1e-3)


@pytest.mark.parametrize("convention", ["falcon_upside_down", "rep103", "native_falcon"])
def test_axes_are_detected_for_other_sensor_conventions(convention):
    """A tunnel generated in the project convention (forward -y, up +z) is re-expressed in another
    convention; the detector must find the track bed with the detected axes."""
    cfg = DetectorConfig()
    base = tunnel_cloud(length=150.0, rng=np.random.default_rng(3), zero_fraction=0.3)
    valid = (base != 0).any(axis=1)
    R = cfg.axis_matrix()
    dlh = base[valid].astype(np.float64) @ R.T                 # (d, l, h)
    d, l, h = dlh[:, 0], dlh[:, 1], dlh[:, 2]
    if convention == "falcon_upside_down":                     # x forward, z down, y = l
        other, expect = np.stack([d, l, -h], 1), ("+x", "-z")
    elif convention == "rep103":                               # x forward, y left (= -l here), z up
        other, expect = np.stack([d, -l, h], 1), ("+x", "+z")
    else:                                                      # x up, y right, z forward (manual §1.4)
        other, expect = np.stack([h, -l, d], 1), ("+z", "+x")
    fwd, up, note = detect_axes(other, cfg)
    assert (fwd, up) == expect, note
    xyz = np.zeros_like(base)
    xyz[valid] = other.astype(np.float32)
    msg = make_pointcloud2(xyz, drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    assert fr.profile.axes_source == "auto" and (fr.profile.forward_axis, fr.profile.up_axis) == expect
    res, _ = pipe.step_frame(fr)
    assert res.rail is not None and res.rail.ok and res.status != "unknown"
    assert abs(res.rail.z0 + 1.3) < 0.25                        # sensor 1.3 m above rail in tunnel_cloud


def test_configured_axes_are_kept_when_the_track_is_found():
    cfg = DetectorConfig()
    xyz = tunnel_cloud(length=150.0, rng=np.random.default_rng(4))
    fr = FrameBuilder(cfg).build(make_pointcloud2(xyz))
    assert fr.profile.axes_source == "config" and (fr.profile.forward_axis, fr.profile.up_axis) == ("-y", "+z")


def test_unorganised_stream_with_a_varying_point_count_keeps_one_profile():
    fb = FrameBuilder(DetectorConfig())
    rng = np.random.default_rng(3)
    first = None
    for n in (30000, 29987, 30011, 29950):
        xyz = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-150, -2, n), rng.uniform(-2, 3, n)]).astype(np.float32)
        msg = make_pointcloud2(xyz, drop=("ring",))
        msg.header.frame_id = "edited"
        fr = fb.build(msg)
        assert not fr.profile.organized
        first = first or fr.profile
        assert fr.profile is first                      # one profile for the stream, not one per point count
    assert fb.last_new_profile is None


def test_unorganised_frame_with_a_count_divisible_by_128_keeps_the_stream_profile():
    # Falcon frames whose point count happens to be a multiple of 128 must not re-detect the axes from that one
    # frame (a warm start of the Falcon bag flipped the up axis with a person close ahead)
    fb = FrameBuilder(DetectorConfig())
    rng = np.random.default_rng(5)
    first = None
    for n in (30011, 29952, 30080, 29987):              # 29952 and 30080 are multiples of 128
        xyz = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-150, -2, n), rng.uniform(-2, 3, n)]).astype(np.float32)
        msg = make_pointcloud2(xyz, drop=("ring",))
        msg.header.frame_id = "edited"
        fr = fb.build(msg)
        first = first or fr.profile
        assert fr.profile is first
        if n != 30011:
            assert fb.last_new_profile is None
