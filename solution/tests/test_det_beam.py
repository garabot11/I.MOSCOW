"""Beam detector D2 on synthetic tunnels (binned range image path)."""
import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.det_beam import BeamDetector
from metro_obstacle_detector.pipeline import DetectionPipeline
from metro_obstacle_detector.range_image import build_range_image

from synthetic import make_pointcloud2, tunnel_cloud


def _d2_on(obstacle, seed=5, length=200.0):
    cfg = DetectorConfig()
    xyz = tunnel_cloud(length=length, obstacle=obstacle, rng=np.random.default_rng(seed), zero_fraction=0.3)
    msg = make_pointcloud2(xyz, drop=("ring",))        # unorganised -> angle-binned range image
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    assert res.rail is not None and res.rail.ok and res.curve is not None
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    return BeamDetector(cfg).process(ri, res.rail, res.curve), res


def test_d2_finds_a_box_on_the_track_and_grounds_it():
    cands, res = _d2_on((60.0, 0.0, 0.8, 1.8))
    near = [c for c in cands if abs(c.d - 60.0) < 2.5 and c.cls == "obstacle"]
    assert near, [(round(c.d, 1), c.n_support, c.cls) for c in cands]
    best = max(near, key=lambda c: c.n_support)
    assert best.n_support >= 6 and best.score >= 0.3
    assert abs(best.lateral_offset) < 0.5 and 0.2 < best.h_above_rail < 1.3
    assert best.info["attached_frac"] <= 0.35
    assert best.slots is not None and len(best.slots) == best.n_support


def test_d2_sees_a_small_far_object_the_cluster_detector_rejects():
    cands, res = _d2_on((150.0, 0.2, 0.5, 1.7), length=260.0)
    near = [c for c in cands if abs(c.d - 150.0) < 4.0 and c.cls == "obstacle"]
    assert near, [(round(c.d, 1), c.n_support) for c in cands]
    assert max(c.n_support for c in near) >= 2


def test_d2_reports_nothing_in_an_empty_tunnel_beyond_the_near_range():
    cands, res = _d2_on(None)
    far = [c for c in cands if c.d > 30.0 and c.cls == "obstacle" and c.score > 0.3]
    assert not far, [(round(c.d, 1), c.n_support, round(c.score, 2), c.info) for c in far]


def test_d2_runs_in_a_few_milliseconds():
    import time
    cfg = DetectorConfig()
    xyz = tunnel_cloud(length=200.0, obstacle=(80.0, 0.0, 0.6, 1.5), rng=np.random.default_rng(1), zero_fraction=0.3)
    msg = make_pointcloud2(xyz, drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    det = BeamDetector(cfg)
    det.process(ri, res.rail, res.curve)
    t = time.perf_counter()
    for _ in range(5):
        det.process(ri, res.rail, res.curve)
    assert (time.perf_counter() - t) / 5 < 0.1


def test_two_objects_a_few_metres_apart_stay_two_groups():
    # a person at 34 m and a box at 38 m are neighbours in the range image: bridging pixel gaps must not merge them
    cfg = DetectorConfig()
    xyz = tunnel_cloud(obstacle=(34.0, 0.5, 0.5, 1.7), rng=np.random.default_rng(5), zero_fraction=0.3)
    rng = np.random.default_rng(7)
    n = 400
    box = np.column_stack([-0.1 + rng.uniform(-0.3, 0.3, n), -37.9 - rng.uniform(0, 0.05, n), -1.3 + rng.uniform(0.05, 1.2, n)]).astype(np.float32)
    msg = make_pointcloud2(np.vstack([xyz, box]), drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    out = BeamDetector(cfg).process(ri, res.rail, res.curve)
    near34 = [c for c in out if abs(c.d - 34.2) < 1.5 and c.n_support >= 5]
    near38 = [c for c in out if abs(c.d - 37.9) < 1.5 and c.n_support >= 5]
    assert near34 and near38, [(c.d, c.n_support, c.cls) for c in out]


def test_a_small_far_fragment_beside_a_near_object_does_not_swallow_it():
    # a person at 34 m with a 7-pixel wall fragment 16 m behind it, adjacent in the image: bridging must not
    # stretch the group along the track until max_extent_d drops it (the reviewer's reproduction)
    cfg = DetectorConfig()
    xyz = tunnel_cloud(obstacle=(34.0, 0.5, 0.5, 1.7), rng=np.random.default_rng(5), zero_fraction=0.3)
    rng = np.random.default_rng(11)
    frag = np.column_stack([0.95 + rng.uniform(-0.05, 0.05, 9), -50.0 - rng.uniform(0, 0.1, 9), -1.3 + 1.0 + rng.uniform(0, 0.4, 9)]).astype(np.float32)
    msg = make_pointcloud2(np.vstack([xyz, frag]), drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    out = BeamDetector(cfg).process(ri, res.rail, res.curve)
    near = [c for c in out if abs(c.d - 34.2) < 1.5 and c.n_support >= 4 and c.cls == "obstacle"]
    assert near, [(c.d, c.n_support, c.cls, c.extent_dlh) for c in out]
    assert all(c.extent_dlh[0] < 3.0 for c in near)                     # nothing stretched towards the fragment
    assert not any(abs(c.d - 50.0) < 2.0 and c.n_support >= 20 for c in out)


def _sparse_insert(xyz, d, width, height, lateral=0.0, spacing=0.24, bottom=0.3):
    """A planar surface facing the sensor at forward distance d, sampled every ``spacing`` metres (one return
    per range-image pixel with a one-pixel gap between them at 75 m, like the organisers' inserted wall)."""
    ls = np.arange(-width / 2, width / 2 + 1e-6, spacing) + lateral
    hs = np.arange(bottom, bottom + height + 1e-6, spacing)
    L, H = np.meshgrid(ls, hs)
    pts = np.column_stack([L.ravel(), np.full(L.size, -d), -1.3 + H.ravel()]).astype(np.float32)
    return np.vstack([xyz, pts])


def _d2_groups(cloud):
    cfg = DetectorConfig()
    msg = make_pointcloud2(cloud, drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    return BeamDetector(cfg).process(ri, res.rail, res.curve)


def test_a_sparsely_sampled_face_across_the_corridor_is_one_d2_group():
    base = tunnel_cloud(rng=np.random.default_rng(5), zero_fraction=0.3)
    out = _d2_groups(_sparse_insert(base, 75.0, width=2.8, height=3.5))
    face = [c for c in out if abs(c.d - 75.0) < 2.0 and c.n_support >= 12 and c.cls == "obstacle"]
    assert face, [(c.d, c.n_support, c.cls) for c in out if abs(c.d - 75.0) < 2.0]
    assert face[0].extent_dlh[1] >= 1.5 and face[0].extent_dlh[2] >= 1.5


def test_a_sparsely_sampled_narrow_column_stays_fragments():
    base = tunnel_cloud(rng=np.random.default_rng(5), zero_fraction=0.3)
    out = _d2_groups(_sparse_insert(base, 75.0, width=0.3, height=3.0, lateral=1.2))
    assert not any(abs(c.d - 75.0) < 2.0 and c.n_support >= 12 for c in out), [(c.d, c.n_support, c.cls) for c in out if abs(c.d - 75.0) < 2.0]
