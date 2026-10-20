"""Version-2 obstacle rules (WP-R): every rule is behind a parameter; ``V1`` below is the v1 setting of all of them.

* ``d2_bridge_empty_bins`` (default on; binned images of the Hesai class only, ``d2_bridge_min_vspan_deg``): D2 joins
  small in-box fragments separated by empty bins and looks through empty bins for their neighbours (the organisers'
  inserted plate fell apart into 1-2 px fragments);
* ``d1_plate_exception`` (on): a plate across the corridor is not a "flat patch at range";
* ``far_weak_d1_vote`` (on): train standing - a far D1 cluster rejected only for its support votes when D2 sees an
  obstacle at the same place (the dark person at 150 m on the Falcon bag);
* ``far_vote_d1_min_d`` (140 m): train moving - a far D1 cluster with only a few voxels adds evidence but does not
  vote (sparse wall returns at 156-165 m on the 20-minute ride);
* ``compact_pair_vote`` (on): a small compact D1 cluster (also one rejected only for its support or as a short edge
  sliver) and an isolated D2 obstacle group at the same place both vote (the organisers' 0.3 m cubes).
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from metro_obstacle_detector.candidates import DetCandidate
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.det_beam import BeamDetector, BeamDetectorParams, _bridge_empty_bins
from metro_obstacle_detector.detector import Candidate, FrameResult, ObstacleDetector
from metro_obstacle_detector.fusion import FusionFilter
from metro_obstacle_detector.pipeline import DetectionPipeline
from metro_obstacle_detector.range_image import build_range_image
from metro_obstacle_detector.track_model import RailLevel

from synthetic import make_pointcloud2, tunnel_cloud

R = DetectorConfig().axis_matrix()
RAIL = RailLevel(-1.3, 0.0, 0.0, 10, True, 0.02, 60.0)
# the v1 setting of every WP-R rule (the helpers below start from it; a test switches on what it checks)
V1 = dict(d2_bridge_empty_bins=False, d1_plate_exception=False, far_weak_d1_vote=False, far_vote_d1_min_d=0.0,
          compact_pair_vote=False)
# the synthetic test tunnel spans 30.6 deg vertically (the Hesai class 39.5, the Falcon Prime 26.2): bridging tests lower the gate
BRIDGE = dict(d2_bridge_empty_bins=True, d2_bridge_min_vspan_deg=28.0)


def test_v2_rule_defaults():
    cfg = DetectorConfig()
    assert cfg.d2_bridge_empty_bins and cfg.d1_plate_exception and cfg.far_weak_d1_vote
    assert cfg.far_vote_d1_min_d == 140.0 and cfg.d2_bridge_min_vspan_deg == 33.0
    assert cfg.compact_pair_vote


# ---------------------------------------------------------------- the plate (D2 bridging, D1 exception)
def _plate(d, width=1.3, depth=0.3, above_rail=0.75, step_l=0.22, step_d=0.15, layers=(0.0, 0.15)):
    """A thin horizontal sheet across the track at forward distance d, sampled as a coarse grid of points like
    the organisers' inserted plate: every other bin of the angle-binned image, two point layers 0.15 m apart
    (sensor 1.3 m above rail in ``tunnel_cloud``; the box bottom at 66 m lies ~0.6 m above the synthetic rail)."""
    ls = np.arange(-width / 2, width / 2 + 1e-6, step_l)
    ds = np.arange(d, d + depth + 1e-6, step_d)
    L, D, Z = np.meshgrid(ls, ds, np.asarray(layers))
    return np.column_stack([L.ravel(), -D.ravel(), -1.3 + above_rail + Z.ravel()]).astype(np.float32)


def _d2(cloud, **cfg_kw):
    cfg = dataclasses.replace(DetectorConfig(), d3_enabled=False, **{**V1, **cfg_kw})
    msg = make_pointcloud2(cloud, drop=("ring",))            # unorganised -> angle-binned range image
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    assert res.rail is not None and res.rail.ok and res.curve is not None
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    return BeamDetector(cfg).process(ri, res.rail, res.curve), ri, res


def _plate_groups(out, d):
    return [c for c in out if abs(c.d - d) < 2.0 and abs(c.lateral_offset) < 0.8]


def test_a_coarsely_sampled_plate_is_one_d2_obstacle_group_with_empty_bin_bridging():
    base = tunnel_cloud(rng=np.random.default_rng(5), zero_fraction=0.3)
    cloud = np.vstack([base, _plate(66.0)])
    off, _, _ = _d2(cloud)
    on, _, _ = _d2(cloud, **BRIDGE)
    best_off = max((c.n_support for c in _plate_groups(off, 66.0)), default=0)
    plate_on = [c for c in _plate_groups(on, 66.0) if c.cls == "obstacle"]
    assert plate_on, [(round(c.d, 1), c.n_support, c.cls) for c in _plate_groups(on, 66.0)]
    best = max(plate_on, key=lambda c: c.n_support)
    assert best.n_support > best_off                                   # the fragments are joined
    assert best.extent_dlh[1] >= 1.0 and best.score >= 0.3
    assert best.info["n_bg"] > 0                                       # free space behind/under it is seen through the empty bins


def _same(a, b):
    key = lambda cs: [(round(c.d, 3), c.n_support, c.cls, round(c.score, 4)) for c in cs]
    return key(a) == key(b)


def test_empty_bin_bridging_never_applies_to_organised_images_or_below_the_stream_gate():
    base = tunnel_cloud(rng=np.random.default_rng(5), zero_fraction=0.3)
    cfg = dataclasses.replace(DetectorConfig(), d3_enabled=False, **{**V1, **BRIDGE})
    msg = make_pointcloud2(np.vstack([base, _plate(66.0)]), drop=("ring",))
    pipe = DetectionPipeline(cfg)
    fr = pipe.build_frame(msg)
    res = pipe.detector.process(fr.xyz)
    ri = build_range_image(fr, az_limit_deg=cfg.d2_az_limit_deg)
    v1 = BeamDetector(dataclasses.replace(cfg, d2_bridge_empty_bins=False)).process(ri, res.rail, res.curve)
    assert not _same(BeamDetector(cfg).process(ri, res.rail, res.curve), v1)          # the rule does change this image
    ri_org = dataclasses.replace(ri, organized=True)                   # pretend: an empty pixel is a fired beam
    v1_org = BeamDetector(dataclasses.replace(cfg, d2_bridge_empty_bins=False)).process(ri_org, res.rail, res.curve)
    assert _same(BeamDetector(cfg).process(ri_org, res.rail, res.curve), v1_org)
    # a stream with the Falcon Prime's vertical span (26 deg) is not bridged even when the flag is on
    gate = dataclasses.replace(cfg, d2_bridge_min_vspan_deg=33.0)
    assert float(ri.el_deg.max() - ri.el_deg.min()) < 33.0
    assert _same(BeamDetector(gate).process(ri, res.rail, res.curve), v1)


def test_empty_bin_bridging_adds_nothing_in_an_empty_tunnel():
    base = tunnel_cloud(rng=np.random.default_rng(5), zero_fraction=0.3)
    on, _, _ = _d2(base, **BRIDGE)
    far = [c for c in on if c.d > 30.0 and c.cls == "obstacle" and c.score > 0.3]
    assert not far, [(round(c.d, 1), c.n_support, round(c.score, 2), c.info) for c in far]


def _d1_near(cloud, d, **cfg_kw):
    res = ObstacleDetector(dataclasses.replace(DetectorConfig(), **{**V1, **cfg_kw})).process(cloud)
    acc = [c for c in res.candidates if abs(c.forward_m - d) < 2.0]
    rej = [c for c in res.rejected if abs(c.forward_m - d) < 2.0]
    return acc, rej


def _dense_plate(d, width=1.3, depth=0.3, above_rail=0.75, n=60, seed=2):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-width / 2, width / 2, n), -d - rng.uniform(0, depth, n),
                            -1.3 + above_rail + rng.uniform(0, 0.01, n)]).astype(np.float32)


def test_d1_plate_exception_accepts_a_plate_seen_edge_on_at_range():
    cloud = np.vstack([tunnel_cloud(), _dense_plate(66.0)])
    acc, rej = _d1_near(cloud, 66.0)
    assert not acc and any("flat" in c.rejected_reason or "grazing" in c.rejected_reason for c in rej)
    acc, rej = _d1_near(cloud, 66.0, d1_plate_exception=True)
    assert acc, [c.rejected_reason for c in rej]
    assert all(c.plate for c in acc)        # accepted only by the exception: the fusion asks D2 for free space behind it
    assert all(c.to_dict().get("plate") for c in acc)


def test_d1_plate_exception_keeps_a_long_floor_strip_rejected():
    # 3 m along the track, 1.2 m wide, 0.75 m above rail at 66 m: a floor patch lifted into the box, not a plate
    cloud = np.vstack([tunnel_cloud(), _dense_plate(66.0, width=1.2, depth=3.0, n=120)])
    acc, rej = _d1_near(cloud, 67.0, d1_plate_exception=True)
    assert not acc, [c.to_dict() for c in acc]


# ---------------------------------------------------------------- far weak D1 votes (fusion)
def _cand(d, l=0.0, n=4, conf=1.0, weak=False, reason=""):
    h = -1.3 + 1.5
    return Candidate(distance_m=float(np.hypot(d, l)), forward_m=d, lateral_m=l, height_m=h, center_sensor=R.T @ np.array([d, l, h]),
                     extent=np.array([0.1, 0.5, 0.13]), n_points=n, lateral_offset_from_track=l, height_above_rail=1.5,
                     confidence=conf, rejected_reason=reason, weak=weak)


def _d2c(d, l=0.0, n=6, score=1.0, cls="obstacle"):
    return DetCandidate("d2", d, l, 1.0, (0.4, 0.7, 0.2), float(np.hypot(d, l)), n, score, cls=cls, lateral_offset=l)


def _run(frames, **cfg_kw):
    """frames: list of (accepted D1, rejected D1, D2, static_frac) -> statuses."""
    f = FusionFilter(dataclasses.replace(DetectorConfig(), **{**V1, **cfg_kw}))
    out = []
    for acc, rej, c2, sf in frames:
        res = FrameResult(status="clear")
        res.candidates, res.rejected, res.static_frac = list(acc), list(rej), sf
        out.append(f.update(res, c2, RAIL, R).status)
    return out


def test_a_weak_far_d1_cluster_votes_with_d2_only_on_a_standing_train_when_enabled():
    seq = [([], [_cand(150.0 - 0.1 * i, weak=True, reason="single-ring return at range (0.13 m tall)")], [_d2c(150.0 - 0.1 * i)], 0.88)
           for i in range(8)]
    assert "detected" not in _run(seq)                                    # v1: D2 alone never confirms
    assert _run(seq, far_weak_d1_vote=True)[-1] == "detected"
    moving = [(a, r, c, 0.5) for a, r, c, _ in seq]
    assert "detected" not in _run(moving, far_weak_d1_vote=True)         # a moving train keeps the v1 rule


def test_a_weak_far_d1_cluster_without_d2_at_the_same_place_does_not_vote():
    seq = [([], [_cand(150.0, weak=True, reason="too few points (3<4)")], [_d2c(150.0, l=2.0)], 0.9)] * 8
    assert "detected" not in _run(seq, far_weak_d1_vote=True)


def test_a_rejected_cluster_that_is_not_weak_never_votes():
    seq = [([], [_cand(150.0, weak=False, reason="floating far object")], [_d2c(150.0)], 0.9)] * 8
    assert "detected" not in _run(seq, far_weak_d1_vote=True)


def test_a_few_far_d1_voxels_do_not_vote_on_a_moving_train():
    seq = [([_cand(165.0 - 1.8 * i, n=4)], [], [_d2c(165.0 - 1.8 * i)], 0.55) for i in range(7)]
    assert _run(seq)[-1] == "detected"                                    # v1 confirms D1 (4 voxels) + D2 at 160 m
    assert "detected" not in _run(seq, far_vote_d1_min_d=140.0)
    seq5 = [([_cand(165.0 - 1.8 * i, n=5)], [], [_d2c(165.0 - 1.8 * i)], 0.55) for i in range(7)]
    assert _run(seq5, far_vote_d1_min_d=140.0)[-1] == "detected"          # five voxels still vote
    standing = [(a, r, c, 0.9) for a, r, c, _ in seq]
    assert _run(standing, far_vote_d1_min_d=140.0)[-1] == "detected"     # the rule is for a moving train only


def test_a_confirmed_weak_track_is_reported_as_a_synthetic_candidate():
    seq = [([], [_cand(150.0, weak=True, reason="single-ring return at range (0.13 m tall)")], [_d2c(150.0)], 0.9)] * 8
    f = FusionFilter(dataclasses.replace(DetectorConfig(), **{**V1, "far_weak_d1_vote": True}))
    for acc, rej, c2, sf in seq:
        res = FrameResult(status="clear")
        res.candidates, res.rejected, res.static_frac = list(acc), list(rej), sf
        dec = f.update(res, c2, RAIL, R)
    assert dec.status == "detected"
    assert len(res.candidates) == 1 and res.candidates[0].confirmed and not res.candidates[0].weak
    assert all(not c.confirmed for c in res.rejected)


def _plate_cand(d, l=0.0, n=8, conf=1.0):
    c = _cand(d, l=l, n=n, conf=conf)
    c.plate = True
    c.extent = np.array([0.4, 1.3, 0.05])
    return c


def test_a_plate_accepted_d1_cluster_votes_only_with_a_free_space_d2_group():
    # the 0.2 m step across the bed at 69 m of the organisers' platform frames: D1 accepts it as a plate, the D2 group
    # there is embedded in the surface behind it (background 5 of 24 ring pixels) -> D1 adds evidence but no vote
    step = [([_plate_cand(69.0 - 0.7 * i)], [], [_iso(69.0 - 0.7 * i, n=9, score=0.38, bg=5, ring=24)], 0.8) for i in range(8)]
    assert "detected" not in _run(step, d1_plate_exception=True)
    # the plate on the bed casts a shadow: background behind its D2 group -> both vote
    plate = [([_plate_cand(69.0 - 0.7 * i)], [], [_iso(69.0 - 0.7 * i, n=9, score=0.38, bg=20, ring=24)], 0.8) for i in range(8)]
    assert _run(plate, d1_plate_exception=True)[-1] == "detected"
    # a cluster accepted on its own merits votes as before, whatever the D2 group's surroundings
    plain = [([_cand(69.0 - 0.7 * i, n=8)], [], [_iso(69.0 - 0.7 * i, n=9, score=0.38, bg=5, ring=24)], 0.8) for i in range(8)]
    assert _run(plain, d1_plate_exception=True)[-1] == "detected"


# ---------------------------------------------------------------- empty-bin bridging helper (labels)
def _hook_image(vspan=39.5, organized=False):
    """A (6, 30) in-box mask with three 1-px fragments one empty bin apart, a 40-px group and a fragment one empty bin
    from it, as a ``RangeImage`` stand-in for the labelling hook (crop window = the whole image)."""
    inbox = np.zeros((8, 32), bool)
    inbox[2, 3] = inbox[2, 5] = inbox[4, 4] = True          # three 1-px fragments one empty bin apart
    inbox[2:6, 11:21] = True                                # a 40-px group
    inbox[2, 22] = True                                     # a fragment one empty bin from the big group
    valid = inbox.copy()                                    # every other pixel is an empty bin
    ri = SimpleNamespace(valid=np.stack([valid, valid]), organized=organized,
                         el_deg=np.linspace(vspan / 2, -vspan / 2, inbox.shape[0]).astype(np.float32))
    return inbox, ri


def test_bridge_empty_bins_joins_only_small_fragments_and_keeps_big_labels():
    inbox, ri = _hook_image()
    win = (slice(1, 7), slice(1, 25))                       # the labelling runs on a crop of the image
    box = inbox[win]
    plain, n_plain = ndimage.label(box, structure=np.ones((3, 3), bool))
    p = BeamDetectorParams(bridge_empty_bins=True, bridge_min_vspan_deg=33.0, bridge_fragment_px=8, bridge_empty_px=1)
    lab, n, n_big = _bridge_empty_bins(plain, n_plain, box, ri, p, win)
    assert n_big == n_plain and n > n_plain and lab.shape == box.shape
    big = plain[2, 12]
    assert (lab[1:5, 10:20] == big).all()                   # the big group keeps its plain label
    assert lab[1, 2] == lab[1, 4] == lab[3, 3] and lab[1, 2] > n_big   # the fragments are one joined group
    assert lab[1, 21] > n_big and lab[1, 21] != lab[1, 2]  # a fragment is never joined to a big group
    assert (lab > 0).sum() == box.sum()                     # no pixel outside the in-box mask is labelled
    lab0, n0, nb0 = _bridge_empty_bins(plain, 0, np.zeros_like(box), ri, p, win)
    assert n0 == 0 and nb0 == 0


def test_bridge_empty_bins_splits_a_joined_group_where_the_range_jumps():
    """Background seen through a gap in a person (the tunnel 204 m behind a person at 20 m, one empty bin from his
    fragments) is not joined to him: the group would be 185 m deep and dropped as structure (val_sweep)."""
    inbox, ri = _hook_image()
    rng = np.full(inbox.shape, 20.0, np.float32)
    rng[2, 5] = 204.4                                       # the middle fragment is the background
    ri.rng = np.stack([rng, rng])
    win = (slice(1, 7), slice(1, 25))
    box = inbox[win]
    plain, n_plain = ndimage.label(box, structure=np.ones((3, 3), bool))
    p = BeamDetectorParams(bridge_empty_bins=True, bridge_min_vspan_deg=33.0, bridge_fragment_px=8, bridge_empty_px=1)
    lab, n, n_big = _bridge_empty_bins(plain, n_plain, box, ri, p, win)
    assert lab[1, 2] == lab[3, 3] and lab[1, 2] > n_big     # the two 20 m fragments stay one group
    assert lab[1, 4] > n_big and lab[1, 4] != lab[1, 2]     # the 204 m fragment is a group of its own
    assert (lab[1:5, 10:20] == plain[2, 12]).all() and (lab > 0).sum() == box.sum()
    rng[2, 5] = 20.3                                        # within the attachment margin: one surface, one group
    ri.rng = np.stack([rng, rng])
    lab2, _n2, _ = _bridge_empty_bins(plain, n_plain, box, ri, p, win)
    assert lab2[1, 2] == lab2[1, 4] == lab2[3, 3]
    off = BeamDetectorParams(bridge_empty_bins=True, bridge_min_vspan_deg=33.0, bridge_split_range=False)
    rng[2, 5] = 204.4
    ri.rng = np.stack([rng, rng])
    lab3, _n3, _ = _bridge_empty_bins(plain, n_plain, box, ri, off, win)
    assert lab3[1, 2] == lab3[1, 4] == lab3[3, 3]          # switched off: joined as before


def test_bridge_empty_bins_is_the_identity_when_gated_off():
    inbox, ri = _hook_image()
    win = (slice(0, 8), slice(0, 32))
    plain, n_plain = ndimage.label(inbox, structure=np.ones((3, 3), bool))
    on = BeamDetectorParams(bridge_empty_bins=True, bridge_min_vspan_deg=33.0)
    assert _bridge_empty_bins(plain, n_plain, inbox, ri, on, win)[1] > n_plain
    for p, img in ((BeamDetectorParams(bridge_empty_bins=False, bridge_min_vspan_deg=33.0), ri),   # flag off
                   (on, _hook_image(organized=True)[1]),                                           # organised image
                   (on, _hook_image(vspan=26.2)[1])):                                              # Falcon Prime span
        lab, n, n_big = _bridge_empty_bins(plain, n_plain, inbox, img, p, win)
        assert n == n_plain and n_big == n_plain and lab is plain


# ---------------------------------------------------------------- compact pair votes (small objects)
def _cube_face(d, l0, l1, h0, h1, step=0.03):
    """A dense front face of a small box at forward distance d (sensor 1.3 m above rail in ``tunnel_cloud``)."""
    ls = np.arange(l0, l1 + 1e-6, step)
    hs = np.arange(h0, h1 + 1e-6, step)
    L, Hh = np.meshgrid(ls, hs)
    return np.column_stack([L.ravel(), np.full(L.size, -d), -1.3 + Hh.ravel()]).astype(np.float32)


def test_d1_marks_a_small_cube_at_the_corridor_edge_compact_but_not_a_long_edge_strip():
    cube = _cube_face(30.0, 1.0, 1.28, 1.0, 1.3)
    acc, rej = _d1_near(np.vstack([tunnel_cloud(), cube]), 30.0)
    assert not acc and rej and all(not c.compact for c in rej)                      # v1: a thin sliver at the edge
    assert any(c.rejected_reason.startswith("thin sliver") for c in rej)
    acc, rej = _d1_near(np.vstack([tunnel_cloud(), cube]), 30.0, compact_pair_vote=True)
    assert not acc and any(c.compact for c in rej), [(c.rejected_reason, c.extent.tolist()) for c in rej]
    strip = np.vstack([_cube_face(30.0 + k * 0.1, 1.15, 1.28, 1.0, 1.3) for k in range(21)])   # 2 m along the track
    acc, rej = _d1_near(np.vstack([tunnel_cloud(), strip]), 31.0, compact_pair_vote=True)
    assert not acc and rej and not any(c.compact for c in rej)


def _iso(d, l=0.0, n=9, score=0.15, att=0.0, bg=24, ring=24, ext=(0.05, 0.25, 0.2)):
    return DetCandidate("d2", d, l, 1.4, ext, float(np.hypot(d, l)), n, score, cls="obstacle", lateral_offset=l,
                        info={"attached_frac": att, "n_bg": bg, "n_ring": ring})


def _small(d, l=0.0, n=6, conf=0.2, compact=False, reason=""):
    c = _cand(d, l=l, n=n, conf=conf, reason=reason)
    c.extent = np.array([0.05, 0.25, 0.2])
    c.compact = compact
    return c


def test_a_small_d1_cluster_and_an_isolated_d2_group_confirm_together_when_enabled():
    seq = [([_small(40.0 - 1.7 * i)], [], [_iso(40.0 - 1.7 * i)], 0.5) for i in range(6)]
    assert "detected" not in _run(seq)                                       # v1: both scores below the vote threshold
    assert _run(seq, compact_pair_vote=True)[-1] == "detected"


def test_a_compact_pair_needs_an_isolated_d2_group_at_the_same_place():
    attached = [([_small(40.0 - 1.7 * i)], [], [_iso(40.0 - 1.7 * i, att=0.2)], 0.5) for i in range(6)]
    assert "detected" not in _run(attached, compact_pair_vote=True)
    embedded = [([_small(40.0 - 1.7 * i)], [], [_iso(40.0 - 1.7 * i, bg=10, ring=24)], 0.5) for i in range(6)]
    assert "detected" not in _run(embedded, compact_pair_vote=True)
    beside = [([_small(40.0 - 1.7 * i)], [], [_iso(40.0 - 1.7 * i, l=0.9)], 0.5) for i in range(6)]
    assert "detected" not in _run(beside, compact_pair_vote=True)
    large = [([_small(40.0 - 1.7 * i)], [], [_iso(40.0 - 1.7 * i, ext=(0.3, 1.6, 1.0))], 0.5) for i in range(6)]
    assert "detected" not in _run(large, compact_pair_vote=True)


def _big(d, l, n=120):
    c = _cand(d, l=l, n=n, conf=1.0)
    c.extent = np.array([1.0, 1.5, 1.8])
    return c


def test_a_compact_pair_needs_an_isolated_d1_cluster():
    # a dust cloud (val_dust_001 frame 8): the 15-voxel fragment and the 5-px speck sit 0.9 m across / 1.3 m along from
    # 100-voxel clusters of the same cloud - no pair, so only D1 votes and nothing is confirmed (v1 behaviour)
    cloud = [([_small(40.0 - 1.7 * i), _big(40.0 - 1.7 * i + 1.3, 0.1), _big(40.0 - 1.7 * i, -0.9)], [], [_iso(40.0 - 1.7 * i)], 0.5)
             for i in range(6)]
    assert "detected" not in _run(cloud, compact_pair_vote=True)
    # scattered small fragments around it (rejected, 4 voxels each) count too
    frag = lambda d, l: _small(d, l=l, n=4, reason="too few points (4<5)")
    scattered = [([_small(40.0 - 1.7 * i)], [frag(40.0 - 1.7 * i + 2.0, 0.2), frag(40.0 - 1.7 * i - 0.5, 1.2)], [_iso(40.0 - 1.7 * i)], 0.5)
                 for i in range(6)]
    assert "detected" not in _run(scattered, compact_pair_vote=True)
    # a 2-voxel speck 1 m away or a big cluster beyond the isolation window does not break the pair
    speck = lambda d, l: _small(d, l=l, n=2, reason="too few points (2<5)")
    alone = [([_small(40.0 - 1.7 * i), _big(40.0 - 1.7 * i + 4.0, 0.0)], [speck(40.0 - 1.7 * i + 1.0, 0.8)], [_iso(40.0 - 1.7 * i)], 0.5)
             for i in range(6)]
    assert _run(alone, compact_pair_vote=True)[-1] == "detected"


def test_a_compact_rejected_d1_cluster_votes_only_with_an_isolated_d2_group_and_stays_rejected():
    rej = lambda d: _small(d, n=4, compact=True, reason="too few points (4<5)")
    seq = [([], [rej(52.0 - 1.8 * i)], [_iso(52.0 - 1.8 * i, n=6)], 0.5) for i in range(6)]
    assert "detected" not in _run(seq)
    f = FusionFilter(dataclasses.replace(DetectorConfig(), compact_pair_vote=True))
    for acc, rj, c2, sf in seq:
        res = FrameResult(status="clear")
        res.candidates, res.rejected, res.static_frac = list(acc), list(rj), sf
        dec = f.update(res, c2, RAIL, R)
    assert dec.status == "detected"
    assert all(not c.confirmed for c in res.rejected) and any(c.confirmed for c in res.candidates)
    alone = [([], [rej(52.0 - 1.8 * i)], [], 0.5) for i in range(6)]
    assert "detected" not in _run(alone, compact_pair_vote=True)
    not_compact = [([], [_small(52.0 - 1.8 * i, n=4, reason="too few points (4<5)")], [_iso(52.0 - 1.8 * i, n=6)], 0.5) for i in range(6)]
    assert "detected" not in _run(not_compact, compact_pair_vote=True)


def test_a_compact_pair_does_not_bypass_the_far_vote_minimum_on_a_moving_train():
    # code review: a far 4-voxel cluster and an isolated D2 group voted through the compact pair although
    # far_vote_d1_min_d keeps such a cluster from voting on a moving train (accepted and rejected compact clusters alike)
    acc = [([_small(165.0 - 1.8 * i, n=4, conf=1.0)], [], [_iso(165.0 - 1.8 * i)], 0.55) for i in range(7)]
    assert "detected" not in _run(acc, far_vote_d1_min_d=140.0, compact_pair_vote=True)
    rej = [([], [_small(165.0 - 1.8 * i, n=3, compact=True, reason="too few points (3<4)")], [_iso(165.0 - 1.8 * i)], 0.55)
           for i in range(7)]
    assert "detected" not in _run(rej, far_vote_d1_min_d=140.0, compact_pair_vote=True)
    standing = [(a, r, c, 0.9) for a, r, c, _ in acc]
    assert _run(standing, far_vote_d1_min_d=140.0, compact_pair_vote=True)[-1] == "detected"   # standing: the pair votes
    near = [([_small(60.0 - 1.8 * i, n=4, conf=1.0)], [], [_iso(60.0 - 1.8 * i)], 0.55) for i in range(7)]
    assert _run(near, far_vote_d1_min_d=140.0, compact_pair_vote=True)[-1] == "detected"       # within 140 m it votes


def test_a_weak_and_compact_rejected_cluster_is_one_d1_hit():
    # code review: a far cluster both weak and compact was emitted twice (weak hit + compact extra) -> two tracks
    c = lambda d: _small(d, n=4, compact=True, reason="single-ring return at range (0.13 m tall)")
    f = FusionFilter(dataclasses.replace(DetectorConfig(), far_weak_d1_vote=True, compact_pair_vote=True))
    for i in range(6):
        k = c(150.0 - 0.1 * i)
        k.weak = True
        res = FrameResult(status="clear")
        res.candidates, res.rejected, res.static_frac = [], [k], 0.9
        dec = f.update(res, [_iso(150.0 - 0.1 * i, score=1.0)], RAIL, R)
    assert dec.status == "detected"
    assert len(f.tracks) == 1, [(t.track_id, t.distance_m) for t in f.tracks]
