"""Version-3 obstacle rules (the organisers' marker bags): every rule is behind a parameter; ``V2`` is the v2 setting.

* ``axis_cross_exempt`` (on): a D1 cluster spanning the track axis and thin along it is not "attached to structure at
  the box edge" (the organisers' beam across the rails sticks out of the low zone and touches the wall on one side);
* ``hanging_central_exempt`` (on): a D1 cluster hanging into the middle of the corridor at short range, with free
  space below it, is neither "extends above the box top" nor attached structure (their tilted cable from the ceiling);
* ``straddle_compact_exempt`` (on, compact path only): the returns outside the box next to a small edge cluster are the
  rest of the same object when, grown through the outside returns, they stay compact with it (their cube on the left
  rail, two thirds of it beyond the low zone); such a cluster (``Candidate.straddle``) pairs with a D2 group whose
  ring is at least ``straddle_min_bg_frac`` background (the rest of the object fills one side of it);
* ``straddle_thin_exempt`` (on, with the above): an edge sliver too thin for the compact path (the top face of a cube
  whose body lies beyond the low-zone edge) is compact when the object it forms beyond the edge is compact and thick.
"""
import dataclasses

import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector

from synthetic import tunnel_cloud

V2 = dict(axis_cross_exempt=False, hanging_central_exempt=False, straddle_compact_exempt=False)


def _d1_near(cloud, d, **cfg_kw):
    res = ObstacleDetector(dataclasses.replace(DetectorConfig(), **{**V2, **cfg_kw})).process(cloud)
    acc = [c for c in res.candidates if abs(c.forward_m - d) < 2.0]
    rej = [c for c in res.rejected if abs(c.forward_m - d) < 2.0]
    return acc, rej


def _beam(d, l0, l1, h0=0.53, h1=0.67, depth=0.14, step=0.04):
    """A square beam across the track (sensor 1.3 m above rail in ``tunnel_cloud``), front and top faces sampled."""
    ls = np.arange(l0, l1 + 1e-6, step)
    hs = np.arange(h0, h1 + 1e-6, step)
    ds = np.arange(d, d + depth + 1e-6, step)
    L, Hh = np.meshgrid(ls, hs)
    front = np.column_stack([L.ravel(), np.full(L.size, -d), -1.3 + Hh.ravel()])
    L, D = np.meshgrid(ls, ds)
    top = np.column_stack([L.ravel(), -D.ravel(), np.full(L.size, -1.3 + h1)])
    return np.vstack([front, top]).astype(np.float32)


def _cable(d, l_top, l_bottom, h_top, h_bottom, step=0.03):
    """A thin tilted cable hanging from the ceiling: a 0.1 m thick line from (l_top, h_top) to (l_bottom, h_bottom)."""
    t = np.arange(0.0, 1.0 + 1e-6, step / max(abs(h_top - h_bottom), 1e-3))
    pts = []
    for dl in (-0.05, 0.0, 0.05):
        l = l_top + (l_bottom - l_top) * t + dl
        h = h_top + (h_bottom - h_top) * t
        pts.append(np.column_stack([l, np.full(len(t), -d), -1.3 + h]))
    return np.vstack(pts).astype(np.float32)


def test_v3_rule_defaults():
    cfg = DetectorConfig()
    assert cfg.axis_cross_exempt and cfg.axis_cross_min_m == 0.25 and cfg.axis_cross_max_depth_m == 0.5
    assert cfg.hanging_central_exempt and cfg.hanging_central_max_d == 70.0
    assert cfg.hanging_central_min_bottom_m == 1.8 and cfg.hanging_central_edge_m == 0.5
    assert cfg.straddle_compact_exempt and cfg.straddle_grow_m == 0.15 and cfg.straddle_grow_per_m == 0.004
    assert cfg.straddle_thin_exempt and cfg.straddle_max_points == 3000 and cfg.straddle_min_bg_frac == 0.4
    assert cfg.straddle_chunk == 128


def test_a_beam_across_the_rails_touching_the_wall_is_accepted_only_with_the_axis_exemption():
    # 25 m ahead, from 0.6 m left of the axis to the tunnel wall on the right (1.7 m at this height), 0.53-0.67 m above rail
    cloud = np.vstack([tunnel_cloud(), _beam(25.0, -0.6, 1.75)])
    acc, rej = _d1_near(cloud, 25.0)
    assert not acc and any(c.rejected_reason.startswith("attached") for c in rej), [c.rejected_reason for c in rej]
    acc, rej = _d1_near(cloud, 25.0, axis_cross_exempt=True)
    assert acc, [c.rejected_reason for c in rej]


def test_the_axis_exemption_does_not_apply_to_a_deep_cluster():
    # a block 1.2 m deep along the track across the axis to the wall (a dust cloud is as deep): judged as without the exemption
    cloud = np.vstack([tunnel_cloud(), _beam(25.0, -0.6, 1.75, h0=0.4, h1=1.0, depth=1.2)])
    acc0, rej0 = _d1_near(cloud, 25.0)
    acc, rej = _d1_near(cloud, 25.0, axis_cross_exempt=True)
    assert not acc0 and rej0, [c.to_dict() for c in acc0]
    assert not acc and [c.rejected_reason for c in rej] == [c.rejected_reason for c in rej0]


def test_the_axis_exemption_does_not_apply_to_an_edge_cluster():
    # the same beam stub only between 0.5 m and the wall: it does not reach across the axis and stays structure
    cloud = np.vstack([tunnel_cloud(), _beam(25.0, 0.5, 1.75)])
    acc, rej = _d1_near(cloud, 25.0, axis_cross_exempt=True)
    assert not acc and rej, [c.to_dict() for c in acc]


def test_a_cable_hanging_into_the_corridor_is_accepted_only_with_the_hanging_exemption():
    # 30 m ahead, from the ceiling (4.4 m above rail, out of the box) down to 2.0 m above rail, 0.1-0.6 m right of the axis
    cloud = np.vstack([tunnel_cloud(), _cable(30.0, 0.1, 0.6, 4.4, 2.0)])
    acc, rej = _d1_near(cloud, 30.0)
    assert not acc and any(c.rejected_reason.startswith(("extends above", "attached")) for c in rej), \
        [c.rejected_reason for c in rej]
    acc, rej = _d1_near(cloud, 30.0, hanging_central_exempt=True)
    assert acc, [c.rejected_reason for c in rej]


def test_the_hanging_exemption_needs_short_range_free_space_below_and_a_central_position():
    on = dict(hanging_central_exempt=True)
    far = np.vstack([tunnel_cloud(), _cable(90.0, 0.1, 0.6, 4.4, 2.0, step=0.02)])
    acc, _ = _d1_near(far, 90.0, **on)
    assert not acc                                                    # beyond hanging_central_max_d
    low = np.vstack([tunnel_cloud(), _cable(30.0, 0.1, 0.6, 4.4, 1.2)])
    acc, rej = _d1_near(low, 30.0, **on)
    assert not acc, [c.to_dict() for c in acc]                        # reaches down to 1.2 m: no free space below
    edge = np.vstack([tunnel_cloud(), _cable(30.0, 1.2, 1.1, 4.4, 2.0)])
    acc, rej = _d1_near(edge, 30.0, **on)
    assert not acc, [c.to_dict() for c in acc]                        # at the box edge: a wall-side cable or a column


def _box(d0, d1, l0, l1, h0, h1, step=0.04):
    """Front (d0), inner side (l0) and top (h1) faces of a box, h above rail (sensor 1.3 m above rail)."""
    ds, ls, hs = (np.arange(a, b + 1e-6, step) for a, b in ((d0, d1), (l0, l1), (h0, h1)))
    faces = []
    L, H = np.meshgrid(ls, hs); faces.append(np.column_stack([L.ravel(), np.full(L.size, -d0), -1.3 + H.ravel()]))
    D, H = np.meshgrid(ds, hs); faces.append(np.column_stack([np.full(D.size, l0), -D.ravel(), -1.3 + H.ravel()]))
    D, L = np.meshgrid(ds, ls); faces.append(np.column_stack([L.ravel(), -D.ravel(), np.full(D.size, -1.3 + h1)]))
    return np.vstack(faces).astype(np.float32)


def _edge_cands(cloud, d, **kw):
    acc, rej = _d1_near(cloud, d, **kw)
    return acc, [c for c in rej if 0.6 <= c.lateral_offset_from_track <= 1.1]


def test_a_cube_straddling_the_low_zone_edge_is_compact_only_with_the_straddle_exemption():
    # 0.3 m cube 20 m ahead, 0.75-1.05 m left of the axis, 0.36-0.66 m above rail: the box edge is at 0.94 m there
    cloud = np.vstack([tunnel_cloud(), _box(20.0, 20.3, 0.75, 1.05, 0.36, 0.66)])
    acc, rej = _edge_cands(cloud, 20.0)
    assert not acc and rej and not any(c.compact for c in rej), [(c.rejected_reason, c.compact) for c in rej]
    acc, rej = _edge_cands(cloud, 20.0, straddle_compact_exempt=True)
    assert not acc and any(c.compact and c.straddle for c in rej), [(c.rejected_reason, c.compact) for c in rej]   # D2 must still agree
    assert all(c.to_dict().get("straddle") for c in rej if c.straddle)


def test_compact_straddle_grows_through_the_rest_of_a_cube_but_not_along_a_cover():
    # outside-the-box returns (columns d, l, h) next to an edge cluster 20 m ahead (in-box part of a cube, l 0.75-0.94 m)
    from scipy.spatial import cKDTree
    cfg = DetectorConfig()
    own = _box(20.0, 20.3, 0.75, 0.94, 0.36, 0.66)
    own_dlh = np.column_stack([-own[:, 1], own[:, 0], own[:, 2]])
    rest = _box(20.0, 20.3, 0.94, 1.05, 0.36, 0.66)                 # the rest of the cube beyond the edge
    cover = _box(17.0, 23.0, 1.05, 1.5, 0.36, 0.66)                 # a 6 m cover the cube touches
    for pts, compact in ((rest, True), (np.vstack([rest, cover]), False)):
        dlh = np.column_stack([-pts[:, 1], pts[:, 0], pts[:, 2]])
        tree = cKDTree(dlh)
        seeds = set()
        for lst in tree.query_ball_point(own_dlh, r=cfg.attach_radius_m + cfg.attach_radius_per_m * 20.0):
            seeds.update(lst)
        assert seeds
        got = ObstacleDetector._compact_straddle(tree, seeds, own_dlh[:, 0], own_dlh[:, 1], own_dlh[:, 2], 20.0, cfg)
        assert (got is not None) is compact
        if compact:
            assert np.all(got <= cfg.compact_max_extent_m) and got[1] >= 0.25          # the whole cube (0.75-1.02 m sampled)


def test_a_straddling_compact_cluster_pairs_with_a_d2_group_bordered_by_the_rest_of_the_object():
    from test_rules_v2 import FusionFilter, FrameResult, RAIL, R, _iso, _small

    def run(straddle, bg, **kw):
        f = FusionFilter(dataclasses.replace(DetectorConfig(), **kw))
        for i in range(6):
            c = _small(20.0 - 1.5 * i, l=0.85, n=6, compact=True, reason="thin sliver at the lateral box edge (platform edge / duct)")
            c.straddle = straddle
            res = FrameResult(status="clear")
            res.candidates, res.rejected, res.static_frac = [], [c], 0.5
            dec = f.update(res, [_iso(20.0 - 1.5 * i, l=0.85, n=20, bg=bg, ring=24)], RAIL, R)
        return dec.status

    assert run(True, 12) == "detected"                                   # half of the D2 ring is the cube beyond the edge
    assert run(False, 12) != "detected"                                  # any other compact pair keeps compact_min_bg_frac
    assert run(True, 12, straddle_compact_exempt=False) != "detected"
    assert run(True, 8) != "detected"                                    # a third of the ring background: embedded
    assert run(False, 23) == "detected"


def test_the_top_face_of_a_cube_beyond_the_low_zone_edge_is_compact_only_with_the_thin_exemption():
    # a cube 20 m ahead, 1.02-1.40 m beside the axis, 0.50-0.86 m above rail (like the organisers' switch cube #1): below
    # 0.8 m it is beyond the +-1.0 m low zone, above it only a 6 cm slice of its top lies inside the +-1.4 m body zone.
    # A wide tunnel: in the default one (radius 2.7 m) the wall is 1.57 m from the axis at 0.5 m and the cube would touch
    # it (grown into the wall it is structure, rightly)
    cloud = np.vstack([tunnel_cloud(radius=4.5), _box(20.0, 20.3, 1.02, 1.40, 0.50, 0.86)])
    def edge(**kw):
        acc, rej = _d1_near(cloud, 20.0, **kw)
        return acc, [c for c in rej if 1.0 <= c.lateral_offset_from_track <= 1.4]
    acc, rej = edge(straddle_compact_exempt=True, straddle_thin_exempt=False)
    assert not acc and rej and not any(c.compact for c in rej), [(c.rejected_reason, np.round(c.extent, 2).tolist()) for c in rej]
    acc, rej = edge(straddle_compact_exempt=True)
    assert not acc and any(c.compact and c.straddle for c in rej), [(c.rejected_reason, np.round(c.extent, 2).tolist()) for c in rej]


def test_compact_straddle_stops_early_on_dense_structure_with_the_same_verdict():
    # a dense wall 0.36 m beside a small edge cluster: the growth hits the point cap after one chunk (bounded cost),
    # and a chunk size larger than the frontier (one query per pass, the original loop) gives the same verdict
    import time
    from scipy.spatial import cKDTree
    cfg = DetectorConfig()
    g = np.arange(0.0, 3.0, 0.01)
    D, H = np.meshgrid(4.0 + g, 0.3 + g)
    wall = np.column_stack([D.ravel(), np.full(D.size, 1.36), H.ravel()])
    wall = np.vstack([wall, wall + 1e-4])                                   # dual-return duplicates
    tree = cKDTree(wall)
    own = np.array([[5.0, 1.0, 1.0], [5.05, 1.0, 1.05], [5.0, 0.98, 1.1]])
    seeds = set()
    for lst in tree.query_ball_point(own, r=cfg.attach_radius_m + cfg.attach_radius_per_m * 5.0):
        seeds.update(lst)
    t0 = time.perf_counter()
    got = ObstacleDetector._compact_straddle(tree, seeds, own[:, 0], own[:, 1], own[:, 2], 5.0, cfg)
    dt = time.perf_counter() - t0
    big = dataclasses.replace(cfg, straddle_chunk=10 ** 9)
    assert got is None and ObstacleDetector._compact_straddle(tree, seeds, own[:, 0], own[:, 1], own[:, 2], 5.0, big) is None
    assert dt < 0.5, dt
