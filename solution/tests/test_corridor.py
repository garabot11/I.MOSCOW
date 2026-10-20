"""WP-C corridor: extension weight (ramp instead of the step), corridor support, quasi-static curvature,
status text placement and the coasting-track geometry memory."""
import dataclasses

import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import Candidate, FrameResult
from metro_obstacle_detector.pipeline import DetectionPipeline
from metro_obstacle_detector.track_model import (RailLevel, RailLevelExtended, TrackCurve, corridor_support,
                                                 extension_weight)


def _ext_rail(far=141.0, d_switch=24.0):
    base = RailLevel(z0=-2.0, grade=-0.04, curv=0.0, n_bins=8, ok=True, residual_rms=0.03, max_d_observed=far)
    return RailLevelExtended(**dataclasses.asdict(base), far_coef=(2.5, -0.04, 0.0), height=4.6,
                             d_switch=d_switch, far_max_d=far, n_far_bins=20, bed_max_d=d_switch)


def _curve(b, walls=81.0, own=None):
    return TrackCurve(l0=0.0, a=0.0, b=b, ok=True, n_used=20, max_d_supported=walls,
                      max_d_absolute=walls if own is None else own, max_d_walls=walls)


def test_extension_weight_is_v1_below_the_thresholds_and_fades_above():
    cfg = DetectorConfig()
    assert extension_weight(0.9e-4, False, cfg) == 1.0 and extension_weight(1.4e-4, True, cfg) == 1.0
    # v1 step: 0 at the threshold; now a linear fade over corridor_ext_ramp_frac (0.2) x threshold
    assert abs(extension_weight(1.1e-4, False, cfg) - 0.5) < 1e-9
    assert abs(extension_weight(1.65e-4, True, cfg) - 0.5) < 1e-9
    assert extension_weight(1.8e-4, True, cfg) == 0.0 and extension_weight(3e-4, False, cfg) == 0.0
    w = [extension_weight(b, True, cfg) for b in np.linspace(1.4e-4, 1.9e-4, 51)]
    assert all(x >= y for x, y in zip(w, w[1:]))                               # monotone, no jump > ramp slope
    assert max(abs(x - y) for x, y in zip(w, w[1:])) <= 0.1e-5 / 0.3e-4 + 1e-9
    step = dataclasses.replace(cfg, corridor_ext_ramp_frac=0.0)
    assert extension_weight(1.49e-4, True, step) == 1.0 and extension_weight(1.5e-4, True, step) == 0.0


def test_corridor_support_blends_walls_and_ceiling_reach_never_below_the_detector():
    rail = _ext_rail(141.0)
    assert corridor_support(_curve(1e-4), rail, 1.0) == 141.0
    assert corridor_support(_curve(1e-4), rail, 0.0) == 81.0
    assert abs(corridor_support(_curve(1e-4), rail, 0.5) - 111.0) < 1e-9
    # the detector extended this frame's support already (v1 rule): never less than that
    assert corridor_support(_curve(1e-4, own=141.0), rail, 0.0) == 141.0
    # a plain rail has no ceiling reach: the wall support only
    plain = RailLevel(z0=-2.0, grade=-0.04, curv=0.0, n_bins=8, ok=True, residual_rms=0.03, max_d_observed=30.0)
    assert corridor_support(_curve(1e-4), plain, 1.0) == 81.0
    # curves built without max_d_walls (older callers) fall back to max_d_absolute
    c = TrackCurve(l0=0.0, a=0.0, b=0.0, ok=True, max_d_absolute=90.0)
    assert corridor_support(c, rail, 1.0) == 141.0 and corridor_support(c, rail, 0.0) == 90.0


def _pipe():
    return DetectionPipeline(DetectorConfig(d3_enabled=False))


def _fit_result(b):
    res = FrameResult(status="clear")
    res.fit = (_ext_rail(), _curve(b))
    return res


def test_quasi_static_curvature_decides_on_the_median_of_the_stop():
    p = _pipe()
    rng = np.random.default_rng(0)
    bs = np.clip(rng.normal(1.15e-4, 0.3e-4, 200), 0.5e-4, 2.2e-4)    # Falcon-like scatter around a straight tunnel
    sup = []
    for b in bs:
        p._track_curvature(_fit_result(b), 0.88)
        sup.append(p._corridor_support(_ext_rail(), _curve(b)))
    assert min(sup[20:]) == 141.0                                    # no flip once the stop has a curvature estimate
    # the same scatter on a moving train (static fraction below quasi_static_frac): per-frame decisions
    p = _pipe()
    sup = []
    for b in bs:
        p._track_curvature(_fit_result(b), 0.5)
        sup.append(p._corridor_support(_ext_rail(), _curve(b)))
    assert min(sup) < 141.0


def test_quasi_static_median_never_averages_a_real_curve_away():
    p = _pipe()
    for _ in range(50):
        p._track_curvature(_fit_result(0.5e-4), 0.9)
    # this frame's own model is clearly curved (|b| > corridor_static_b_max): no extension
    assert p._corridor_support(_ext_rail(), _curve(4e-4)) == 81.0
    # the scene moves: the collected curvature is dropped
    p._track_curvature(_fit_result(2e-4), 0.6)
    assert p._b_long == []


def test_status_text_ahead_and_above_the_sensor():
    from metro_obstacle_detector.markers import status_text_position
    cfg = DetectorConfig()                       # Hesai: forward -y, up +z
    pos = status_text_position(FrameResult(status="clear"), cfg, cfg.axis_matrix())
    assert np.allclose(pos, [0.0, -cfg.marker_status_ahead_m, cfg.marker_status_height_m])
    falcon = DetectorConfig(forward_axis="+x", up_axis="+z")
    pos = status_text_position(FrameResult(status="clear"), falcon, falcon.axis_matrix())
    assert np.allclose(pos, [falcon.marker_status_ahead_m, 0.0, falcon.marker_status_height_m])


def _cand(tid, ext, n, confirmed=True):
    return Candidate(distance_m=50.0, forward_m=50.0, lateral_m=0.0, height_m=0.0, center_sensor=np.zeros(3),
                     extent=np.array(ext, dtype=float), n_points=n, lateral_offset_from_track=0.0,
                     height_above_rail=1.0, track_id=tid, confirmed=confirmed)


def test_coasting_track_keeps_its_last_substantial_geometry():
    from metro_obstacle_detector.markers import ObstacleGeometryMemory
    mem = ObstacleGeometryMemory(max_age=5)
    mem.apply([_cand(7, (0.3, 0.6, 1.5), 12)])
    c = _cand(7, (0.0, 0.0, 0.0), 1)                 # the fusion's write-back of a 1-pixel D2 hit
    mem.apply([c])
    assert np.allclose(c.extent, (0.3, 0.6, 1.5)) and c.n_points == 12
    other = _cand(8, (0.0, 0.0, 0.0), 1)             # unknown track: left as it is
    mem.apply([other])
    assert other.n_points == 1
    for _ in range(6):
        mem.apply([])
    late = _cand(7, (0.0, 0.0, 0.0), 1)              # too old: not carried any more
    mem.apply([late])
    assert late.n_points == 1 and np.allclose(late.extent, 0.0)
