"""WP-C C2: curvature shrink of the corridor (shared by D1, D2 and RViz) and the chain-shape range."""
import dataclasses

import numpy as np

from metro_obstacle_detector.config import DetectorConfig, curv_factor
from metro_obstacle_detector.track_model import TrackCurve, chain_shape_range, curvature_shrink


def _curve(b, shape_ok=0.0):
    return TrackCurve(l0=0.1, a=0.01, b=b, ok=True, n_used=20, max_d_supported=90.0, max_d_absolute=60.0, shape_ok_d=shape_ok)


def test_default_is_bit_identical_to_v1():
    cfg = DetectorConfig()
    for dtype in (np.float32, np.float64):
        d = np.linspace(3.0, 200.0, 777).astype(dtype)
        for b in (0.0, 5e-5, -2.8e-4, 3.4e-4, 5.7e-4, -1.3e-3, np.float64(6.1e-4)):
            for ok in (0.0, 75.0):
                c = _curve(b, ok)
                v1 = curv_factor(cfg, c.b) * abs(c.b) * d * d
                new = curvature_shrink(cfg, c, d)
                assert new.dtype == v1.dtype and np.array_equal(new, v1)
    # per-cluster scalar use (D1) too
    c = _curve(5.7e-4, 75.0)
    assert curvature_shrink(cfg, c, 58.0) == curv_factor(cfg, c.b) * abs(c.b) * 58.0 * 58.0


def test_shape_factor_scales_only_inside_the_shape_range():
    cfg = dataclasses.replace(DetectorConfig(), corridor_curv_shape_factor=0.0)
    d = np.array([20.0, 58.0, 75.0, 75.5, 100.0])
    c = _curve(5.7e-4, 75.0)
    v1 = curv_factor(cfg, c.b) * abs(c.b) * d * d
    new = curvature_shrink(cfg, c, d)
    assert np.all(new[:3] == 0.0) and np.array_equal(new[3:], v1[3:])
    assert np.array_equal(curvature_shrink(cfg, _curve(5.7e-4, 0.0), d), v1)   # no shape range: v1


def _chain(l0, a, b, d_anchor, ds, err=None, off=0.0):
    f = lambda x: l0 + a * x + b * x * x
    out = []
    for dc in ds:
        e = err(dc) if err else 0.0
        out.append((dc, f(dc) - f(d_anchor) + off + e, 0.9))
    return out


def test_chain_shape_range_follows_a_matching_bend_and_stops_where_it_departs():
    cfg = DetectorConfig()
    l0, a, b = -0.37, 0.0295, 5.7e-4
    ds = [4.5] + [7.5 + 3.0 * k for k in range(35)]           # 4.5 ... 109.5 m
    # the chain follows the fitted shape up to a constant anchor offset (-0.4 m): the whole chain range
    ch = _chain(l0, a, b, 4.5, ds, off=-0.4)
    ch[0] = (4.5, 0.0, 1.0)
    assert chain_shape_range(ch, 4.5, l0, a, b, 120.0, cfg) == ds[-1]
    assert chain_shape_range(ch, 4.5, l0, a, b, 80.0, cfg) == max(x for x in ds if x <= 80.0)
    # the tunnel bends more than the quadratic beyond 70 m: the range ends before the residual exceeds the tolerance
    ch = _chain(l0, a, b, 4.5, ds, err=lambda x: 0.002 * max(0.0, x - 70.0) ** 2)
    r = chain_shape_range(ch, 4.5, l0, a, b, 120.0, cfg)
    assert 70.0 <= r < 82.0
    # a gap in the chain ends it
    ch = [x for x in _chain(l0, a, b, 4.5, ds) if not (50.0 < x[0] < 66.0)]
    assert chain_shape_range(ch, 4.5, l0, a, b, 120.0, cfg) < 50.0
    # too few samples in the reference window: not established
    assert chain_shape_range(_chain(l0, a, b, 4.5, [4.5, 40.0, 50.0]), 4.5, l0, a, b, 120.0, cfg) == 0.0
    assert chain_shape_range([], None, l0, a, b, 120.0, cfg) == 0.0
