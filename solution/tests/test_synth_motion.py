"""synth/motion.py: 1-DoF along-track displacement between consecutive real frames (used by fake_scenes.ego_motion).

Regression for the val_fake_style crash of 2026-10-18 23:44 (cloud_with_fake_obj 900-1049, a creeping train at
0-0.1 m/frame): the Viterbi path point of pair 919 -> 920 was not a local minimum of its own cost curve (the
smoothness prior overruled it), the parabolic refinement through the monotone triple (0.2996, 0.4808, 0.6627) with
curvature 7e-4 put the step at -5.05 m, the two-frame closure window around the sum of two steps (-4.98 m) lay outside
the -0.2..6 m grid and np.argmin raised on an empty sequence.  Pure NumPy/SciPy, no bag access."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))

from synth.motion import _closure_window, _refine, compact_frame, frame_shift, segment_motion_dp  # noqa: E402

STEP = 0.02
GRID = np.arange(-0.2, 3.0 + 1e-9, STEP)


def test_refine_keeps_grid_value_on_monotone_triple():
    # exact normalised costs of pair 19 (frames 919 -> 920) of the crashed run; path point j = 14 (0.08 m)
    j = 14
    c = np.zeros(len(GRID))
    c[j - 1:j + 2] = [0.2996, 0.4808, 0.6627]
    assert _refine(c, j, GRID, STEP) == GRID[j]
    c[j - 1:j + 2] = [0.6627, 0.4808, 0.2996]            # mirrored
    assert _refine(c, j, GRID, STEP) == GRID[j]
    assert _refine(c, 0, GRID, STEP) == GRID[0]           # grid edge: no refinement
    assert _refine(c, len(GRID) - 1, GRID, STEP) == GRID[-1]


def test_refine_local_minimum_stays_within_half_step():
    rng = np.random.default_rng(3)
    for _ in range(50):
        x0 = rng.uniform(0.3, 2.5)
        curv = 10 ** rng.uniform(-4, 1)                  # from nearly flat to sharp
        c = curv * (GRID - x0) ** 2
        j = int(np.argmin(c))
        s = _refine(c, j, GRID, STEP)
        assert abs(s - GRID[j]) <= 0.5 * STEP + 1e-12
        assert abs(s - x0) < 1e-6                         # exact for a parabola
    # flat triple (zero curvature) and a tiny-curvature minimum: grid value / clamped, never far away
    c = np.zeros(len(GRID))
    assert _refine(c, 20, GRID, STEP) == GRID[20]
    c[19:22] = [0.0, -1e-9, 1e-3]
    assert abs(_refine(c, 20, GRID, STEP) - GRID[20]) <= 0.5 * STEP + 1e-12


def test_closure_window_never_empty():
    g2 = np.arange(-0.2, 6.0 + 1e-9, STEP)
    for exp2 in (-5.0, -0.42, -0.2, 0.0, 0.074 - 5.050, 3.3, 6.0, 6.02, 40.0):
        lo, hi = _closure_window(g2, exp2, STEP)
        assert 0 <= lo < hi <= len(g2)
        if -0.2 <= exp2 <= 6.0:
            assert np.all(np.abs(g2[lo:hi] - exp2) <= 0.5 + STEP)


def _tunnel(rng, n=5000):
    """Static structure in (d, l, h): two walls, a ceiling and periodic ribs every 1.0 m (the aliasing case)."""
    d = rng.uniform(0.0, 90.0, n)
    side = rng.choice([-1.0, 1.0], n)
    l = side * 2.2 + rng.normal(0, 0.03, n)
    h = rng.uniform(-0.5, 2.0, n)
    ceil = rng.random(n) < 0.3
    l[ceil] = rng.uniform(-2.0, 2.0, ceil.sum())
    h[ceil] = 2.6 + rng.normal(0, 0.03, ceil.sum())
    rib = rng.random(n) < 0.3
    d[rib] = np.round(d[rib]) + rng.normal(0, 0.01, rib.sum())
    return np.stack([d, l, h], axis=1)


def _frames(rng, world, s_true):
    out = []
    for k, s in enumerate(s_true):
        keep = rng.random(len(world)) < 0.7                       # the returns differ frame to frame
        p = world[keep] - np.array([s, 0.0, 0.0]) + rng.normal(0, 0.01, (keep.sum(), 3))
        out.append(compact_frame(p, seed=k))
    return out


def test_segment_motion_dp_creeping_train_no_crash_and_accurate():
    rng = np.random.default_rng(20261019)
    world = _tunnel(rng)
    steps = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0, 0.0, 0.02])
    s_true = np.cumsum(steps)
    res = segment_motion_dp(_frames(rng, world, s_true))
    assert np.all(np.isfinite(res["s"])) and np.all(np.isfinite(res["closure_err"]))
    assert len(res["uninformative"]) == 0
    assert np.all(np.abs(res["step"] - steps) <= 0.03), res["step"]
    assert abs(res["s"][-1] - s_true[-1]) < 0.1
    assert np.all(res["closure_err"] < 0.1)


def test_segment_motion_dp_fast_train_with_periodic_structure():
    rng = np.random.default_rng(7)
    world = _tunnel(rng)
    steps = np.array([0.0, 1.40, 1.42, 1.45, 1.47, 1.50, 1.52, 1.55, 1.57, 1.60])
    s_true = np.cumsum(steps)
    res = segment_motion_dp(_frames(rng, world, s_true))
    assert np.all(np.abs(res["step"] - steps) <= 0.03), res["step"]
    assert abs(res["s"][-1] - s_true[-1]) < 0.1


def test_segment_motion_dp_bridges_an_empty_frame():
    rng = np.random.default_rng(11)
    world = _tunnel(rng)
    steps = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    fr = _frames(rng, world, np.cumsum(steps))
    fr[2] = np.zeros((0, 3))                              # a frame without returns
    res = segment_motion_dp(fr)
    assert np.all(np.isfinite(res["s"]))
    assert set(res["uninformative"].tolist()) == {2, 3}
    assert np.all(np.abs(res["step"][[1, 4, 5]] - 0.5) <= 0.03)
    assert abs(res["s"][-1] - 2.5) < 0.15                 # the prior bridges the gap with the neighbouring steps


def test_frame_shift_without_structure_is_nan():
    s, c, costs = frame_shift(np.zeros((0, 3)), np.zeros((0, 3)))
    assert s == 0.0 and np.isnan(c) and np.all(np.isnan(costs))
