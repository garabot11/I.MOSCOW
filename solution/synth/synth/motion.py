"""1-DoF along-track displacement of the train between real frames (for insertion into moving segments).

Between two frames the sensor moves along the track by s (0..3 m at <= 25 m/s, 10 Hz).  Static structure seen in
frame k at analysis coordinate d appears in frame k+1 at d - s.  The shift is found by aligning the near-field
structure (walls, ceiling, equipment; 4..45 m ahead, |l| < 4 m, h above the floor) of frame k+1, shifted by +s,
with frame k: robust nearest-neighbour cost on a subsample, coarse grid (2 cm) then parabolic refinement.
Lateral/vertical motion and yaw are neglected (straight or gently curved track, 0.1 s).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _structure_points(xyz_dlh: np.ndarray, rng, n: int = 6000, d_rng=(4.0, 45.0)):
    d, l, h = xyz_dlh.T
    m = (d > d_rng[0]) & (d < d_rng[1]) & (np.abs(l) < 4.0) & (h > -0.9)
    p = xyz_dlh[m]
    if len(p) > n:
        p = p[rng.choice(len(p), n, replace=False)]
    return p


def frame_shift(a_dlh: np.ndarray, b_dlh: np.ndarray, s_max: float = 3.0, step: float = 0.02, seed: int = 0,
                trunc: float = 0.15) -> Tuple[float, float, np.ndarray]:
    """Along-track displacement from frame a to frame b (b observed later): returns (s, cost_min, costs)."""
    rng = np.random.default_rng(seed)
    q = _structure_points(b_dlh, rng)
    grid = np.arange(-0.2, s_max + 1e-9, step)
    if len(q) == 0 or len(a_dlh) == 0:
        # no structure in one of the frames: the pair carries no information (NaN costs, handled by the callers)
        return 0.0, float("nan"), np.full(len(grid), np.nan)
    tree = cKDTree(a_dlh)
    costs = np.empty(len(grid))
    for i, s in enumerate(grid):
        dist, _ = tree.query(q + np.array([s, 0.0, 0.0]), k=1, distance_upper_bound=trunc)
        costs[i] = np.mean(np.minimum(dist, trunc))
    j = int(np.argmin(costs))
    return _refine(costs, j, grid, step), float(costs[j]), costs


def _refine(c: np.ndarray, j: int, grid: np.ndarray, step: float) -> float:
    """Sub-grid position of the minimum of cost curve ``c`` around grid point ``j``: vertex of the parabola through
    the three samples j-1, j, j+1.  Applied only when ``c[j]`` is the lowest of the three - then the vertex lies
    within half a step of ``grid[j]`` (and is clamped to that anyway).  A Viterbi path point need not be a local
    minimum of its own curve (the smoothness prior may overrule it); on a monotone triple with a tiny curvature the
    vertex can be metres away (-5.05 m for a 0.08 m step in cloud_with_fake_obj 919 -> 920), so the grid value is
    kept there."""
    m = len(c)
    if not (0 < j < m - 1):
        return float(grid[j])
    c0, c1, c2 = float(c[j - 1]), float(c[j]), float(c[j + 1])
    if not (c1 <= c0 and c1 <= c2):
        return float(grid[j])
    den = c0 - 2.0 * c1 + c2
    if not den > 1e-12:
        return float(grid[j])
    off = 0.5 * step * (c0 - c2) / den
    return float(grid[j] + min(max(off, -0.5 * step), 0.5 * step))


def _closure_window(g2: np.ndarray, exp2: float, step: float, half_width: float = 0.5):
    """Index range [lo, hi) of the grid ``g2`` within ``half_width`` of the expected two-frame shift ``exp2``,
    centred on the grid point nearest to ``exp2`` (clipped into the grid), so that it is never empty."""
    w = int(round(half_width / step))
    jc = int(round((exp2 - g2[0]) / step))
    jc = min(max(jc, 0), len(g2) - 1)
    return max(0, jc - w), min(len(g2), jc + w + 1)


def segment_motion(frames_dlh: List[np.ndarray], seed: int = 0) -> dict:
    """Cumulative along-track position s_k (s_0 = 0) of the sensor for consecutive frames, plus the
    consistency check |s(k -> k+2) - (s(k -> k+1) + s(k+1 -> k+2))| on every second pair."""
    struct = []
    step = [0.0]
    for k in range(1, len(frames_dlh)):
        s, c, _ = frame_shift(_dense(frames_dlh[k - 1]), frames_dlh[k], seed=seed + k)
        step.append(s)
    pos = np.cumsum(step)
    chk = []
    for k in range(0, len(frames_dlh) - 2, 2):
        s2, _, _ = frame_shift(_dense(frames_dlh[k]), frames_dlh[k + 2], s_max=6.0, seed=seed + 1000 + k)
        chk.append(abs(s2 - (step[k + 1] + step[k + 2])))
    return {"s": pos, "step": np.array(step), "closure_err": np.array(chk)}


def _dense(xyz_dlh):
    d, l, h = xyz_dlh.T
    m = (d > 2.0) & (d < 50.0) & (np.abs(l) < 4.5) & (h > -1.0)
    return xyz_dlh[m]


def compact_frame(xyz_dlh: np.ndarray, seed: int = 0, cap: int = 60000) -> np.ndarray:
    """Memory-light frame for ``segment_motion*``: only the region the estimator uses (``_dense``), float32,
    at most ``cap`` random points (long segments otherwise need GBs)."""
    p = _dense(np.asarray(xyz_dlh)).astype(np.float32)
    if len(p) > cap:
        p = p[np.random.default_rng(seed).choice(len(p), cap, replace=False)]
    return p.astype(np.float64) if len(p) < 2000 else p


def segment_motion_dp(frames_dlh: List[np.ndarray], seed: int = 0, s_max: float = 3.0, step: float = 0.02,
                      accel_sigma: float = 0.03) -> dict:
    """Like ``segment_motion`` but robust to periodic structure (tunnel ribs alias the shift by ~1 m): the
    per-pair cost curves are combined by a Viterbi pass with a Gaussian prior on the change of the step between
    frames (``accel_sigma`` m/frame, i.e. 3 m/s^2 at 10 Hz).  Emission = (cost - min) / noise, noise = median
    absolute deviation of the curve."""
    if len(frames_dlh) < 2:
        return {"s": np.zeros(len(frames_dlh)), "step": np.zeros(len(frames_dlh)), "closure_err": np.zeros(0),
                "uninformative": np.zeros(0, dtype=np.int64)}
    grid = np.arange(-0.2, s_max + 1e-9, step)
    m = len(grid)
    curves = []
    uninformative = []
    for k in range(1, len(frames_dlh)):
        _, _, c = frame_shift(_dense(frames_dlh[k - 1]), frames_dlh[k], s_max=s_max, step=step, seed=seed + k)
        if not np.all(np.isfinite(c)):
            # a pair without structure returns (empty frame): flat emission, the prior alone bridges it
            uninformative.append(k)
            curves.append(np.zeros(m))
            continue
        noise = max(np.median(np.abs(np.diff(c))) * 3, 1e-4)
        curves.append((c - c.min()) / noise)
    n = len(curves)
    D = (grid[:, None] - grid[None, :]) / accel_sigma
    trans = 0.5 * D * D
    acc = curves[0].copy()
    back = np.zeros((n, m), dtype=np.int64)
    for k in range(1, n):
        tot = acc[:, None] + trans            # from state i (rows) to state j (cols)
        back[k] = np.argmin(tot, axis=0)
        acc = tot[back[k], np.arange(m)] + curves[k]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmin(acc))
    for k in range(n - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    # parabolic refinement around the chosen grid point of each curve (only where it is a local minimum, see _refine)
    ref = np.array([_refine(curves[k], int(j), grid, step) for k, j in enumerate(path)])
    step_all = np.concatenate([[0.0], ref])
    # independent check: two-frame shifts (search restricted to +-0.5 m around the sum of the two steps)
    chk = []
    g2 = np.arange(-0.2, 6.0 + 1e-9, step)
    for k in range(0, len(frames_dlh) - 2, 2):
        exp2 = float(step_all[k + 1] + step_all[k + 2])
        _, _, c2 = frame_shift(_dense(frames_dlh[k]), frames_dlh[k + 2], s_max=6.0, step=step, seed=seed + 1000 + k)
        if not np.all(np.isfinite(c2)):
            chk.append(np.nan)
            continue
        lo, hi = _closure_window(g2, exp2, step)
        s2 = g2[lo + int(np.argmin(c2[lo:hi]))]
        chk.append(abs(s2 - exp2))
    return {"s": np.cumsum(step_all), "step": step_all, "closure_err": np.array(chk),
            "uninformative": np.array(uninformative, dtype=np.int64)}
