"""Geometric model of the track ahead of the train, estimated from one cloud.

Analysis frame: d = along the track (forward), l = lateral, h = up.

* Rail level  h_rail(d) = z0 + g d + c d^2  -- robust polynomial through the
  per-bin median of track-bed returns near the track centre (c bounded by a
  minimum vertical-curve radius).  It absorbs sensor pitch and track grade.
* Sensor lateral offset l0 -- centre of the drainage trough (lowest returns
  between the rails) in the near range; falls back to a configured value.
* Track centre-line  l_c(d) = l0 + a d + b d^2  -- the tunnel is built around
  the track, so the lateral profile of its structure follows the track through
  curves.  Adjacent range bins have almost identical profiles, so the lateral
  shift between neighbouring bins is measured by normalised cross-correlation
  and chained; a bounded-curvature parabola is fitted to the chain.  Chaining
  adjacent bins (rather than matching a single near-range template) survives
  changes of tunnel type (round -> rectangular -> double-track -> station).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import DetectorConfig


@dataclass
class RailLevel:
    z0: float
    grade: float
    curv: float
    n_bins: int
    ok: bool
    residual_rms: float = float("nan")
    max_d_observed: float = 0.0

    def __call__(self, d):
        """Rail level at range d; beyond the observed track bed the curve is extended
        linearly along its tangent (a quadratic must not be extrapolated)."""
        D = self.max_d_observed
        dd = np.minimum(d, D) if D > 0 else d
        base = self.z0 + self.grade * dd + self.curv * dd * dd
        if D > 0:
            beyond = np.maximum(d - D, 0.0)
            return base + (self.grade + 2.0 * self.curv * D) * beyond
        return base


@dataclass
class TrackCurve:
    l0: float
    a: float
    b: float
    ok: bool
    n_used: int = 0
    max_d_supported: float = 0.0
    offset_from_trough: bool = False
    max_d_absolute: float = 0.0   # farthest symmetry / wall / ceiling sample used (chain excluded)
    samples: List[Tuple[float, float, float]] = field(default_factory=list)  # (d, shift, weight)

    def __call__(self, d):
        return self.l0 + self.a * d + self.b * d * d


# ----------------------------------------------------------------------------
def fit_rail_level(d, l, h, cfg: DetectorConfig, l_center=0.0) -> RailLevel:
    """Fit h_rail(d) from per-bin medians of returns below the sensor near the track centre.

    ``l_center`` is either a constant lateral offset or a callable l_c(d) (the track
    curve), so that in curves the window follows the track instead of the wall base."""
    lc = l_center(d) if callable(l_center) else l_center
    m = ((d >= cfg.d_min) & (d <= cfg.floor_fit_d_max) & (np.abs(l - lc) <= cfg.floor_lateral_halfwidth)
         & (h <= cfg.floor_h_max))
    if m.sum() < cfg.floor_min_points * cfg.floor_min_bins:
        return RailLevel(float("nan"), 0.0, 0.0, 0, False)
    dd, hh = d[m], h[m]
    edges = np.arange(cfg.d_min, cfg.floor_fit_d_max + cfg.floor_bin_m, cfg.floor_bin_m)
    idx = np.digitize(dd, edges) - 1
    # per-bin percentile without a Python loop over bins: sort by (bin, height), index at the quantile
    order = np.lexsort((hh, idx))
    idx_s, hh_s = idx[order], hh[order]
    bounds = np.searchsorted(idx_s, np.arange(len(edges)))
    counts = np.diff(bounds)
    centres = 0.5 * (edges[:-1] + edges[1:])
    need = np.where(centres <= cfg.floor_near_d, cfg.floor_min_points, cfg.floor_min_points_far)
    okb = counts >= need
    pos = bounds[:-1][okb] + np.floor((counts[okb] - 1) * cfg.floor_percentile / 100.0).astype(int)
    xs, ys, ws = list(centres[okb]), list(hh_s[pos]), list(np.minimum(counts[okb], 500))
    if len(xs) < cfg.floor_min_bins:
        return RailLevel(float("nan"), 0.0, 0.0, len(xs), False)
    x, y, w = np.array(xs), np.array(ys), np.array(ws, dtype=float)
    # the quadratic (vertical-curve) term is only identifiable with a long observed track bed
    cmax = 1.0 / (2.0 * cfg.floor_min_vertical_radius_m) if x.max() >= cfg.floor_quad_min_range_m else 0.0
    # stage 1: near bins only (dense, reliable) - robust initialisation from the median level,
    # then tighten the residual gate
    near = x <= cfg.floor_near_d
    keep = near & (np.abs(y - np.median(y[near] if near.any() else y)) <= 4 * cfg.floor_residual_m)
    coef = None
    for thr in (2 * cfg.floor_residual_m, cfg.floor_residual_m, cfg.floor_residual_m):
        if keep.sum() < max(2, cfg.floor_min_bins):
            break
        coef = _fit_poly_bounded(x[keep], y[keep], np.sqrt(w[keep]), cmax)
        res = y - _poly(coef, x)
        new_keep = near & (np.abs(res) <= thr)
        if new_keep.sum() < cfg.floor_min_bins:
            break
        keep = new_keep
    if coef is None or keep.sum() < cfg.floor_min_bins:
        return RailLevel(float("nan"), 0.0, 0.0, int(keep.sum()), False)
    # stage 2: sparse far bins are accepted only close to the near-range extrapolation
    # (an object standing on the track must not lift the floor model at its range)
    res = y - _poly(coef, x)
    gate_far = cfg.floor_far_residual_m + cfg.floor_residual_per_m * (x - cfg.floor_near_d)
    far_ok = (~near) & (np.abs(res) <= gate_far)
    # continuity: a far bin is accepted only if the previous accepted bin is close enough
    # (an isolated far bin - e.g. an object standing on the track - must not lever the fit)
    last_d = float(x[keep].max()) if keep.any() else -1e9
    for i in np.flatnonzero(far_ok):
        if x[i] - last_d <= cfg.floor_max_gap_m:
            last_d = float(x[i])
        else:
            far_ok[i] = False
    if far_ok.any():
        keep2 = keep | far_ok
        coef2 = _fit_poly_bounded(x[keep2], y[keep2], np.sqrt(w[keep2]), cmax)
        res2 = y - _poly(coef2, x)
        keep2 = keep2 & (np.abs(res2) <= np.where(near, cfg.floor_residual_m, gate_far))
        if keep2.sum() >= cfg.floor_min_bins:
            coef = _fit_poly_bounded(x[keep2], y[keep2], np.sqrt(w[keep2]), cmax)
            keep = keep2
    res = y[keep] - _poly(coef, x[keep])
    return RailLevel(float(coef[0]), float(coef[1]), float(coef[2]), int(keep.sum()), True,
                     float(np.sqrt(np.mean(res ** 2))), float(x[keep].max()))


def _poly(coef, x):
    return coef[0] + coef[1] * x + coef[2] * x * x


def _fit_poly_bounded(x, y, w, cmax):
    """Weighted LSQ of z0 + g x + c x^2 with |c| <= cmax (refit linear if the bound binds)."""
    if len(x) >= 4 and cmax > 0:
        A = np.column_stack([np.ones_like(x), x, x * x]) * w[:, None]
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        if abs(sol[2]) <= cmax:
            return sol
        c = float(np.clip(sol[2], -cmax, cmax))
        A = np.column_stack([np.ones_like(x), x]) * w[:, None]
        sol2, *_ = np.linalg.lstsq(A, (y - c * x * x) * w, rcond=None)
        return np.array([sol2[0], sol2[1], c])
    A = np.column_stack([np.ones_like(x), x]) * w[:, None]
    sol2, *_ = np.linalg.lstsq(A, y * w, rcond=None)
    return np.array([sol2[0], sol2[1], 0.0])


# ----------------------------------------------------------------------------
def estimate_lateral_offset(d, l, h, rail: RailLevel, cfg: DetectorConfig, hr=None) -> Tuple[float, bool, float]:
    """Centre of the drainage trough (lowest track-bed returns) in the near range -> (l0, ok, n)."""
    d0, d1 = cfg.trough_search_d
    hr = rail(d) if hr is None else hr
    m = (d >= d0) & (d <= d1) & (np.abs(l) <= cfg.trough_search_l_max) & (h <= hr - cfg.trough_depth_min) & (h >= hr - cfg.trough_depth_max)
    n = int(m.sum())
    if n < cfg.trough_min_points:
        return cfg.sensor_lateral_offset, False, float(n)
    ll = l[m]
    # robust centre: median of the returns inside the densest 1.2 m window
    step = 0.1
    edges = np.arange(-cfg.trough_search_l_max, cfg.trough_search_l_max + step, step)
    hist, _ = np.histogram(ll, bins=edges)
    k = int(round(cfg.trough_width_m / step))
    if k < 1 or len(hist) <= k:
        return cfg.sensor_lateral_offset, False, float(n)
    win = np.convolve(hist, np.ones(k), mode="valid")
    j = int(np.argmax(win))
    lo, hi = edges[j], edges[j + k]
    sel = (ll >= lo) & (ll <= hi)
    if sel.sum() < cfg.trough_min_points or win[j] < 0.5 * n:
        return cfg.sensor_lateral_offset, False, float(n)
    return float(np.median(ll[sel])), True, float(n)


# ----------------------------------------------------------------------------
def symmetry_centres(d, l, h, rail: RailLevel, cfg: DetectorConfig, hr=None) -> List[Tuple[float, float, float]]:
    """Axis of symmetry of the track-bed height profile per range bin -> [(d, centre, weight)].

    Two rails and the drainage trough form a profile that is mirror-symmetric
    about the track centre, whereas one-sided equipment (contact rail, cable
    ducts) lies outside the +-1.1 m window and does not bias the estimate.
    """
    out: List[Tuple[float, float, float]] = []
    hr = rail(d) if hr is None else hr
    half, cell = cfg.sym_half_window_m, cfg.sym_cell_m
    span = cfg.sym_center_range_m + half + 0.1
    base = (np.abs(l) <= span) & (h <= hr + 0.5) & (h >= hr - 1.0)
    edges_l = np.arange(-span, span + cell, cell)
    centres_l = 0.5 * (edges_l[:-1] + edges_l[1:])
    us = np.arange(0.1, half, cell)
    cands = np.arange(-cfg.sym_center_range_m, cfg.sym_center_range_m + 1e-6, 0.02)
    d_edges = np.asarray(cfg.sym_d_edges, dtype=float)
    for k in range(len(d_edges) - 1):
        m = base & (d >= d_edges[k]) & (d < d_edges[k + 1])
        n = int(m.sum())
        if n < cfg.sym_min_points:
            continue
        ll, hh = l[m], (h - hr)[m]
        idx = np.digitize(ll, edges_l) - 1
        # per-cell percentile without a Python loop: sort by (cell, height) and index at the quantile
        order = np.lexsort((hh, idx))
        idx_s, hh_s = idx[order], hh[order]
        bounds = np.searchsorted(idx_s, np.arange(len(centres_l) + 1))
        counts = np.diff(bounds)
        prof = np.full(len(centres_l), np.nan)
        okc = counts >= 3
        pos = bounds[:-1][okc] + np.floor((counts[okc] - 1) * cfg.sym_percentile / 100.0).astype(int)
        prof[okc] = hh_s[pos]
        # evaluate all candidate axes at once on a 0.01 m resampled profile (NaN where no data)
        fine = np.arange(-span, span + 0.005, 0.01)
        valid_cells = np.isfinite(prof)
        if valid_cells.sum() < 4:
            continue
        pf = np.interp(fine, centres_l[valid_cells], prof[valid_cells])
        # mark fine samples without a valid cell nearby as NaN
        cell_ok = np.interp(fine, centres_l, valid_cells.astype(float)) > 0.99
        pf[~cell_ok] = np.nan
        ic = np.round((cands + span) / 0.01).astype(int)[:, None]
        iu = np.round(us / 0.01).astype(int)[None, :]
        ip, im = np.clip(ic + iu, 0, len(fine) - 1), np.clip(ic - iu, 0, len(fine) - 1)
        hp, hm = pf[ip], pf[im]
        okm = np.isfinite(hp) & np.isfinite(hm)
        n_ok = okm.sum(axis=1)
        diff = np.where(okm, np.abs(hp - hm), 0.0).sum(axis=1)
        scores = np.where(n_ok >= 0.7 * len(us), diff / np.maximum(n_ok, 1), np.inf)
        j = int(np.argmin(scores))
        best, best_c = float(scores[j]), float(cands[j])
        if not np.isfinite(best) or best > cfg.sym_max_score:
            continue
        w = min(4.0, (cfg.sym_ref_score / max(best, 0.005)) ** 2) * min(1.0, n / 500.0)
        out.append((0.5 * (d_edges[k] + d_edges[k + 1]), best_c, w))
    return out


# ----------------------------------------------------------------------------
def wall_centres(d, l, h, rail: RailLevel, predict, cfg: DetectorConfig, hr=None) -> List[Tuple[float, float, float]]:
    """Absolute track-centre samples from the tunnel cross-section -> [(d, centre, weight)].

    Two cues per range bin:
    * walls: centre of a free lateral interval bounded by structure on both
      sides in the band 1.5-3.0 m above rail level (single-track walls, the
      column row of a double-track tunnel, gate frames);
    * ceiling: lateral centre of the ceiling returns (3.0-5.0 m above rail).
    Only intervals of plausible width are accepted.  Bins are visited from
    near to far and each prediction is the local extrapolation of the samples
    accepted so far (``predict`` seeds the near range), so strong curves are
    followed step by step while jumps to unrelated structure (stations,
    junctions, gaps between columns) fall outside the gate and are skipped.
    """
    hr = rail(d) if hr is None else hr
    dh = h - hr
    d_edges = np.asarray(cfg.profile_d_edges, dtype=float)
    cell = cfg.profile_bin_m
    l_edges = np.arange(-cfg.profile_l_max, cfg.profile_l_max + cell * 0.5, cell)
    centres_l = 0.5 * (l_edges[:-1] + l_edges[1:])
    d_centres = 0.5 * (d_edges[:-1] + d_edges[1:])
    kernel = np.ones(3)
    wmin, wmax = cfg.wall_width_range
    # ---- candidate intervals per bin (walls)
    lo, hi = cfg.wall_band_above_rail
    band = (dh >= lo) & (dh <= hi) & (np.abs(l) <= cfg.profile_l_max) & (d >= cfg.d_min) & (d <= cfg.d_max)
    H, _, _ = np.histogram2d(d[band], l[band], bins=[d_edges, l_edges])
    per_bin: List[List[Tuple[float, float, str]]] = [[] for _ in range(len(d_centres))]  # (centre, strength, kind)
    for k in range(H.shape[0]):
        row = H[k]
        n = float(row.sum())
        if n < cfg.wall_min_points:
            continue
        sm = np.convolve(row, kernel, mode="same")
        thr = max(cfg.wall_occ_min, cfg.wall_occ_frac * n)
        occ = sm >= thr
        j = 0
        while j < len(occ):
            if occ[j]:
                j += 1
                continue
            j_start = j
            while j < len(occ) and not occ[j]:
                j += 1
            j_end = j
            if j_start == 0 or j_end >= len(occ):
                continue
            L, R = centres_l[j_start - 1], centres_l[j_end]
            width = R - L
            if wmin <= width <= wmax:
                strength = min(sm[j_start - 1], sm[j_end]) / thr
                per_bin[k].append((0.5 * (L + R), cfg.wall_weight * min(1.0, n / 100.0) * min(2.0, strength), "wall"))
    # ---- candidate intervals per bin (ceiling)
    lo, hi = cfg.ceiling_band_above_rail
    band = (dh >= lo) & (dh <= hi) & (np.abs(l) <= cfg.profile_l_max) & (d >= cfg.d_min) & (d <= cfg.d_max)
    if band.any():
        dd, ll = d[band], l[band]
        idx = np.digitize(dd, d_edges) - 1
        for k in np.unique(idx):
            if k < 0 or k >= len(d_centres):
                continue
            sel = idx == k
            n = int(sel.sum())
            if n < cfg.ceiling_centre_min_points:
                continue
            L, R = np.percentile(ll[sel], [5, 95])
            if wmin <= (R - L) <= wmax:
                per_bin[k].append((0.5 * float(L + R), cfg.ceiling_weight * min(1.0, n / 60.0), "ceiling"))
    # ---- sequential acceptance with local extrapolation
    out: List[Tuple[float, float, float]] = []
    accepted: List[Tuple[float, float]] = []  # (d, centre)
    for k, dc in enumerate(d_centres):
        if not per_bin[k]:
            continue
        if len(accepted) >= 2:
            recent = np.array(accepted[-4:])
            slope = np.polyfit(recent[:, 0], recent[:, 1], 1)[0] if len(recent) >= 2 else 0.0
            d_last, c_last = accepted[-1]
            gap = dc - d_last
            c0 = c_last + slope * gap
            gate = cfg.wall_track_gate_base_m + gap * (dc / cfg.curve_min_radius_m)
        else:
            c0 = float(predict(dc))
            gate = cfg.wall_prior_gate_base_m + cfg.wall_prior_gate_per_m * dc
        best = None
        for c, w, kind in per_bin[k]:
            dev = abs(c - c0)
            if dev <= gate and (best is None or dev < best[0]):
                best = (dev, c, w)
        if best is None:
            continue
        accepted.append((float(dc), float(best[1])))
        out.append((float(dc), float(best[1]), float(best[2]) / (1.0 + dc / cfg.wall_weight_decay_d)))
    return out


# ----------------------------------------------------------------------------
def estimate_track_curve(d, l, h, rail: RailLevel, trough: Tuple[float, bool], cfg: DetectorConfig,
                         prev: Optional[TrackCurve] = None, hr=None) -> TrackCurve:
    """Estimate l_c(d) = l0 + a d + b d^2 from track-bed symmetry (absolute, near range),
    chained profile shifts (relative, all ranges) and the trough centre (absolute prior)."""
    lo, hi = cfg.profile_band_above_rail
    hr = rail(d) if hr is None else hr
    band = (h >= hr + lo) & (h <= hr + hi) & (np.abs(l) <= cfg.profile_l_max) & (d >= cfg.d_min) & (d <= cfg.d_max)
    d_edges = np.asarray(cfg.profile_d_edges, dtype=float)
    l_edges = np.arange(-cfg.profile_l_max, cfg.profile_l_max + cfg.profile_bin_m * 0.5, cfg.profile_bin_m)
    H, _, _ = np.histogram2d(d[band], l[band], bins=[d_edges, l_edges])
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    counts = H.sum(axis=1)
    L = H.shape[1]
    pred_a, pred_b = (prev.a, prev.b) if (prev is not None and prev.ok) else (0.0, 0.0)
    chain: List[Tuple[float, float, float]] = []   # (d, shift relative to anchor, weight)
    shift = 0.0
    ref_row, ref_d = None, None
    d_anchor = None
    for r in range(H.shape[0]):
        row = H[r]
        n = float(counts[r])
        dc = float(d_centers[r])
        if n < cfg.profile_min_points:
            continue
        if ref_row is None:
            ref_row, ref_d, d_anchor = row, dc, dc
            chain.append((dc, 0.0, 1.0))
            continue
        gap = dc - ref_d
        smax = cfg.chain_shift_slack_m + gap * (dc / cfg.curve_min_radius_m)
        K = max(1, int(smax / cfg.profile_bin_m))
        rn, tn = float(np.linalg.norm(row)), float(np.linalg.norm(ref_row))
        corr_ok = False
        if rn > 0 and tn > 0:
            full = np.correlate(row / rn, ref_row / tn, mode="full")
            lo_i, hi_i = max(0, L - 1 - K), min(len(full), L + K)
            seg = full[lo_i:hi_i]
            if len(seg):
                j = int(np.argmax(seg))
                corr = float(seg[j])
                if corr >= cfg.chain_min_corr:
                    shift += ((lo_i + j) - (L - 1)) * cfg.profile_bin_m
                    chain.append((dc, shift, corr * corr * min(1.0, n / 150.0)))
                    corr_ok = True
        if not corr_ok:
            shift += (pred_a * (dc - ref_d) + pred_b * (dc * dc - ref_d * ref_d))
        ref_row, ref_d = row, dc
    sym = symmetry_centres(d, l, h, rail, cfg, hr=hr)
    l_trough, trough_ok = trough
    default_l0 = l_trough if trough_ok else cfg.sensor_lateral_offset

    def joint_fit(walls, use_chain=True, linear=False):
        rows, rhs, wts, gates, kinds = [], [], [], [], []
        for dc, c, w in sym:
            rows.append([1.0, dc, dc * dc]); rhs.append(c); wts.append(w); gates.append(cfg.sym_residual_base_m + cfg.curve_residual_per_m * dc); kinds.append("sym")
        for dc, c, w in walls:
            rows.append([1.0, dc, dc * dc]); rhs.append(c); wts.append(w); gates.append(cfg.wall_residual_base_m + cfg.curve_residual_per_m * dc); kinds.append("wall")
        last_abs = max([dc for dc, _, _ in sym] + [dc for dc, _, _ in walls] + [0.0])
        if d_anchor is not None and use_chain:
            for dc, sh, w in chain:
                if dc <= min(cfg.curve_fit_d_max, last_abs + cfg.chain_extra_range_m) and dc != d_anchor:
                    rows.append([0.0, dc - d_anchor, dc * dc - d_anchor * d_anchor]); rhs.append(sh); wts.append(w * cfg.chain_weight)
                    gates.append(cfg.curve_residual_base_m + cfg.curve_residual_per_m * dc); kinds.append("chain")
        if trough_ok:
            dt = 0.5 * (cfg.trough_search_d[0] + cfg.trough_search_d[1])
            rows.append([1.0, dt, dt * dt]); rhs.append(l_trough); wts.append(cfg.trough_weight); gates.append(0.4); kinds.append("trough")
        # priors: yaw and curvature change slowly from frame to frame (expressed as lateral offsets at 100 m)
        a_prev = prev.a if (prev is not None and prev.ok) else 0.0
        b_prev_ = prev.b if (prev is not None and prev.ok) else 0.0
        rows.append([0.0, 100.0, 0.0]); rhs.append(100.0 * a_prev); wts.append(cfg.yaw_prior_weight); gates.append(1e9); kinds.append("prior")
        if not linear:
            rows.append([0.0, 0.0, 1.0e4]); rhs.append(1.0e4 * b_prev_); wts.append(cfg.curv_prior_weight); gates.append(1e9); kinds.append("prior")
        A_all = np.array(rows, dtype=float).reshape(-1, 3)
        y_all, w_all, g_all = np.array(rhs, dtype=float), np.array(wts, dtype=float), np.array(gates, dtype=float)
        l0_, a_, b_ = default_l0, 0.0, 0.0
        ok_, n_used_, max_d_ = False, 0, 0.0
        bmax = 1.0 / (2.0 * cfg.curve_min_radius_m)
        n_min = 2 if linear else 3
        n_real = sum(1 for k in kinds if k != "prior")
        if n_real >= n_min and any(k not in ("chain", "prior") for k in kinds):
            keep = np.ones(len(rows), bool)
            for _ in range(4):
                if keep.sum() < n_min:
                    break
                sw = np.sqrt(w_all[keep])
                if linear:
                    sol2, *_ = np.linalg.lstsq(A_all[keep][:, :2] * sw[:, None], y_all[keep] * sw, rcond=None)
                    sol = np.array([sol2[0], sol2[1], 0.0])
                else:
                    sol, *_ = np.linalg.lstsq(A_all[keep] * sw[:, None], y_all[keep] * sw, rcond=None)
                l0_ = float(np.clip(sol[0], -cfg.sym_center_range_m, cfg.sym_center_range_m))
                if trough_ok:
                    l0_ = float(np.clip(l0_, l_trough - cfg.trough_l0_gate_m, l_trough + cfg.trough_l0_gate_m))
                a_ = float(np.clip(sol[1], -cfg.curve_max_abs_a, cfg.curve_max_abs_a))
                b_ = float(np.clip(sol[2], -bmax, bmax))
                res = np.abs(A_all @ np.array([l0_, a_, b_]) - y_all)
                new_keep = res <= g_all
                if new_keep.sum() < n_min or np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            n_used_ = int(sum(1 for i in range(len(rows)) if keep[i] and kinds[i] != "prior"))
            ok_ = n_used_ >= n_min
            if ok_:
                ds = [A_all[i, 1] + (d_anchor if kinds[i] == "chain" else 0.0) for i in range(len(rows)) if keep[i] and kinds[i] != "prior"]
                max_d_ = float(max(ds)) if ds else 0.0
        max_abs_ = 0.0
        if ok_:
            ds_abs = [A_all[i, 1] for i in range(len(rows)) if keep[i] and kinds[i] in ("sym", "wall")]
            max_abs_ = float(max(ds_abs)) if ds_abs else 0.0
        return l0_, a_, b_, ok_, n_used_, max_d_, max_abs_

    # first pass: a straight line (l0, a) through the absolute near-range cues (symmetry + trough).
    # Curvature is not observable within ~35 m, so it is taken from the previous frame for the
    # prediction that gates the far-range wall/ceiling samples; those samples then refine it.
    b_prev = prev.b if (prev is not None and prev.ok) else 0.0
    l0, a, b, ok, n_used, max_d, max_abs = joint_fit([], use_chain=False, linear=True)
    if ok:
        first = (l0, a, b_prev)
        predict = lambda dd: first[0] + first[1] * dd + first[2] * dd * dd
    elif prev is not None and prev.ok:
        predict = prev
    else:
        predict = lambda dd: default_l0 + 0.0 * dd
    walls = wall_centres(d, l, h, rail, predict, cfg, hr=hr)
    l0, a, b, ok, n_used, max_d, max_abs = joint_fit(walls, use_chain=True)
    if not ok and prev is not None and prev.ok:
        l0, a, b, ok, max_d, max_abs = prev.l0, prev.a, prev.b, True, prev.max_d_supported, prev.max_d_absolute
    elif ok and prev is not None and prev.ok and cfg.curve_smoothing > 0:
        wgt = cfg.curve_smoothing
        l0 = wgt * prev.l0 + (1 - wgt) * l0
        a = wgt * prev.a + (1 - wgt) * a
        b = wgt * prev.b + (1 - wgt) * b
    samples = [(dc, c, w) for dc, c, w in sym] + [(dc, c, w) for dc, c, w in walls] + [(dc, sh, -w) for dc, sh, w in chain]
    return TrackCurve(l0=l0, a=a, b=b, ok=ok, n_used=n_used, max_d_supported=max_d, offset_from_trough=bool(trough_ok),
                      max_d_absolute=max_abs, samples=samples)
