"""Single-frame geometric obstacle detector.

Pipeline (see docs/algorithm.md):
 1. validate the cloud (finite, non-zero returns) and move it into the analysis
    frame d/l/h (along track, lateral, up);
 2. fit the rail-level line h_rail(d) and find the sensor's lateral offset;
 3. estimate the track centre-line l_c(d) from the tunnel profile;
 4. select returns inside the clearance box that follows the track;
 5. cluster them with a range-dependent radius and reject clusters that look
    like infrastructure (elongated along the track, attached to structure at
    the box edge, too few points);
 6. report the remaining clusters with their nearest-surface distance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from .config import DetectorConfig
from .track_model import RailLevel, TrackCurve, estimate_lateral_offset, estimate_track_curve, fit_rail_level


@dataclass
class Candidate:
    """A cluster of returns inside the clearance box."""

    distance_m: float           # Euclidean range from the sensor origin to the nearest reliable surface
    forward_m: float            # d of the nearest surface
    lateral_m: float            # cluster centre lateral position (analysis frame)
    height_m: float             # cluster centre height (analysis frame)
    center_sensor: np.ndarray   # (3,) centre in the sensor frame
    extent: np.ndarray          # (3,) size along d, l, h
    n_points: int
    lateral_offset_from_track: float
    height_above_rail: float
    confidence: float = 0.0
    track_id: int = -1
    confirmed: bool = False
    rejected_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance_m": round(float(self.distance_m), 3),
            "forward_m": round(float(self.forward_m), 3),
            "lateral_m": round(float(self.lateral_m), 3),
            "height_m": round(float(self.height_m), 3),
            "center_xyz": [round(float(v), 3) for v in self.center_sensor],
            "extent_dlh": [round(float(v), 3) for v in self.extent],
            "n_points": int(self.n_points),
            "lateral_offset_from_track": round(float(self.lateral_offset_from_track), 3),
            "height_above_rail": round(float(self.height_above_rail), 3),
            "confidence": round(float(self.confidence), 3),
            "track_id": int(self.track_id),
            "confirmed": bool(self.confirmed),
        }


@dataclass
class FrameResult:
    status: str                       # 'clear' | 'detected' | 'unknown' (before temporal filtering: raw)
    reason: str = ""
    candidates: List[Candidate] = field(default_factory=list)
    rejected: List[Candidate] = field(default_factory=list)
    rail: Optional[RailLevel] = None
    curve: Optional[TrackCurve] = None
    n_total: int = 0
    n_valid: int = 0
    n_zero: int = 0
    n_nonfinite: int = 0
    n_roi: int = 0
    n_box: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    box_points_sensor: Optional[np.ndarray] = None  # candidate returns (sensor frame) for debug output

    @property
    def nearest(self) -> Optional[Candidate]:
        return min(self.candidates, key=lambda c: c.distance_m) if self.candidates else None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "status_raw": self.status,
            "reason": self.reason,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_zero": self.n_zero,
            "n_nonfinite": self.n_nonfinite,
            "n_roi": self.n_roi,
            "n_box": self.n_box,
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [dict(c.to_dict(), reason=c.rejected_reason) for c in self.rejected],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }
        if self.rail is not None:
            out["rail"] = {"z0": round(self.rail.z0, 3) if math.isfinite(self.rail.z0) else None,
                           "grade": round(float(self.rail.grade), 5), "curv": round(float(self.rail.curv), 7),
                           "n_bins": int(self.rail.n_bins), "ok": bool(self.rail.ok),
                           "max_d_observed": round(float(self.rail.max_d_observed), 1)}
        if self.curve is not None:
            out["curve"] = {"l0": round(float(self.curve.l0), 3), "a": round(float(self.curve.a), 5), "b": round(float(self.curve.b), 7),
                            "ok": bool(self.curve.ok), "n_used": int(self.curve.n_used),
                            "max_d_supported": round(float(self.curve.max_d_supported), 1),
                            "max_d_absolute": round(float(self.curve.max_d_absolute), 1),
                            "offset_from_trough": bool(self.curve.offset_from_trough)}
        return out


class ObstacleDetector:
    """Stateless per-frame geometry (a little state for curve smoothing only)."""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.R = self.cfg.axis_matrix()
        self.Rinv = self.R.T
        # each analysis axis is a signed sensor axis: (index, sign) per row of R
        self._perm_idx = [int(np.argmax(np.abs(self.R[i]))) for i in range(3)]
        self._perm_sign = [np.float32(np.sign(self.R[i, self._perm_idx[i]])) for i in range(3)]
        self._prev_curve: Optional[TrackCurve] = None

    def reset(self) -> None:
        self._prev_curve = None

    # ------------------------------------------------------------------
    def process(self, xyz: np.ndarray) -> FrameResult:
        cfg = self.cfg
        t = time.perf_counter()
        timings: Dict[str, float] = {}
        res = FrameResult(status="unknown")
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        res.n_total = int(len(xyz))
        # column-wise tests (row reductions over (N,3) are very slow in older NumPy)
        cols = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        x0, y0, z0 = cols
        finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)
        zero = (x0 == 0) & (y0 == 0) & (z0 == 0)
        valid = finite & ~zero
        res.n_nonfinite = int((~finite).sum())
        res.n_zero = int(zero.sum())
        res.n_valid = int(valid.sum())
        if res.n_valid < cfg.min_valid_points:
            res.reason = f"too few valid points ({res.n_valid})"
            timings["total"] = (time.perf_counter() - t) * 1e3
            res.timings_ms = timings
            return res
        # analysis frame: axis specs are signed permutations, so no matrix product is needed.
        # The ROI is tested on the full columns and all arrays are gathered once with the same
        # indices (no intermediate (N,3) copies; values and order identical to filtering in steps).
        d = self._perm_sign[0] * cols[self._perm_idx[0]]
        l = self._perm_sign[1] * cols[self._perm_idx[1]]
        h = self._perm_sign[2] * cols[self._perm_idx[2]]
        with np.errstate(invalid="ignore"):  # NaN of invalid returns compare False, they are masked anyway
            roi = valid & (d >= cfg.d_min) & (d <= cfg.d_max) & (np.abs(l) <= cfg.l_abs_max) & (h >= cfg.h_abs_min) & (h <= cfg.h_abs_max)
        sel = np.flatnonzero(roi)
        if cfg.max_analysed_points > 0 and len(sel) > cfg.max_analysed_points:
            sel = sel[::-(-len(sel) // cfg.max_analysed_points)]  # every k-th point, deterministic
        d, l, h = d[sel], l[sel], h[sel]
        sensor_pts = xyz[sel]
        res.n_roi = int(len(d))
        timings["prepare"] = (time.perf_counter() - t) * 1e3
        t1 = time.perf_counter()

        # ---- rail level and lateral offset (two passes: offset refines the floor window)
        rail = fit_rail_level(d, l, h, cfg, 0.0)
        l0, l0_ok = cfg.sensor_lateral_offset, False
        if rail.ok:
            hr = rail(d)
            l0, l0_ok, _ = estimate_lateral_offset(d, l, h, rail, cfg, hr=hr)
            if l0_ok and abs(l0) > 0.15:
                rail2 = fit_rail_level(d, l, h, cfg, l0)
                if rail2.ok:
                    rail = rail2
                    hr = rail(d)
        res.rail = rail
        if not rail.ok:
            res.reason = "rail level not observable"
            timings["track"] = (time.perf_counter() - t1) * 1e3
            timings["total"] = (time.perf_counter() - t) * 1e3
            res.timings_ms = timings
            return res
        curve = estimate_track_curve(d, l, h, rail, (l0, bool(l0_ok)), cfg, self._prev_curve, hr=hr)
        self._prev_curve = curve
        res.curve = curve
        # refit the rail level along the estimated track curve (in curves the straight
        # window drifts onto the wall base and bends the floor model)
        if curve.ok and (abs(curve.a) > 0.005 or abs(curve.b) > 2e-5):
            rail2 = fit_rail_level(d, l, h, cfg, curve)
            if rail2.ok:
                rail = rail2
                res.rail = rail
                hr = rail(d)
        timings["track"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()

        # ---- clearance box that follows the track
        lc = curve(d)
        dl = l - lc
        dh = h - hr
        halfwidth = np.where(dh < cfg.low_zone_top_above_rail, cfg.gauge_halfwidth_low, cfg.gauge_halfwidth_body)
        # centre-line uncertainty grows with range and, in curves, with the curvature term itself
        # (the corridor "closes" where the track can no longer be seen around the bend)
        lat_unc = cfg.lateral_shrink_per_m * d + cfg.lateral_curv_factor * abs(curve.b) * d * d
        halfwidth = halfwidth - lat_unc
        closed = halfwidth < cfg.lateral_min_halfwidth
        halfwidth = np.where(closed, -1.0, halfwidth)
        beyond = np.maximum(0.0, d - rail.max_d_observed)
        rms_term = cfg.margin_rms_factor * (rail.residual_rms if math.isfinite(rail.residual_rms) else 0.0)
        bottom = cfg.box_bottom_above_rail + cfg.bottom_margin_per_m * d + cfg.bottom_margin_beyond_per_m * beyond + rms_term
        top = cfg.box_top_above_rail - cfg.top_margin_per_m * d - cfg.top_margin_beyond_per_m * beyond - rms_term
        top = np.minimum(top, self._ceiling_cap(d, dl, dh))
        inbox = (np.abs(dl) <= halfwidth) & (dh >= bottom) & (dh <= top)
        res.n_box = int(inbox.sum())
        timings["box"] = (time.perf_counter() - t2) * 1e3
        t3 = time.perf_counter()
        if res.n_box == 0:
            res.status = "clear"
            timings["cluster"] = 0.0
            timings["total"] = (time.perf_counter() - t) * 1e3
            res.timings_ms = timings
            return res

        # ---- clustering with a range-dependent radius
        # duplicate returns (identical coordinates) are common in these clouds, and dense
        # near-range clutter would make pair search quadratic: keep one return per voxel
        # whose size grows with range (far returns are sparser than the voxel anyway)
        box_idx = np.flatnonzero(inbox)
        vox = cfg.voxel_base_m + cfg.voxel_per_m * d[box_idx]
        keys = np.column_stack([np.floor(d[box_idx] / vox), np.floor(l[box_idx] / vox), np.floor(h[box_idx] / vox)]).astype(np.int64)
        _, uniq = np.unique(keys, axis=0, return_index=True)
        box_idx = box_idx[np.sort(uniq)]
        cd, cl, ch = d[box_idx], l[box_idx], h[box_idx]
        cdl, cdh, ctop = dl[box_idx], dh[box_idx], top[box_idx]
        cpts = np.column_stack([cd, cl, ch])
        labels = self._cluster(cpts, cd)
        # structure outside the box but near it (for the attachment test)
        # structure that may leak into the box: laterally beside it or above it (never the
        # track bed below the box bottom, which belongs to the floor or to the object itself)
        near_out = (~inbox) & (np.abs(dl) <= 4.0) & (dh >= bottom)
        out_pts = np.column_stack([d[near_out], l[near_out], dh[near_out]]) if near_out.any() else None
        out_tree = cKDTree(np.column_stack([d[near_out], l[near_out], h[near_out]])) if near_out.any() else None
        cands: List[Candidate] = []
        rejected: List[Candidate] = []
        accepted_sel = np.zeros(len(labels), bool)
        for lab in np.unique(labels):
            if lab < 0:
                continue
            sel = labels == lab
            n = int(sel.sum())
            pd_, pl, ph = cd[sel], cl[sel], ch[sel]
            rng = np.sqrt(pd_ ** 2 + pl ** 2 + ph ** 2)
            ext = np.array([pd_.max() - pd_.min(), pl.max() - pl.min(), ph.max() - ph.min()])
            d_near = float(np.median(pd_))
            min_pts = cfg.min_points_near if d_near < cfg.far_range_m else cfg.min_points_far
            hw_sel = (np.where(cdh[sel] < cfg.low_zone_top_above_rail, cfg.gauge_halfwidth_low, cfg.gauge_halfwidth_body)
                      - cfg.lateral_shrink_per_m * pd_ - cfg.lateral_curv_factor * abs(curve.b) * pd_ * pd_)
            dist = float(np.percentile(rng, cfg.distance_percentile)) if n >= 10 else float(rng.min())
            center_dlh = np.array([pd_.mean(), pl.mean(), ph.mean()])
            center_sensor = self.R.T @ center_dlh  # (x,y,z) = R^T (d,l,h)
            cand = Candidate(
                distance_m=dist, forward_m=float(pd_.min()), lateral_m=float(center_dlh[1]), height_m=float(center_dlh[2]),
                center_sensor=center_sensor, extent=ext, n_points=n,
                lateral_offset_from_track=float(np.mean(cdl[sel])), height_above_rail=float(np.mean(cdh[sel])),
            )
            reason = ""
            d_limit = curve.max_d_absolute + cfg.report_beyond_support_m
            if n < min_pts:
                reason = f"too few points ({n}<{min_pts})"
            elif d_near > d_limit:
                reason = f"beyond the range supported by the track model ({d_near:.0f} > {d_limit:.0f} m)"
            elif ext[0] > cfg.cluster_max_extent_d:
                reason = f"elongated along track ({ext[0]:.1f} m)"
            elif max(ext[1], ext[2]) < cfg.cluster_min_extent_m and n < 3 * min_pts:
                reason = "no spatial extent"
            elif ext[2] < cfg.low_flat_max_height_m and ext[0] > cfg.low_flat_min_length_m:
                reason = f"low flat structure along the track ({ext[0]:.1f} m long, {ext[2]:.2f} m tall)"
            elif ext[1] < cfg.edge_sliver_max_width_m and np.max(np.abs(cdl[sel])) > np.min(hw_sel) - cfg.edge_sliver_margin_m:
                reason = "thin sliver at the lateral box edge (platform edge / duct)"
            elif d_near >= cfg.far_range_m and ext[2] < cfg.cluster_min_extent_h_far:
                reason = f"single-ring return at range ({ext[2]:.2f} m tall)"
            elif d_near >= cfg.shape_rule_beyond_d and ext[2] < cfg.shape_min_height_ratio * max(ext[0], ext[1]):
                reason = f"flat/elongated patch at range (h {ext[2]:.2f} vs {max(ext[0], ext[1]):.2f} m): floor/ceiling/wall"
            elif d_near >= cfg.shape_rule_beyond_d and out_tree is not None and self._grazing_surface(out_tree, pd_, pl, ph, d_near, cfg):
                reason = "grazing planar surface (wall/floor seen at a shallow angle)"
            elif out_tree is not None and self._extends_above(out_pts, pd_, pl, cdh[sel], cfg):
                reason = "extends above the box top (wall/column/ceiling structure)"
            elif out_tree is not None:
                # attachment test: an edge cluster whose outside neighbourhood (walls, columns,
                # platform, ceiling) holds more returns than the cluster itself is structure
                # leaking into the box (curve / floor-model error), not a compact object
                hw = hw_sel
                top_here = ctop[sel]
                near_edge = (np.abs(cdl[sel]) > hw - cfg.attach_edge_margin_m) | (cdh[sel] > top_here - cfg.attach_edge_margin_m)
                if near_edge.any() or d_near >= cfg.attach_all_beyond_d:
                    r_att = cfg.attach_radius_m + cfg.attach_radius_per_m * d_near
                    neigh = out_tree.query_ball_point(np.column_stack([pd_, pl, ph]), r=r_att)
                    outside = set()
                    for lst in neigh:
                        outside.update(lst)
                    n_out = len(outside)
                    if n_out > cfg.attach_ratio * n:
                        reason = f"attached to structure at box edge ({n_out} outside vs {n} own)"
            if reason:
                cand.rejected_reason = reason
                rejected.append(cand)
            else:
                # confidence: point support relative to what a ~0.5 m wide object yields at this range
                expected = max(3.0, 60.0 / max(1.0, dist / 20.0) ** 2)
                cand.confidence = float(min(1.0, n / expected))
                cands.append(cand)
                accepted_sel |= sel
        cands.sort(key=lambda c: c.distance_m)
        res.candidates = cands
        res.rejected = rejected
        res.status = "detected" if cands else "clear"
        if cfg.debug_cloud:
            # returns of the accepted candidates only (rejected structure is in the JSON log)
            res.box_points_sensor = sensor_pts[box_idx[accepted_sel]]
        timings["cluster"] = (time.perf_counter() - t3) * 1e3
        timings["total"] = (time.perf_counter() - t) * 1e3
        res.timings_ms = timings
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _grazing_surface(out_tree, pd_, pl, ph, d_near, cfg) -> bool:
        """PCA of the cluster plus its neighbourhood: a thin plane whose normal is nearly
        perpendicular to the line of sight is a wall or floor patch seen at a shallow angle."""
        r = cfg.attach_radius_m + cfg.attach_radius_per_m * d_near
        own = np.column_stack([pd_, pl, ph])
        neigh = set()
        for lst in out_tree.query_ball_point(own, r=r):
            neigh.update(lst)
        pts = own if not neigh else np.vstack([own, out_tree.data[sorted(neigh)]])
        if len(pts) < 6:
            return False
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        evals, evecs = np.linalg.eigh(cov)
        planarity = float(evals[0] / max(evals.sum(), 1e-9))
        normal = evecs[:, 0]
        ray = c / max(np.linalg.norm(c), 1e-9)
        return planarity < cfg.grazing_planarity_max and abs(float(normal @ ray)) < cfg.grazing_cos_max

    # ------------------------------------------------------------------
    @staticmethod
    def _extends_above(out_pts, pd_, pl, pdh, cfg) -> bool:
        """True when returns continue right above the cluster's own top inside its footprint.

        Walls, columns, gates and hanging cables continue upwards past the box
        top; a person, a bag or a trolley ends in free air.  ``out_pts`` columns
        are (d, l, h_above_rail)."""
        d0, d1 = pd_.min() - cfg.above_footprint_d_m, pd_.max() + cfg.above_footprint_d_m
        l0, l1 = pl.min() - cfg.above_footprint_l_m, pl.max() + cfg.above_footprint_l_m
        top = float(np.max(pdh)) + 0.1
        m = ((out_pts[:, 0] >= d0) & (out_pts[:, 0] <= d1) & (out_pts[:, 1] >= l0) & (out_pts[:, 1] <= l1)
             & (out_pts[:, 2] > top) & (out_pts[:, 2] <= top + cfg.above_band_m))
        return int(m.sum()) >= cfg.above_min_points

    # ------------------------------------------------------------------
    def _ceiling_cap(self, d: np.ndarray, dl: np.ndarray, dh: np.ndarray) -> np.ndarray:
        """Per-point upper bound: observed ceiling height (above rail) minus a margin.

        Guards against the rail-level extrapolation error at long range, where the
        tunnel ceiling would otherwise drift into the box."""
        cfg = self.cfg
        edges = np.asarray(cfg.profile_d_edges, dtype=float)
        cap = np.full(len(d), np.inf)
        near_axis = np.abs(dl) <= 2.0
        idx = np.digitize(d, edges) - 1
        valid = (idx >= 0) & (idx < len(edges) - 1)
        high = near_axis & (dh >= cfg.ceiling_min_height_m) & valid
        if not high.any():
            return cap
        hi_idx, hi_dh = idx[high], dh[high]
        order = np.lexsort((hi_dh, hi_idx))
        hi_idx, hi_dh = hi_idx[order], hi_dh[order]
        nb = len(edges) - 1
        bounds = np.searchsorted(hi_idx, np.arange(nb + 1))
        counts = np.diff(bounds)
        okb = counts >= cfg.ceiling_min_points
        pos = bounds[:-1][okb] + np.floor((counts[okb] - 1) * cfg.ceiling_percentile / 100.0).astype(int)
        cap_per_bin = np.full(nb, np.inf)
        cap_per_bin[okb] = hi_dh[pos] - cfg.ceiling_margin_m
        cap[valid] = cap_per_bin[idx[valid]]
        return cap

    # ------------------------------------------------------------------
    def _cluster(self, pts: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Connected components with a range-dependent linking radius (union-find)."""
        cfg = self.cfg
        n = len(pts)
        if n == 0:
            return np.zeros(0, dtype=int)
        eps = cfg.cluster_eps_base_m + cfg.cluster_eps_per_m * d
        r_max = float(eps.max())  # eps of the farthest candidate actually present
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r_max, output_type="ndarray")
        parent = np.arange(n)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        if len(pairs):
            pv = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
            thr = np.maximum(eps[pairs[:, 0]], eps[pairs[:, 1]])
            for i, j in pairs[pv <= thr]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
        roots = np.array([find(i) for i in range(n)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels
