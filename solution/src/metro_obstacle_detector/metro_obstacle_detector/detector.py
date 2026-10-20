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

from .config import DetectorConfig, curv_factor
from .track_model import curvature_shrink, median_model, wall_sweep_limit, RailLevel, TrackCurve, estimate_lateral_offset, estimate_track_curve, extend_rail_by_ceiling, fit_rail_level


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
    weak: bool = False          # rejected only for its point support / shape at range (see far_weak_d1_vote)
    compact: bool = False       # small compact cluster rejected only for its support or as an edge sliver (see compact_pair_vote)
    plate: bool = False         # accepted only by the plate exception: votes only with a free-space D2 group (see plate_d2_min_bg_frac)
    straddle: bool = False      # compact only through straddle_compact_exempt (part of the object lies beyond the box edge, see straddle_min_bg_frac)

    def to_dict(self) -> Dict[str, Any]:
        d = {
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
        if self.weak:
            d["weak"] = True
        if self.compact:
            d["compact"] = True
        if self.plate:
            d["plate"] = True
        if self.straddle:
            d["straddle"] = True
        return d


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
    d2: List[Any] = field(default_factory=list)      # DetCandidate list of the beam detector (D2)
    d3: List[Any] = field(default_factory=list)      # DetCandidate list of the neural detector (D3)
    fusion: Dict[str, Any] = field(default_factory=dict)  # track summary of the fusion stage
    held_model: bool = False                         # rail/curve of a previous frame were used (fit failed or static scene)
    static_frac: float = 0.0                         # fraction of unchanged range-image pixels vs the previous frame
    verified_to_m: float = float("nan")              # range up to which the corridor is built and checked
    bend_limit_m: float = float("inf")               # a wall sweeps into the corridor here (bend not followed by the fit)
    bend_info: Dict[str, Any] = field(default_factory=dict)
    fit: Any = None                                  # this frame's own (rail, curve) fit before smoothing
    smoothed: bool = False                           # rail/curve are the median of recent fits (quasi-static scene)

    @property
    def nearest(self) -> Optional[Candidate]:
        return min(self.candidates, key=lambda c: c.distance_m) if self.candidates else None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "status_raw": self.status,
            "bend_limit_m": None if not math.isfinite(self.bend_limit_m) else round(float(self.bend_limit_m), 1),
            "smoothed": self.smoothed,
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
            "d2": [c.to_dict() for c in self.d2],
            "d3": [c.to_dict() for c in self.d3],
            "fusion": self.fusion,
            "held_model": self.held_model,
            "static_frac": round(float(self.static_frac), 3),
            "verified_to_m": None if not math.isfinite(self.verified_to_m) else round(float(self.verified_to_m), 1),
        }
        if self.rail is not None:
            out["rail"] = {"z0": round(self.rail.z0, 3) if math.isfinite(self.rail.z0) else None,
                           "grade": round(float(self.rail.grade), 5), "curv": round(float(self.rail.curv), 7),
                           "n_bins": int(self.rail.n_bins), "ok": bool(self.rail.ok),
                           "max_d_observed": round(float(self.rail.max_d_observed), 1),
                           "ceiling_ext": ({"height": round(float(self.rail.height), 2), "d_switch": round(float(self.rail.d_switch), 1),
                                            "far_max_d": round(float(self.rail.far_max_d), 1), "n_far_bins": int(self.rail.n_far_bins)}
                                           if hasattr(self.rail, "far_coef") else None)}
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
        self._extended_prev = False

    def reset(self) -> None:
        self._prev_curve = None
        self._extended_prev = False

    def _straight(self, curve) -> bool:
        """Straight section for the ceiling-based support extension, with hysteresis on |b|."""
        thr = self.cfg.straight_keep_abs_b if self._extended_prev else self.cfg.straight_max_abs_b
        return bool(curve.ok and abs(curve.b) < thr)

    def _curv_factor(self, b: float) -> float:
        return curv_factor(self.cfg, b)

    # ------------------------------------------------------------------
    def process(self, xyz: np.ndarray, fixed_model=None, smooth_with=None) -> FrameResult:
        """Single-frame geometry.  ``fixed_model`` = (rail, curve) skips the fits and uses that model
        (the pipeline passes the previous model while the scene is static).  ``smooth_with`` = list of
        recent (rail, curve) fits: this frame's fit is replaced by the element-wise median of them and
        itself before the clearance box is built (quasi-static scenes; the fit itself is kept in
        ``res.fit`` for the pipeline's history)."""
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
        extended_now = False
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

        if fixed_model is not None:
            rail, curve = fixed_model
            hr = rail(d)
            self._prev_curve = curve
            res.rail, res.curve, res.held_model = rail, curve, True
            far_fixed = float(getattr(rail, "far_max_d", 0.0) or 0.0)
            extended_now = bool(curve.ok and far_fixed > 0.0 and curve.max_d_absolute >= far_fixed - 1e-6)
        else:
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
            # far rail level from the ceiling profile (dark / grazing bed, vertical curves)
            rail3 = extend_rail_by_ceiling(d, l, h, rail, curve(d) if curve.ok else np.zeros_like(d), cfg)
            if rail3 is not rail:
                rail = rail3
                res.rail = rail
                hr = rail(d)
                # in a straight section the centre-line is a line: with the vertical profile carried by the
                # ceiling, the corridor is supported as far as the ceiling is seen (curves keep the stricter
                # wall/ceiling-centre support, which is where the lateral uncertainty lives)
                if self._straight(curve) and rail.far_max_d > curve.max_d_absolute:
                    curve.max_d_absolute = float(rail.far_max_d)
                    curve.max_d_supported = max(curve.max_d_supported, float(rail.far_max_d))
                    extended_now = True
        res.fit = (rail, curve)
        if smooth_with and fixed_model is None and curve.ok:
            rail, curve = median_model(list(smooth_with) + [(rail, curve)], straight_keep_abs_b=cfg.straight_keep_abs_b)
            res.rail, res.curve = rail, curve
            hr = rail(d)
            res.smoothed = True
            # the straight-section support rule once more on the median curvature (a per-frame |b| hovering
            # around the threshold flipped the support of a standing train between 81 and 141 m)
            far_reach = float(getattr(rail, "far_max_d", 0.0) or 0.0)
            if far_reach > 0.0:
                if self._straight(curve) and far_reach > curve.max_d_absolute:
                    curve.max_d_absolute = far_reach
                    curve.max_d_supported = max(curve.max_d_supported, far_reach)
                    extended_now = True
                if curve.max_d_absolute >= far_reach - 1e-6:
                    extended_now = True      # the median carried the extension: keep the hysteresis engaged
        # a tunnel wall sweeping into the corridor beyond the near range = a bend the centre-line did not follow:
        # the corridor is not trusted from there on (the pipeline caps the verified range with it)
        # (the model itself is not touched: a frozen static-scene model must not carry a cap of its warm-up
        # frames through a whole stop; the check runs on every frame's returns)
        if cfg.wall_sweep_check and curve.ok:
            lim, info = wall_sweep_limit(d, l - curve(d), h - hr, cfg)
            if math.isfinite(lim):
                res.bend_limit_m, res.bend_info = float(lim), info
        timings["track"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()

        # ---- clearance box that follows the track
        lc = curve(d)
        dl = l - lc
        dh = h - hr
        halfwidth = np.where(dh < cfg.low_zone_top_above_rail, cfg.gauge_halfwidth_low, cfg.gauge_halfwidth_body)
        # centre-line uncertainty grows with range and, in curves, with the curvature term itself
        # (the corridor "closes" where the track can no longer be seen around the bend)
        lat_unc = cfg.lateral_shrink_per_m * d + curvature_shrink(cfg, curve, d)
        self._extended_prev = extended_now     # the hysteresis only follows the rule's own decision
        halfwidth = halfwidth - lat_unc
        closed = halfwidth < cfg.lateral_min_halfwidth
        halfwidth = np.where(closed, -1.0, halfwidth)
        bed_max = getattr(rail, "bed_max_d", rail.max_d_observed)
        beyond_bed = np.maximum(0.0, d - bed_max)
        beyond = np.maximum(0.0, d - rail.max_d_observed)        # beyond the ceiling-covered range too
        rms_term = cfg.margin_rms_factor * (rail.residual_rms if math.isfinite(rail.residual_rms) else 0.0)
        bottom = (cfg.box_bottom_above_rail + cfg.bottom_margin_per_m * d + cfg.bottom_margin_ext_per_m * (beyond_bed - beyond)
                  + cfg.bottom_margin_beyond_per_m * beyond + rms_term)
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
                      - cfg.lateral_shrink_per_m * pd_ - curvature_shrink(cfg, curve, pd_))
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
            # a face across the corridor (wide, thin along the track, centred): a train standing ahead, a closed
            # gate, a fallen panel - it may well reach the ceiling and have nothing behind it, and it is an obstacle
            face_across = (ext[1] >= cfg.face_min_width_m and ext[0] <= cfg.face_max_depth_m and n >= cfg.face_min_points
                           and ext[2] >= cfg.face_min_height_m and float(cdh[sel].min()) <= cfg.face_max_bottom_m
                           and abs(float(np.mean(cdl[sel]))) <= cfg.face_max_center_offset_m)
            # a plate across the corridor (the organisers' 1.3 m wide sheet 0.5 m above the rails): seen edge-on from
            # afar it is as flat as a floor patch, but it is short along the track, centred and clearly above the bed
            plate_across = (cfg.d1_plate_exception and ext[1] >= cfg.plate_min_width_m and ext[0] <= cfg.plate_max_depth_m
                            and n >= cfg.plate_min_points and abs(float(np.mean(cdl[sel]))) <= cfg.plate_max_center_offset_m
                            and float(np.mean(cdh[sel])) <= cfg.plate_max_height_above_rail_m)
            # a small compact cluster (a 0.3 m box): short along the track and not a line in either cross direction;
            # at the corridor edge such a cluster is not a platform edge or a duct, which run along the track
            short_compact = (ext[0] <= cfg.compact_sliver_max_length_m
                             and min(ext[1], ext[2]) >= cfg.compact_min_thickness_m)
            # a cluster across the track axis and thin along it (the organisers' beam over the rails) is in the train's path whatever
            # sticks out of the box beside it; a thin object hanging into the middle of the corridor nearby (their tilted cable) is not a wall
            crosses_axis = (cfg.axis_cross_exempt and float(cdl[sel].min()) <= -cfg.axis_cross_min_m
                            and float(cdl[sel].max()) >= cfg.axis_cross_min_m and ext[0] <= cfg.axis_cross_max_depth_m)
            hanging_central = (cfg.hanging_central_exempt and d_near <= cfg.hanging_central_max_d
                               and float(cdh[sel].min()) >= cfg.hanging_central_min_bottom_m
                               and float(np.max(np.abs(cdl[sel]) - hw_sel)) <= -cfg.hanging_central_edge_m)
            straddle_memo: List[Optional[np.ndarray]] = []

            def straddle_extent() -> Optional[np.ndarray]:
                """Extent (d, l, h) of the compact object the cluster forms with its outside neighbours grown through the
                returns outside the box, or None when they grow into structure (computed once per cluster)."""
                if not straddle_memo:
                    ext_u = None
                    if out_tree is not None:
                        r_att = cfg.attach_radius_m + cfg.attach_radius_per_m * d_near
                        seeds = set()
                        for lst in out_tree.query_ball_point(np.column_stack([pd_, pl, ph]), r=r_att):
                            seeds.update(lst)
                        if seeds:
                            ext_u = self._compact_straddle(out_tree, seeds, pd_, pl, ph, d_near, cfg)
                    straddle_memo.append(ext_u)
                return straddle_memo[0]

            # an edge sliver too thin for short_compact (the top face of a cube whose body lies beyond the low-zone edge)
            # is compact when the object it forms with the returns beyond the edge is compact and thick enough
            short_thin = (cfg.straddle_compact_exempt and cfg.straddle_thin_exempt and ext[0] <= cfg.compact_sliver_max_length_m
                          and max(ext[1], ext[2]) >= cfg.compact_min_thickness_m)

            def thin_straddles() -> bool:
                ext_u = straddle_extent()
                return ext_u is not None and min(float(ext_u[1]), float(ext_u[2])) >= cfg.compact_min_thickness_m

            def judge(weak_pass: bool, compact_pass: bool = False, plate_ok: bool = True, straddle_ok: bool = True) -> str:
                """The structure-rejection rules; ``weak_pass`` skips the rules that judge a cluster by its point
                support and shape (too few points, single ring, flat patch, grazing plane), which a few far
                returns can never satisfy, and keeps every rule that recognises structure.  ``compact_pass`` skips
                only the support rule and, for a short compact cluster, the lateral edge-sliver rule.
                ``plate_ok=False`` judges without the plate exception, ``straddle_ok=False`` without the straddle one."""
                if n < min_pts and not weak_pass and not compact_pass:
                    return f"too few points ({n}<{min_pts})"
                if d_near > d_limit:
                    return f"beyond the range supported by the track model ({d_near:.0f} > {d_limit:.0f} m)"
                if ext[0] > cfg.cluster_max_extent_d:
                    return f"elongated along track ({ext[0]:.1f} m)"
                if (ext[2] < cfg.flat_sliver_max_h_m and max(ext[0], ext[1]) >= cfg.flat_sliver_min_w_m
                        and float(np.mean(cdh[sel])) >= cfg.flat_sliver_min_h_above_rail_m):
                    return f"flat sliver ({ext[2]:.2f} m thick, {max(ext[0], ext[1]):.1f} m wide) hanging high: cable or duct"
                if d_near >= cfg.floating_beyond_d and float(cdh[sel].min()) >= cfg.floating_bottom_above_rail:
                    return f"floating far object (lowest point {float(cdh[sel].min()):.1f} m above rail): hanging infrastructure"
                if max(ext[1], ext[2]) < cfg.cluster_min_extent_m and n < 3 * min_pts:
                    return "no spatial extent"
                if ext[2] < cfg.low_flat_max_height_m and ext[0] > cfg.low_flat_min_length_m:
                    return f"low flat structure along the track ({ext[0]:.1f} m long, {ext[2]:.2f} m tall)"
                if (ext[1] < cfg.edge_sliver_max_width_m and np.max(np.abs(cdl[sel])) > np.min(hw_sel) - cfg.edge_sliver_margin_m
                        and not (compact_pass and short_compact)
                        and not (compact_pass and straddle_ok and short_thin and thin_straddles())):
                    return "thin sliver at the lateral box edge (platform edge / duct)"
                if not weak_pass:
                    if d_near >= cfg.far_range_m and ext[2] < cfg.cluster_min_extent_h_far:
                        return f"single-ring return at range ({ext[2]:.2f} m tall)"
                    if (d_near >= cfg.shape_rule_beyond_d and ext[2] < cfg.shape_min_height_ratio * max(ext[0], ext[1])
                            and not (plate_across and plate_ok)):
                        return f"flat/elongated patch at range (h {ext[2]:.2f} vs {max(ext[0], ext[1]):.2f} m): floor/ceiling/wall"
                    if (d_near >= cfg.shape_rule_beyond_d and out_tree is not None and not (plate_across and plate_ok)
                            and self._grazing_surface(out_tree, pd_, pl, ph, d_near, cfg)):
                        return "grazing planar surface (wall/floor seen at a shallow angle)"
                if (out_tree is not None and not face_across and not hanging_central
                        and self._extends_above(out_pts, pd_, pl, cdh[sel], cfg)):
                    return "extends above the box top (wall/column/ceiling structure)"
                if out_tree is not None and not face_across and not crosses_axis and not hanging_central:
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
                        if n_out > cfg.attach_ratio * n and not (
                                compact_pass and straddle_ok and cfg.straddle_compact_exempt and straddle_extent() is not None):
                            return f"attached to structure at box edge ({n_out} outside vs {n} own)"
                return ""

            reason = judge(False)
            # a cluster accepted only by the plate exception (a step in the bed looks the same to D1: the 0.2 m step at
            # 69 m in the organisers' platform frames) votes in the fusion only with a D2 group that has free space
            # behind it (plate_d2_min_bg_frac)
            if not reason and plate_across and judge(False, plate_ok=False):
                cand.plate = True
            # a far cluster rejected only for its point support (a dark person at 150 m gives one scan line of 3-5
            # returns) and recognised as structure by none of the structure rules is kept as a WEAK candidate: the
            # fusion may count it as D1's vote when D2 sees an obstacle at the same place (far_weak_d1_vote)
            if (reason and cfg.far_weak_d1_vote and d_near >= cfg.far_range_m and n >= cfg.far_weak_d1_min_points
                    and reason.startswith(("too few points", "single-ring")) and not judge(True)):
                cand.weak = True
            # a small compact cluster rejected only for its support or as an edge sliver, and recognised as structure by
            # no other rule, is kept as a COMPACT candidate: the fusion counts it as D1's vote only when D2 sees an
            # isolated obstacle group at the same place in the same frame (compact_pair_vote; the organisers' 0.3 m
            # cubes give 6-9 returns at 50 m and sit 1.0-1.3 m from the axis)
            if (reason and cfg.compact_pair_vote and n >= cfg.compact_min_d1_points and float(ext.max()) <= cfg.compact_max_extent_m
                    and reason.startswith(("too few points", "thin sliver")) and not judge(False, compact_pass=True)):
                cand.compact = True
                # the rest of the object beyond the box edge also borders the D2 group: the fusion asks less free space around it
                cand.straddle = bool(cfg.straddle_compact_exempt and judge(False, compact_pass=True, straddle_ok=False))
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
    def _compact_straddle(out_tree, seeds, pd_, pl, ph, d_near, cfg) -> Optional[np.ndarray]:
        """The extent (d, l, h) of the cluster together with the returns outside the box next to it when they belong
        to one compact object: grown from ``seeds`` through the outside returns (step straddle_grow_m +
        straddle_grow_per_m * d), they stay within compact_max_extent_m in every direction.  Structure leaking into
        the box (a cover, a platform edge, a wall) runs on along the track or up and grows past the limit: None."""
        r = cfg.straddle_grow_m + cfg.straddle_grow_per_m * d_near
        lo = np.array([pd_.min(), pl.min(), ph.min()])
        hi = np.array([pd_.max(), pl.max(), ph.max()])
        region = set(seeds)
        frontier = np.fromiter(region, dtype=np.int64, count=len(region))
        while len(frontier):
            pts = out_tree.data[frontier]
            lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
            if float(np.max(hi - lo)) > cfg.compact_max_extent_m or len(region) > cfg.straddle_max_points:
                return None
            # the frontier is queried in chunks so that dense structure (raw returns, not voxels) stops the growth
            # after one chunk instead of after a query over thousands of points; the outcome is the same, since the
            # region only grows and the next pass would return None at the cap check above
            new: List[int] = []
            for i in range(0, len(pts), cfg.straddle_chunk):
                added = set().union(*out_tree.query_ball_point(pts[i:i + cfg.straddle_chunk], r=r)) - region
                region |= added
                new.extend(added)
                if len(region) > cfg.straddle_max_points:
                    return None
            frontier = np.fromiter(new, dtype=np.int64, count=len(new))
        return hi - lo

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
