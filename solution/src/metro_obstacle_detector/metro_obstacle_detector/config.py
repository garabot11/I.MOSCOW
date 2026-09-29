"""Detector configuration.

Every parameter has a unit and a documented effect (see docs/algorithm.md and
config/detector.yaml). Unknown keys are reported instead of silently ignored.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class DetectorConfig:
    # ---- coordinate convention (sensor frame -> analysis frame d/l/h) ----
    forward_axis: str = "-y"  # sensor axis pointing along the track ahead of the train
    up_axis: str = "+z"       # sensor axis pointing up
    # ---- region of interest, metres ----
    d_min: float = 3.0        # ignore returns closer than this (own train body / near-field clutter)
    d_max: float = 300.0      # maximum analysed range along the track
    l_abs_max: float = 12.0   # lateral half-width of the analysed slab
    h_abs_min: float = -8.0   # absolute height limits relative to the sensor
    h_abs_max: float = 10.0
    min_valid_points: int = 2000  # fewer valid points -> status "unknown"
    max_analysed_points: int = 0  # 0 = off (recommended); else keep every k-th point of the analysed region so that at most this many remain. Last-resort load reduction: 180000 halved the 921k-point sensor and the obstacle at 56 m was then found in 12 of 47 frames instead of 47
    # ---- rail-level (track surface) model: h_rail(d) = z0 + grade * d ----
    floor_fit_d_max: float = 120.0     # bins up to this range are used for the fit (sparser bins get lower weight)
    floor_bin_m: float = 2.0           # longitudinal bin size for the floor percentiles
    floor_lateral_halfwidth: float = 1.0  # lateral half-width around the track centre
    floor_percentile: float = 50.0     # median of track-bed returns ~ rail-head level
    floor_min_points: int = 40         # minimum returns per bin to trust its percentile (near range)
    floor_min_points_far: int = 8      # minimum returns per bin beyond floor_near_d
    floor_quad_min_range_m: float = 60.0  # the quadratic term is fitted only when the bed is observed at least this far
    floor_near_d: float = 45.0
    floor_min_bins: int = 4            # minimum bins for a valid fit, else "unknown"
    floor_residual_m: float = 0.25     # outlier rejection threshold for the fit (near range)
    floor_residual_per_m: float = 0.003 # far gate widens with range beyond floor_near_d
    floor_far_residual_m: float = 0.15  # base gate for sparse far bins around the near-range extrapolation
    floor_max_gap_m: float = 20.0       # far bins must continue the accepted bins without a larger gap
    floor_min_vertical_radius_m: float = 2500.0  # bounds the quadratic term of the rail-level fit
    floor_h_max: float = -0.3          # only returns below the sensor (h <= this) can be track bed
    # ---- lateral offset of the sensor from the track centre (drainage-trough centre) ----
    trough_search_d: Tuple[float, float] = (4.0, 25.0)
    trough_search_l_max: float = 2.0
    trough_depth_min: float = 0.15     # trough returns lie this far below rail level ...
    trough_depth_max: float = 0.9      # ... and not deeper than this (drain pits, gaps)
    trough_width_m: float = 1.2        # densest window of this width defines the trough
    trough_min_points: int = 150
    sensor_lateral_offset: float = 0.0  # used when no trough is found (metres, +l = towards +lateral axis)
    # ---- track centre-line curve l_c(d) = l0 + a d + b d^2 (profile correlation) ----
    profile_band_above_rail: Tuple[float, float] = (0.4, 3.6)  # heights above rail level used for the profile
    profile_l_max: float = 12.0
    profile_bin_m: float = 0.1
    profile_d_edges: Tuple[float, ...] = field(default_factory=lambda: tuple(
        list(np.arange(3.0, 60.0, 3.0)) + list(np.arange(60.0, 150.0, 6.0)) + list(np.arange(150.0, 316.0, 15.0))))
    profile_min_points: int = 25
    chain_min_corr: float = 0.5        # adjacent-bin profiles must correlate at least this well
    chain_shift_slack_m: float = 0.3   # extra lateral slack per bin pair beyond the curvature bound
    curve_fit_d_max: float = 160.0
    curve_max_abs_a: float = 0.06       # |yaw slope| bound (rad-ish)
    curve_min_radius_m: float = 200.0   # |b| <= 1 / (2 R_min)
    curve_residual_base_m: float = 1.0  # residual gate for chained-shift samples (base + per_m * d)
    curve_residual_per_m: float = 0.02
    curve_smoothing: float = 0.0        # EMA weight for the previous frame's coefficients (0 = none; the fit priors do the smoothing)
    chain_weight: float = 0.3           # relative weight of chained profile shifts vs absolute samples
    chain_extra_range_m: float = 40.0   # chain samples are used at most this far beyond the last absolute sample
    trough_weight: float = 2.0          # weight of the trough-centre prior
    trough_l0_gate_m: float = 0.5       # the fitted l0 may deviate at most this much from the trough centre
    yaw_prior_weight: float = 0.3       # weak prior pulling the yaw term towards the previous frame (weight of a 1 m offset at 100 m)
    curv_prior_weight: float = 0.3      # weak prior pulling the curvature term towards the previous frame (same units)
    sym_d_edges: Tuple[float, ...] = field(default_factory=lambda: tuple(np.arange(4.0, 46.0, 6.0)))
    sym_half_window_m: float = 1.1      # mirror window: rails at +-0.76 m and the trough are inside it
    sym_cell_m: float = 0.05
    sym_center_range_m: float = 1.2     # search range for the symmetry axis
    sym_percentile: float = 20.0        # per-cell height statistic (robust to objects above the bed)
    sym_min_points: int = 300
    sym_max_score: float = 0.04         # mean |h(c+u) - h(c-u)| above this -> bin discarded
    sym_ref_score: float = 0.02         # score giving unit weight
    sym_residual_base_m: float = 0.3
    wall_band_above_rail: Tuple[float, float] = (1.5, 3.0)  # band where walls/columns/gates bound the track
    wall_width_range: Tuple[float, float] = (3.6, 5.8)      # plausible free width between the two sides (gate 4.0 .. round 5.4)
    wall_min_points: int = 20
    wall_occ_min: float = 2.0           # smoothed (3-cell) count that marks a lateral cell as occupied ...
    wall_occ_frac: float = 0.01         # ... or this fraction of the bin's points, whichever is larger
    wall_weight: float = 0.5
    wall_weight_decay_d: float = 60.0   # wall/ceiling sample weight ~ 1 / (1 + d / this)
    wall_prior_gate_base_m: float = 0.8  # seed gate (near range) around the linear near-range fit
    wall_prior_gate_per_m: float = 0.02
    wall_track_gate_base_m: float = 0.6  # bin-to-bin gate around the local extrapolation of accepted samples
    wall_residual_base_m: float = 0.5
    ceiling_band_above_rail: Tuple[float, float] = (3.0, 5.0)
    ceiling_centre_min_points: int = 15
    ceiling_weight: float = 0.5
    # ---- danger box (train clearance gauge) relative to track centre and rail level ----
    gauge_halfwidth_low: float = 1.0    # |l - l_c| bound below low_zone_top (contact-rail covers start ~1.2 m from the track centre)
    gauge_halfwidth_body: float = 1.40  # |l - l_c| bound for the car-body zone (car half-width 1.35; platform edges start at ~1.45 m)
    low_zone_top_above_rail: float = 0.8
    box_bottom_above_rail: float = 0.30 # minimum height above rail level to be a candidate
    box_top_above_rail: float = 3.40    # car roof ~3.7 m; pressure-gate lintel ~3.8 m
    bottom_margin_per_m: float = 0.002  # extra bottom margin per metre of range where the track bed is observed ...
    bottom_margin_beyond_per_m: float = 0.008  # ... and per metre beyond the last observed track-bed bin (extrapolation)
    margin_rms_factor: float = 1.0      # margins also grow with the rail-level fit residual (rms) times this factor
    top_margin_per_m: float = 0.002     # box top lowers by this per metre of range (same reasons) ...
    top_margin_beyond_per_m: float = 0.006
    lateral_shrink_per_m: float = 0.003 # box half-width shrink per metre (centre-line error at range)
    lateral_curv_factor: float = 0.25   # additional shrink = factor * |b| * d^2 (curvature uncertainty: no view around the bend)
    lateral_min_halfwidth: float = 0.7
    report_beyond_support_m: float = 15.0  # candidates farther than the last wall/ceiling/symmetry sample + this are not reported
    # ---- candidate clustering ----
    cluster_eps_base_m: float = 0.35
    cluster_eps_per_m: float = 0.004    # point spacing grows ~0.0026 m per m (0.15 deg)
    min_points_near: int = 5
    min_points_far: int = 4
    far_range_m: float = 100.0
    cluster_max_extent_d: float = 6.0   # longer clusters are infrastructure (rails, cables, walls in curves)
    cluster_min_extent_m: float = 0.10  # at least one of lateral/height extents must reach this
    low_flat_max_height_m: float = 0.20 # clusters lower than this and longer than low_flat_min_length_m along the track are
    low_flat_min_length_m: float = 1.0  # track-bed infrastructure (cable ducts, train stops, guard rails), not objects
    edge_sliver_max_width_m: float = 0.35  # thin slivers hugging the lateral box edge (platform edges, ducts) are rejected
    edge_sliver_margin_m: float = 0.15
    cluster_min_extent_h_far: float = 0.25  # beyond far_range_m a candidate must span >= 2 scan rings
    shape_rule_beyond_d: float = 60.0   # beyond this range flat patches (floor/ceiling/grazing walls) are rejected ...
    grazing_planarity_max: float = 0.03 # far clusters (with neighbours) whose PCA planarity is below this ...
    grazing_cos_max: float = 0.35       # ... and whose normal is nearly perpendicular to the line of sight are grazing surfaces
    shape_min_height_ratio: float = 0.5 # ... unless height >= ratio * max(horizontal extents)
    attach_radius_m: float = 0.35       # neighbourhood used to test attachment to structure outside the box
    attach_radius_per_m: float = 0.008  # grows with range (point spacing and centre-line error)
    attach_ratio: float = 1.0           # reject an edge cluster when outside neighbours > ratio * own points
    ceiling_margin_m: float = 0.6       # box top is capped this far below the observed ceiling
    above_footprint_d_m: float = 0.8    # footprint slack (along track) for the 'extends above the top' test
    above_footprint_l_m: float = 0.4    # footprint slack (lateral)
    above_min_points: int = 3           # returns above the box top inside the footprint -> structure
    above_band_m: float = 1.2           # returns must lie within this band right above the cluster
    ceiling_min_height_m: float = 2.5   # only returns this high above rail define the ceiling
    ceiling_min_points: int = 10
    voxel_base_m: float = 0.08          # candidate voxel size at range 0 ...
    voxel_per_m: float = 0.002          # ... growing with range
    ceiling_percentile: float = 98.0
    attach_edge_margin_m: float = 0.3   # near range: only clusters this close to the box edge are tested ...
    attach_all_beyond_d: float = 60.0   # ... beyond this range every cluster is tested (centre-line error grows)
    distance_percentile: float = 10.0   # robust "nearest surface" range inside a cluster (min if < 10 points)
    # ---- temporal confirmation ----
    confirm_hits: int = 2               # hits needed inside the window to confirm a track
    confirm_window: int = 3             # frames
    confirm_hits_far: int = 3           # stricter confirmation for candidates beyond far_range_m ...
    confirm_window_far: int = 4         # ... (sparse returns, larger track-model error)
    track_max_misses: int = 3
    assoc_lateral_m: float = 1.2
    assoc_d_closer_m: float = 8.0       # object may come closer by this much between frames (train speed)
    assoc_d_farther_m: float = 2.5
    max_gap_s: float = 1.5              # larger gaps between frames reset temporal state
    stale_timeout_s: float = 1.5        # node-only: no input for this long -> "unknown"
    # ---- output ----
    debug_cloud: bool = True
    marker_corridor: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise KeyError(f"unknown detector parameters: {unknown}")
        kwargs = {}
        for k, v in data.items():
            if isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "DetectorConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # accept both a flat mapping and a ROS parameter file layout
        if "detector" in data and isinstance(data["detector"], dict):
            data = data["detector"]
        if "ros__parameters" in data:
            data = data["ros__parameters"]
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                v = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            out[f.name] = v
        return out

    def axis_matrix(self) -> np.ndarray:
        """3x3 matrix R such that (d, l, h) = R @ (x, y, z)."""
        fwd = _axis_vector(self.forward_axis)
        up = _axis_vector(self.up_axis)
        if abs(float(fwd @ up)) > 1e-6:
            raise ValueError("forward_axis and up_axis must be orthogonal")
        lat = np.cross(up, fwd)  # right-handed: d x l = h  <=>  l = h x d
        return np.vstack([fwd, lat, up]).astype(np.float64)


def _axis_vector(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    name = spec.lstrip("+-")
    idx = {"x": 0, "y": 1, "z": 2}.get(name)
    if idx is None:
        raise ValueError(f"bad axis spec '{spec}', expected e.g. '-y' or '+z'")
    v = np.zeros(3)
    v[idx] = sign
    return v
