"""Detector configuration.

Every parameter has a unit and a documented effect (see docs/algorithm.md and
config/detector.yaml). Unknown keys are reported instead of silently ignored.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


def curv_factor(cfg, b: float) -> float:
    """Shrink factor of the corridor half-width per |b|*d^2: ramps from lateral_curv_factor to
    lateral_curv_factor_strong over +-0.5e-4 around lateral_curv_strong_b (a step changed the
    half-width by 0.2 m at 70 m when a creeping train's b crossed the threshold).  Shared by D1,
    D2 and the RViz corridor so that all three draw the same corridor."""
    t = (abs(b) - cfg.lateral_curv_strong_b) / cfg.lateral_curv_ramp_b + 0.5
    t = min(1.0, max(0.0, t))
    return cfg.lateral_curv_factor + t * (cfg.lateral_curv_factor_strong - cfg.lateral_curv_factor)


@dataclass
class DetectorConfig:
    # ---- coordinate convention (sensor frame -> analysis frame d/l/h) ----
    forward_axis: str = "-y"  # sensor axis pointing along the track ahead of the train
    up_axis: str = "+z"       # sensor axis pointing up
    axes_mode: str = "auto_if_needed"  # "config" = always forward_axis/up_axis; "auto_if_needed" = keep them while the track bed is found with them, otherwise detect the axes from the cloud once per sensor
    forward_crop_deg: float = 80.0  # organised clouds: keep firing columns within this azimuth of forward before any processing (|az| > 80 deg cannot reach d >= d_min inside |l| <= l_abs_max); 0 = off
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
    floor_step_up_max_m: float = 0.0   # > 0: a dense bed bin whose level rises by more than this over the last bed bin is not bed (an
                                       # object on it: Falcon, the person's torso bins at 28-30 m, 1.2 m above the dark bed, levered the
                                       # fit; 0.5 fixes it) - 0 = v1: at 0.5 dust bins above the bed shortened the near run and val_dust
                                       # 0-20 m recall fell 0.505 -> 0.474 (C-fal-2); enable per deployment, not by default
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
    curve_max_abs_a: float = 0.045      # |yaw slope| bound (rad): mounting yaw + car yaw against the local tangent; 0.025 was too tight (synthetic R=600 m curve, Falcon mount a=-0.024), 0.06 let the fit drift at stations
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
    lateral_curv_factor: float = 0.1   # additional shrink = factor * |b| * d^2 (curvature uncertainty: no view around the bend) (0.25 before: at stations a spurious curvature of 1/700 m shrank the corridor to 0.36 m at 50 m)
    lateral_curv_factor_strong: float = 0.25  # the factor in strong curves (|b| >= lateral_curv_strong_b, R <= 1.7 km): the quadratic over-bends there
    lateral_curv_strong_b: float = 3.0e-4     # (roundT_pressureGate: a creeping train in an R = 880 m curve put wall returns 0.4 m from the extrapolated centre at 73 m)
    lateral_curv_ramp_b: float = 1.0e-4       # width of the ramp between the two factors
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
    face_min_width_m: float = 1.5       # a cluster this wide across the corridor, thin along the track and centred is a face
    face_max_depth_m: float = 1.0       # (a train standing ahead, a closed gate, a panel): never 'structure reaching the ceiling'
    face_max_center_offset_m: float = 0.9
    face_min_points: int = 20
    flat_sliver_max_h_m: float = 0.12   # a cluster this thin vertically and at least flat_sliver_min_w_m wide/long is a cable, duct or floor patch
    flat_sliver_min_w_m: float = 0.4    # (with D2 bridging pixel gaps, 2-3 cm thick slivers at 3 m height were confirmed on the edited bag)
    flat_sliver_min_h_above_rail_m: float = 1.5  # only slivers hanging this high are structure: a flat plate 0.5 m above the rails is an obstacle
    face_min_height_m: float = 1.5      # ... and tall inside the box with its lowest returns low (a lintel or a grid is neither)
    face_max_bottom_m: float = 1.5
    d1_plate_exception: bool = True     # a plate across the corridor (wide, short along the track, centred, low) is not a "flat patch at range" / "grazing surface" (the organisers' 2 m plate 0.5 m above the rails, seen edge-on from 90 m); D2 must still agree
    plate_min_width_m: float = 0.8      # plate: lateral extent at least this ...
    plate_max_depth_m: float = 0.6      # ... extent along the track at most this ...
    plate_max_center_offset_m: float = 0.6  # ... centre this close to the centre-line ...
    plate_max_height_above_rail_m: float = 1.2  # ... mean height above rail at most this ...
    plate_min_points: int = 6           # ... and at least this many voxels
    plate_d2_min_bg_frac: float = 0.5   # ... and a cluster accepted only by the exception votes only with a voting D2 obstacle group at the same place that has at least this fraction of its boundary ring in the background (a plate on the bed casts a shadow; the 0.2 m step at 69 m in the organisers' platform frames is continuous with the surface behind it)
    ceiling_min_height_m: float = 2.5   # only returns this high above rail define the ceiling
    ceiling_min_points: int = 10
    voxel_base_m: float = 0.08          # candidate voxel size at range 0 ...
    voxel_per_m: float = 0.002          # ... growing with range
    # ---- far rail level from the ceiling profile (track_model.extend_rail_by_ceiling) ----
    ceiling_extension: bool = True      # beyond the observed bed take rail = ceiling profile - tunnel height
    ceiling_ext_bin_m: float = 5.0      # range bins for the ceiling profile
    ceiling_ext_lateral_m: float = 1.5  # ceiling samples within this lateral distance of the centre-line
    ceiling_height_tol_m: float = 0.3   # tunnel height must be constant within this in the near range; far bins are gated by it (+2 mm/m)
    ceiling_ext_min_bins: int = 3       # at least this many ceiling bins beyond the bed, else no extension
    ceiling_ext_max_gap_m: float = 30.0 # the ceiling profile must be continuous (no larger gap between accepted bins)
    ceiling_ext_max_drift_m: float = 0.5   # ceiling-derived rail may differ from the bed extrapolation by this ...
    ceiling_ext_drift_per_m: float = 0.005 # ... plus this per metre beyond the bed (a vertical curve), else the extension stops
    bottom_margin_ext_per_m: float = 0.004 # box bottom growth per metre beyond the observed bed while the ceiling carries the rail (0.008 beyond that)
    wall_sweep_check: bool = True       # cap the corridor where a tunnel wall sweeps into it (a bend the centre-line fit
                                        # did not follow, or the straight extrapolation drifting into a wall)
    wall_sweep_min_d: float = 40.0      # the check starts here (nearer, the fit is supported and a standing train must stay reportable)
    wall_sweep_bin_m: float = 5.0
    wall_sweep_h_min: float = 0.5       # wall-height band above rail used by the check (floor and ceiling excluded)
    wall_sweep_h_max: float = 3.0
    wall_sweep_l_max: float = 4.0       # returns farther than this from the centre-line are not walls of this tunnel
    wall_sweep_min_pts: int = 4         # returns per bin and side needed to place the wall's inner edge (far walls are sparse columns)
    wall_sweep_span_min_m: float = 1.2  # a wall is tall: height span of the bin's inner surface
    wall_sweep_margin_m: float = 0.1    # a wall closer than gauge_halfwidth_body + this to the centre-line is inside the corridor
    wall_sweep_total_m: float = 0.8     # the inner edge must come in by at least this over the three bins ...
    wall_sweep_deep_total_m: float = 0.5  # ... or by this when it ends deep inside (below the inside bound - wall_sweep_deep_m):
    wall_sweep_deep_m: float = 0.4      # round-tunnel walls read 0.3-0.5 m closer where lower rings sample them, never 0.8 m
    wall_sweep_step_m: float = 0.15     # the inner edge must approach the centre-line by at least this per 5 m over three bins
    straight_max_abs_b: float = 1.0e-4  # |b| below this (R > 5 km) = straight section: the corridor support extends to the ceiling-covered range
    straight_keep_abs_b: float = 1.5e-4 # ... and once extended it stays extended while |b| is below this (hysteresis: the Falcon tunnel fits
                                        # b = 1.0-1.35e-4 and with a single threshold the support flipped 81 <-> 141 m from frame to frame)
    floating_beyond_d: float = 100.0      # beyond this range a cluster whose lowest point is >= floating_bottom_above_rail ...
    floating_bottom_above_rail: float = 1.5  # ... above the rail is hanging infrastructure (lamps, signs, cables), not an object on the track
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
    # ---- fusion of the detectors (D1 clusters, D2 beams, D3 network) ----
    fusion_enabled: bool = True   # False = the D1 temporal filter alone (confirm_hits / confirm_window)
    far_weak_d1_vote: bool = True       # train standing (static fraction >= far_weak_d1_static_frac): beyond far_range_m a D1 cluster rejected only for its point support (one scan line, < min_points_far) votes when a voting D2 obstacle group lies at the same place in the same frame (the dark person at 150 m on the Falcon bag)
    far_weak_d1_min_points: int = 3     # ... such a weak cluster needs at least this many voxels
    far_weak_d1_static_frac: float = 0.85  # ... and the scene must be at least this static (unchanged range-image pixels)
    far_vote_d1_min_d: float = 140.0    # 0 = off; train moving (static fraction < far_weak_d1_static_frac): beyond this range a D1 cluster with fewer than far_vote_d1_min_points voxels adds evidence but does not vote (sparse far wall returns at 156-165 m on the 20-minute ride)
    far_vote_d1_min_points: int = 5
    compact_pair_vote: bool = True      # a small compact D1 cluster (also one rejected only for its support or as a short edge sliver) and an isolated D2 obstacle group at the same place in the same frame both vote, whatever their size-normalised scores (a 0.3 m cube scores 0.15-0.25 in both detectors)
    compact_max_extent_m: float = 1.0   # ... both at most this large in every direction
    compact_min_d1_points: int = 3      # ... the D1 cluster has at least this many voxels
    compact_sliver_max_length_m: float = 0.6  # ... an edge "sliver" this short along the track ...
    compact_min_thickness_m: float = 0.1      # ... and at least this thick laterally and vertically is a compact object, not a platform edge
    compact_min_d2_px: int = 4          # ... the D2 group has at least this many pixels,
    compact_max_attached: float = 0.05  # ... no attached neighbours (at most this fraction),
    compact_min_bg_frac: float = 0.9    # ... and at least this fraction of its boundary ring is background (free space behind it)
    compact_pair_dd_m: float = 1.5      # ... "same place": along-track ...
    compact_pair_dl_m: float = 0.5      # ... and lateral distance between the D1 and D2 centres
    compact_isolation_d_m: float = 2.5  # ... and the D1 cluster stands alone: no larger D1 cluster, and no other cluster of compact_min_d1_points outside the pair window, within this distance along the track ...
    compact_isolation_l_m: float = 2.0  # ... and this distance across (a dust cloud breaks into many small clusters a metre apart; a 0.3 m cube has nothing around it)
    axis_cross_exempt: bool = True      # a D1 cluster spanning the track axis (by >= axis_cross_min_m on both sides of the centre-line) is in the train's path:
    axis_cross_min_m: float = 0.25      # the attachment test does not apply (structure leaking in at the box edge never reaches across the axis; the organisers' beam across the rails sticks out of the low zone on one side and had more returns outside than inside)
    axis_cross_max_depth_m: float = 0.5  # ... when it is thin along the track (their beam: <= 0.11 m; a dust cloud across the corridor 1.2-1.6 m was confirmed through it on val_dust)
    hanging_central_exempt: bool = True  # a D1 cluster hanging into the middle of the corridor at short range is not structure: neither the 'extends above' nor the attachment test applies
    hanging_central_max_d: float = 70.0  # ... (the organisers' tilted cable from the ceiling down to 2.0 m above the rails): within this range,
    hanging_central_min_bottom_m: float = 1.8  # ... lowest return at least this high above the rail (free space below it: nothing stands on the track there),
    hanging_central_edge_m: float = 0.5  # ... every return at least this far inside the lateral box edge (walls and columns stand at the edge)
    straddle_compact_exempt: bool = True  # compact path only (compact_pair_vote): the 'outside' neighbours of a small cluster at the box edge are the rest
    straddle_grow_m: float = 0.15       # of the same object when, grown through the returns outside the box (step straddle_grow_m + straddle_grow_per_m * d),
    straddle_grow_per_m: float = 0.004  # they stay within compact_max_extent_m together with the cluster: not attached to structure (the organisers' cube on
    straddle_max_points: int = 3000     # the left rail, two thirds of it beyond the low zone); a cover, a platform edge or a wall grows past it
    straddle_chunk: int = 128           # frontier points per neighbour query of the growth (bounds the cost on dense structure, result unchanged)
    straddle_thin_exempt: bool = True   # ... and an edge sliver too thin for the compact path (min(width, height) < compact_min_thickness_m: the top face of a cube
                                        # whose body is beyond the low-zone edge, 0-9 cm of it inside) is compact when that grown object is at least that thick
    straddle_min_bg_frac: float = 0.4   # such a cluster pairs with a D2 group whose boundary ring is at least this much background (the object's own part beyond
                                        # the edge fills one side of the ring: 0.40-0.55 on the cube at 5-29 m; compact_min_bg_frac for every other pair)
    d2_enabled: bool = True       # beam detector on the range image
    d2_az_limit_deg: float = 25.0 # range-image sector around forward used by D2 (the corridor at d_min=3 m spans +-25 deg)
    d2_bridge_empty_bins: bool = True   # unorganised (binned) clouds only: D2 joins small in-box fragments separated by empty bins and looks through empty bins for their neighbours (an object sampled coarser than the bins - the organisers' inserted plate - fell apart into 1-2 px fragments with no neighbour)
    d2_bridge_min_vspan_deg: float = 33.0  # ... only on a stream whose binned image spans at least this many degrees vertically: the Hesai class (OT128 39.5 deg); the Falcon Prime (25 deg) is not bridged (the rule cost it frames where the corridor model breaks)
    d3_enabled: bool = True       # neural detector on the range image (needs PyTorch and a model file; silently off otherwise)
    d3_model_path: str = ""      # TorchScript model; empty = METRO_D3_MODEL, /workspace/models/d3_unet.pt or <repo>/solution/models/d3_unet.pt
    d3_min_pixels: int = 10       # smaller D3 components only add weak evidence (score x0.3) and do not vote
    d3_corridor_margin_m: float = 0.3  # D3 components farther than this outside the clearance (laterally / in height) are dropped
    d3_cpu_stride: int = 3        # in CPU modes the network runs on every n-th frame
    d3_max_ms: float = 120.0      # three consecutive D3 calls above this disable D3 (real time first)
    model_hold_s: float = 1.0     # keep the last rail/track model this long when the fit fails (object blocking the near bed)
    static_hold: bool = True      # while the scene is static (train standing) keep the rail/track model of the previous frame instead of re-fitting it
    static_min_frac: float = 0.9  # fraction of range-image pixels whose range did not change (within static_tol) for the scene to count as static
    static_tol_m: float = 0.05    # ... plus 0.2 % of the range
    static_warmup: int = 5        # static frames whose per-frame models are estimated before their median is frozen
    quasi_static_frac: float = 0.85  # above this (but below static_min_frac) the model is the median of the last quasi_static_frames fits:
                                     # a raster lidar's returns flicker (Falcon standing: 0.88-0.89), the corridor must not wobble with them
    quasi_static_frames: int = 5     # 0.5 s (15 frames lagged a slowly moving train into 5 false frames on roundT_pressureGate)
    # ---- output ----
    debug_cloud: bool = True
    marker_corridor: bool = True
    marker_corridor_volume: bool = True  # draw the clearance corridor as a translucent volume (floor + side walls) up to verified_to_m
    # ---- corridor support and stability (pipeline verified range; track_model.extension_weight / median_model) ----
    corridor_ext_ramp_frac: float = 0.2      # the straight-section extension of the verified range fades out linearly over this fraction of the
                                             # straight_max_abs_b / straight_keep_abs_b threshold above it (0 = the v1 step: 81 <-> 141 m flips on Falcon)
    corridor_static_b_window: int = 300      # while the scene is quasi-static, the extension is decided on the median |b| of up to this many
                                             # per-frame fits (a standing train: the curvature cannot change; Falcon's per-frame |b| scatters 0.7-2.2e-4)
    corridor_static_b_min_frames: int = 10   # ... once at least this many quasi-static fits are collected
    corridor_static_b_max: float = 2.5e-4    # ... and only while the frame's own |b| is below this (a real curve is never averaged away)
    corridor_curv_shape_factor: float = 1.0  # curvature shrink (curv_factor |b| d^2) is scaled by this where the fitted centre-line follows the
                                             # measured shape of the bend (track_model.chain_shape_range); 1.0 = v1 (no reduction anywhere)
    corridor_curv_shape_ref_d: Tuple[float, float] = (15.0, 30.0)  # chain residuals over this range give the anchor offset
    corridor_curv_shape_tol_m: float = 0.25  # ... the shape is followed while a chain residual stays within this of the offset
    corridor_curv_shape_max_gap_m: float = 12.0  # ... and the chain samples are not farther apart than this
    marker_status_ahead_m: float = 20.0      # RViz status text: this far ahead of the sensor along the track (at the sensor it sat on the view's bottom edge)
    marker_status_height_m: float = 7.0      # ... and this high above the sensor (above the tunnel cross-section in the default view)

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
