# Algorithm

> **Кратко по-русски.** Детектор геометрический: (1) проверка облака и переход в систему «вдоль пути / вбок / вверх»; (2) уровень рельса как полином по медианам полотна (двухэтапная робастная подгонка, линейная экстраполяция за пределами видимого полотна); (3) ось пути `l_c(d)=l0+a·d+b·d²` из четырёх источников — симметрия профиля полотна (ближняя зона), центр лотка, центры свободного пространства между стенами/потолком по бинам дальности (все дальности, последовательный трекинг), цепочка сдвигов профиля между соседними бинами; (4) габаритная коробка вокруг оси (полуширина 1,0 м внизу / 1,40 м по кузову, 0,3–3,4 м над рельсом, с запасами, растущими с дальностью и кривизной); (5) воксельная кластеризация; (6) правила отбраковки конструкций (вытянутые вдоль пути, продолжающиеся до потолка, привязанные к стенам, плоские/скользящие поверхности, тонкие слайверы у края, за пределами опоры модели); (7) расстояние — евклидово от лидара до ближайшей достоверной поверхности; (8) временное подтверждение 2 из 3 кадров (3 из 4 за 100 м), сброс при разрывах времени. Ограничения — в конце файла.

## Problem statement

Input: a stream of 3D LiDAR clouds from a sensor on the front of a metro train (300 k – 900 k
returns, ≈ 10 Hz). Output per cloud: `clear` / `detected` / `unknown`, and for `detected` the
Euclidean distance from the sensor to the nearest confirmed foreign object inside the train's
clearance ahead, plus the object's position and size.

"Obstacle" is defined geometrically: **a compact group of returns standing above the track bed
inside the clearance box of the car, following the track ahead, that is not part of the tunnel
structure**. No object classes are learned; the method describes the *normal* tunnel (track
bed, walls, ceiling, side equipment) and reports what does not fit. This is the direction the
task statement suggests for a homogeneous environment with very few positive examples.

## Notation and frames

Sensor frame `(x, y, z)` → analysis frame `(d, l, h)` with `d` along the track ahead, `l`
lateral, `h` up: `(d, l, h)ᵀ = R (x, y, z)ᵀ`, `R` built from `forward_axis` (−y) and `up_axis`
(+z). All quantities are in metres.

* `h_rail(d)` – rail-head level (track surface) as a function of range;
* `l_c(d)` – lateral position of the track centre-line;
* `Δh = h − h_rail(d)`, `Δl = l − l_c(d)` – height above the track and offset from the centre-line.

## Pipeline

### 1. Validation and ROI

`cloud_io` parses the mixed 26-byte layout. Returns that are non-finite or exactly `(0,0,0)`
(placeholder returns, 40–60 % of the cloud) are discarded; a cloud with fewer than
`min_valid_points` valid returns yields `unknown`. The ROI keeps `d ∈ [d_min, d_max]` (3–300 m),
`|l| ≤ 12 m`, `h ∈ [−8, 10] m`. The own train body (returns closer than 3 m) is excluded.

### 2. Rail level `h_rail(d)`

Per 2 m range bin the **median height of returns within ±1 m of the track centre and below
the sensor** is taken (this sits at rail-head/sleeper level; the trough is lower, the rails
slightly higher). A polynomial

`h_rail(d) = z₀ + g·d + c·d²`, `|c| ≤ 1/(2·R_v,min)` (R_v,min = 2500 m)

is fitted robustly: stage 1 uses only bins up to 45 m (dense) with iterative residual gating
(0.5 → 0.25 m); stage 2 adds sparse far bins (≥ 8 returns) only if they lie within
`0.15 + 0.003·(d−45)` m of the near-range extrapolation and continue the accepted bins without
a gap > 20 m, so that an object standing on the track cannot lift the floor model at its own
range. The quadratic term is fitted only if the bed is observed to ≥ 60 m; beyond the last
observed bin the curve is extended along its tangent (no quadratic extrapolation). When the
track curve (step 3) shows a curve, the fit is repeated with the window following `l_c(d)`
instead of a straight strip. If no fit is possible → `unknown` ("rail level not observable").

### 3. Track centre-line `l_c(d) = l₀ + a·d + b·d²`

Bounded curvature `|b| ≤ 1/(2·200 m)`, yaw `|a| ≤ 0.06`. Four independent cues are fused in
one weighted least-squares problem with iterative outlier rejection and weak priors
(`a ≈ a_prev`, `b ≈ b_prev`):

1. **Bed symmetry (absolute, 4–46 m).** In each 6 m bin the height profile of the track bed
   `p(l)` (20th percentile of `Δh` per 5 cm lateral cell) is mirrored: the axis `c` minimising
   `mean_u |p(c+u) − p(c−u)|`, `u ∈ (0.1, 1.1) m`, is the track centre (two rails at ±0.76 m and
   the trough are symmetric; the one-sided contact rail lies outside the window). Bins with a
   symmetry score > 0.04 m are discarded (weight `(0.02/score)²`).
2. **Trough centre (absolute prior).** Densest 1.2 m window of the lowest returns
   (0.15–0.9 m below rail level) at 4–25 m gives `l₀`; the fitted `l₀` is kept within ±0.5 m
   of it.
3. **Free-space centres from walls and ceiling (absolute, all ranges).** In the band
   1.5–3.0 m above rail the lateral occupancy histogram of each range bin is scanned for free
   intervals bounded by structure on both sides with a plausible width (3.6–5.8 m: pressure
   gate 4.0 m … round tunnel 5.4 m; the double-track wall/column pair 4.5 m); their centre is a
   sample. The ceiling band (3.0–5.0 m) gives the centre of its 5–95 % lateral extent under the
   same width test. Bins are visited near → far, each prediction being the local extrapolation
   of the samples accepted so far (gate `0.6 + Δd·d/R_min`), so strong curves are followed step
   by step while stations and junctions (implausible widths or jumps) are skipped. Weights decay
   as `1/(1 + d/60)`.
4. **Chained profile shifts (relative, all ranges).** The lateral histogram of all structure
   0.4–3.6 m above rail is cross-correlated between adjacent range bins; the accumulated shift is
   a relative sample of `l_c(d) − l_c(d_anchor)` (normalised correlation ≥ 0.5, else the bin is
   bridged with the previous curve). Used with a lower weight and only up to 40 m beyond the last
   absolute sample.

`max_d_absolute` – the farthest absolute sample used – defines the range up to which the model
is *supported*; candidates farther than `max_d_absolute + 15 m` are not reported (this is what
limits the range at stations, junctions and behind sharp curves).

### 4. Clearance box

A return is a candidate if

* `|Δl| ≤ hw(Δh, d)` with `hw = 1.0 m` below 0.8 m above rail (contact-rail covers begin at
  1.2 m) and `hw = 1.40 m` above (car half-width 1.35 m; platform edges begin at ≈1.45 m),
  reduced by the centre-line uncertainty `0.003·d + 0.25·|b|·d²` (the corridor "closes" where
  it can no longer be seen around a bend, `hw < 0.7 m`);
* `Δh ≥ 0.30 m + 0.002·d + 0.008·max(0, d − d_obs) + rms_rail` (above the track bed, with
  margins for the floor-model error beyond the observed bed);
* `Δh ≤ 3.40 m − 0.002·d − 0.006·max(0, d − d_obs) − rms_rail`, additionally capped 0.6 m
  below the observed ceiling of the bin (ceiling returns ≥ 2.5 m above rail).

### 5. Clustering

Candidate returns are voxelised (`0.08 + 0.002·d` m; removes duplicates and bounds the cost)
and linked by a KD-tree with a range-dependent radius `0.35 + 0.004·d` m (point spacing grows
≈ 0.0026 m per metre); connected components are clusters.

### 6. Structure rejection rules (what a "normal tunnel" looks like)

A cluster is discarded, with the reason recorded in the log, if it is

| rule | rationale |
|---|---|
| fewer than 5 (near) / 4 (≥100 m) voxels | noise, single rays |
| beyond `max_d_absolute + 15 m` | track model unsupported there |
| longer than 6 m along the track | rails, ducts, cables, walls entering the box in a curve |
| lower than 0.2 m and longer than 1 m | train stops, guard rails, cable ducts on the bed |
| a sliver < 0.35 m wide hugging the lateral box edge | platform edges, ducts |
| ≥ 100 m and < 0.25 m tall | single scan ring |
| ≥ 60 m and height < 0.5 × horizontal extent | floor / ceiling / wall patches at grazing angles |
| ≥ 60 m and a thin plane whose normal is ⟂ to the line of sight (PCA with neighbours) | grazing wall/floor surface |
| returns continue right above the cluster (within 1.2 m) in its footprint | walls, columns, gates, hanging cables reach the ceiling; objects end in free air |
| near the box edge (always beyond 60 m) and more returns in the outside neighbourhood (`0.35 + 0.008·d` m) than its own | structure leaking into the box because of a centre-line/floor error |

Everything else is an obstacle candidate. A confidence 0…1 compares the voxel count with the
expectation for a ~0.5 m wide object at that range.

### 7. Distance

For each candidate: **Euclidean range from the sensor origin to the nearest reliable surface
of the object** – the 10th percentile of its returns' ranges (minimum if fewer than 10 returns),
so a single stray return cannot shorten the distance. `forward_m` (along-track distance of the
nearest return), the centre in the sensor frame, the extents along `d/l/h` and the height above
rail are reported as well. Distances are from the sensor, not from the front of the train (the
sensor-to-coupler offset is unknown).

### 8. Temporal confirmation

Candidates are associated to short-lived tracks (lateral gate 1.2 m, along-track gate −8/+2.5 m
per frame). A track is confirmed after `confirm_hits = 2` hits within the last
`confirm_window = 3` frames (beyond `far_range_m` = 100 m: 3 hits within 4 frames, because far
candidates rest on few returns and a larger track-model error); a track dies after 3 misses. The status is `detected` if any
confirmed track exists and the reported distance is that of the nearest confirmed candidate.
At 10 Hz confirmation costs ≈ 0.1–0.2 s. The state is reset on time running backwards (bag
loop, restart, next bag) or gaps > `max_gap_s` = 1.5 s, and `unknown` frames never count as
`clear`. The header stamp is checked **before** the frame is processed, and a discontinuity resets
both the temporal tracks and the geometric state of the detector (the previous track curve used
to chain the corridor), so the first frame of a new sequence is processed exactly like the first
frame of a fresh detector. The node additionally resets everything when no cloud arrives for
`stale_timeout_s`. Both adapters – the ROS node and the offline evaluator – run the same
per-frame class (`pipeline.DetectionPipeline`), so their state handling is identical; this is
covered by regression tests (`tests/test_state_and_streaming.py`: jump backwards and long gap,
offline run of a synthetic bag with a jump compared record by record with the pipeline).

## Parameters

All parameters live in `config/detector.yaml` (generated from `config.py`, one comment per key)
and can be overridden as ROS parameters. The most influential ones:

| parameter | default | effect |
|---|---|---|
| `forward_axis`, `up_axis` | `-y`, `+z` | sensor mounting convention |
| `d_min`, `d_max` | 3, 300 m | analysed range |
| `gauge_halfwidth_low`, `gauge_halfwidth_body`, `low_zone_top_above_rail` | 1.0, 1.40, 0.8 m | clearance box width vs height |
| `box_bottom_above_rail`, `box_top_above_rail` | 0.30, 3.40 m | lowest / highest detectable object part |
| `bottom_margin_per_m`, `bottom_margin_beyond_per_m`, `margin_rms_factor` | 0.002, 0.008, 1.0 | how fast the box bottom rises with range beyond the observed bed |
| `lateral_shrink_per_m`, `lateral_curv_factor` | 0.003, 0.25 | corridor narrowing with range and curvature |
| `min_points_near`, `min_points_far`, `far_range_m` | 5, 4, 100 m | minimum cluster support |
| `cluster_max_extent_d` | 6 m | elongated-structure rule |
| `attach_radius_m`, `attach_radius_per_m`, `attach_ratio`, `attach_all_beyond_d` | 0.35, 0.008, 1.0, 60 m | attachment rule |
| `report_beyond_support_m` | 15 m | reporting range beyond the last absolute track sample |
| `confirm_hits`, `confirm_window`, `confirm_hits_far`, `confirm_window_far`, `track_max_misses` | 2, 3, 3, 4, 3 | temporal confirmation (near / beyond `far_range_m`) |
| `stale_timeout_s`, `max_gap_s` | 1.5, 1.5 s | unknown on stale input, reset on gaps |
| `wall_width_range` | 3.6–5.8 m | plausible tunnel widths for the free-space centre |
| `max_analysed_points` | 0 (off) | optional decimation of the analysed region (every k-th point). Not recommended: with 180 000 (halves only the 921 k-point sensor) the obstacle at 56 m was found in 12 of 47 frames instead of 47 – the far-range rules count points |

## Limitations and assumptions

* **Clearance geometry is assumed, not supplied.** Car half-width 1.35 m, height 3.4 m above
  rail, lowest detectable object 0.3 m (plus range margins), contact-rail covers at ≥ 1.2 m.
  Objects between 1.0 and 1.35 m laterally and below 0.8 m are missed on purpose.
* **Detection range is limited by physics and by the model support**: the track bed is seen
  to ≈ 60–120 m, so low objects are only detectable to about that range; person-sized objects
  give ≥ 4 voxels up to ≈ 150–200 m in straight tunnels; behind curves the corridor closes;
  at stations and junctions the reporting range ends where no plausible free interval exists.
* **Objects touching structure** (a person leaning on a wall or column) beyond 60 m are
  rejected by the attachment rule; within 60 m only if they hug the box edge.
* **Objects hanging from the ceiling** below 3.4 m above rail but above the observed ceiling
  cap are missed; objects longer than 6 m along the track are treated as infrastructure.
* **Single-frame geometry**: no ego-motion, no map, no accumulation. Deskew is not applied
  (the per-point timestamps' semantics are undocumented).
* Only one positive example exists; thresholds for far-range rules were tuned on the six
  development bags and are not independently validated (see `docs/experiments.md`).
* `unknown` is reported when the track bed cannot be observed (blocked sensor, empty cloud);
  a wrong `forward_axis`/`up_axis` also ends in `unknown`, never in `clear`.
