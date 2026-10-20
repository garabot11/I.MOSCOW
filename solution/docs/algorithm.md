# Algorithm

> **Кратко по-русски.** Три детектора на общей геометрической модели тоннеля и слияние во времени. Общая часть: профиль датчика (раскладка облака, двойное эхо, оси — автоматически и для другого лидара, обрезка вперёд), range-image, уровень рельса (полином по полотну, за пределами видимого полотна — по профилю потолка минус высота тоннеля), ось пути `l_c(d)=l0+a·d+b·d²` по симметрии полотна, лотку, свободному пространству между стенами/потолком и цепочке сдвигов профиля; на стоящем поезде модель замораживается (медиана первых 5 кадров), при сбое подгонки удерживается 1 с. **Д1** — кластеры в габаритной коробке (полуширина 1,0/1,40 м, 0,3–3,4 м над рельсом) с правилами отбраковки конструкций. **Д2** — лучевой: группы пикселей range-image внутри коридора, оцениваемые по окружению в изображении (фон / привязка к поверхности / опора на полотно / продолжение вверх / тонкие слайверы и столбики / висящие), хватает 1–2 лучей; признак пыли — второе эхо за объектом. **Д3** — нейросеть U-Net на вырезке range-image (3 кадра × 7 каналов, в т.ч. высота над рельсом и смещение от оси по модели Д1), классы объект/пыль; GPU, при сбое — CPU. **Слияние**: треки в (d, l) накапливают логарифм отношения правдоподобия; подтверждение до 120 м — два детектора с весом в последних 5 кадрах и не менее двух кадров, дальше 120 м — устойчивость одного детектора 4 кадра из 5; гистерезис, вето пыли. Выход — `CLEAR/DETECTED/UNKNOWN`, расстояние, объекты, «путь проверен до X м», режим GPU/CPU. Ограничения — в конце файла.

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

Bounded curvature `|b| ≤ 1/(2·200 m)`, yaw `|a| ≤ 0.06`. A section is *straight* for the
ceiling-based support extension when `|b| < 1e-4` (R > 5 km), and once extended it stays so
while `|b| < 1.5e-4` (hysteresis, decided on the median curvature of the quasi-static model: the
Falcon tunnel fits b = 1.0–1.35e-4 and with a single threshold the support flipped between 81
and 141 m from frame to frame; a single higher threshold let a real R ≈ 3 km curve on the
20-minute ride extend to 175 m and confirm wall returns at 134 m). Four independent cues are fused in
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

**Wall sweeping into the corridor (`track_model.wall_sweep_limit`).** A quadratic cannot
represent a bend that *begins* beyond its support: on the pressure-gate bag the tunnel turns
left at ≈110 m while the fit (|b| < 1e-4, "straight") extrapolated the corridor right, into the
right wall, whose returns then became "obstacles" at 108–145 m. Every frame therefore checks
the walls against the corridor: per 5 m bin and side, the inner edge of the wall band
(0.5–3.0 m above rail, ≥ 6 returns forming a surface ≥ 1.2 m tall) is the 20th percentile of
|Δl|; three consecutive bins (one empty bin allowed) whose inner edge approaches the
centre-line by ≥ 0.15 m per 5 m, in total by ≥ 0.8 m (or ≥ 0.5 m when it ends ≥ 0.4 m inside
the car half-width + 0.1 m), reaching inside the corridor, are a wall crossing it. The start of
the first bin inside is the frame's `bend_limit_m`; the pipeline caps `verified_to_m` with the
median of the last 5 frames (no "+15 m" beyond it, flicker-free), D1/D2/D3 candidates beyond are
not reported and neither is a track confirmed a few frames earlier that now lies beyond the cap
(on the 20-minute `new_data` ride such a track coasted into a wall for 7 frames). A dense surface across the corridor (a face: ≥ 20 returns in the nearest
1 m slab of a bin, ≥ 0.8 m wide, ≥ 1 m tall; a train's visible side or a person in the same bin
does not spoil the test) vetoes the cap for the frame – a train standing ahead never caps
itself; single bins
(gate posts, cabinets, a person) and round-tunnel walls read 0.3–0.5 m closer by lower rings
never qualify (the inner edge must come in by ≥ 0.8 m over the three bins, or by ≥ 0.5 m ending
deep inside); far walls are accepted as single columns per bin (≥ 4 returns), since that is how
a wall 150 m away is sampled. The check starts at 40 m. The model itself is not modified, so a frozen static-scene model
cannot carry a cap of its warm-up frames through a whole stop.

### 4. Clearance box

A return is a candidate if

* `|Δl| ≤ hw(Δh, d)` with `hw = 1.0 m` below 0.8 m above rail (contact-rail covers begin at
  1.2 m) and `hw = 1.40 m` above (car half-width 1.35 m; platform edges begin at ≈1.45 m),
  reduced by the centre-line uncertainty `0.003·d + f·|b|·d²` with `f = 0.1`, or `0.25` in
  strong curves (`|b| ≥ 3e-4`, R ≤ 1.7 km, where the quadratic over-bends: on
  roundT_pressureGate a creeping train in an R = 880 m curve put wall returns 0.4 m from the
  extrapolated centre at 73 m) – the corridor "closes" where it can no longer be seen around a
  bend, `hw < 0.7 m`;
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

#**Face across the corridor.** The organisers' edited bag `cloud_with_fake_obj` inserts a planar
2.9 × 4 m wall across the track (a closed gate or a train standing ahead look the same): it reaches
above the clearance top and has nothing behind it, so the "extends above the box top" and
"attached to structure" rules rejected it. A cluster that is wide (≥ 1.5 m across the corridor),
thin along the track (≤ 1 m), tall inside the box (≥ 1.5 m) with its lowest returns ≤ 1.5 m above
rail, centred (|Δl| ≤ 0.9 m) and dense (≥ 20 returns) is a *face across the corridor*: those two
rules are skipped for it. A lintel or a ventilation grid is wide but hangs high; a column is
narrow; a wall in a curve is long along the track – none of them qualifies. D2 applies the same
exception (≥ 12 pixels) before its "no free space behind" and "reaches above" verdicts. D2 groups
stay plain 8-connected pixel groups (the behaviour of every evaluation before the organisers'
edited bag); bridging across one-pixel gaps is applied only to clusters of small fragments
(≤ 8 px each, ≥ 2 of them, ≥ 12 px together) whose union is a face across the corridor by the
same geometry – the inserted face is sampled every other pixel and fell apart into 1–4 px pieces.
Bridging everything merged far persons into wall groups (`val_hesai_insert_tunnel` 50–100 m
recall 0.32 → 0.05) and, with range splits, turned wall pieces into "obstacles" (4 false frames on
the platform bag, 10 on the 20-minute ride). Two more D2 verdicts: a flat sliver (< 12 cm
thick over ≥ 2 image rows, ≥ 0.4 m wide, hanging ≥ 1.5 m above rail) and a long narrow group at
the corridor edge (≥ 3 m along the track, ≤ 1 m wide, ≥ 3× longer than wide) are structure – a 6 × 0.7 ×
2.6 m wall chunk at the edge of the corridor scored 0.83 on the 20-minute ride. A flat sliver (< 12 cm thick, ≥ 0.4 m wide)
is structure only when it hangs ≥ 1.5 m above rail (cables, ducts); the same bag carries a 1.3 m
wide, 1 cm thick plate 0.5 m above the rails, which D1 clusters but D2 does not vote for (see Limitations). The face is confirmed from 76 m (first visible
at 89 m as a 0.7 × 1 m patch at 2.2–3.2 m height, which stays structure by these rules).

## 7. Distance

For each candidate: **Euclidean range from the sensor origin to the nearest reliable surface
of the object** – the 10th percentile of its returns' ranges (minimum if fewer than 10 returns),
so a single stray return cannot shorten the distance. `forward_m` (along-track distance of the
nearest return), the centre in the sensor frame, the extents along `d/l/h` and the height above
rail are reported as well. Distances are from the sensor, not from the front of the train (the
sensor-to-coupler offset is unknown).

### 8. Temporal confirmation

Candidates are associated to short-lived tracks (lateral gate 1.2 m, along-track gate −8/+2.5 m
per frame). **With the fusion stage enabled (the default) this filter is bypassed** – the
fusion's own tracks and hysteresis (§15) decide; the rules below apply only with
`fusion_enabled: false`. A track is then confirmed after `confirm_hits = 2` hits within the last
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

## 9. Sensor profile, axes and forward crop (`sensor_profile.py`, `frame.py`)

The first cloud of a stream (frame_id, fields, point count) defines a profile that is reused
for every following cloud. For the organised Hesai-like clouds (slot `i` → ring `i % 128`,
column `i // 128`; adjacent column pairs are the far/near echoes of one firing) the profile
holds per-ring elevations, per-ring azimuth offsets and per-column azimuths measured from the
returns, so that every slot – also a no-return slot – has a direction. Firing columns outside
±80° of forward are dropped before any processing (the 921 k clouds shrink to 410 k points).

The configured axes (`forward_axis`, `up_axis`) are kept while the track bed is found with
them. Otherwise (another lidar) the axes are detected: forward = the signed axis with the
largest extent; up = one of the two signs of the axis with the narrower angular spread (the
vertical field of view of these lidars is the narrow one) whose rail-level fit succeeds **and
has returns beyond the fitted surface** (drainage trough, ballast between sleepers – a ceiling
has nothing beyond it; on the Falcon Prime recording the dense ceiling fits a plane better than
the dark track bed, so the fit quality alone would choose the wrong sign). Among the remaining
candidates the known conventions win (REP-103 `+x/+z`, Falcon native `+z/+x`, the development
bags `-y/+z`). The Falcon Prime bag (`x` forward, `z` up, pitched up 2°) is handled this way
without any configuration.

## 10. Range image (`range_image.py`)

A `(2, H, W)` image of near and far returns: for organised clouds rows are the rings sorted by
elevation and columns true-azimuth bins of 0.1° (column azimuth + the ring's offset, so a
vertical edge in the scene is vertical in the image); for unorganised clouds (Falcon Prime,
no no-return slots) the returns are binned by angle (0.09° × 0.08°) keeping the nearest and the
farthest return of a cell. Every pixel keeps its message slot. Only the sector of
`d2_az_limit_deg` (±25°) around forward is built: the corridor at `d_min` = 3 m spans less.
Cost: 4–6 ms for an organised cloud, ≈ 15 ms binned.

## 11. Static scene and model hold (`pipeline.py`)

The fraction of range-image pixels whose range did not change since the previous frame
(compared with the 3×3 neighbourhood of the previous image, because a raster lidar jitters by
a fraction of a pixel) tells whether the train stands. While it stands (`static_min_frac` =
0.9) the first `static_warmup` = 5 frames are still estimated and the element-wise median of
their rail/curve models is then frozen: on stationary data the per-frame centre-line estimate
used to drift (yaw to its bound and a spurious curvature on a straight station within 2 s),
and the corridor drifted away from an object standing 34 m ahead. The last good model is also
held for `model_hold_s` = 1 s when the rail fit fails (an object right in front of the train
hides the near bed): D2 and the fusion keep working and the status can be `detected`, never
`clear`.

**Quasi-static smoothing.** A raster lidar's returns flicker: on the Falcon Prime bag the
standing train never exceeds 0.88–0.89 unchanged pixels, so the freeze never engages and the
per-frame fits wobbled the far corridor (centre-line standard deviation 0.20 m at 100 m and
0.51 m at 150 m, `verified_to_m` jumping between 96 and 156 m). Lowering the freeze threshold
is not an option (moving Hesai trains show 0.85–0.9 in a third of their frames). Instead, while
the unchanged fraction is ≥ `quasi_static_frac` = 0.85 and the model is not frozen, the box of
each frame is built with the element-wise median of the last `quasi_static_frames` = 5 fits
(this frame's own fit included; only successful fits enter the list, and it is cleared as soon
as the scene moves or a freeze starts), i.e. a bounded lag of ≤ 0.25 s at creeping speed and no
lag at all once the train moves. The frame's own fit is kept (`FrameResult.fit`) for the
history. The median is taken over rails of one kind only (plain or ceiling-extended – an
extended rail stores its reach as the observed range, and a mixed median once claimed the bed
to 150 m); for extended rails the whole far part (ceiling profile, height, switch point) is
that of the rail with the median reach, with the base fit medianed on top. (`track_model.extend_rail_by_ceiling`)

The track bed is seen to 25–60 m on a dark or grazing floor, the ceiling to 150–250 m, and the
height of a tunnel section is constant. Per 5 m bin the 98th percentile of the returns ≥ 2.5 m
above the rail within ±1.5 m of the centre-line is the ceiling; where the bed is observed the
median difference is the tunnel height `H` (accepted only if constant within 0.3 m); the
ceiling profile is fitted outwards bin by bin within a gate of 0.3 m + 2 mm/m and a continuity
limit of 30 m; the far rail level is the ceiling profile minus `H`, shifted to meet the near
fit where the bed ends. A pressure gate, a station or a change of cross-section breaks the
constant-height test and the extension stops there. On the Falcon Prime bag the bed is seen to
24 m and the extension carries the rail to 140 m with an error of 0.1–0.2 m at 100–150 m; the
Hesai bags extend to 130–150 m.

## 13. D2 – the beam detector (`det_beam.py`)

Every return of the near layer inside the clearance corridor (same geometry as the box, with a
smaller bottom growth beyond the observed bed) is a candidate pixel; candidates are grouped by
8-neighbourhood in the image. A group is judged by what surrounds it in the image:

| cue | meaning |
|---|---|
| background | neighbours with no return (organised sensors) or a return ≥ 3 m / 5 % farther – the object stands in free space in front of the tunnel |
| attached | neighbours at a similar range outside the corridor – a continuous surface poking in; a group with > 35 % of its known neighbours attached is structure, unless the material it is connected to is compact (≤ 2.5 m along the track and ≤ 6 × the group): an object straddling the corridor edge |
| grounded | the first return below the group is the bed at the same distance (positive evidence while the bed is seen) |
| extends above | the first return above the group, at the same range, lies above the box top – column, wall, gate, cable reaching the ceiling |
| sliver / pole | thin and long along the track at the lateral edge (platform coping, duct), or a thin vertical post at the edge (signs) |
| hanging | the lowest pixel ≥ 1.6 m above the rail beyond 50 m – lamps, cables |
| see-through | ≥ 50 % of the group's beams also return from the tunnel behind it (second echo ≥ 1 m / 3 % farther): a semi-transparent medium → class `dust` |

The score is the pixel count relative to a 0.5 m object at that range times (1 − attached
fraction), scaled down for hanging / structure classes. One or two beams are enough to become
a candidate; the fusion decides over time. 2–8 ms per frame.

## 14. D3 – the neural detector (`det_nn.py`, `nn_input.py`, `nn_model.py`)

A U-Net (0.67 M parameters, plain convolutions) on a 64 × 256 crop of the range image around
the forward direction (the dense band of the Hesai sensor; resampled for other sensors) with
7 channels per frame – near and far range, intensity, validity, height above the rail and
lateral offset from the centre-line (D1's model of the same frame), the range residual – for
the current and the two previous frames (21 channels). Output: per-pixel obstacle / dust /
background; connected components become candidates with their pixels' geometry. Trained on
synthetic scenes (objects ray-cast into real tunnel frames and parametric tunnels, see
`train/`); runs in fp16 on the GPU (≈ 1 ms forward, ≈ 5 ms end to end) or on the CPU
(≈ 10–15 ms), switching to the CPU on a CUDA fault. Components below `d3_min_pixels` only add
weak evidence.

## 15. Fusion over time (`fusion.py`)

Candidates of the three detectors are associated to tracks in (d, l) (gates: −8 / +2.5 m along
the track per frame, 1.2 m laterally). Each track accumulates a log-likelihood ratio: a hit of
detector *k* adds `llr_hit[k] · score` (2.0 / 1.2 / 1.0 for D1 / D2 / D3; D1 scores have a floor
of 0.3 because its candidates are already vetted by its rules), two detectors with score ≥ 0.3
in the same frame add 0.6, a frame without a hit subtracts 0.8, a dust hit subtracts 1.5; the
sum is clipped to [−2, 6]. A track is confirmed when it was hit in at least two frames and

* **two distinct detectors** voted (score ≥ 0.3) in the last 5 frames – "minimum two of three"
  – at every range; the LLR must reach 3.0 within 120 m and 4.0 beyond, where a track must also
  have been seen in ≥ 4 of the last 5 frames (3-frame blips of far wall returns at 160 m on the
  20-minute ride were confirmed with 3);
* the earlier exception "beyond 120 m one detector persisting in 4 of 5 frames is the second
  vote" is off (`far_single_detector`): on the pressure-gate bag it confirmed a wall swept into
  the corridor at 133 m on D1 alone while D2 called the same returns structure. The cost on the
  Falcon bag is a few frames (the dark person is confirmed at 147 m, D2 first, D1 a little
  later); on `val_hesai_param` the 100–150 m recall fell from 0.18 to 0.08.
* D3 adds evidence (LLR) but does not vote (`vote_min_score_det d3 = 1.01`): with the first
  model its votes produced 56 false frames on the platform bag. D3 candidates outside the
  corridor (±0.3 m) or beyond `verified_to_m` are dropped before fusion.

A confirmed track stays confirmed while its LLR is above 1.2 and D1 or D2 still hits it in the
last 5 frames – weak hits below the voting score count here (the Falcon person walking 100 →
4 m is hit only weakly in ≈ 190 of 710 frames; requiring a voting hit lost them), D3-only hits
do not (small D3 blobs with score 0.3 kept a stale track alive for 40 frames on a synthetic
scene) – and disappears after ≈ 8 missed frames. Track geometry (range, lateral position) is
taken from D1, else D2; a frame with D3 hits only adds evidence and leaves the track in place,
because D3 components are coarse range-image blobs (on untextured synthetic floors they spanned
10–20 m and dragged tracks away from their object), and D3 hits never start a track (a track
born from a blob one frame earlier took the D2 hits away from the real, confirmed track, which
then coasted and died while the blob's track could not confirm). D3 components longer than 6 m
along the track are dropped before fusion. Confirmed tracks are
written back into the frame result as confirmed candidates (synthetic ones for tracks hit only
by D2/D3 or coasting), so the message, the markers and the offline tool are unchanged. A single
detector never confirms within 120 m however persistent: on the real clear bags static
structure (niches, brackets) persisted with high D2 scores for seconds.

## 16. Output of the corridor

`verified_to_m` – the range up to which the corridor is built and checked in this frame – is
the minimum of the track-model support (median of the last 5 frames + 15 m), the far rail
validity (ceiling extension or observed bed + 15 m), the wall-sweep cap (median of the last 5
frames, §3) and `d_max`. The RViz corridor (centre line, edges and the translucent volume) is
drawn up to it; "CLEAR" never claims more. Medians on the real bags: 170 m (doubleT_obstacle),
176 m (platform + switch), 102 m (pressure gate, the bend caps it at 85–115 m when it comes
into view), 84–90 m (round tunnels with curves), 156 m (Falcon).

## 17. Rules for the organisers' insertion style

The organisers announced that they will test on data like `cloud_with_fake_obj`: objects inserted
into real Hesai frames (a wall, 0.3 m cubes, side boxes, overhead boxes, a low plate, a hanging pole;
`work/specs/11_fake_style.md`). The rules below handle that style; each one is a parameter in
`config/detector.yaml` and can be switched off. No cue
of the organisers' generator is used (intensity exactly 1.0, off-grid angles, the appended tail of
the cloud): geometry and physics only.

**Speed of unorganised clouds (`range_image.py`, `pipeline.py`).** The edited clouds have a varying
point count (307 187 / 307 174 / 307 200), so the range image is binned. The binning is now exact and
fast on the CPU, and runs on the GPU (torch scatter) while the device manager is in GPU mode; a GPU
error moves the binning back to the CPU for the rest of the run (hot fallback). The GPU and CPU binnings give
the same image (same winner per bin, same ties). D1 runs in a worker thread beside the range
image, and D3 beside D2 (`METRO_PARALLEL=0` runs everything in one thread; results are identical
frame by frame). D3 is warmed up in the worker thread it runs in (its kernel setup is per thread: a
warm-up in the calling thread left 4 s on the first frame). D2's connected components are labelled on the
bounding box of the in-box pixels only.

**D2 empty-bin bridging (`d2_bridge_empty_bins`).** An object sampled more coarsely than the bins
(the inserted plate) falls apart into 1–2 px fragments with empty bins between them and no
neighbour. On binned streams of the Hesai class (vertical span ≥ 33°) D2 joins small in-box
fragments across empty bins and looks through empty bins for their neighbours. The Falcon Prime
(25°) is not bridged: there the rule joined the person's fragments to structure in frames where the
corridor model breaks. A joined group is split where the mean ranges of its fragments, sorted, jump by more
than the attachment margin (`bridge_split_range`, 2026-10-20): through the gap between the head and the
shoulder of a person 20 m ahead the tunnel returns 204 m behind him are in-box fragments one empty bin away,
and joined to him they made a group 185 m deep that was dropped as structure (no D2 vote, the person was never
confirmed). The same split of *plain* 8-connected groups (a 2 × 2 m wall at 15 m joined with station returns
27–65 m away above its top edge) was tried and rejected: it splits station structure as well and gave 7 false
frames at 160 m on roundT_squareT_pressureGate_squareT.

**Plate exception in D1 (`d1_plate_exception`).** A cluster that is wide (≥ 0.8 m), short along the
track (≤ 0.6 m), centred (≤ 0.6 m from the centre-line), low (≤ 1.2 m above the rail) and has ≥ 6
voxels is not rejected as a "flat patch at range" or a "grazing surface". It votes only together with
a D2 obstacle group at the same place whose boundary ring is at least half background
(`plate_d2_min_bg_frac`): a plate on the bed casts a shadow, while the 0.2 m step of a platform edge
is continuous with the surface behind it.

**Far weak D1 vote on a standing train (`far_weak_d1_vote`).** While the scene is static (≥ 85 %
unchanged range-image pixels), beyond `far_range_m` a D1 cluster rejected only for its point support
(one scan line, ≥ 3 voxels) votes when a voting D2 obstacle group lies at the same place in the same
frame. This is the dark person at 150 m on the Falcon bag.

**Far D1 vote minimum on a moving train (`far_vote_d1_min_d`, 140 m).** While the train moves,
beyond 140 m a D1 cluster with fewer than 5 voxels adds evidence but does not vote. This removes the
sparse far wall returns at 156–165 m on the 20-minute ride (otherwise 5 false frames in 2 episodes).

**Compact pair vote (`compact_pair_vote`).** A small compact D1 cluster (≤ 1 m in every direction,
≥ 3 voxels, also one rejected only for its support or as a short edge sliver at least 0.1 m thick) and
an isolated D2 obstacle group (≥ 4 px, no attached neighbours, ≥ 90 % background ring) at the same
place in the same frame both vote, whatever their size-normalised scores (a 0.3 m cube scores
0.15–0.25 in both detectors). The D1 cluster must stand alone: no larger cluster and no other cluster
of ≥ 3 voxels within 2.5 m along and 2 m across – a dust cloud breaks into many small clusters a metre
apart, a cube has nothing around it.

**Corridor stability (`track_model.py`, `pipeline.py`).** The straight-section extension of the
verified range no longer switches as a step at the curvature threshold: it fades out linearly over
20 % of the threshold (`corridor_ext_ramp_frac`), and while the scene is quasi-static it is decided on
the median |b| of up to 300 frame fits (the tunnel cannot bend while the train stands; Falcon's
per-frame |b| scatters 0.7–2.2·10⁻⁴ around the thresholds and flipped the verified range between 81
and 141 m). A frame whose own fit is clearly curved (|b| > 2.5·10⁻⁴) is never averaged away. The
median model of a static stop keeps the rail kind of the last fit unless a plain last fit faces an
extended majority, and only while the window's median |b| is below the straight-section threshold
(in bends the plain last fit is kept). Off by default: the curvature shrink along the
measured shape of a bend (`corridor_curv_shape_factor`) and the floor step-up rule of the rail fit
(`floor_step_up_max_m`, 0.5 m fixes a corridor break on the Falcon bag at 4193–4233 but shortened the
near run on synthetic dust).

**RViz and status cosmetics.** The status text is placed 20 m ahead and 7 m above the sensor (it sat
on the bottom edge of the default view), the emergency-CPU text 2 m below it. A confirmed track that
coasts, or is hit only by a 1–2 px D2 group, is published with the extent and point count of its last
substantial observation (`ObstacleGeometryMemory`, 30 frames) instead of "0.0×0.0 m, 1 pts".

## 18. Exemptions found with the organisers' ground-truth markers

On 2026-10-20 the organisers sent their three synthetic bags again (`dataset/org_synth_markers`, clouds
byte-identical to `org_synth`) with `/synthetic_obstacle_markers`: a box around every inserted object
in every frame it is visible. Scored against them, three objects inside our gauge were missed, and in
each case a D1 structure rule rejected a cluster that D2 saw in almost every frame. One exemption per
case is added, and a fourth one for a cube that is inside the gauge only by its top face. Each one is a
parameter in `config/detector.yaml` and can be switched off.

**A beam across the rails (`axis_cross_exempt`).** The beam lies 0.53–0.67 m above the rail, from
0.62 m right to 1.64 m left of the axis. It sticks out of the ±1.0 m low zone on one side and touches
the wall there, so the attachment test found more returns outside the box than inside and rejected
it as "attached to structure at box edge". The attachment test is now skipped for a cluster that
reaches at least 0.25 m to both sides of the centre-line and is at most 0.5 m deep along the track.
Structure that leaks in at the box edge (a curve or floor-model error) never reaches across the axis.
The depth limit was added after a dust cloud across the corridor, 1.2–1.6 m deep, was confirmed
through the first version of the rule on `val_dust` (the beam is at most 0.11 m deep).

**A cable hanging into the corridor (`hanging_central_exempt`).** A tilted cable runs from the
ceiling down to 2.0 m above the rails, 0.1–0.6 m from the axis. It continues above the box top, so
"extends above the box top (wall/column/ceiling structure)" rejected it. Within 70 m, a cluster whose
lowest return is at least 1.8 m above the rail and whose every return is at least 0.5 m inside the
lateral box edge now skips both the "extends above" and the attachment test. Walls, columns and
gates stand at the box edge or reach the floor. With the rule on, the six real bags and the
20-minute ride give the same reported frames as without it, but not the same D1 candidates. On the ride D1 now
accepts hanging clusters it rejected before (frame 9946: two at 44.6–45.9 m, 2.4 m above the rail,
confidence 0.75 / 0.88), and only the fusion (D2's vote, the temporal rule) keeps them unreported. Beyond 70 m the
rule is off: there the corridor model is less certain and the "floating far object" rule (beyond
100 m) takes over.

**A cube on the left rail (`straddle_compact_exempt`).** The cube spans 0.83–1.17 m from our
centre-line, so only a 0.1–0.28 m strip of it lies inside the low zone. That strip is a "thin sliver at
the lateral box edge"; the compact pair path (§17) is meant to rescue such slivers, but the attachment
test counted the cube's own outside part as structure (34–64 outside returns against 6–9 own). In the
compact path only, the outside neighbours are now grown through the outside returns, with a step of
0.15 m + 0.004 m per metre of range. If the cluster and the grown set together stay within
`compact_max_extent_m` (1 m) in every direction, they are one object and the attachment test does not
reject it. A cover, a platform edge or a wall continues along the track or upwards and grows past
the limit. Such a cluster is marked `straddle`. Its D2 partner may have as little as 40 % background
in its boundary ring, because the rest of the cube fills one side of the ring (0.40–0.55 measured at
5–29 m). Every other compact pair keeps the 90 % rule. D2 must still agree in the same frame, and the
isolation rule of §17 still applies.

**The top face of a cube beyond the edge (`straddle_thin_exempt`).** The switch cube #1 stands
1.02–1.49 m from the axis and 0.50–0.86 m above the rail. Below 0.8 m it is outside the ±1.0 m low zone,
so near the train only its top 0–9 cm reach the ±1.4 m body zone. That slice is too thin for the compact
path (`compact_min_thickness_m`, 0.1 m), and it was rejected as an edge sliver at 16–48 m. In the compact
path a short sliver (at most 0.6 m along the track) now passes when the object it forms with the outside
returns, grown as above, is compact and at least 0.1 m wide and tall. The thickness is then judged on
the whole object rather than on the slice inside the box. A cube that touches a wall grows into it and
stays rejected.

**Not done: widening the low zone.** The organisers call all 13 objects obstacles; 6 of them stand
1.1–2.3 m from the axis at 0–0.85 m above the rail, outside our ±1.0 m low zone. A full chain with the
low zone widened to ±1.7 m (`--set gauge_halfwidth_low=1.7`, without these exemptions) found more of the marker
objects (rail cube 0 → 51 frames, beam 0 → 31, low cube 0 → 22) and nothing new on the real bags
or the 20-minute ride. However, it lost 75 frames on `cloud_with_fake_obj`: cubes 113 → 58, side boxes
54 → 28, plate 49 → 35, pole 30 → 14. The duct / platform edge at 1.35–1.55 m entered the box, merged
with objects next to it and broke the isolation of the small ones. It also gave 37 false frames in the
switch bag (trackside equipment at 1.45 m, 0.45–0.6 m above the rail, as the train passes it) and lost
the cable. The cubes at 1.2–1.7 m in the platform bag stayed undetected anyway. The low zone therefore
stays at ±1.0 m (`experiments.md`, "Widening the low zone").

**Not done: large blocks entering the box at its edge, and far objects seen without their lower
part.** Both were measured on the existing runs before any code was written.

- The side boxes of `cloud_with_fake_obj` (2.4 × 2 m, inner face 0.9–1.3 m from the axis) show up in
  D1 as a tall, narrow vertical strip at the body-zone edge, 1.4–1.5 m tall. Within 60 m it is rejected
  as an edge sliver in 27 of 29 frames, while D2 sees a group in every frame. The same strip is common
  on the real recordings: 555 such clusters on the 20-minute ride (105 with a D2 obstacle group at the
  same place), 156 on squareT, and a 2.3 m column with a D2 score of 1.0 in a double-track section of
  roundT_doubleT. Geometrically, an inserted box at the edge cannot be told from a column or a wall face
  that a corridor error brings into the box.
- The far person of the platform bag (100–150 m) is rejected as a "floating far object" (only returns
  above 1.5–1.7 m). In none of those frames does D2 see an obstacle there, so relaxing the rule with D2's
  agreement would not recover him. On the 20-minute ride it would let through hanging structure at
  120–145 m in 27 frames. At that range the person is limited by the verified corridor (100 m in this
  bag) and by 1–4 returns per frame, not by the rule.

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
* **No ego-motion estimate, no map.** Evidence is accumulated per track over time (fusion) and
  the track model is frozen while the train stands, but frames are not registered to each
  other; deskew is not applied (the per-point timestamps' semantics are undocumented).
* **Detection range is bounded by the sensor's returns**: the Hesai-like bags contain no return
  beyond ≈ 209 m; a dark-clothed person on the Falcon Prime bag gives 2–8 returns at 100 m, which
  the fusion confirms in 510 of 510 frames while he stands and in 710 of 710 while he walks
  from 96 to 4 m. Objects beyond ≈ 150 m on the Hesai sensor and beyond ≈ 100 m for dark
  targets on the Falcon are detected only when their returns persist over several frames.
* **"Two of three" is strict at every range**: an object seen by one detector only (e.g. a
  crouching person at 100 m with 8 returns, below D1's voxel threshold, or a far object D2 sees
  two frames before D1) is not reported until a second detector agrees; this is the price of
  zero false positives on the six development bags and of the organisers' rule "minimum two
  algorithms agree".
* **Far sparse wall returns** (two episodes of 3 and 2 frames at ≈ 160 m in a double-track
  section of the 20-minute ride without the rule) are not confirmed: on a moving train a D1 cluster of fewer
  than 5 voxels beyond 140 m does not vote (§17). The price: a small object beyond 140 m is
  confirmed on a moving train only once D1 sees 5 voxels on it.
* **Dust was validated on synthetic single-return dust only**: there is no real dust recording.
  The see-through and roughness cues need a second echo or range noise; on synthetic dust without
  them D3's dust class and the fusion's dust veto are what remains (objects behind dust are
  reported in 18 % of the 20–50 m frames). The veto does not fire on dense clouds inside the corridor: 89
  confirmed detections lie on a labelled dust cloud with no object within 2 m,
  50 of them in one scene where the cloud sits on the track axis.
* **The organisers' low plate** (1 cm thick, 0.5 m above the rails) is confirmed from 90.8 m
  through the plate exception and the empty-bin bridging (§17); the bridging works only on
  binned Hesai-class streams. Of the other inserted objects, the 0.3 m cubes are confirmed below
  ≈ 50 m (1–9 returns beyond that); the side boxes (inner face 0.9–1.3 m from the axis, rejected
  by D1 as attached to the corridor edge), the overhead boxes (bottom 3.1–3.3 m above the bed) and
  the 7 cm hanging pole are not confirmed – real lintels and signs live at that height, and zero
  false alarms on the real recordings came first.
* **Distance sweep in the insertion style** (`experiments.md`, `val_sweep`): a standing person is
  confirmed from 5 to 80 m, not at 100–150 m (10 → 4 rendered points, mostly below the clearance
  floor there); nothing is reported beyond the verified range (≈ 170 m in a straight station
  section) – a wall at 200–270 m still gets 28–45 points from the renderer, but there is no tunnel
  model there to decide whether they are in the clearance; a 0.5 m box ends 2 cm above the
  clearance floor; a wall filling the view at 5 m hides the bed and the status is `UNKNOWN`; a
  near object whose top edge touches in-box returns of far structure (a 2 × 2 m wall at 15 m below
  a station's far ceiling structure) is not confirmed by D2 when it appears without an approach.
* **Bends beginning beyond the model support close the corridor** at the first bin where a
  wall crosses it (§3): objects standing in the bend beyond that point are reported only once
  the train is close enough for the centre-line to follow the curve. The check needs the walls
  to be sampled along the track (sparse far walls seen as single columns do not qualify – the
  two-detector rule is the backstop there) and runs only with D2 enabled (it lives in the
  verified-range computation).
* **Stations**: the corridor model is weaker where the free-space cue finds no plausible tunnel
  width; the ceiling extension stops where the cross-section changes.
* Only one positive example exists; thresholds for far-range rules were tuned on the six
  development bags and are not independently validated (see `docs/experiments.md`).
* `unknown` is reported when the track bed cannot be observed (blocked sensor, empty cloud);
  a wrong `forward_axis`/`up_axis` also ends in `unknown`, never in `clear`.
