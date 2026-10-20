# Synthetic LiDAR data (synth-builder)

Labelled rosbag2 scenes for testing and training the obstacle detectors: objects inserted into **real** Hesai
frames, **parametric** tunnels for the Hesai-like sensor and for the **Falcon Prime**, and **dust** that must not
be detected. Interface B of `work/specs/00_interfaces.md`; the beam models (interface A) live in
`src/metro_obstacle_detector/metro_obstacle_detector/beam_model.py`.

## Run

```bash
conda create -n imos-synth python=3.10 && conda activate imos-synth
pip install numpy==1.26.4 scipy warp-lang open3d rosbags mcap pyyaml pytest
cd solution/synth
python -m synth.generate --config configs/val_hesai_param.yaml --out /home/gag/i.mos/synthetic/val_hesai_param
#   --only 0,3 (scene indices)   --max-frames N (smoke test)
tools/gen_sets.sh val_hesai_insert val_hesai_param ...     # several sets, logs in synthetic/_logs
tools/check_sets.sh val_hesai_param                         # verify_set.py + container_check.sh
```

If the host shell sources ROS (`PYTHONPATH` from `/opt/ros`), run with `env -u PYTHONPATH`. Ray casting uses
NVIDIA Warp (GPU); without CUDA it falls back to Open3D, then to brute-force NumPy (`IMOS_RAYCAST=open3d|numpy`).
The three backends gave identical hit counts for all five object classes at 20 m (one check during development;
`tests/test_synth_scene.py` compares meshes with analytic boxes/capsules). Scene *i* of a set depends only on
`default_rng([seed, i])` and the generator version. `--workers N` renders scenes in parallel processes.

## Sets (`/home/gag/i.mos/synthetic/<set>/<set>_<i>_<pos|neg>[_dust]/`)

| set | content |
|---|---|
| `val_hesai_insert` | objects in real frames of the two stationary segments (stations: `squareT_platform_squareT_switch` 323-674, `doubleT_platform` 269-344), 8 scenes |
| `val_hesai_insert_tunnel` | objects in the real double-track tunnel of `doubleT_obstacle` with the real object removed, 307k layout, 6 scenes; header stamps are the real ones, so every scene reproduces the recording's 200 ms and 400 ms gaps between messages (median 100 ms) |
| `val_hesai_param` | parametric tunnels (circular / rectangular / double track, straight and R 300-2000 m, platforms), ego 0-20 m/s, 10 scenes |
| `val_falcon_param` | the same for the Falcon Prime, humans dark (5-15 %) and normal (15-60 %), 10 scenes |
| `val_dust` | dust clouds, single- and dual-return scenes, objects inside / behind dust, 10 scenes |
| `train_*` | larger sets with other seeds (`train_hesai_insert`, `train_hesai_param`, `train_falcon_param`, `train_dust`, `train_dust_falcon`) |

Counts (`tools/set_stats.py /home/gag/i.mos/synthetic`, `results/set_stats.json`, 2026-10-17); "object-frames" counts every
labelled object in every frame, "visible obstacle-frames" those with `is_obstacle` and `n_hits >= 3` by `distance_m`:

| set | sensor | scenes (pos/neg) | frames | GB | object-frames by class | visible obstacle-frames by distance (m) | backgrounds / tunnels |
|---|---|---|---|---|---|---|---|
| val_hesai_insert | hesai128 | 8 (7/1) | 784 | 1.08 | box 348, debris 468, crouching 196, standing 392 | 0-50: 544, 50-100: 318, 100-150: 194 | stations: squareT_platform 4, doubleT_platform 4 (stationary) |
| val_hesai_insert_tunnel | hesai128 | 6 (5/1) | 600 | 0.94 | box 100, crouching 300, lying 200, standing 300 | 0-50: 400, 50-100: 328 | cleaned doubleT_obstacle tunnel 6 (stationary) |
| val_hesai_param | hesai128 | 10 (8/2) | 1200 | 1.84 | box 480, debris 120, crouching 120, lying 240, standing 840 | 0-50: 241, 50-100: 229, 100-150: 437, 150-210: 306 | double 4, circular 2, rectangular 4; curves 5, moving 8 |
| val_falcon_param | falcon_prime | 10 (8/2) | 1000 | 1.59 | box 300, debris 300, crouching 400, lying 100, standing 500 | 0-50: 185, 50-100: 371, 100-150: 415, 150-210: 206, >210: 102 | double 5, circular 3, rectangular 2; curves 3, moving 6 |
| val_hesai_curve | hesai128 | 3 (2/1) | 360 | 0.55 | box 360, crouching 120, standing 240 | 0-50: 177, 50-100: 215, 100-150: 4 | double R -331 / rectangular R -405 / double R 518 m; sight limit 74 / 86 / 88 m |
| val_hesai_far_dark | hesai128 | 3 (3/0) | 360 | 0.53 | one standing person, rho 0.07-0.10, at 183.6 / 200.1 / 203.3 m | person slots/frame mean 10.8 / 1.5 / 1.2 | straight level track, stationary; beam divergence 0 |
| val_hesai_far_dark_div | hesai128 | 3 (3/0) | 360 | 0.53 | same seeds and plans | person slots/frame mean 7.8 / 4.8 / 1.3 | beam divergence 0.1 deg (unmeasured for Hesai; bracket) |
| val_dust | hesai128 | 10 (10/0) | 1000 | 1.58 | dust 1800, box 700, crouching 300, standing 200 | 0-50: 817, 50-100: 221 | 5 single-return / 5 dual-return scenes; moving 6 |
| train_hesai_insert | hesai128 | 24 (22/2) | 2208 | 3.20 | box 1352, debris 528, crouching 904, lying 928, standing 1656 | 0-50: 2512, 50-100: 1428, 100-150: 569 | squareT_platform 8, doubleT_platform 8, cleaned tunnel 8 |
| train_hesai_param | hesai128 | 36 (32/4) | 4320 | 6.66 | box 1920, debris 840, crouching 2760, lying 1200, standing 3600 | 0-50: 1408, 50-100: 1874, 100-150: 1453, 150-210: 713 | circular 14, rectangular 10, double 12; curves 19, moving 29 |
| train_falcon_param | falcon_prime | 20 (18/2) | 2000 | 3.17 | box 900, debris 600, crouching 500, lying 400, standing 1800 | 0-50: 476, 50-100: 639, 100-150: 648, 150-210: 296, >210: 316 | double 9, circular 7, rectangular 4; curves 5, moving 16 |
| train_dust | hesai128 | 14 (14/0) | 1120 | 1.79 | dust 2320, box 480, crouching 400, standing 720 | 0-50: 666, 50-100: 474, 100-150: 17, 150-210: 32 | 5 single-return scenes; moving 11 |
| train_dust_falcon | falcon_prime | 4 (4/0) | 320 | 0.52 | dust 560, box 240, crouching 80, standing 80 | 0-50: 5, 50-100: 190, 100-150: 28, >210: 30 | 1 single-return scene; moving 2 |

All sets: 0 consistency errors (`tools/verify_set.py`: one label line per message with the header stamp,
`n_hits == slot labels` per frame and object, labelled slots are returns and fired slots, obstacles inside the
gauge, dust never an obstacle). Insertion sets contain no visible obstacle beyond 150 m: the stationary real
backgrounds see the track to ~125-135 m (station ends, the tunnel's sight limit); far Hesai positives (150-210 m)
are in the parametric sets only.

Parametric `scene.json` has `grade_profile`: constant grade per scene (|grade| <= 2 %), rail level z(s) = grade * s
along the centre-line arc length, the horizontal profile (straight lead-in, curve radius, where the curve starts),
and `sensor_frame_rail_level_m` = the rail line in the sensor's analysis frame, h(d) ~ -mount_height +
(grade - tan(pitch)) d. The sensor frame is not pitched with the grade, so a constant grade looks like a sensor
pitch of -atan(grade). `labels.jsonl` carries `sight_limit_m` for every frame. Check against the chord formula
S = 2 sqrt(2 R a - a^2) (a = distance to the inner wall / column row) in `val_hesai_curve`: R 331 m, a 2.4 m -> 79.6
m formula vs 74 m measured; R 405 m, a 2.3 m -> 86.2 vs 86; R 518 m, a ~1.9 m -> 88.7 vs 88.

Every scene: `bag/` (MCAP, zstd chunks, rosbag2 metadata version 5 as written by Humble; ~1.35-1.6 MB/frame),
`scene.json`, `labels.jsonl`, `slot_labels.npz`. Proven readable in `metro-lidar:gpu-dev` with `ros2 bag info`,
`ros2 bag play` and `ros2 run metro_obstacle_detector offline --bag <scene>/bag` (`tools/container_check.sh`;
for Falcon scenes add `--set forward_axis=+x --set up_axis=+z`).

### Labels

`labels.jsonl` has one line per message (same order, `stamp_ns` = header stamp) with the interface-B keys and
* `center_sensor`: object centre in the sensor frame at the header time; `extent_dlh`: box along track /
  lateral / up; `d_along_m`: along-track distance (arc length) sensor -> object centre; `lateral_m`,
  `height_above_rail_m`: object **centre** relative to the track centre-line and rail level (as the detector's
  candidates); `bottom_above_rail_m`, `top_above_rail_m` in addition;
* `n_hits` = number of message points labelled with the object (`slot_labels.npz`, checked by `tools/verify_set.py`);
  `distance_m` = range of the nearest returned point, else the geometric nearest surface;
* `in_clearance`: the object box overlaps the gauge (|l| <= 1.0 m below 0.8 m, <= 1.4 m up to 3.4 m, bottom
  **0.10 m** = train underbody, so lying persons count); `is_obstacle` = `in_clearance` and not dust;
* `sight_limit_m`: parametric - farthest along-track distance where the track (1 m above rail) is in line of
  sight within range and field of view; insert_real - 99.5th percentile of returns in the tunnel cross-section;
* insert_real extras: `placement` (`local_floor` | `rail_model` | `rail_model_extrapolated`), `occluded_by_scene_frac`
  (fraction of the object's beams whose real return is nearer; placements above 0.25 are re-drawn), `real_frame`.
  `rail_model*` objects stand at rail level + the scene's `floor_offset_m` (median local floor - rail model on the
  centre-line, -0.05 m when unknown; since b6d5b60 - the shipped insertion sets used rail - 0.15 m, i.e.
  `bottom_above_rail_m` = -0.15 for those objects);
* `occluded_by_objects_frac` (all object labels, since b6d5b60): fraction of the object's geometric centre-ray
  beams on which another object is nearer; an object hidden completely gets 1.0 and, having no returned points,
  `n_hits = 0`, `visible = false` (not present in the shipped sets, which were generated before);
* dust: class `dust`, `is_obstacle: false`, `reflectivity` = median equivalent reflectivity of its soft returns,
  `extinction_peak_per_m`, `visibility_m`, `object_id_behind`.

`slot_labels.npz`: `frame`, `slot` (message point index; for the Hesai layouts the organised slot index), `obj_id`
(object ids 1.., dust ids 100..).

## Sensor models

**Hesai-like (dataset sensor).** `hesai128(2400|7200)`: exact separable beam model (ring elevation, ring azimuth
offset +-7.7 deg, column azimuth -0.1 deg per firing, dual-return column pairs, ring firing parity, +-50 deg point-
azimuth crop of the 307k topic); every valid return of all six bags is reproduced to < 5e-5 deg, no return falls
into a slot the model calls unfired. Output = the real `/lidar_points` layout (26-byte points, column-major,
(0,0,0) no-return slots with ring and timestamp, header 10 Hz, point time = header + firing time). Mount 1.3 m above
rail on the centre-line; every beam is cast from the sensor position at its own firing time.

**Falcon Prime.** `FalconBeamModel` = slot table of the real recording (`data/falcon_prime_seyond.npz`, 95 618
(scan_id, scan_idx, flags) slots, median direction and time). Each frame draws a per-line azimuth shift (sigma 0.06
deg, measured). Output = the recorded layout (`/iv_points`, frame `seyond`, point step 48, fields x y z timestamp
intensity flags elongation scan_id scan_idx is_2nd_return, no zero points, header = end of scan, point times -100..-5
ms); `coordinate_mode`: `as_recorded` (x forward, y left, z up; default), `upside_down`, `native` (x up, y right, z
forward). Mount 1.98 m above rail, pitched up 2.33 deg (as recorded), beam divergence 0.1 deg (datasheet; 7 sub-rays
on objects).

**Return model** (`synth/returns.py`). P = logistic in ln(rho cos^0.45 f / r^2), 10 % at 200 m (Hesai) / 250 m
(Falcon) -> P = 0.9; dropout 2.5 % / 2 %; no Hesai return beyond 209 m; range noise 2 cm + 4 mm / 5 mm quantisation;
intensity = 100 rho cos^0.15 x lognormal(0.45) (Hesai: x min(1, (60/r)^1.3)). Edge mixed pixels: a beam next to a
farther neighbour gets that range as a second echo with p = 0.35 (Hesai) / 0.30 (Falcon).

**Dust** (`synth/dust.py`): Gaussian puffs of extinction, analytic optical depth per ray (Hahner et al. fog model
applied to puffs); soft return at the peak of alpha e^-2tau / t^2 with rho_eq = pi alpha/40 sr x 0.75 m x e^-2tau,
hard returns behind survive with P(g e^-2tau)/P(g); dual return = dust near / background far, single return =
strongest; puffs drift and are re-drawn every frame. The soft echo's measured range is clipped to the surviving hard
echo minus one quantisation step, so near <= far holds in every beam (since b6d5b60; before it, range noise
could put a soft return up to a few cm behind its hard echo).

## Realism (measured, not assumed)

Gate (`tools/gate_hesai.py`, `results/gate_hesai.json`): the real object in `doubleT_obstacle` (frames 30-45, per-slot
difference to 6 same-parity background frames) has **145.8** slots/frame (131-176); a 0.9 x 0.9 x 0.85 m box at its
position, ray-cast with the beam model into background frames and counted the same way: **136.6** (ratio **0.937**;
bottom -0.1/+0.1 m: 0.855/0.950; lateral +-0.2 m: 0.921). That count assumes every hit returns (P = 1). Through the
production renderer (`tools/gate_hesai_production.py`: `overlay_objects` + calibrated `ReturnModel.hesai()`, rho 0.3,
2.5 % dropout, dual-return echoes) the same box gives **132.8** measured slots (130.0 labelled `n_hits`), ratio
**0.911**. The real object spans rings 53-62 (taller and narrower than the box): totals agree, the per-ring profile
does not.

Falcon gate (`tools/gate_falcon.py`, `results/gate_falcon.json`): spec box at 50/55/60/65 m -> 83/80/85/80 slots on
target; an upper-body box 0.5 x 0.8 x 0.95 m gives 1.07-1.16 x the datasheet ROI raster (148/124/109/95 slots vs
138/114/96/82 at 50-65 m). The real walking person (dark clothing) returns 96-156 first returns at 75-50 m. The
procedural 1.75 m human (clothing girth 1.15, the generator draws 1.0-1.3) gives real/synthetic 1.39-1.75 when only
centre rays count and **1.03-1.20** with the datasheet beam divergence of 0.1 deg, counting a slot as on target when
**at least 1 of its 7 sub-rays** hits the human (with "at least 2 of 7" the ratio is 1.10-1.29), i.e. within +-20 % in
all 14 compared frames. The generator itself does not use a hard count rule: the footprint fraction f enters the
detection probability, which at these ranges is close to the 1-of-7 rule. (Before the girth term the model was
0.46 m wide vs 0.67 m real, ratio 1.17-1.40.)

Far range (`tools/object_hits.py`, `results/far_dark.json`): a dark (rho 0.07-0.10) standing person on straight level
track gives 10.8 / 1.5 / 1.2 slots per frame (mean) at 183.6 / 200.1 / 203.3 m without beam divergence and 7.8 / 4.8 /
1.3 with 0.1 deg (`val_hesai_far_dark`, `val_hesai_far_dark_div`, same seeds). Around 200 m the count is set by where
the 307k raster falls on the silhouette (0.35 m between firings, ~0.45 m between rings): 1 centre ray at 200 m vs 6 at
183.6 m; the Hesai divergence is not measured, so the two sets bracket it.

Parametric vs real (`tools/realism_hesai.py` on `roundT_squareT_pressureGate_squareT`, `tools/realism_falcon.py` on
the Falcon recording; numbers in `results/realism_*.json`):

| quantity | Hesai real / synthetic | Falcon real / synthetic |
|---|---|---|
| no-return fraction of fired slots, rings 0-24 / 25-88 / 89-127 | 0.015 / 0.025, 0.034 / 0.032, 0.038 / 0.024 | table slots: 0.028 / 0.025 |
| beams with two different echoes / 2nd-return fraction | 0.0137 / 0.0124 | 0.021 / 0.024 |
| farthest on-axis return (p99.9) / corridor points beyond 100, 150, 200 m | 105.9 / 107.9 m | 330, 81, 30 / 392, 69, 26 |
| floor / wall / ceiling points per frame 2-10 m | 21 722 / 123 080 / 232 vs 21 163 / 116 804 / 250 | |
| same 30-80 m | 1 270 / 2 254 / 562 vs 1 535 / 3 010 / 616 | |
| same 80-210 m | 79 / 295 / 190 vs 81 / 297 / 300 | |
| intensity median floor / wall / ceiling 2-10 m | 9 / 9 / 7 vs 8 / 8 / 8 | wall 9 vs 8; ceiling 10-30 m: 16 vs 7 |

Known limits: synthetic 30-80 m returns are 10-34 % too many and far ceiling returns +58 % (Hesai); Falcon ceiling
intensity is 2-5x higher in the recording, its left wall almost disappears beyond 30 m (629 vs 2 572 points at 30-80
m) and the real far end (269 m) is shorter than the synthetic tunnel; tunnels are idealised (no signals, switches,
cables hanging, no other trains); dust realism is **not** measured (no real dust in any recording); insertion uses
only stationary segments (stations + the cleaned `doubleT_obstacle` tunnel), moving segments are skipped (a 1-DoF
along-track displacement estimator, `synth/motion.py` `segment_motion_dp`, exists: Viterbi over per-pair alignment
costs with a 3 m/s^2 prior; on 4 real segments at 0-15 m/s its two-frame closure error is p90 <= 0.02 m, max 0.15 m
when the train stops; it is not wired into the insertion yet); the rail level beyond the observed track bed is
extrapolated (`placement` label); parametric grades are constant per scene (no vertical curves).
**Background overlap:** only three stationary real segments exist, so `train_hesai_insert` re-uses the background
frames of `val_hesai_insert` (315 of 332 squareT frames, 76 of 76 doubleT_platform frames) and of
`val_hesai_insert_tunnel` (104 of 104); objects and seeds differ. Use the insertion val sets for detection on known
backgrounds, the parametric val sets and the six real bags for unseen backgrounds.

Speed (307k layout, Warp on the RTX 5060 laptop): **1 129 frames/min wall with `--workers 4`** (8 parametric
scenes x 120 frames of `val_hesai_param`, 313 frames/min per process, other agents' containers running);
423-444 frames/min for one process; 468 frames/min wall with `--workers 3` on 60-frame scenes under heavy load
(scene start-up dominates short scenes); dust scenes ~65 frames/min per process (per-ray puff integrals on the CPU).

## Tests

`tests/test_beam_model.py` (beam models vs all bags, < 0.02 deg; Falcon table on held-out frames),
`tests/test_synth_writer.py` (CDR round trip through `cloud_io`, MCAP + Humble metadata),
`tests/test_synth_scene.py` (return-model calibration, ray-box/capsule vs meshes, object sizes, label consistency
`n_hits == slot labels`, dust never an obstacle). Host: `env -u PYTHONPATH python -m pytest tests/test_beam_model.py
tests/test_synth_*.py`; in `metro-lidar:gpu-dev` the Warp/Open3D/MCAP-dependent tests skip.
