# Д3 model card — range-image U-Net

* created (UTC): 2026-10-17T01:32:26Z, git 63fa595, torch 2.11.0+cu128
* architecture: RangeUNet widths [24, 48, 96, 160], decoder `add`, 665923 parameters; input (21, 64, 256) = current frame + 2 previous × channels ['rng_near', 'rng_far', 'intensity_near', 'valid_near', 'h_above_rail', 'lateral_from_centreline', 'residual']; output logits ['background', 'obstacle', 'dust']
* crop: {"n_rows": 64, "n_cols": 256, "el_top_deg": 1.97, "el_bottom_deg": -6.07, "az_centre_deg": 0.0, "az_pitch_deg": 0.1, "az_descending": true, "native_tolerance": 0.02}
* range-image builder used for training: `range_image.build_range_image` {"az_step_deg": 0.1, "binned_res_deg": [0.09, 0.08], "az_limit_deg": null} — inference must use range images built the same way (det_nn warns otherwise); sensors: ['hesai128']
* aux channels populated in training: ['h_above_rail', 'lateral_from_centreline'] (whole-channel dropout p = 0.3); other aux channels are zeroed by det_nn
* obstacle class definition: `object` (see d3_preprocess.target_class); labels from the near-return slot of each pixel
* threshold: {'obstacle': 0.5, 'dust': 0.5}
* training data: {"data": "/home/gag/i.mos/synthetic", "train_sets": ["train_hesai_insert"], "heldout_sets": ["val_hesai_insert", "val_hesai_insert_tunnel"], "scenes": {"train": 20, "val_split": 4}, "frames": {"train": 1856, "val_split": 352, "val_hesai_insert": 784, "val_hesai_insert_tunnel": 600}, "generator_versions": ["synth-1.0"], "split_seed": 0}
* hyper-parameters: {"epochs": 60, "batch": 32, "lr": 0.002, "wd": 0.0001, "focal_gamma": 2.0, "class_weights": "1,4,2", "far_weight": 1.0, "dice_obstacle": 1.0, "dice_dust": 0.5, "margin_cols": 32, "aux": "d1"}

## Validation metrics

### val_split

| band, m | objects | object recall | pixel P | pixel R | pixel IoU | FP comp./frame |
|---|---:|---:|---:|---:|---:|---:|
| 0-20 | 0 | – | 0.000 | – | 0.000 | 0.006 |
| 20-50 | 176 | 1.000 | 0.876 | 0.866 | 0.772 | 0.014 |
| 50-100 | 317 | 0.918 | 0.994 | 0.799 | 0.795 | 0.009 |
| 100-150 | 99 | 0.758 | 0.872 | 0.293 | 0.281 | 0.009 |
| 150-200 | 0 | – | 0.000 | – | 0.000 | 0.000 |
| 200-300 | 0 | – | – | – | – | 0.000 |

frames 352, FP components per frame 0.037, frames with FP 0.037, dust px -> obstacle –, FP comps on dust 0

### val_hesai_insert

| band, m | objects | object recall | pixel P | pixel R | pixel IoU | FP comp./frame |
|---|---:|---:|---:|---:|---:|---:|
| 0-20 | 0 | – | 0.000 | – | 0.000 | 0.004 |
| 20-50 | 620 | 0.805 | 0.992 | 0.832 | 0.826 | 0.003 |
| 50-100 | 468 | 0.547 | 0.980 | 0.787 | 0.775 | 0.008 |
| 100-150 | 196 | 0.990 | 0.974 | 0.934 | 0.911 | 0.008 |
| 150-200 | 0 | – | – | – | – | 0.000 |
| 200-300 | 0 | – | – | – | – | 0.000 |

frames 784, FP components per frame 0.022, frames with FP 0.022, dust px -> obstacle –, FP comps on dust 0

### val_hesai_insert_tunnel

| band, m | objects | object recall | pixel P | pixel R | pixel IoU | FP comp./frame |
|---|---:|---:|---:|---:|---:|---:|
| 0-20 | 0 | – | – | – | – | 0.000 |
| 20-50 | 400 | 1.000 | 0.998 | 0.816 | 0.814 | 0.028 |
| 50-100 | 450 | 0.956 | 0.895 | 0.968 | 0.869 | 0.015 |
| 100-150 | 0 | – | 0.000 | – | 0.000 | 0.077 |
| 150-200 | 0 | – | – | – | – | 0.000 |
| 200-300 | 0 | – | – | – | – | 0.000 |

frames 600, FP components per frame 0.120, frames with FP 0.115, dust px -> obstacle –, FP comps on dust 0

## Limitations

* trained on synthetic data only (interface B); real-data quality is unknown until checked on the real bags (the only real positive is the box at 56 m in doubleT_obstacle);
* the network segments foreign objects; whether they are inside the clearance gauge is decided downstream (fusion);
* residual channel (r_expected) is not populated in training.

## Caveat: validation backgrounds overlap with training

`val_hesai_insert` reuses the real background frames of `train_hesai_insert` (squareT_platform_squareT_switch:
286 of 332 frames, doubleT_platform: 76 of 76; `val_hesai_insert_tunnel`: doubleT_obstacle_clean307 104 of 104).
These sets therefore measure NEW OBJECTS ON SEEN BACKGROUNDS: recall is meaningful, false-positive figures are
optimistic.  The scenes are static, so "objects" counts object x frame instances: `val_hesai_insert` contains
only 14 distinct objects (20-50 m: 7, 50-100 m: 5, 100-150 m: 2) and no object beyond 150 m.
Breakdown on `val_hesai_insert` (tau 0.5, object x frame): persons (standing / crouching) detected in 100 % of
frames at 20-150 m; boxes 61 % (20-50 m) / 80 % (50-100 m); debris 70 % (20-50 m) / 7 % (50-100 m, median 5 px);
objects outside the clearance gauge mostly missed (20-50 m: 0 of 76 instances, one object).

## Real-data check (no synthetic labels; single frames, no temporal filtering)

`train/eval_real_d3.py`, node frame path (build_frame -> Д1 -> build_range_image -> NNDetector.step), tau 0.5,
in gauge = |l - l_c| <= 1.4 m and 0.1 <= h_above_rail <= 3.4 m from Д1's track model:

| bag | frames | result |
|---|---:|---|
| doubleT_obstacle (real object, 921k/360 deg topic; only its cleaned background was used in training) | 201 | object detected in 47 / 47 annotated obstacle frames (n_support 55-64 px, score 1.0); 6 of these 47 frames also have another in-gauge candidate (all <= 3 px) |
| roundT_doubleT (tunnel NOT used in training) | 252 | in-gauge false candidates in 121 frames (1.14 / frame); 74 % have <= 3 px; with n_support >= 5: 37 frames, >= 10: 23 frames |
| roundT_pressureGate_roundT (tunnel NOT used in training) | 268 | in-gauge false candidates in 74 frames (1.21 / frame); 87 % have <= 3 px; with n_support >= 5: 13 frames, >= 10: 1 frame |

False candidates on unseen tunnels sit near the gauge edge (median |l - l_c| 1.1-1.3 m, ~0.5 m above the rail:
contact-rail covers / ducts).  Use Д3 in fusion with its n_support and score, not as a standalone detector.

## More held-out sets (tau 0.5, object x frame recall per band | FP components per frame)

| set | 20-50 m | 50-100 m | 100-150 m | 150-200 m | 200-300 m | FP / frame |
|---|---:|---:|---:|---:|---:|---:|
| val_hesai_param (parametric tunnels, unseen geometry) | 0.93 | 0.42 | 0.04 | 0.00 | 0.00 | 2.60 |
| val_hesai_curve | 1.00 | 0.66 | 0.00 | 0.00 | – | 2.01 |
| val_hesai_far_dark | – | – | – | 0.00 | 0.00 | 1.48 |
| val_dust | 0.50 | 0.77 | – | – | – | 1.81 |
| val_falcon_param (binned range image) | 0.22 | 0.56 | 0.46 | 0.00 | 0.00 | 2.05 |

This model was trained on objects inserted into three real backgrounds only (all objects <= ~127 m): it does
not detect anything beyond ~150 m, produces many false components on parametric / dust / Falcon scenes and is
not trained for Falcon at all.  Real Falcon walking person (falcon_test_second_dark frames 3800-4400, person
at 72 -> 17 m, D3 alone, in gauge, n_support >= 10): within 3 m of the person in 172 / 600 frames, other in-gauge
candidates in 365 / 600 frames.

## Later runs (NOT exported; models in /home/gag/i.mos/work/ml_d3/runs/)

Run 2 = + train_hesai_param + train_dust; run 3 = + Falcon param/dust + real obstacle-free frames of
roundT_pressureGate_roundT and roundT_squareT_pressureGate_squareT as negatives, split by background.
All keep model 1's architecture and input.  None is "not worse on val_hesai_insert_tunnel", the export rule:

| model | val_hesai_insert_tunnel FP/frame, recall 50-100 m | val_hesai_insert recall 20-50 / 50-100 / 100-150 | val_hesai_param recall 150-300 m | roundT_doubleT (real, unseen) frames with false in-gauge cand. n>=10 | doubleT_obstacle object (n>=10) |
|---|---|---|---|---:|---:|
| model 1 (this file) | 0.120, 0.96 | 0.80 / 0.55 / 0.99 | 0.00 | 23 / 252 | 47 / 47 |
| run 2 | 0.012, 0.94 | 0.75 / 0.68 / 0.99 | 0.80 | 29 / 252 | 47 / 47 |
| run 3 (best = epoch 2) | 0.213, 0.94 | 0.75 / 0.46 / 0.61 | 0.79 | 0 / 252 | 47 / 47 |
| run 3 (last epoch) | 0.172, 0.95 | 0.88 / 0.64 / 1.00 | 0.86 | 39 / 252 | 47 / 47 |
