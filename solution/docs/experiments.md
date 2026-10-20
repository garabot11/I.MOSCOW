# Experiments

> **Кратко по-русски.** Все числа — образ поставки `metro-lidar:gpu-dev` (ID `a0e1e7e33476`), офлайн-прогон
> каждого кадра без подмонтированного кода. Шесть записей организаторов (2488 кадров) и 20-минутная поездка
> `new_data` (11 271 кадр) — **0 ложных кадров**; объект на 56 м — с кадра 11 до ухода с пути. Falcon Prime:
> тёмный человек подтверждён с **154 м**, стоя на 100 м — 510 из 510 кадров, подход 100 → 4 м — 710 из 710,
> до его появления — 0 ложных из 2723. Объекты, вставленные организаторами (`cloud_with_fake_obj`): стена — с
> 76 м, кубы 0,3 м — с 37–52 м, плита — с 91 м, чистая часть — 0 ложных. Синтетика организаторов с разметкой:
> найдены все 7 объектов внутри габарита (49 % объект-кадров), 8 из 13 по всему списку (35 %), 0 ложных.
> Время на 8 закреплённых ядрах ноутбука: p95 39–75 мс, ни одного кадра дольше 100 мс из 3480. Всё измерено на
> ноутбуке; на стенде организаторов (i7-9700E) не проверялось.

All numbers below were produced on the development laptop (Intel Core 7 240H, 16 logical CPUs, 15 GiB RAM,
RTX 5060 Laptop 8 GB, Ubuntu 24.04 host, **Ubuntu 22.04 + ROS 2 Humble + PyTorch 2.11 cu128 inside Docker**)
with the delivery image `metro-lidar:gpu-dev`, ID `a0e1e7e33476`. Nothing has been run on the organisers'
stand (i7-9700E, RTX 4070 Ti SUPER); their control recordings have not been seen. The detector does not use
file names, timestamps or frame counts; every recording is processed with the same configuration
(`config/detector.yaml`).

## Data and labels

* **Six organisers' recordings** (Hesai, 2488 clouds, 250 s, `manifests/bags.yaml`). One of them
  (`doubleT_obstacle`, 921 600 points per cloud, `lidar_livox` frame, train standing) contains a foreign object
  (~0.9 × 0.85 m, probably a crouching person) at 55.5–56.7 m that later walks off the track. Labels:
  `annotations/events.yaml` (see `docs/data_notes.md`).
* **Falcon Prime** (`falcon_test_second_dark`, Seyond, 4443 clouds, ~95 k points, axes +X forward / +Z up): a
  person in dark clothes appears at ~154 m (frame 2723), walks to 100 m, stands there (frames 3181–3690) and
  walks to 4 m (3691–4400).
* **The organisers' additional data:** `cloud_with_fake_obj` (1510 edited Hesai clouds, unorganised, with
  inserted objects: a 2 × 2 m wall, 0.3 m cubes, side and overhead boxes, a low plate, a hanging pole; frames
  804–1509 clean), `new_data` (a 20-minute ride, 11 271 clouds, 221 files) and the synthetic rides
  `synthetic_{platform,switch,tunnels}` with the ground-truth topic `/synthetic_obstacle_markers` (13 objects:
  a person, cubes, a cube on the rail, a hanging cable, a 2 m cube, a beam across the rails; dust).
* **Our synthetic validation sets** (`synth/`): objects inserted into real Hesai frames, parametric tunnels and
  curves for Hesai and Falcon, dust, far dark targets, the organisers' insertion style (`val_fake_style`) and a
  distance / object-count sweep (`val_sweep`, 75 scenes, 4 460 frames).
* **No independent split is possible with one real obstacle.** The thresholds were tuned on the same
  recordings; the numbers are development results. The organisers' control recordings are the only
  independent test.

## Protocol

* **Offline runner** `ros2 run metro_obstacle_detector offline`: every message in order, no DDS, no losses, the
  same per-frame pipeline as the ROS node. One container per chain (`scripts/eval_chain.sh`), results compared
  with `scripts/chain_summary.py` and `scripts/scoreboard.py`.
* **Reported detection** = a confirmed candidate in a frame with status `detected`. **False frame** = a frame
  with a reported detection and no labelled object at that place.
* **Synthetic scoring** (`scoreboard.py`): positives are labelled obstacles inside the clearance with the top
  ≥ 0.3 m above the rail; recall per object-frame in range bands.
* **Marker bags** (`work/markers/score.py`): a reported detection matches an object when its centre lies within
  the object's box widened by 1 m across, max(2 m, 3 % of the range) along the track and 1 m vertically. "In
  gauge" = the object's box overlaps our danger box (0.30–3.40 m above the rail; |lateral| ≤ 1.0 m below 0.8 m,
  ≤ 1.4 m above) by at least 5 cm.
* **Distance** is Euclidean, from the sensor to the nearest reliable surface of the object.
* **Processing time**: monotonic clock around the whole pipeline per cloud, without message decoding.

## Results

### Organisers' recordings

| recording | frames | reported frames | first detection | verified range p50, m |
|---|---:|---:|---|---:|
| doubleT_obstacle (921 k points, standing) | 201 | 65 (the object at 56 m, frames 11–75) | frame 11, 55.6 m | 170.5 |
| doubleT_platform | 345 | 0 | – | 83 |
| roundT_doubleT | 252 | 0 | – | 91 |
| roundT_pressureGate_roundT | 268 | 0 | – | 84 |
| roundT_squareT_pressureGate_squareT | 545 | 0 | – | 102 |
| squareT_platform_squareT_switch (307 k points, moving) | 877 | 0 | – | 175.5 |
| `new_data` (20-minute ride) | 11 271 | **0** | – | 93 |

In `doubleT_obstacle` the object stands on the track until ≈ frame 66, then walks to the left between the tracks;
from frame 76 no return lies inside the clearance and the status is `clear`.

### Falcon Prime (`falcon_test_second_dark`)

| phase | frames | reported |
|---|---:|---:|
| before the person appears (false-alarm phase) | 2723 | **0** |
| appearance at 154 m (2723–2729) | 7 | 3 |
| walk 150 → 100 m | 451 | 390 |
| standing at 100 m | 510 | **510** |
| walk 100 → 4 m | 710 | **710** |
| first detection | | **frame 2727, 154.2 m** |

The verified range stays at 155.5 m in 98 % of the frames (30 jumps of more than 20 m in 4433 frames), so a
person at 100 m is inside the checked corridor in every standing frame. D1 sees 1–13 returns of the person on
the far walk; D2 must agree before a report.

### Objects inserted by the organisers (`cloud_with_fake_obj`, 1510 frames)

| object (frames where it is in the clouds) | reported frames | first report |
|---|---:|---|
| wall 2 × 2 m across the track (0–231) | 115 | frame 117, **76.1 m**, tracked to 3 m |
| four 0.3 m cubes on the track (236–532) | 113 | each cube from 47.5 / 51.5 / 36.9 / 52.2 m to ≈ 3.5 m |
| side boxes (435–633) | 0 (the frames counted in this window are the third and fourth cubes) | – |
| overhead boxes, bottom 3.1–3.3 m above the bed (534–682) | 0 (frames 534–536: the fourth cube's track coasting at 4.2 m) | – |
| low plate 1.3 m wide, 0.5 m above the rails (682–744) | 49 | frame 691, **90.8 m**, to 17.5 m |
| hanging pole (710–803) | 0 (frames 710–739: the plate) | – |
| clean part (804–1509) | **0** | – |

### The organisers' synthetic rides with ground-truth markers (1667 frames)

| object | in our gauge | reported / visible frames, first report | CPU mode (D3 every 3rd frame) |
|---|---|---|---|
| platform: person | in | 220/386, from 119 m | 220 |
| platform: cubes right / left (1.2–1.7 m from the axis) | marginal | 0 / 0 | 0 / 0 |
| switch: cube right #1 | in | 65/256, from 110 m (17 of 42 frames at 0–50 m) | 65 |
| switch: cube on the left rail #2 | in | 19/111, from 29 m (13 of 14 frames at 0–20 m) | 15 |
| switch: cable hanging from the ceiling #3 | in | 30/140, from 50 m (every frame at 0–50 m) | 30 |
| switch: cubes left #6 / right #9 | out | 0 / 52 | 0 / 51 |
| tunnels: 2 m cube #6 | in | 34/167, from 61 m | 34 |
| tunnels: beam across the rails #8 | in | 39/50, from 74 m (every frame at 0–50 m) | 38 |
| tunnels: low cube right #5 | in | 13/103, from 30 m | 13 |
| tunnels: cubes left #1 / #9 (1.8–2.3 m from the axis) | out | 0 / 0 | 0 / 0 |
| **objects in our gauge: object-frames within 150 m** | | **420/850 (49 %)**, found 7 of 7 | 415/850 |
| all 13 objects of the organisers' list | | 472/1354 (35 %), found 8 of 13 | 466/1354 |
| false detections / alarms on dust | | **0 / 0** | 0 / 0 |

The detector stops reporting a beam's track 4 frames after the beam leaves the view (frame 493, held at 3.0 m);
the switch ride has two such holds (frames 416–419 and 653–655). These are stale holds of real objects, not
false alarms.

**Widening the low zone was measured and rejected.** Six of the 13 objects stand 1.1–2.3 m from the axis, up
to 0.85 m above the rail, outside our low zone (±1.0 m; the conductor-rail duct begins at 1.2 m). With
`--set gauge_halfwidth_low=1.7` (chain `G-wide17`):

| recording | ±1.0 m (delivered) | ±1.7 m |
|---|---|---|
| six real recordings, `new_data` | 0 false frames | 0 false frames |
| `cloud_with_fake_obj`: cubes / plate | 113 / 49 | **58 / 35** (the duct / platform edge at 1.35–1.55 m enters the box and merges with the objects) |
| marker bags: objects in gauge | 420/850 | 438/850, but the cable is lost |
| marker bags: false frames | 0 | **37** (trackside equipment 1.45 m from the axis on the switch ride) |

The style the organisers announced for testing loses a third of its detections, and their own switch scene gains
false alarms, so ±1.0 m is kept; the beam, the cable and the cube on the rail are covered by the exemptions of
`algorithm.md` §18 instead.

### Synthetic validation

| set | scenes / frames | recall 0–20 m | 20–50 m | 50–100 m | 100–150 m | false frames |
|---|---|---:|---:|---:|---:|---:|
| val_hesai_insert (people / boxes inserted into real Hesai frames) | 8 / 784 | – | 0.83 | 0.38 | 0.61 | 0 |
| val_hesai_insert_tunnel | 6 / 600 | – | 0.50 | 0.32 | – | 0 |
| val_hesai_param (parametric tunnels) | 10 / 1200 | 1.00 | 0.33 | 0.50 | 0.29 | 0 |
| val_hesai_curve | 3 / 360 | 1.00 | 0.91 | 0.03 | 0.00 | 0 |
| val_falcon_param | 10 / 1000 | 0.31 | 0.19 | 0.05 | 0.00 | 4 |
| val_dust (single-return dust) | 10 / 1000 | 0.51 | 0.18 | 0.10 | – | 71 (strict count) |
| val_hesai_far_dark (dark person 150–210 m) | 3 / 360 | – | – | – | 0.00 | 0 |
| val_fake_style (the organisers' insertion style) | 12 / 1797 | 0.91 | 0.90 | 0.07 | 0.03 | 0 |
| val_sweep (distances 5–300 m, 1–16 objects) | 75 / 4460 | 0.58 | 0.44 | 0.24 | 0.04 | 0 |

**Dust.** The strict count of val_dust marks a frame false when the reported object is labelled invisible behind
the dust. Of the reported detections, 89 lie on a labelled dust cloud with no object within 2 m (50 of them in
val_dust_006, a cloud on the track axis): the dust filter does not hold on dense clouds in the corridor. In the
organisers' marker bags the dust is sparse (1–12 points) and gave no alarm.

**Sweep, one object at a fixed distance, standing train** (corridor verified to 170.5 m; 40 frames per cell,
the first frame is the confirmation delay; in brackets the median number of rendered points):

| object | 5 m | 10 m | 20 m | 40 m | 60 m | 80 m | 100 m | 125 m | 150 m | 175 m+ |
|---|---|---|---|---|---|---|---|---|---|---|
| person 1.7 m | 39 [3243] | 39 [700] | 39 [239] | 39 [58] | 39 [28] | 39 [24] | 0 [10] | 0 [9] | 0 [4] | 0 (beyond the verified range) |
| box 0.5 m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cube 0.3 m, 1 m above the bed | 39 [728] | 39 [167] | 39 [42] | 39 [9] | 0 [4] | 0 [1] | 0 [2] | 0 [1] | 0 [1] | 0 |
| wall 2.9 × 4 m | 0 (UNKNOWN) | 0 | 39 [3462] | 39 | 39 | 39 | 39 [216] | 37 [126] | 37 [84] | 0 (beyond) |

What limits each row:

* **Beyond ≈ 170 m nothing is reported – by design.** There is no tunnel model there to decide whether the
  returns are inside the clearance (real Hesai returns end near 200 m). A person gives 0–3 points at 200–300 m.
* **Person at 100–150 m:** 10 → 4 rendered points, most of them below the clearance floor at that range.
* **Box 0.5 m:** its top ends 0.41 m above the rail, the clearance floor at this station is 0.39 m: 2 cm of it
  is inside. A design limit (the floor keeps rail heads, sleepers and bed debris out).
* **Cube beyond 40 m:** ≤ 4 rendered points; the compact pair vote needs 3 D1 voxels.
* **Wall at 5 m** hides the whole bed: the rail level cannot be estimated and the status is `UNKNOWN`, not
  `CLEAR`. On an approach the wall is tracked from 76 m down to 3 m.

**Several objects per frame** (1–16 objects at 15–150 m, inside and beside the clearance): no false frame in
any scene; on a standing train 59 of 60 frames detect, on a moving one 147 / 148 / 92 / 86 of 150 frames with
1 / 4 / 8 / 16 objects (the added objects are the ones the single sweep already misses, and the insertion style
draws objects that do not occlude each other). The processing time does not grow with the number of objects.

**Moving train from 300 m** (`new_data` 4750–4899 and 1250–1399, person / box / cube / wall): nothing is
reported and no false frame either; the base runs into a curve, so the verified range falls from 172 to 87 m and
the object stays beyond it.

### Corridor stability

Frames whose verified range differs from the previous frame by more than 20 m, and the p95 lateral move of the
centre-line between frames (only where both frames verify that far):

| recording | jumps > 20 m | verified p50, m | ≥ 100 m | centre-line move at 50 / 100 m, m |
|---|---:|---:|---:|---|
| doubleT_obstacle | 0 % | 170.5 | 100 % | 0.00 / 0.00 |
| squareT_platform_squareT_switch | 7.0 % | 175.5 | 93 % | 0.09 / 0.16 |
| roundT_squareT_pressureGate_squareT | 10.7 % | 102 | 52 % | 0.14 / 0.29 |
| roundT_doubleT (curve) | 15.9 % | 91 | 39 % | 0.33 / 0.85 |
| `new_data` | 9.2 % | 93 | 47 % | 0.20 / 0.59 |
| Falcon Prime | 0.7 % | 155.5 | 98 % | 0.06 / 0.18 |

In curves the corridor closes where the tunnel wall sweeps into it, so the verified range moves with the view.

## Performance

**Idle machine, 8 pinned cores** (`docker run --cpuset-cpus 0-7`, nothing else running, 20.10.2026, two rounds,
`docker/bench_frames.py`: the same 150 frames of each recording read into memory, then the whole pipeline per
frame like the offline runner). Median / p95 in ms:

| recording | GPU mode |
|---|---|
| doubleT_obstacle (921 k points, train standing) | 27 / 39 |
| squareT_platform_squareT_switch (307 k, moving through a station and a switch – the stand's format) | 49–53 / 65–70 |
| Falcon Prime (95 k, binned range image) | 53 / 56 |
| `cloud_with_fake_obj` (307 k edited = unorganised, binned) | 64–66 / 71–75 |
| val_sweep, 16 objects | 61–62 / 67–68 |

The CPU modes (D3 every third frame) stay within p95 44–77 ms on the same frames. **No frame took longer than
100 ms: 0 of 3480, worst 92 ms.** D3 on the GPU costs 8–21 ms and runs in a worker
thread beside D2; in the CPU modes it runs every third frame and switches itself off after three calls over
120 ms. The stand's i7-9700E (2.6 GHz, no hyper-threading) has slower cores – expect ≈ 1.3–1.5× these numbers
there: the organised 307 k clouds stay within the 100 ms budget, the binned clouds come to ≈ 95–110 ms p95, so
a 10 Hz stream of edited clouds may lose some frames (freshest-frame policy, see below). Other work on the same
cores slows the node down 1.5–2× (measured): the stand should not run anything heavy beside it.

## Live operation (ROS 2 node + `metro play`, 10 Hz)

The node subscribes with a keep-last queue of depth 1: when a frame takes longer than the frame interval, the
waiting frame is replaced by the next one and the detector always works on the freshest cloud. Measured on the
delivery image `a0e1e7e33476` through `./run.sh` on 20.10.2026 (`docker/failure_matrix.sh`, 17 of 17 cases
PASS, 0 tracebacks in every log):

| scenario | result |
|---|---|
| GPU start | `doctor` OK, `D3 on: d3_unet.pt (gpu), warm-up 3.8–6.3 s`, "detector started" after ≈ 10 s |
| `doubleT_obstacle`, rate 1.0, GPU | 196 of 201 frames processed, 59 reported from frame 8 at 55.7 m, p50 28 ms, p95 44 ms; `/obstacles/status` at 10.05 Hz |
| `squareT_platform_squareT_switch`, 8 cores (`--cpuset-cpus 0-7`), GPU | 608 frames in 62 s, p50 20 ms, p95 62 ms, 0 frames over 100 ms |
| `doubleT_platform`, rate 2.0, GPU | 322 frames, 0 reported, p95 55 ms |
| emergency CPU (no NVIDIA Container Toolkit), `roundT_doubleT` rate 2.0 | banner RU + EN with the reason, the fix and the contact; `ВНИМАНИЕ_GPU.txt`; 211 frames, 0 reported, p95 77 ms |
| bag switching (platform → obstacle 921 k → Falcon → roundT) | 468 frames, 3 resets, three different `frame_id`, p95 94 ms |
| two cloud topics (SQLite3, MCAP), MCAP | 27 / 25 / 28 of 30 frames |
| GPU failure at run time | hot switch `gpu` → `cpu_emergency`, warning and file |

## What was hard, what was learned

* **Far range is dominated by tunnel structure seen at grazing angles**: sparse wall, floor and ceiling
  patches, single scan rings, columns and signal posts. What keeps it quiet: no reports beyond the last
  identified tunnel cross-section (`verified_to_m`), "returns continue above the cluster" (walls, columns,
  gates), the attachment test, the flat / grazing tests, two detectors agreeing at every range.
* **The track centre-line is the crux.** A straight corridor sweeps the outer wall of every curve into the box
  beyond ~60 m. The centre-line fuses bed symmetry (near), wall / ceiling free-space centres tracked bin by bin,
  chained profile shifts and the trough centre, with bounded curvature and weak temporal priors.
* **The floor model must not be pulled by the object it should reveal**: a two-stage fit (near bins first, far
  bins only in a gate) and the far rail level from the ceiling profile.
* **Platform edges are only ~1.45 m from the track centre** (car half-width 1.35 m): a wider body clearance gives
  continuous alarms in stations; the low-zone widening above is the same trade-off near the bed.
* **The organisers' insertion style needs its own handling**: edited clouds are unorganised (binned range image,
  exact fast binning), inserted objects do not occlude the tunnel behind them, small objects need a compact
  pair vote of D1 and D2.

## What was not measured (and why)

* The stand itself (i7-9700E, RTX 4070 Ti SUPER, sm_89): the start-up probe (matmul + conv2d against the CPU)
  checks the GPU on site; the image's CUDA libraries carry sm_86 cubins that sm_89 runs by binary compatibility
  (`docs/launch.md`, "Совместимость GPU").
* Precision / recall on independent data and false alarms per hour: the organisers' control recordings are the
  only independent test.
* End-to-end latency from the sensor: the recordings use a different clock domain; only the processing inside
  the node is measured.

## Reproducibility

* Full evaluation chain: `scripts/eval_chain.sh <tag> [gpu|cpu]` (every recording and synthetic set, offline
  runner in the image), summary `scripts/chain_summary.py`, synthetic scores `scripts/scoreboard.py`.
* Launch checks: `docker/failure_matrix.sh --data <dataset>`.
* 8-core timing: `docker/bench_frames.py` (`docker/bench_ab.sh` for an A/B of two images).
* Demo video: `docker/record_demo.sh` (needs an X11 desktop session and ffmpeg).
* Tests: `docker run --rm --gpus all metro-lidar:gpu-dev python3 -m pytest -q /workspace/tests` → 255 passed,
  17 skipped (20.10.2026, GPU).
