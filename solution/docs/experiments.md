# Experiments

> **Кратко по-русски.** Все 2488 кадров шести записей обработаны в Humble-контейнере: единственное препятствие (объект ~0,9×0,85 м на 56 м, поезд стоит) обнаружено во всех 47 размеченных кадрах, первое подтверждение через 1,6 с от начала записи; одна ложная тревога длиной 1 кадр (край платформы на 58,6 м); точность по кадрам 0,979, полнота 1,0 (на размеченных кадрах, 286 кадров помечены как неопределённые и не считались). Обработка 25–67 мс на кадр (медиана, зависит от режима питания CPU ноутбука; отдельные кадры до ~220 мс) на одном ядре; с учётом декодирования сообщения узел выдерживает ≈ 19 кадров/с для датчика 921 тыс. точек и ≈ 29 кадров/с для 307 тыс. на ноутбуке без привязки к ядрам и ≈ 12 / 19 кадров/с на его E-ядрах (замена более старых ядер стенда, не замер стенда). Потоковый режим при 10 Гц, весь bag, независимый учёт по timestamp (`bag_tools coverage`): обработано 3392 из 3394 отправленных кадров в 10 прогонах всех шести записей; два необработанных кадра — на стрелке, где один кадр обрабатывался 167 мс и следующий заменил ожидающий (политика «самый свежий кадр»); статусы всех обработанных кадров совпали с офлайн-прогоном. Строго нулевых потерь это не гарантирует. Матрица экспериментов: без временного подтверждения — 4 ложных события, без правил отбраковки конструкций — 11, без зависимости порогов от дальности — 12, без модели поворота — препятствие не найдено вовсе. Данные ноутбука; на стенде организаторов не проверялось; дальность обнаружения при приближении измерить не на чем (объект стоит на фиксированных 56 м).

All numbers below were produced on the development laptop (Intel Core 7 240H, 16 logical
CPUs, 15 GiB RAM, Ubuntu 24.04 host, **Ubuntu 22.04 + ROS 2 Humble inside Docker**,
NumPy 1.21.5 / SciPy 1.8.0 of Jammy, one thread). Nothing has been run on the organisers'
stand (i7-9700E); the hidden control bag has not been seen. The detector does not use file
names, timestamps or frame counts; every bag is processed with the same configuration
(`config/detector.yaml`).

## Data, labels and split

* Six development bags, 2488 clouds, 250 s (`manifests/bags.yaml`). One bag
  (`doubleT_obstacle`) contains a foreign object; it was also recorded with a different sensor
  (921 600 returns, `lidar_livox` frame, other mounting) – so it doubles as a test of the
  sensor-agnostic geometry.
* Labels: `annotations/events.yaml`, made by us from projections of every 20th frame of all
  bags and a frame-difference analysis of the stationary `doubleT_obstacle` bag (see
  `docs/data_notes.md`). The object there is ~0.9 m wide, ~0.85 m high (probably a crouching
  person) and moves across the track at 55.5–56.7 m from the sensor. Frames in which it is at
  the very edge of the car clearance (0–15) or beside the track (63–200) are labelled
  *uncertain*, as is the junction/switch scene at the end of `squareT_platform_squareT_switch`
  (frames 745–876) whose semantics we cannot judge. All other frames are labelled *clear*.
* **No frame-level train/test split is meaningful with one positive event**: the thresholds
  were tuned by inspecting all six bags (whole-bag leave-nothing-out). The numbers are
  therefore *development* results; the organisers' control bag is the only independent test.
  Tuning happened in the order dev1…dev9 (strided runs, see `results/`), the final run is on
  every frame.

## Protocol

* Offline: `ros2 run metro_obstacle_detector offline` reads every message sequentially in the
  Humble container (no DDS, no losses); the node and the offline tool run the same per-frame
  pipeline (`pipeline.DetectionPipeline`), so the quality numbers apply to both (confirmed in
  the stream tests: every processed frame had the offline status).
* Frame rule: positive = frame inside an *obstacle* interval, negative = *clear* interval;
  *uncertain* frames are excluded from TP/FP/FN/TN; `unknown` on a positive frame is a miss.
* Event rule: an alarm event is a maximal run of `detected` frames (gaps of one processed
  frame bridged); an alarm entirely on negative frames is a false-alarm event; an obstacle
  event counts as detected when at least one alarm frame lies in its interval; the first such
  frame gives the first-detection time and distance.
* Distance: Euclidean range sensor → nearest reliable surface (10th percentile of the
  cluster's ranges). Reference distances come from the same clouds (±0.3 m), not from
  independent metrology.
* Processing time: monotonic clock around `ObstacleDetector.process` + temporal filter;
  read/deserialisation is measured separately (`read_ms`). No sensor-vs-system clock
  differences enter any number.
* Implementation: `scripts/evaluate.py`, `scripts/ablation.sh`, `scripts/make_report.py`.

## Results – final configuration, every frame (`results/final_offline_v2`)

| bag | frames | TP | FP | FN | TN | uncertain | false-alarm events | processing ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle (921 k pts) | 201 | 47 | 0 | 0 | 0 | 154 | 0 | 61 / 82 / 89 |
| doubleT_platform | 345 | 0 | 1 | 0 | 344 | 0 | 1 (frame 73, t = 7.3 s, 58.6 m, 1 frame) | 38 / 48 / 66 |
| roundT_doubleT | 252 | 0 | 0 | 0 | 252 | 0 | 0 | 43 / 60 / 67 |
| roundT_pressureGate_roundT | 268 | 0 | 0 | 0 | 268 | 0 | 0 | 40 / 61 / 75 |
| roundT_squareT_pressureGate_squareT | 545 | 0 | 0 | 0 | 545 | 0 | 0 | 33 / 62 / 69 |
| squareT_platform_squareT_switch | 877 | 0 | 0 | 0 | 745 | 132 | 0 (one 2-frame alarm on uncertain switch frames) | 29 / 65 / 175 |
| **total** | **2488** | **47** | **1** | **0** | **2154** | 286 | **1** | – |

Frame-level precision 0.979, recall 1.0 (on the labelled frames). Timing above was measured
with three containers running in parallel; single-container timing is in the next section.

**The obstacle event.** Detected in 47 of its 47 labelled frames (raw single-frame candidates
exist from frame 12; the temporal filter confirms at frame 16, t = 1.6 s after the start of
the bag, i.e. 0.4 s after the object enters the clearance as labelled). Reported distance
55.66–56.33 m against a reference of 55.4–56.7 m (the object moves), lateral offset from the
estimated track centre −0.1…−0.9 m, height 0.8–1.1 m above rail (the lowest 0.3–0.5 m is below
the box bottom by design), 13–23 voxels (60–250 raw returns). The train is stationary in this
bag, so **"detection range" here is the fixed 56 m at which the object was placed** – the data
contain no approach to the object, and no object at 100–300 m exists in the development set
(see "What was not measured").

**The false alarm.** Frame 73 of `doubleT_platform` (t = 7.3 s, entering the station): a
0.2 m-wide, 0.5 m-tall fragment of the platform edge 0.76 m left of the estimated track
centre at 58 m. The centre-line error in the station (≈ 0.6 m) let it inside the box; two hits
in three frames confirmed it for one frame (0.1 s). Ablation shows that the structure rules
remove all other platform-edge fragments; this one is neither elongated, nor a sliver at the
box edge (0.21 m wide but 0.1 m inside the edge margin), nor attached to enough outside
returns. Raising the confirmation to 3 of 4 frames removes it at the price of +0.1 s latency
for every detection; we kept 2 of 3 for near range (< 100 m) and use 3 of 4 beyond 100 m.

## Performance (single container, no other load)

| cloud | valid returns | prepare | track model | box | clustering | total median / p95 |
|---|---:|---:|---:|---:|---:|---|
| `doubleT_obstacle` (921 600 pts, 62 % zeros) | ≈ 347 k | 13 ms | 22 ms | 5 ms | 3–10 ms | **46 / 49 ms** |
| `roundT_doubleT` (307 200 pts) | ≈ 190 k | 5 ms | 18 ms | 3 ms | 1–8 ms | **32 / 42 ms** |

These are single-container numbers on the development laptop under good conditions. Repeated
offline runs vary with the CPU power state: the audit re-run gave medians of ≈ 26–56 ms per bag
with maxima up to 217 ms, the run of the corrected image (`results/final_offline_v3`, three
containers in parallel, `powersave` governor) 30–38 ms for the 307 k clouds and 66 ms for the
921 k clouds, maximum 187 ms (a frame at the switch). Single frames above 100 ms therefore
exist; see the streaming section for their effect.

Memory and CPU of the node, measured with psutil every 0.5 s during a stream at 10 Hz
(`results/recheck/resources/`, 2026-09-29, the laptop was **loaded by other applications**
during this measurement – a browser used ≈ 1.3 cores): RSS ≈ 76 MB idle, up to 128 MB with the
307 k-point clouds and up to 220 MB with the 921 k-point clouds, no growth over the bag. CPU:
median 52 % of one core for 307 k clouds (251 of 252 frames processed, processing
median 40 ms); for 921 k clouds the node saturated one core (median 106 %), processing rose to
132 ms median and only 104 of 201 frames were processed – the freshest-frame policy keeps the
latency bounded but halves the rate. On the same machine without that load an hour earlier
the 921 k bag gave 47 ms median and 201/201 frames. The detector is single-threaded
(`OMP/OPENBLAS_NUM_THREADS=1` in the image, DDS adds its own threads); GPU not used. The margin
for the dense sensor at 10 Hz is therefore ≈ 2× on an idle laptop core and disappears under
foreign load; the stand's i7-9700E (older cores, lower single-thread clock) was not measured.

The first implementation took 90–120 ms per 307 k cloud on the host and 170–200 ms in the
Humble container (NumPy 1.21 is slow on row reductions of (N,3) arrays); vectorising the
per-cell percentiles, replacing row reductions by column operations, replacing the axis
matrix product by a signed permutation and reusing the rail-level evaluation brought it to the
numbers above without changing a single result (verified frame by frame).

### Throughput, CPU and slower cores (final image, 2026-09-29)

The processing time above is not the whole cost of a frame in the node: before the callback,
rclpy converted every 8 / 24 MB message into a Python message in the node's only thread. The
node now takes the serialised message (`raw_decode`, default) and reads it with a zero-copy CDR
parser; the prepare stage no longer makes intermediate `(N,3)` copies. Both changes are
result-neutral: the parser gives the same arrays as rclpy on all 2488 recorded messages
(`results/recheck/perf/cdr_parser_equality.json`), and the full offline run of the final image
is identical frame by frame to `final_offline_v2` (`results/recheck/offline_v4_vs_v2.json`).

Measured in a whole-bag stream at 10 Hz (`results/recheck/load/final_*`, psutil every 0.5 s;
"node CPU per frame" = CPU time of the whole node process – decoding, DDS threads, detector –
divided by the processed frames; **throughput = 1000 / node CPU per frame**, the rate the node
could sustain if frames came faster). The laptop's E-cores (Gracemont, ≤ 4.0 GHz, no
hyper-threading) stand in for the older cores of the stand's i7-9700E (Coffee Lake, 2.6–4.4
GHz); this is a proxy, not a measurement of the stand. The node process is pinned to all four
E-cores with `taskset` (its DDS threads can run in parallel, as on the 8-core stand); the player
runs on the other cores.

| cores | cloud | decoding | processed | processing median / p95 | node CPU per frame | throughput | RSS max |
|---|---|---|---:|---|---:|---:|---:|
| unpinned (all 16 CPUs) | 921 k (`doubleT_obstacle`) | raw CDR | 201 / 201 | 48 / 52 ms | 54 ms | ≈ 19 fps | 177 MB |
| unpinned (all 16 CPUs) | 307 k (`roundT_doubleT`) | raw CDR | 252 / 252 | 24 / 45 ms | 34 ms | ≈ 29 fps | 121 MB |
| 4 E-cores | 921 k | rclpy (before) | 192 / 201 | 73 / 80 ms | 107 ms | ≈ 9 fps | 230 MB |
| 4 E-cores | 921 k | raw CDR | 199 / 201 | 73 / 76 ms | 84 ms | ≈ 12 fps | 216 MB |
| 4 E-cores | 307 k | raw CDR | 252 / 252 | 39 / 69 ms | 53 ms | ≈ 19 fps | 121 MB |

On the slower cores the dense sensor needed 107 ms of node CPU per frame with rclpy decoding –
more than the 100 ms frame interval, so frames were dropped (192 / 201); with raw decoding
84 ms (199 / 201). The margin for the dense sensor on such cores is therefore ≈ 1.2×; for the
307 k sensor ≈ 1.9×. Measurements with the node pinned to a single E-core, where its own DDS
threads compete with the detector, are in `results/recheck/load/image_fix2/` (same code): 106 ms
per frame and 150 / 201 frames processed – the node should not be confined to one core.

**Decimation is not a remedy.** The optional `max_analysed_points` (every k-th point of the
analysed region, off by default) cuts the processing of the 921 k clouds from 72 to 44 ms at
120 000 points, but the far-range rules count points: with 180 000 (which halves only the
921 k-point sensor and leaves the other bags bit-identical) the obstacle at 56 m was found in
12 of 47 labelled frames instead of 47 (`results/recheck/offline_cap180k/metrics.json`). It is
documented as a last resort and not used.

## Streaming (ROS 2 node + `ros2 bag play --rate 1.0`, same container)

`scripts/run_stream_test.sh`, `roundT_doubleT` (252 messages, 25 s, 10 Hz), subscription
keep-last depth 1:

| QoS of the subscriber | frames processed | inter-arrival median / p95 | processing median / p95 |
|---|---:|---|---|
| best-effort | 85 of ~195 published in the window | 0.107 s / – | 85 / 118 ms (before optimisation) |
| **reliable (default)** | 211 of 252 | 0.114 s / 0.164 s | 101 / 147 ms (before optimisation) |

With best-effort QoS the 8 MB messages are fragmented on the loopback and whole frames are
lost; with the recorded *reliable* profile 84 % of the frames were processed while the
detector still needed ~100 ms per frame; the remaining 16 % were replaced by fresher frames
(depth-1 queue).

**Earlier run of the optimised detector** (`results/humble/stream_final_*`, reliable QoS,
rate 1.0, playback cut by `timeout`): 244 and 192 clouds were logged. This was reported as
"no drops", which is **not supported**: the logs cover only part of each bag (source indices
0 and 245–251 of `roundT_doubleT`, 0–2 and 195–200 of `doubleT_obstacle` are absent), and the
number of clouds *sent* in that window was not counted independently – the node's own
received counter cannot see frames that never arrived. An independent re-run of the whole bags
(audit of 2026-09-29) processed 251/252 and 198/201.

**Corrected protocol** (`scripts/run_stream_test.sh`, since 2026-09-29): the whole bag is
played (no time limit; the player must exit with 0), QoS *reliable*, depth 1, in one container;
a separate subscriber (`status_monitor --jsonl`) records every published status;
`bag_tools coverage` matches the header stamps of the bag messages (= frames sent), of the node
log (= processed) and of the observer (= published). Two player artefacts of Humble's
`ros2 bag play` were found and removed from the measurement:

* started unpaused, the player runs its clock while it fills the read-ahead queue and then sends
  the first 1–3 clouds in a burst – the depth-1 subscription keeps only the last of them
  (measured: frames 0–1 or 0–2 lost with `--delay 1` and even with `--delay 5`, so it is not
  DDS discovery; frame 0 only with `--read-ahead-queue-size 2`, but then more losses mid-bag
  because reading 24 MB messages from SQLite enters the playback loop). Fix: `--start-paused`
  and `/rosbag2_player/resume` after 3 s;
* the player exits right after publishing the last message, before the reliable transfer of a
  large message completes – the last frame was lost in the audit run. Fix:
  `--wait-for-all-acked 5000`.

**Results of the corrected protocol** (`results/recheck/stream/*/coverage.json`, first corrected image `sha256:8a6f9d0a…`, before raw decoding and `auto`, now kept as
`metro-lidar:pre-fix2-8a6f9d0a`; 2026-09-29, development laptop, all six bags + two repetitions
of the two main bags):

| bag (run) | sent | processed = published | not processed (source index) | processing median / p95 / max | status = offline status | result |
|---|---:|---:|---|---|---:|---|
| `doubleT_obstacle` (3 runs, 921 k pts) | 3 × 201 | 3 × 201 | – | 47–48 / 53–55 / 56–73 ms | 201/201 | 52 × detected, first alarm at bag frame 13 (55.65 m) as offline |
| `doubleT_platform` | 345 | 345 | – | 23 / 40 / 44 ms | 345/345 | the known 1-frame false alarm (frame 73, 58.56 m) |
| `roundT_doubleT` (3 runs) | 3 × 252 | 3 × 252 | – | 26 / 43–44 / 47–50 ms | 252/252 | clear |
| `roundT_pressureGate_roundT` | 268 | 268 | – | 23 / 37 / 45 ms | 268/268 | clear |
| `roundT_squareT_pressureGate_squareT` | 545 | 545 | – | 23 / 39 / 44 ms | 545/545 | clear |
| `squareT_platform_squareT_switch` | 877 | 875 | 798, 800 | 21 / 44 / 167 ms | 875/875 | 2 frames on uncertain frames 776–777, as offline |
| **total** | **3394** | **3392 (99.94 %)** | 2 | | all equal | |

In every run the observer saw 7–9 `UNKNOWN` statuses ("no input …") before the first cloud and
3–4 ("input stale …") after the input stopped. The two unprocessed frames follow a frame of
the switch area that took 167 ms: with a 0.1 s frame interval the next cloud replaced the
waiting one (freshest-frame policy). Processing times vary with the laptop's CPU power state
(`powersave` governor): in a slower CPU state earlier that day the same 921 k-point frames took
53–67 ms median (p95 75–101 ms); with the corrected protocol such a run processed 200 of 201 frames
(`results/recheck/exploration/paused_doubleT_obstacle`), with the old unpaused player 181 of 201
(log not kept; the runs with `--delay 5` and `--read-ahead-queue-size 2` are in
`results/recheck/exploration/`). The streaming numbers are therefore laptop measurements, not a
guarantee of zero loss; the stand (i7-9700E) was not measured.

With RViz2 rendering in software in the same container (the demo videos) the detector's
processing time roughly doubles (≈ 90 ms per 307 k cloud) because RViz competes for the CPU;
without a GUI, or with hardware OpenGL, the numbers above apply.

**Final image** (`sha256:69acee4e…`, raw CDR decoding, `input_topic:=auto`,
`results/recheck/stream_final/`): all six bags and an MCAP copy of `roundT_doubleT` (converted
with `ros2 bag convert` in the image) – **2738 of 2740** frames processed; the node found the
cloud topic by itself in every run (7–8 `UNKNOWN` "waiting for a PointCloud2 topic" before
playback), every processed frame has the offline status, and the only two unprocessed frames
are again 798 and 800 of `squareT_platform_squareT_switch` after a 183 ms frame at the switch.
The MCAP copy gave the same offline records as the SQLite original
(`results/recheck/mcap/mcap_vs_sqlite.json`: 0 differences except the bag name).

At 10 Hz the median processing time leaves a 1.5–4× margin, but single frames above 100 ms
occur (up to 167 ms in the stream, 187–217 ms offline with three containers in parallel), so
the median alone does not guarantee that every frame is processed. The end-to-end reaction is
the frame interval (0.1 s) + processing (25–70 ms) + confirmation (one extra frame).
End-to-end latency from the sensor cannot be measured on the recordings (different clock
domains), only the parts inside the node.

## Experiment matrix (`scripts/ablation.sh`, every 2nd frame, `results/ablation/*`)

The table with all numbers is appended at the end of this file (generated by
`scripts/make_report.py`; every 2nd frame, 1246 frames, 24 labelled positive frames). Summary:

| variant | FP frames | false-alarm events | obstacle found | reading |
|---|---:|---:|---|---|
| `full` | 0 | 0 | yes, frame 16, 55.7 m | reference |
| `no_temporal` | 7 | 4 | yes, frame 16 | single-frame flukes at 60–150 m appear; confirmation costs one frame (0.1 s) |
| `box_only` | 111 | 11 | yes | without the structure rules the corridor beyond ~60 m is full of walls, columns, floor patches: precision 0.18 |
| `fixed_thresholds` | 54 | 12 | yes | one clustering radius / voxel / margin for all ranges: near-range values are too tight at 100 m and the box margins too small for the floor-model error |
| `straight_corridor` | 28 | 8 | **no** (0 of 24 frames) | without the track curve the box at 56 m in `doubleT_obstacle` lies 1.5 m beside the object (sensor yaw + curve), and the outer walls of every curve enter the box |
| `no_voxel` | 2 | 2 | yes, frame 16 | raw (duplicated) returns as cluster support: the point-count thresholds are effectively halved, two far flukes pass; clustering is slower |

Variants:

| variant | what changes |
|---|---|
| `full` | final configuration |
| `no_temporal` | raw single-frame decisions (no confirmation) |
| `box_only` | clearance box + clustering only: all structure-rejection rules and the supported-range limit disabled |
| `fixed_thresholds` | no range dependence: constant clustering radius, voxel, box margins, corridor width, min points |
| `straight_corridor` | no curve model (yaw and curvature fixed at zero) |
| `no_voxel` | clustering on raw (duplicated) returns |

## Quality across development iterations (retrospective)

How the quality changed while the algorithm was improved. The runs `results/dev` … `dev9` were
made during development on the host (ROS Jazzy Python, every 4th frame, the code of that day);
they are re-scored here with `scripts/evaluate.py` against the **current** labels, which did
not exist yet at that time – a retrospective view, not the numbers used for decisions then.
`dev` covered only 404 of the 625 strided frames. What changed is read from the configuration
stored in each run's summary (new or changed parameters); code changes without a new parameter
are not visible there.

| run | frames | TP | FP | FN | false-alarm events | precision | recall | main change (from the stored configuration) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dev` | 404 | 0 | 118 | 12 | 19 | 0.00 | 0.00 | first version: rail level, a track curve fitted to the bed, clustering of raw returns, attachment test |
| `dev2` | 625 | 12 | 208 | 0 | 31 | 0.06 | 1.00 | track centre from bed symmetry, trough, wall / ceiling free space and chained profile shifts; voxelised clustering; "returns above" structure test |
| `dev3` | 625 | 3 | 231 | 9 | 27 | 0.01 | 0.25 | minimum vertical extent far away; `min_points_far` 3 → 4 |
| `dev4` | 625 | 2 | 93 | 10 | 22 | 0.02 | 0.17 | attachment rule, ceiling centres and longer chaining in the track model, shape rule |
| `dev5` | 625 | 2 | 21 | 10 | 9 | 0.09 | 0.17 | margins beyond the observed bed, edge-sliver and low-flat rules; body half-width 1.55 → 1.40 m |
| `dev6` | 625 | 11 | 59 | 1 | 14 | 0.16 | 0.92 | margin tuning (`margin_rms_factor`, shape rule from 60 m) |
| `dev7` | 625 | 11 | 37 | 1 | 13 | 0.23 | 0.92 | grazing / planarity rule, yaw and curvature priors of the track curve |
| `dev8` | 625 | 12 | 0 | 0 | 0 | 1.00 | 1.00 | corridor narrowing with curvature, wider curve residual gates, no curve smoothing |
| `dev9` | 625 | 12 | 0 | 0 | 0 | 1.00 | 1.00 | two-stage floor fit, reporting limited to the supported range |
| `final_offline` | 2488 | 47 | 3 | 0 | 2 | 0.94 | 1.00 | every frame, Humble container |
| `final_offline_v2` | 2488 | 47 | 1 | 0 | 1 | 0.979 | 1.00 | 3-of-4 confirmation beyond 100 m, wider sliver rule |

The low recall of `dev3`–`dev5` is the obstacle being cut by rules that were meant for tunnel
structure (and the track curve missing it at 56 m); `dev6`–`dev8` recovered it while removing
the far-range false alarms. The same components appear, one at a time, in the experiment matrix
above. All numbers are on the development data.

## What was hard, what failed, what was learned

* **Far range (> 80 m) is dominated by tunnel structure seen at grazing angles**: sparse
  wall/floor/ceiling patches, single scan rings, columns and signal posts. Without the
  structure rules (`box_only`) the detector fires on almost every frame at 100–180 m. The
  rules that mattered most, in order: the supported-range limit (no reports beyond the last
  identified tunnel cross-section), "returns continue above the cluster" (walls, columns,
  gates), the attachment test (structure leaking into the box), the flat/grazing tests, the
  minimum vertical extent (two rings).
* **The track centre-line is the crux.** A straight corridor (`straight_corridor`) sweeps the
  outer wall of every curve into the box beyond ~60 m; a single near-range template
  correlation broke at tunnel-type transitions (round → double-track, station); the final
  fusion of bed symmetry (near), wall/ceiling free-space centres tracked bin by bin (all
  ranges), chained profile shifts (relative) and the trough centre (absolute prior) with
  bounded curvature and weak temporal priors was the fourth design.
* **The floor model must not be pulled by the object it should reveal**: a far bin whose
  median is the object lifts a quadratic floor fit by 0.5 m at 80 m; the two-stage fit
  (near bins first, far bins only in a gate and without gaps, linear extrapolation beyond the
  observed bed) fixed this and was found on synthetic clouds first.
* **Sensor-specific effects**: 40–60 % exact-zero returns (a fake obstacle at the origin if
  not removed), duplicated returns (voxelised support), two alternating scan patterns in
  `doubleT_obstacle`, a sensor mounted off the track centre in that bag (found from the trough
  and the wall/column geometry).
* **Platform edges are only ~1.45 m from the track centre** (the car half-width is 1.35 m),
  so the body clearance is 1.40 m and thin edge fragments at the box edge are rejected as
  slivers; a wider box (1.55 m) produced continuous alarms in both station bags.
* Ideas tried and dropped: raw frame differencing (the sensor moves; only usable in the
  stationary bag for labelling), a rail-pair matched filter at rail-head level (rails are not
  distinct peaks in the lateral histogram), a fixed ceiling cap from the 98th height
  percentile (cut standing objects when the ceiling was sparse).

## What was not measured (and why)

* Detection range vs. distance for approaching objects – no such data; the only object is at a
  fixed 56 m. From the point density (docs/data_notes.md) a person-sized object should give
  ≥ 4 voxels up to ≈ 150–200 m in a straight tunnel; whether the structure rules keep such
  detections cannot be shown without data and is therefore **not claimed**.
* Precision/recall on independent data, false alarms per hour/kilometre (4 min of data, no
  speed), end-to-end latency sensor → output (clock domains differ; only the processing time
  inside the node is measured), behaviour on the organisers' stand.

## Demo and reproducibility

* `scripts/run_demo.sh` starts, in one Humble container, the detector node, RViz2 (X11 cookie
  passed through a private Xauthority file, software OpenGL) and `ros2 bag play --rate 1.0`;
  `scripts/status_window.sh` opens a terminal with `ros2 run metro_obstacle_detector
  status_monitor` (one line per result: status, distance, processing time, nearest object).
  `scripts/record_video.sh` records the screen with ffmpeg while the demo runs.
* Videos in `video/`: `demo_obstacle_doubleT.mp4` (the object at 56 m is found and tracked;
  the corridor, the red candidate returns, the cube + label and the status text are visible)
  and `demo_clear_roundT_doubleT.mp4` (a curved tunnel with a transition to a double-track
  section, status stays CLEAR). Playback rate 1.0 (real time); the recording itself is
  slower than real time only where the software-rendered RViz drops display frames – the
  detector's own processing time is printed in the terminal window.
* `scripts/run_demo.sh` takes the RViz fixed frame from the real `header.frame_id` of the bag
  (`bag_tools info --frame-id`, override `FIXED_FRAME=`), the launch argument `fixed_frame`
  writes it into the RViz config together with `input_topic` (checked on 2026-09-29 with
  `doubleT_obstacle`: fixed frame `lidar_livox`, `results/recheck/demo_obstacle_screen.png`).
* Final image: built from a clean copy of `solution/` with `docker build --no-cache --pull
  --target solution` (see the build record), tests run inside the image, offline run without
  mounting sources. Base image `ros:humble-ros-base-jammy`.

## Build record (final image)

* Rebuilt on 2026-09-29 after the second round of corrections (`ПРОВЕРКА_ПО_PDF_ПОСЛЕ_ИСПРАВЛЕНИЙ.md`:
  MCAP storage plugin, raw CDR decoding, fused prepare stage, `input_topic:=auto`, UDP-only Fast
  DDS profile), from a clean copy of `solution/` (results/ and video/ excluded), **from scratch**
  (no build cache, base image pulled again, all apt packages downloaded; 2 min 34 s) with
  `docker build --no-cache --pull --target solution --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) -t metro-lidar:final .`
* Base image: `ros:humble-ros-base-jammy@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138`
* Image ID: `sha256:69acee4ec78544ff2b2b9ea44227312fc169117932bd6e2397a9118971bd5a86`
  (previous images kept locally: `metro-lidar:pre-fix2-8a6f9d0a` = first corrected image,
  `metro-lidar:pre-fix-68a2bea3` = image of the videos)
* Inside the image: Ubuntu 22.04.5 LTS, ROS_DISTRO=humble, rosbag2 storage plugins `sqlite3`
  and `mcap`, packages `metro_obstacle_detector` (`detector_node`, `offline`, `status_monitor`,
  `bag_tools`) and `metro_obstacle_msgs`; all 47 files under `/workspace` (sources, config
  incl. `fastdds_udp_only.xml`, scripts, tests, entrypoint) are identical to `solution/`
  (`results/recheck/image_vs_sources.txt`); `python3 -m pytest -q /workspace/tests` → 38 passed;
  the full offline evaluation of all 2488 frames with this image (`results/final_offline_v4`,
  via `scripts/eval_all_container.sh` with relative paths) is identical frame by frame to
  `results/final_offline_v2` (`results/recheck/offline_v4_vs_v2.json`: 0 differences in index,
  stamp, status, distance, reset flag and detector output) and gives the same metrics.
* Export for delivery without a registry: `/home/gag/i.mos/dist/metro-lidar-final.tar.gz`
  (`docker save metro-lidar:final | gzip -1`, 462 MB, SHA-256
  `1c05211bbed1f4ecb62c6da302a56b7eff8c32a7c37cf57f42a39b428de1d313`); `gzip -t` passes and
  `docker load -i metro-lidar-final.tar.gz` restores the same image ID `sha256:69acee4e…`.

## Appendix – experiment matrix tables (`scripts/make_report.py results/ablation/*`)

| variant | frames | TP | FP | FN | TN | precision | recall | false-alarm events | obstacle detected (first frame / distance) | proc ms median (max of bags) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| full | 1246 | 24 | 0 | 0 | 1079 | 1.000 | 1.000 | 0 | obstacle_1: yes (16 / 55.662 m) | 72 |
| no_temporal | 1246 | 24 | 7 | 0 | 1072 | 0.774 | 1.000 | 4 | obstacle_1: yes (16 / 55.662 m) | 82 |
| box_only | 1246 | 24 | 111 | 0 | 968 | 0.178 | 1.000 | 11 | obstacle_1: yes (16 / 55.662 m) | 98 |
| fixed_thresholds | 1246 | 24 | 54 | 0 | 1025 | 0.308 | 1.000 | 12 | obstacle_1: yes (16 / 55.676 m) | 83 |
| straight_corridor | 1246 | 0 | 28 | 24 | 1051 | 0.000 | 0.000 | 8 | obstacle_1: NO (- / - m) | 89 |
| no_voxel | 1246 | 24 | 2 | 0 | 1077 | 0.923 | 1.000 | 2 | obstacle_1: yes (16 / 55.67 m) | 78 |

### full
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 71.8 / 86.2 / 88.1 |
| doubleT_platform | 173 | 0 | 0 | 0 | 173 | 0 | - | 38.7 / 67.3 / 73.5 |
| roundT_doubleT | 126 | 0 | 0 | 0 | 126 | 0 | - | 45.1 / 75.5 / 87.5 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 39.8 / 61.5 / 71.6 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 0 | 0 | 273 | 0 | - | 32.7 / 65.6 / 69.0 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 29.2 / 61.1 / 165.7 |

### no_temporal
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 81.8 / 89.9 / 102.3 |
| doubleT_platform | 173 | 0 | 4 | 0 | 169 | 0 | [68, 70] t=6.8-7.0s [63.04, 95.304]; [122, 126] t=12.2-12.6s [102.964, 106.484] | 40.0 / 68.3 / 72.9 |
| roundT_doubleT | 126 | 0 | 1 | 0 | 125 | 0 | [40, 40] t=4.0-4.0s [114.468, 114.468] | 45.3 / 77.5 / 83.6 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 50.0 / 74.9 / 94.4 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 2 | 0 | 271 | 0 | [296, 298] t=30.6-30.8s [108.072, 111.88] | 41.9 / 82.6 / 190.8 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 37.3 / 85.5 / 209.4 |

### box_only
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 98.1 / 105.7 / 109.5 |
| doubleT_platform | 173 | 0 | 30 | 0 | 143 | 0 | [62, 64] t=6.2-6.4s [3.691, 3.874]; [72, 76] t=7.2-7.6s [22.608, 45.632]; [84, 138] t=8.4-13.8s [3.348, 112.668] | 47.0 / 90.3 / 104.3 |
| roundT_doubleT | 126 | 0 | 14 | 0 | 112 | 0 | [52, 54] t=5.2-5.4s [124.656, 131.724]; [114, 126] t=11.4-12.6s [37.058, 62.691]; [232, 244] t=23.2-24.4s [90.528, 116.888] | 54.9 / 92.9 / 100.6 |
| roundT_pressureGate_roundT | 134 | 0 | 2 | 0 | 132 | 0 | [52, 54] t=5.2-5.4s [121.956, 125.0] | 41.7 / 62.6 / 81.7 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 1 | 0 | 272 | 0 | [292, 292] t=30.2-30.2s [145.296, 145.296] | 32.8 / 67.5 / 80.7 |
| squareT_platform_squareT_switch | 439 | 0 | 64 | 0 | 309 | 66 | [96, 174] t=9.6-18.0s [3.718, 27.356]; [182, 228] t=18.8-23.4s [3.497, 4.03]; [236, 236] t=24.2-24.2s [3.366, 3.366] | 28.9 / 74.9 / 183.6 |

### fixed_thresholds
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 83.3 / 88.1 / 91.2 |
| doubleT_platform | 173 | 0 | 24 | 0 | 149 | 0 | [48, 48] t=4.8-4.8s [77.908, 77.908]; [58, 62] t=5.8-6.2s [73.412, 78.712]; [70, 76] t=7.0-7.6s [45.772, 57.888]; [82, 88] t=8.2-8.8s [57.1, 71.044]; [96, 110] t=9.6-11.0s [75.6, 112.632]; [124, 138] t=12.4-13.8s [45.916, 57.528] | 65.1 / 84.6 / 97.0 |
| roundT_doubleT | 126 | 0 | 7 | 0 | 119 | 0 | [112, 124] t=11.2-12.4s [88.996, 99.113] | 66.6 / 84.7 / 96.8 |
| roundT_pressureGate_roundT | 134 | 0 | 1 | 0 | 133 | 0 | [170, 170] t=17.0-17.0s [104.257, 104.257] | 68.5 / 77.7 / 92.2 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 1 | 0 | 272 | 0 | [300, 300] t=31.0-31.0s [115.952, 115.952] | 53.5 / 68.1 / 71.4 |
| squareT_platform_squareT_switch | 439 | 0 | 21 | 0 | 352 | 66 | [98, 124] t=9.8-12.4s [20.348, 26.112]; [132, 144] t=13.2-15.0s [21.228, 27.732]; [154, 154] t=16.0-16.0s [25.116, 25.116] | 45.1 / 83.6 / 226.0 |

### straight_corridor
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 0 | 0 | 24 | 0 | 77 | - | 85.2 / 89.7 / 93.8 |
| doubleT_platform | 173 | 0 | 1 | 0 | 172 | 0 | [210, 210] t=21.0-21.0s [65.996, 65.996] | 62.4 / 84.7 / 94.5 |
| roundT_doubleT | 126 | 0 | 12 | 0 | 114 | 0 | [4, 8] t=0.4-0.8s [78.884, 96.204]; [98, 116] t=9.8-11.6s [22.656, 41.684] | 66.4 / 81.3 / 97.6 |
| roundT_pressureGate_roundT | 134 | 0 | 10 | 0 | 124 | 0 | [106, 112] t=10.6-11.2s [31.08, 52.384]; [120, 122] t=12.0-12.2s [29.424, 32.376]; [164, 172] t=16.4-17.2s [24.872, 33.448] | 89.0 / 119.4 / 128.8 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 5 | 0 | 268 | 0 | [330, 336] t=34.0-34.6s [83.448, 91.98]; [346, 348] t=35.6-35.8s [51.308, 54.049] | 89.3 / 110.4 / 124.7 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 42.7 / 108.7 / 154.9 |

### no_voxel
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 78.4 / 84.2 / 88.1 |
| doubleT_platform | 173 | 0 | 2 | 0 | 171 | 0 | [72, 72] t=7.2-7.2s [60.54, 60.54]; [124, 124] t=12.4-12.4s [57.528, 57.528] | 53.2 / 2235.3 / 3367.5 |
| roundT_doubleT | 126 | 0 | 0 | 0 | 126 | 0 | - | 43.7 / 73.4 / 544.8 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 39.0 / 63.0 / 72.7 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 0 | 0 | 273 | 0 | - | 37.2 / 63.6 / 67.2 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 26.9 / 2567.7 / 14040.7 |

