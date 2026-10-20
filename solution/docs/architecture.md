# Architecture

> **Кратко по-русски.** Один ROS 2 узел (`detector_node`): адаптер PointCloud2 → кадр (профиль датчика: раскладка, двойное эхо, оси, обрезка вперёд) → range-image → три детектора на общей модели пути (Д1 кластеры на CPU, Д2 лучевой на range-image, Д3 нейросеть на GPU/CPU) → слияние во времени (накопление свидетельств по трекам) → публикация `/obstacles/status`, `/obstacles/markers`, `/obstacles/candidates`, таймер «нет данных» → UNKNOWN, JSON-лог. Порядок обработки кадра (проверка времени и сброс → профиль/кадр → range-image → тест неподвижности → Д1 (с удержанной моделью, если сцена статична) → Д2 → Д3 → слияние) задан одним классом `pipeline.DetectionPipeline`, который используют и узел, и офлайн-оценщик. Режим GPU/CPU выбирает `device.DeviceManager`; без GPU решение работает в аварийном режиме на CPU и предупреждает об этом (баннер, поле `mode`, надпись в RViz, файл `ВНИМАНИЕ_GPU.txt`).

```
                 rosbag2 (SQLite3 / MCAP, PointCloud2)          live sensor (future)
                        │  ros2 bag play (metro play)                │
                        ▼                                            ▼
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  ROS 2 node  metro_obstacle_detector/detector_node   (node.py)                     │
  │   subscription: <input_topic | auto>, raw CDR, depth 1; re-discovery when the      │
  │   publishers vanish (next bag); DeviceManager (device.py): gpu | cpu_emergency      │
  │   ┌─ DetectionPipeline (pipeline.py) ──────────────────────────────────────────┐    │
  │   │ frame.py / sensor_profile.py: layout, dual returns, axes (auto), crop      │    │
  │   │ stamp check: jump back / gap > max_gap_s → reset everything                │    │
  │   │ range_image.py: (2,H,W) near/far image; static-scene test (frozen model)   │    │
  │   │ D1 detector.py + track_model.py: rail (+ceiling extension), curve, box,    │    │
  │   │    clusters, rules                                                         │    │
  │   │ D2 det_beam.py: corridor pixels, image-neighbourhood rules, see-through    │    │
  │   │ D3 det_nn.py (+ nn_input.py, nn_model.py): range-view U-Net, GPU/CPU       │    │
  │   │ fusion.py: tracks, LLR accumulation, "two of three", persistence, dust veto│    │
  │   └────────────────────────────────────────────────────────────────────────────┘    │
  │   timer: no cloud for stale_timeout_s → STATUS_UNKNOWN; banner repeat (CPU mode)   │
  └────────┬────────────────────────┬────────────────────────┬──────────────────────────┘
           ▼                        ▼                        ▼
   /obstacles/status         /obstacles/markers       /obstacles/candidates
   (ObstacleStatus: status,  (corridor volume up to   (PointCloud2, returns
    distance, obstacles[],    verified_to_m, cubes,    inside the box)
    verified_to_m, mode …)    status + mode text)
           │
           ▼  optional JSONL log (log_path) ─▶ scripts/evaluate.py / scripts/scoreboard.py ─▶ metrics
```

The offline evaluator (`offline.py`, `ros2 run metro_obstacle_detector offline`) reads a bag
with `rosbag2_py` sequentially and feeds **the same** `DetectionPipeline` (identical state
handling, covered by tests); it is used for exhaustive, loss-free quality measurements, while
the node is the live path. `bag_tools coverage` compares a live run with the bag by header
stamps.

## Components

| Component | File | Responsibility | Knows about |
|---|---|---|---|
| Sensor profile / frame | `sensor_profile.py`, `frame.py` | one profile per input stream (frame_id, fields, point count): organised layout (ring-fastest columns, dual-return column pairs), per-ring elevation / azimuth offsets, forward crop of firing columns (921k clouds → 1/2.25 before any work), axis convention (configured; auto-detected for another lidar: forward = largest extent, up = narrow-FOV axis whose "rail" surface has returns beyond it — floor, not ceiling); `Frame` = points + intensity + ring + original slot indices | point layout and geometry only |
| Range image | `range_image.py` | `(2,H,W)` near/far range image: by slot index for organised clouds (rows = rings by elevation, columns = true azimuth 0.1°, descending), by angular binning for unorganised ones (Falcon Prime); every pixel keeps its message slot | frame + axes |
| D1 cluster detector | `detector.py`, `track_model.py` | rail level (near fit + far extension from the ceiling profile), lateral offset, track centre-line, clearance box, voxel clustering, structure rules, candidates with distance | analysis frame d/l/h |
| D2 beam detector | `det_beam.py` | corridor pixels of the range image grouped by image adjacency; groups judged by their neighbourhood (background / attached / grounded / extends above / sliver / pole / straddling) and by the dual-return see-through cue (dust); `DetCandidate` per group | range image + D1's rail/curve |
| D3 neural detector | `det_nn.py`, `nn_input.py`, `nn_model.py`, `models/d3_unet.pt` | 21-channel crop (3 frames × 7 channels incl. D1 aux) → U-Net → per-pixel obstacle/dust → components → `DetCandidate`; GPU (fp16) or CPU, hot switch to CPU on a CUDA fault | range image, D1 aux, `DeviceManager` |
| Fusion | `fusion.py` | tracks in (d, l); log-likelihood ratio per track from the three detectors; confirmation rule (two scored detectors within 120 m, persistence beyond), hysteresis, dust veto; writes confirmed candidates back into the frame result | `DetCandidate`, D1 `Candidate` |
| Runtime manager | `device.py`, `banner.py`, `doctor.py` | GPU probe (import, CUDA, arch, test kernels, memory), modes gpu / cpu_emergency / cpu_forced, RU+EN emergency banner with the fix and the contact, `ВНИМАНИЕ_GPU.txt`, self-check `doctor` | host/container environment |
| Cloud adapter | `cloud_io.py` | zero-copy parser of the serialised PointCloud2 (CDR, both endiannesses, bounds-checked; used by the node's raw subscription and by the offline reader – identical values to rclpy deserialisation on all 2488 recorded messages); PointCloud2 → `(N,3)` float32 XYZ via a structured dtype built from `fields`/`point_step`; validates buffer size, required fields, endianness, organised clouds; masks for non-finite and `(0,0,0)` returns | ROS message layout only |
| Configuration | `config.py` | `DetectorConfig` dataclass: every parameter with unit and comment; YAML load/save; axis matrix | nothing else |
| Track model | `track_model.py` | rail-level polynomial `h_rail(d)`, sensor lateral offset from the drainage trough, track centre-line `l_c(d)` from bed symmetry, wall/ceiling free-space centres and chained profile shifts | NumPy arrays in the analysis frame |
| Detector core | `detector.py` | frame validation, ROI, clearance box that follows the track, voxelised clustering (SciPy KD-tree), structure-rejection rules, candidate distance, timings | `track_model`, `config` |
| Temporal filter | `temporal.py` | association of candidates to short tracks, confirmation (`confirm_hits` of `confirm_window`), miss handling, time-jump / gap check (`check_time`), never turns `unknown` into `clear` | candidates + timestamps |
| Frame pipeline | `pipeline.py` | per-frame order shared by node and offline: stamp check → reset of temporal **and** geometric state on a discontinuity → detector → temporal filter | detector, temporal |
| ROS node | `node.py` | parameters (YAML + ROS overrides), QoS, raw subscription (`raw_decode`), `input_topic:=auto` discovery of the only PointCloud2 topic (own outputs excluded, best-effort publishers matched), publishers, UNKNOWN timer (before the first cloud and on stale input), line-buffered JSONL log, markers, clean shutdown | rclpy, messages, pipeline |
| Markers | `markers.py` | boxes, labels, corridor line strips, status text in the cloud frame (no TF needed) | visualization_msgs |
| Offline evaluator | `offline.py` | sequential bag reading (SQLite3 or MCAP, storage taken from `metadata.yaml`), same parser and pipeline, JSONL + summary with completeness check (messages read vs metadata, exit code 3 if incomplete) | rosbag2_py, pipeline |
| Bag / stream accounting | `bag_tools.py` | `info`: topics, counts, real `header.frame_id` (for RViz); `coverage`: frames sent (bag) vs processed (node log) vs published (observer), matched by header stamp | rosbag2_py (headers only) |
| Status monitor | `status_monitor.py` | one console line per status; `--jsonl` records every status as an independent observer | rclpy |
| Messages | `metro_obstacle_msgs` | `ObstacleStatus` (header, status enum, distance, obstacles[], diagnostics), `Obstacle` | – |
| Evaluation | `scripts/evaluate.py` | results vs `annotations/events.yaml` → TP/FP/FN/TN, alarm events, first detection distance, timing | JSON files |

## Data flow and memory

One process: a single-threaded ROS executor; inside a frame D1 runs in a worker thread beside the
range image and D3 beside D2 (`METRO_PARALLEL=0` runs everything in one thread with the same result). Per frame: the message buffer (8 MB / 24 MB) → one
float32 XYZ array (3.7 MB / 11 MB) → a float64 analysis-frame array of the valid returns
(≈ 4–8 MB) → boolean masks. Nothing is accumulated across frames except the previous track
curve, the static-scene model window and the fusion tracks; memory does not grow over a bag.

## Where frames can be lost

1. **Recording gaps** already present in the bag (up to 1.1 s) – visible as `inter_arrival_s`
   in the node log; not detector losses.
2. **Transport**: with best-effort QoS the 8–24 MB messages are fragmented and whole frames
   are dropped on the loopback (measured 85 of ~195 frames received); with **reliable** QoS
   (the recorded profile, default) no frame was lost in transport in the whole-bag stream tests.
3. **Player artefacts** (`ros2 bag play` of Humble, not the detector): started unpaused it sends
   the first 1–3 clouds in a burst after filling its read-ahead queue, and it may exit before
   the last large message is delivered. The stream test therefore uses `--start-paused` +
   `/rosbag2_player/resume` and `--wait-for-all-acked`.
4. **Overload policy**: the subscription keeps only the freshest frame (depth 1). When
   processing takes longer than the frame interval the next frame replaces the waiting one.
   The node cannot count such frames itself (its `frame_index` counts only what it received);
   they are counted against the bag by `bag_tools coverage` (header stamps of the bag vs the
   node log vs an independent observer). Measured live on the delivery image
   (`docs/launch.md` §12.7): 196 of 201 frames of `doubleT_obstacle`, 608 frames in 62 s of
   `squareT_platform_squareT_switch` on 8 cores with no frame over 100 ms. Losses grow when other
   work loads the same cores. Offline evaluation processes every frame.
5. **No input**: if no cloud arrives for `stale_timeout_s` (1.5 s) – also right after start,
   before the first cloud – the node publishes `STATUS_UNKNOWN` every 0.5 s with the reason
   and resets its state; "no data" is never reported as "clear" or as silence.

## Processes on the stand

`./run.sh --data <dataset>` starts the persistent container and, inside it, `metro start`: the
self-check, `ros2 launch metro_obstacle_detector detector.launch.py` (node, status monitor, optionally
RViz2) and the console. `./run.sh play <bag>` runs `ros2 bag play` in the same container (the verified
default path). Results and logs are written under `/results` (bind-mounted, `~/metro-lidar-results`).

Playing the bag **outside** the container (on the host or in another container) was checked with
`roundT_doubleT` (frames logged by the node):

| player | node container | frames |
|---|---|---:|
| second Humble container, `--network host`, own `/dev/shm` | default | **0** (discovery works, data never arrives) |
| second Humble container | both with `--ipc host` | 251 / 252 |
| host ROS 2 Jazzy, default transports | default or `--ipc host` | **0** |
| host Jazzy with `ROS_LOCALHOST_ONLY=1` | `ROS_LOCALHOST_ONLY=1` | **0** (no discovery: Jazzy implements localhost-only differently) |
| host Jazzy, `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` | default, no localhost-only | 252 / 252 |
| host Jazzy, default transports | `FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_udp_only.xml`, no localhost-only | **252 / 252** |
| second Humble container, default | same profile, no localhost-only | **251 / 252** |
| second Humble container, `ROS_LOCALHOST_ONLY=1` | same profile + `ROS_LOCALHOST_ONLY=1` | 0 |
| player in the same container (`docker exec`) | same profile + `ROS_LOCALHOST_ONLY=1` | 251 / 252 |

Cause: Fast DDS uses shared memory for peers on the same machine, but a container has its own
`/dev/shm` and other Fast DDS versions use another segment format, so the data is written where
the peer cannot read it. Recipe for an external player (`docs/launch.md` §12.3, `./run.sh --udp`): start the node container
with the UDP-only profile shipped in the image and the same `ROS_DOMAIN_ID`, without
`ROS_LOCALHOST_ONLY` on either side. A native Humble installation on the host was not available
here; the second Humble container on the host network stands in for it.
