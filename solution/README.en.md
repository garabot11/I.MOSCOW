# Metro tunnel LiDAR obstacle detector (ROS 2 Humble, Docker)

A geometric detector that reads the 3D-LiDAR `sensor_msgs/PointCloud2` stream of a metro
train, models the *normal* tunnel around the track (track bed, walls, ceiling, side
equipment) and reports any compact object standing inside the train's clearance ahead,
together with its distance. No neural network, no training data, no GPU: NumPy + SciPy in a
single Python ROS 2 node. The same core runs offline over a bag for exhaustive evaluation.

* Input: any rosbag2 (SQLite3 or MCAP) or live topic with `PointCloud2` (x, y, z float32 fields
  required; the delivered clouds have a 26-byte mixed layout, see `docs/data_notes.md`). The
  topic name need not be known: by default (`input_topic:=auto`) the node subscribes to the only
  `PointCloud2` topic that appears in the ROS graph.
* Output: `/obstacles/status` (`metro_obstacle_msgs/ObstacleStatus`: `UNKNOWN | CLEAR |
  DETECTED`, distance in metres from the sensor, per-object details),
  `/obstacles/markers` (RViz), `/obstacles/candidates` (returns inside the clearance box),
  optional JSON-lines log.
* Environment: Ubuntu 22.04 + ROS 2 Humble inside the image (`ros:humble-ros-base-jammy`),
  CPU only, no GPU; the node sustains ≈ 19 frames/s for the 921 k-point sensor and ≈ 29 frames/s
  for the 307 k-point one on the laptop without core pinning, ≈ 12 / 19 frames/s on its E-cores (section 6),
  120–230 MB RAM for the node (measured); the host only needs Docker (and an X11 session for RViz).

Documentation: `docs/architecture.md`, `docs/algorithm.md`, `docs/experiments.md`,
`docs/data_notes.md`. Labels used for evaluation: `annotations/events.yaml`; bag manifest with
SHA-256 sums: `manifests/bags.yaml`. Short videos of the running system: `video/`.

## 1. Build

```bash
cd solution
docker build --target solution \
  --build-arg APP_UID="$(id -u)" --build-arg APP_GID="$(id -g)" \
  -t metro-lidar:final .
```

The build installs every dependency with apt (ROS packages, NumPy, SciPy, RViz2), copies the
sources, runs `colcon build` and sets an entrypoint that sources Humble and the workspace.
`APP_UID/APP_GID` make files written to mounted result folders owned by you. Network access is
needed only for the build. A from-scratch build (`--no-cache --pull`) was checked on 2026-09-29:
≈ 2.5 min, resulting image `sha256:69acee4e…` (see `docs/experiments.md`, "Build record"). Verify:

```bash
docker run --rm metro-lidar:final ros2 pkg list | grep metro_obstacle
docker run --rm metro-lidar:final python3 -m pytest -q /workspace/tests      # 38 tests
```

Ready image without building and without network: the archive `metro-lidar-final.tar.gz`
(462 MB, SHA-256 `1c05211bbed1f4ecb62c6da302a56b7eff8c32a7c37cf57f42a39b428de1d313`, in `../dist/`
next to the `solution/` folder):

```bash
sha256sum -c metro-lidar-final.tar.gz.sha256
docker load -i metro-lidar-final.tar.gz          # → metro-lidar:final, ID sha256:69acee4e…
```

In the git repository the archive is split into 95 MB parts; join them before the check:
`cat metro-lidar-final.tar.gz.part-* > metro-lidar-final.tar.gz` (see `../dist/README.md`).

## 2. Run (docker run → ros2 bag play → see the result)

The examples mount the bag folder read-only at `/data` and a result folder at `/results`.
A "bag" is a rosbag2 **directory** (`metadata.yaml` + `*.db3`). `docker run -v` needs an
absolute host path (hence `$PWD/...`); the scripts in `scripts/` also accept relative paths
and make them absolute themselves.

### 2a. Headless: detector + playback, result in the terminal and in a log

Terminal 1 – detector (prints the status every 50 frames; the log has one JSON line per frame):

```bash
mkdir -p results    # otherwise Docker creates it owned by root and the node cannot write its log
docker run --rm -it --name metro-lidar --network host --shm-size=1g \
  -e ROS_DOMAIN_ID=78 -e ROS_LOCALHOST_ONLY=1 \
  -v /path/to/bags:/data:ro -v "$PWD/results":/results \
  metro-lidar:final \
  ros2 launch metro_obstacle_detector detector.launch.py log_path:=/results/live.jsonl
```

Terminal 2 – playback in the same container (the recorded QoS is *reliable*, which the node
subscribes with by default). A new `docker exec` process does not go through the entrypoint of
the main process, so the command is started through it explicitly – `/workspace/entrypoint.sh`
sources Humble and the workspace:

```bash
docker exec -it metro-lidar /workspace/entrypoint.sh \
  ros2 bag play /data/roundT_doubleT --rate 1.0 --delay 2 --read-ahead-queue-size 20 --wait-for-all-acked 5000
```

Terminal 3 – watch the result (one line per cloud: status, distance, processing time):

```bash
docker exec -it metro-lidar /workspace/entrypoint.sh ros2 run metro_obstacle_detector status_monitor
docker exec -it metro-lidar /workspace/entrypoint.sh ros2 topic echo /obstacles/status   # full message
```

Started unpaused, Humble's player first fills its read-ahead queue and then sends the first 1–3
clouds in a burst; the depth-1 subscription keeps the last of them, so these frames are not
processed (a player artefact, not a detector loss; measured 251 of 252 with this command). For
a complete count without this effect use `scripts/run_stream_test.sh` (section 5): it starts the
player with `--start-paused` and resumes it through `/rosbag2_player/resume` once the queue is
full.

Shells inside the image source the environment as well: `docker exec -it metro-lidar bash`
(interactive) and `docker exec metro-lidar bash -lc 'ros2 ...'` (login). Already before the
first cloud the node publishes `UNKNOWN` every 0.5 s with the diagnostic
`no input: no cloud received on <topic>` – a wrong topic/QoS or a missing sensor is visible.

The node finds the cloud topic itself: with `input_topic:=auto` (default) it subscribes to the
only `PointCloud2` topic in the graph as soon as the player creates it (its own
`/obstacles/candidates` is ignored; a best-effort-only publisher gets a best-effort
subscription). `--delay 2` gives the node time to subscribe before the first frame. With
several cloud topics the node does not choose and lists them in the `UNKNOWN` diagnostic – then
set `input_topic:=/sensing/lidar/hesai128/pointcloud` (or any other name). The frame id is taken
from the cloud and needs no TF. The bag format (SQLite3 / MCAP) is read from `metadata.yaml`.

### 2b. With RViz2 (X11 on the host)

`scripts/run_demo.sh` does everything in one container – passes the X11 cookie through a
private Xauthority file, takes the fixed frame from the real `header.frame_id` of the first
cloud of the bag (`bag_tools info --frame-id` in the container; override with
`FIXED_FRAME=<frame>`), starts the node with `rviz:=true fixed_frame:=<frame>` and plays the
bag:

```bash
scripts/run_demo.sh /path/to/bags/doubleT_obstacle /sensing/lidar/hesai128/pointcloud 1.0 results/demo metro-lidar:final
scripts/run_demo.sh /path/to/bags/roundT_doubleT   /lidar_points                        1.0 results/demo metro-lidar:final
FIXED_FRAME=my_lidar scripts/run_demo.sh /path/to/bags/other_bag /points 1.0 results/demo   # explicit frame
```

RViz shows the cloud (colour = height), the returns of accepted obstacle candidates in red,
the green track centre-line with the blue box edges, a red cube + label per obstacle and the
status text (`CLEAR` / `OBSTACLE 55.7 m` / `UNKNOWN`). Software OpenGL is used
(`LIBGL_ALWAYS_SOFTWARE=1`); with the NVIDIA container toolkit add `--gpus all` and
`-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility`.

Manual variant: start the container as in 2a but add
`-e DISPLAY -e XAUTHORITY=/tmp/imos.xauth -e LIBGL_ALWAYS_SOFTWARE=1 -e QT_X11_NO_MITSHM=1
-v "$IMOS_XAUTH":/tmp/imos.xauth:ro -v /tmp/.X11-unix:/tmp/.X11-unix:ro` and launch with
`rviz:=true fixed_frame:=<header.frame_id of the clouds>` (see `scripts/run_demo.sh` for the
cookie preparation; `ros2 run metro_obstacle_detector bag_tools info --bag /data/<bag> --frame-id`
prints the frame of a bag).

### 2c. Offline evaluation of a whole bag (every frame, no DDS)

```bash
docker run --rm -v /path/to/bags:/data:ro -v "$PWD/results":/results metro-lidar:final \
  ros2 run metro_obstacle_detector offline --bag /data/roundT_doubleT --out /results/roundT_doubleT.jsonl
```

Prints one line per frame and a summary (status counts, processing-time percentiles, messages
read vs `metadata.yaml`, a `complete` flag); writes `<name>.jsonl` and `<name>_summary.json`;
exit code 3 if the bag was not read to the end. `scripts/eval_all_container.sh <bags> <results>`
runs all bags and checks each one (container exit code, files present, `complete`, JSONL line
count): `ALLDONE` and exit code 0 only if all succeed, otherwise `FAILED` with the list and exit
code 1. `scripts/evaluate.py <results>` compares the results with `annotations/events.yaml`.

### 2d. Playing the bag outside the container (on the host or in another container)

The main, verified path is playback in the same container (2a). If the bag is played outside
(e.g. with the stand's own ROS 2 `ros2 bag play`), discovery works by default but **no data
arrives**: Fast DDS uses shared memory for peers on the same machine, and a container has its
own `/dev/shm`. Start the node container with the UDP-only Fast DDS profile shipped in the
image and without `ROS_LOCALHOST_ONLY` (isolate with a dedicated `ROS_DOMAIN_ID`, the same for
node and player):

```bash
docker run --rm -it --name metro-lidar --network host \
  -e ROS_DOMAIN_ID=78 -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_udp_only.xml \
  -v "$PWD/results":/results metro-lidar:final \
  ros2 launch metro_obstacle_detector detector.launch.py log_path:=/results/live.jsonl
# on the host (ROS 2 Humble/Jazzy), without ROS_LOCALHOST_ONLY:
ROS_DOMAIN_ID=78 ros2 bag play /path/to/bags/roundT_doubleT --rate 1.0 --delay 2 --wait-for-all-acked 5000
```

Checked: a ROS 2 Jazzy player on the host – 252 of 252 frames, a second Humble container – 251
of 252; without the profile – 0 frames (`docs/architecture.md`, "Processes on the stand").

## 3. Output

`metro_obstacle_msgs/ObstacleStatus` (published for every processed cloud, and every 0.5 s
with `STATUS_UNKNOWN` when no cloud arrived for `stale_timeout_s` – both after start before the
first cloud and after the input stops):

| field | meaning |
|---|---|
| `header` | stamp = input cloud `header.stamp` (sensor clock), frame_id = cloud frame |
| `status` | `0 UNKNOWN` (no/invalid/stale input or track surface not observable), `1 CLEAR`, `2 DETECTED` |
| `distance_m` | Euclidean range (m) from the lidar origin to the nearest reliable surface of the nearest **confirmed** obstacle; NaN otherwise |
| `forward_m` | its along-track distance |
| `obstacles[]` | every candidate of the frame: `confirmed`, `distance_m`, `forward_m`, `lateral_offset_m` (from the track centre-line), `height_above_rail_m`, sizes, `num_points`, `confidence`, `center` (sensor frame) |
| `frame_index` | number of the cloud **received** by this node instance (from 1); not the frame index in the bag – match online and offline results by `header.stamp` |
| `processing_ms` | wall-clock time of the detector for this cloud (monotonic clock) |
| `rail_level_m`, `track_offset_m` | estimated rail level and sensor offset from the track centre |
| `diagnostics` | reason for UNKNOWN, resets, format errors |

Distances are measured from the LiDAR origin (the sensor-to-train-front offset is not known).
The JSON-lines log (`log_path`) additionally contains every rejected cluster with the reason,
the track-model coefficients and per-stage timings.

## 4. Parameters and configuration

All detector parameters are in `config/detector.yaml` (one comment per key; regenerated from
`config.py` by `scripts/gen_config.py`). The node reads that file (`config_file`) and every key
can be overridden as a ROS parameter; the offline tool takes `--config` and `--set key=value`.

```bash
# wider clearance and earlier confirmation
ros2 launch metro_obstacle_detector detector.launch.py input_topic:=/lidar_points \
   config_file:=/results/my_detector.yaml
ros2 run metro_obstacle_detector detector_node --ros-args -p input_topic:=/lidar_points \
   -p gauge_halfwidth_body:=1.5 -p confirm_hits:=1
ros2 run metro_obstacle_detector offline --bag /data/x --set gauge_halfwidth_body=1.5 --set confirm_hits=1
```

Launch arguments: `input_topic` (`auto` by default or a topic name), `config_file`, `qos_reliability` (`reliable` default – the
recorded profile; `best_effort` for live sensors publishing best-effort), `qos_depth` (1 =
freshest frame), `log_path`, `use_sim_time`, `rviz`, `rviz_config`, `rviz_geometry`,
`fixed_frame` (RViz fixed frame = `header.frame_id` of the clouds; the RViz cloud display is
switched to `input_topic`).
Key parameters (details in `docs/algorithm.md`):

| parameter | default | unit | effect |
|---|---|---|---|
| `forward_axis` / `up_axis` | `-y` / `+z` | – | sensor axes pointing along the track / up |
| `d_min` / `d_max` | 3 / 300 | m | analysed range |
| `gauge_halfwidth_low` / `gauge_halfwidth_body` | 1.0 / 1.40 | m | clearance half-width below / above 0.8 m over rail |
| `box_bottom_above_rail` / `box_top_above_rail` | 0.30 / 3.40 | m | vertical extent of the clearance box |
| `min_points_near` / `min_points_far` | 5 / 4 | voxels | minimum cluster support (< / ≥ 100 m) |
| `confirm_hits` / `confirm_window` | 2 / 3 | frames | temporal confirmation (< 100 m; `confirm_hits_far`/`confirm_window_far` = 3 / 4 beyond) |
| `stale_timeout_s` / `max_gap_s` | 1.5 / 1.5 | s | UNKNOWN on stale input / state reset on gaps |
| `report_beyond_support_m` | 15 | m | reporting range beyond the last track-model sample |
| `debug_cloud` | true | – | publish `/obstacles/candidates` |
| `max_analysed_points` | 0 (off) | points | decimation of the analysed region (every k-th point, deterministic) – time margin for dense sensors at the price of far-range support, see `docs/experiments.md` |
| `raw_decode` (node parameter) | true | – | parse the cloud's CDR without building a Python message (same values, less work in the node thread); `false` = standard rclpy deserialisation |

QoS/time notes: the subscriber uses keep-last depth 1 (freshest frame) and *reliable* by
default; `use_sim_time` is off (the clouds' `header.stamp` is on a sensor clock decades away
from the bag clock and is only used for matching, never for latency). There is no TF in the
delivered bags; all outputs are in the cloud's own frame.

## 5. Tests and experiments

```bash
docker run --rm metro-lidar:final python3 -m pytest -q /workspace/tests
scripts/eval_all_container.sh /path/to/bags results/final_offline 1 metro-lidar:final
python3 scripts/evaluate.py results/final_offline --out results/final_offline/metrics.json
scripts/run_stream_test.sh /path/to/bags/roundT_doubleT results/stream /lidar_points
```

`run_stream_test.sh` plays the **whole** bag (`--rate 1.0`, QoS `reliable` like the node
default; `QOS=best_effort`, `QOS_DEPTH=<n>` change the profile; an optional 4th argument is a
time limit, and hitting it fails the test) and writes `coverage.json`: frames sent (= messages
of the bag), processed (node log) and published (a separate subscriber,
`status_monitor --jsonl`), matched by `header.stamp`, with the indices of unprocessed frames
and the number of `UNKNOWN` statuses before the first cloud.

Results and the protocol are in `docs/experiments.md`; raw outputs in `results/` are not part
of the image.

## 6. Known limitations and typical problems

* Clearance geometry (car half-width 1.35 m, contact-rail covers ≥ 1.2 m, platform edges
  ≥ 1.45 m) is assumed from the data, not supplied; objects below 0.3 m, or between 1.0 and
  1.35 m laterally below 0.8 m height, are not reported.
* Range is limited by what the tunnel lets the sensor see: behind curves the corridor closes;
  at stations and junctions reporting stops where the tunnel cross-section is no longer
  identified; low objects only where the track bed is observed (≈ 60–120 m).
* `CLEAR` means "no confirmed object inside the analysed clearance region", not a proven free
  track: the region is narrowed in places (see above), and `d_max = 300 m` is a processing
  limit, not a measured detection range. The only obstacle example stands at ~56 m; detection
  at 100–300 m is not demonstrated on the data.
* The metrics (frame precision 0.979 / recall 1.0) are development-data numbers: thresholds were
  tuned on all six bags, one obstacle event is labelled, 286 ambiguous frames are excluded. They
  are not an independent quality estimate; no official labels exist.
* `UNKNOWN` is reported when the track bed cannot be found (empty/blocked cloud, wrong axes) and
  when no cloud arrives – both after start before the first cloud and after the input stops.
* No frames? – the `UNKNOWN` diagnostic in `/obstacles/status` names the reason (no topic,
  several cloud topics, input stopped). Check `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, the topic
  name (`ros2 topic list`) and the QoS. If the bag is played outside the node's container, see
  section 2d (UDP-only profile, no `ROS_LOCALHOST_ONLY`).
* RViz "No transform": set the fixed frame to the clouds' `header.frame_id`
  (`fixed_frame:=<frame>` in the launch; `bag_tools info --frame-id` prints it for a bag); the
  demo script reads it from the bag itself. The yellow `Global Status: Warn` is then expected:
  there is no TF in the data, the cloud and the markers are shown in their own frame.
* No `ALLDONE` / exit code 1 from `eval_all_container.sh` – see `<results>/FAILED` and
  `<results>/<bag>.log`.
* Time and resources (measured in a 10 Hz stream, `docs/experiments.md`, "Throughput, CPU and
  slower cores"): processing median 24 ms (307 k points) and 48 ms (921 k) on the laptop
  without core pinning, 39 / 73 ms on its E-cores; including message decoding the node spends 34 / 54 ms of
  CPU per frame (≈ 29 / 19 frames/s), 53 / 84 ms on the E-cores (≈ 19 / 12 frames/s). The
  E-cores stand in for the older cores of the stand's i7-9700E – an estimate, the stand was not
  measured. The margin of the dense sensor on such cores is ≈ 1.2×; do not confine the node to
  one core (its DDS threads compete with the detector). When a frame takes longer than the frame
  interval the next one replaces the waiting one (freshest-frame policy, depth 1); such frames
  are not processed, and the stream test's `coverage.json` gives their number and indices.
  Foreign CPU load reduces the margin: before the switch to CDR parsing, with a browser loading
  the laptop, only every second frame of the dense sensor was processed (104 of 201).
  Decimation (`max_analysed_points`) is no remedy: the far object is no longer detected.
