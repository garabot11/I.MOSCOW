# Metro tunnel LiDAR obstacle detector (ROS 2 Humble, Docker)

A ROS 2 node reads the 3D-LiDAR `sensor_msgs/PointCloud2` stream of a metro train, models the *normal* tunnel
around the track (track bed, walls, ceiling, side equipment) and reports any object standing inside the train's
clearance ahead, together with its distance and the range up to which the track was checked. Three detectors
are fused over time:

* D1 — voxel clusters inside the clearance with structure-rejection rules (NumPy);
* D2 — a beam detector on the range image: an obstacle casts a shadow on the tunnel behind it;
* D3 — a range-image U-Net (PyTorch, `models/d3_unet.pt`, obstacle / dust classes).

An obstacle is reported when D1 and D2 both see it at the same place; D3 adds weight. **The GPU is the main
mode; the CPU is an emergency mode only**, in which the solution warns the organisers loudly (reason, exact
fix, contact of team IRS-MIREA). The same core runs offline over a whole bag.

The authoritative launch guide is the Russian `docs/launch.md` (it ends with an English summary).

## 1. Get the image

The delivery image is `metro-lidar:gpu-dev`, ID `a0e1e7e33476`: Ubuntu 22.04, ROS 2 Humble, rosbag2 (SQLite3
and MCAP), RViz2, PyTorch 2.11 + CUDA 12.8 (its own CUDA libraries — the host needs only an NVIDIA driver
≥ 570 and the NVIDIA Container Toolkit), Mesa, the D3 weights and the `metro` CLI. The archive (4.5 GB) is in
release [v3](https://github.com/garabot11/I.MOSCOW/releases/tag/v3) as 45 parts of 95 MB:

```bash
mkdir -p dist && cd dist
base=https://github.com/garabot11/I.MOSCOW/releases/download/v3
for f in metro-lidar-final.parts.sha256 metro-lidar-final.tar.gz.sha256 metro-lidar-final.image.txt \
         $(seq -f 'metro-lidar-final.tar.gz.part-%02g' 0 44); do curl -fL -C - -O "$base/$f"; done
sha256sum -c metro-lidar-final.parts.sha256
cat metro-lidar-final.tar.gz.part-* | gunzip | docker load       # -> metro-lidar:gpu-dev, 13.3 GB on disk
```

Alternative: `docker build -t metro-lidar:gpu-dev solution` (internet, ≈ 4.5 GB download; no GPU, no BuildKit).

Check the GPU inside Docker: `docker run --rm --gpus all metro-lidar:gpu-dev nvidia-smi`.

## 2. Run

```bash
cd solution
./run.sh --data /path/to/dataset           # a directory with rosbag2 directories at any depth
```

`run.sh` decides whether the container can get the GPU (`nvidia-smi`, the NVIDIA Container Toolkit, a trial
`docker run --gpus all`), starts the persistent container `metro-lidar` (data read-only in `/data`, logs in
`~/metro-lidar-results`), runs the self-check, starts the detector and RViz and shows the live status console
(Ctrl-C detaches; the detector keeps running). In a second terminal:

```bash
./run.sh bags                              # every bag under /data
./run.sh play doubleT_obstacle             # SPACE pause, UP/DOWN rate +-10 %, RIGHT next message
./run.sh play falcon/falcon_test_second_dark 1.0 --start-offset 262
./run.sh status | console | doctor | stop
```

Another `play` switches the bag without a restart. The cloud topic, the sensor axes (Hesai, Seyond Falcon
Prime) and the frame are detected automatically. Options: `--cpu`, `--no-gui`, `--udp` (bag played outside the
container), `--topic T`, `--cpuset-cpus 0-7`, `--results DIR`, `--image IMG`, `--detach`.

## 3. Output

`/obstacles/status` (`metro_obstacle_msgs/ObstacleStatus`), one message per processed cloud:

| field | meaning |
|---|---|
| `status` | `0 UNKNOWN` (no / stale input, bed not observed), `1 CLEAR`, `2 DETECTED` |
| `distance_m` | Euclidean distance from the sensor to the nearest reliable surface of the nearest confirmed obstacle (NaN if none) |
| `forward_m` | its distance along the track |
| `verified_to_m` | the range up to which the clearance corridor was built and checked in this frame; `CLEAR` applies to that part only |
| `obstacles[]` | every candidate: `confirmed`, distances, lateral offset from the track centre, height above the rail, size, points, confidence |
| `mode`, `mode_reason` | `gpu`, `cpu_emergency`, `cpu_forced`; why not the GPU |
| `processing_ms`, `diagnostics`, `static_fraction`, `rail_level_m`, `track_offset_m` | timing, reason of `UNKNOWN`, standing-train share, track model |

Also `/obstacles/markers` (RViz: green track centre-line, blue clearance edges and volume, red confirmed
obstacle, orange candidate, `CLEAR` / `OBSTACLE <distance>` with `track checked to N m`) and
`/obstacles/candidates` (points of the accepted candidates). Parameters: `config/detector.yaml` (one comment per
key), overridable as ROS parameters or with `offline --set key=value`.

## 4. Results (delivery image, every frame, `docs/experiments.md`)

| data | result |
|---|---|
| 6 organisers' recordings (2488 frames) and the 20-minute ride `new_data` (11 271 frames) | **0 false frames**; the obstacle at 56 m from frame 11 |
| Falcon Prime, person in dark clothes | first confirmed at **154 m**; 0 false frames before he appears; 510 / 510 frames standing at 100 m, 710 / 710 walking 100 → 4 m |
| `cloud_with_fake_obj` (objects inserted by the organisers) | wall from 76 m, low plate from 91 m, 0.3 m cubes from 37–52 m; 0 false frames in the clean part |
| organisers' synthetic rides with ground-truth markers (13 objects) | all 7 objects inside our clearance found (49 % of object-frames), 8 of 13 of their list (35 %); beam across the rails from 74 m, hanging cable from 50 m, person from 119 m; 0 false |
| timing, 8 pinned laptop cores, GPU | p95 39–75 ms per frame, 0 of 3480 frames over 100 ms |
| launch check (`docker/failure_matrix.sh`) | 17 / 17 scenarios pass, 0 tracebacks; 255 tests pass in the image |

## 5. Tests and evaluation

```bash
docker run --rm --gpus all metro-lidar:gpu-dev python3 -m pytest -q /workspace/tests       # 255 passed, 17 skipped
docker run --rm --gpus all -v /path/to/bags:/data:ro -v "$PWD/results":/results metro-lidar:gpu-dev \
  ros2 run metro_obstacle_detector offline --bag /data/for_hackathon/doubleT_obstacle --out /results/x.jsonl
scripts/eval_chain.sh <tag> gpu            # every recording and synthetic set; scripts/chain_summary.py
solution/docker/failure_matrix.sh --data /path/to/dataset
```

## 6. Limitations

* The clearance geometry (car half-width 1.35 m, conductor-rail ducts from 1.2 m, platform edges from 1.45 m)
  is derived from the data, not supplied; objects 1.0–1.35 m from the axis below 0.8 m are not reported.
* Range is limited by what the tunnel lets the sensor see: behind curves the corridor closes, low objects need
  the track bed in view (≈ 60–120 m), 0.3 m cubes are confirmed within ≈ 50 m, nothing is reported beyond the
  verified range (≈ 170 m in a straight tunnel).
* An object seen by one detector only is not reported until the second one agrees — the price of zero false
  alarms on the real recordings.
* Dust was validated on synthetic single-return dust only; a dense dust cloud on the track axis can be reported.
* The organisers' stand (i7-9700E, RTX 4070 Ti SUPER) was not measured: expect 1.3–1.5× the laptop's frame
  times; the start-up GPU probe confirms the sm_89 card on site.
