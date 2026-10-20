# Data notes

> **Кратко по-русски.** Шесть записей, две конфигурации датчика (`/lidar_points`, 307 200 точек, frame `hesai_lidar`; `/sensing/lidar/hesai128/pointcloud`, 921 600 точек, frame `lidar_livox`); шаг точки 26 байт со смешанными типами; 40–60 % точек — ровно (0,0,0); много дублированных точек; вперёд — ось −Y, вверх +Z; уровень головки рельса z≈−1,30 м (hesai), лоток −1,6 м, край платформы ≈1,45 м от оси пути, короб контактного рельса 1,2–1,6 м, проём гермозатвора 4,0×3,8 м; три шкалы времени (bag, header, точки) с разными эпохами; разрывы между сообщениями до 1,1 с.

Facts about the development bags that shaped the implementation. Everything below was measured
on the delivered archive (`датасет.zip` → `archive/for_hackathon.zst` → six rosbag2 directories);
see `manifests/bags.yaml` for per-bag numbers and SHA-256 sums and `annotations/events.yaml`
for the manual labels.

## Bags

| Bag | Topic | frame_id | Points / msg | Msgs | Duration | Content (visual inspection) |
|---|---|---|---:|---:|---:|---|
| `doubleT_obstacle` | `/sensing/lidar/hesai128/pointcloud` | `lidar_livox` | 921 600 | 201 | 20.4 s | double-track tunnel, **train stationary**, a ~0.9 × 0.9 × 0.85 m object moving across the track at 55–57 m |
| `doubleT_platform` | `/lidar_points` | `hesai_lidar` | 307 200 | 345 | 34.4 s | double-track tunnel → station with island platform; train stops in the station |
| `roundT_doubleT` | `/lidar_points` | `hesai_lidar` | 307 200 | 252 | 25.1 s | round tunnel, right-hand curve, transition to double-track tunnel |
| `roundT_pressureGate_roundT` | `/lidar_points` | `hesai_lidar` | 307 200 | 268 | 26.7 s | round tunnel, pressure gate (4.0 × 3.8 m opening), left-hand curve |
| `roundT_squareT_pressureGate_squareT` | `/lidar_points` | `hesai_lidar` | 307 200 | 545 | 55.4 s | round and rectangular sections, pressure gate, mostly straight |
| `squareT_platform_squareT_switch` | `/lidar_points` | `hesai_lidar` | 307 200 | 877 | 88.2 s | rectangular tunnel, side-platform station (stop ≈ 32–66 s), junction/switch at the end |

All six use SQLite3 storage, CDR serialisation and `sensor_msgs/msg/PointCloud2`
(`ros_distro = humble` in the schema). Three bags have `files[0].message_count` larger than
the real number of messages (`doubleT_obstacle` 205 vs 201, `roundT_squareT_pressureGate_squareT`
555 vs 545, `squareT_platform_squareT_switch` 883 vs 877); frame counts in this project always
come from the messages actually read.

## Point layout

`height = 1`, `is_dense = false`, little-endian, **`point_step = 26`** with mixed types:

| field | type | offset |
|---|---|---:|
| x, y, z | float32 | 0, 4, 8 |
| intensity | float32 | 12 |
| ring | uint16 | 16 |
| timestamp | float64 | 18 |

The buffer must be read through a structured dtype with explicit offsets and `itemsize = 26`
(`cloud_io.py`); reshaping the buffer as float32 triplets or letting NumPy align the dtype
corrupts the coordinates.

**Placeholder returns.** 38–48 % of the points (62 % in `doubleT_obstacle`) are exactly
`(0, 0, 0)`. They are finite, so `skip_nans` does not remove them; the detector masks them
explicitly (`n_zero` in every result). A point with only one zero coordinate is kept.

**Duplicate returns.** Many returns appear twice with identical coordinates (dual-return mode).
Cluster support is therefore counted on a voxel grid (`voxel_base_m`), not on raw points.

**Scan patterns.** In `doubleT_obstacle` odd and even frames use two different scan patterns
(the 0.1 s point-timestamp span vs 0.033 s in the other bags also differs); frame differencing
is only meaningful between frames of the same parity.

## Geometry (measured, not supplied)

* **Axes.** The track ahead lies along **−Y**, **+Z** is up, X is lateral (positive to the
  right when looking forward with −Y ahead, as `l = h × d`). This is a measurement, not a
  ROS convention; it is a parameter (`forward_axis`, `up_axis`).
* **Sensor height.** `hesai_lidar` bags: rail-head level at z ≈ −1.30 m, drainage trough at
  −1.6 m, rails at x ≈ ±0.76 m, sensor laterally on the track centre (±0.1 m).
  `doubleT_obstacle`: track bed at z ≈ −2.2…−2.4 m, sensor ≈ 0.3–0.6 m right of the track
  centre (trough at x ≈ −0.6 m, wall at −3.0 m, column row at +1.5 m).
* **Cross-sections.** Round tunnel radius ≈ 2.7 m (ceiling 3.3 m above the sensor);
  rectangular tunnel ≈ 4.6 m wide, ceiling 3.0 m above the sensor; pressure-gate opening
  4.0 m wide × 3.8 m above rail; double-track tunnel: wall 2.4 m and column row 2.1 m from
  the track centre; platform edges start ≈ 1.45 m from the track centre at 1.1 m above rail.
* **Side equipment.** Contact-rail cover 1.2–1.6 m from the track centre, 0.35–0.75 m above
  rail (either side); cable ducts / boxes on the walls at 2.0–2.3 m.
* **Curves.** Lateral displacement of the tunnel reaches 10 m at 130 m (`roundT_doubleT`),
  i.e. radii of roughly 600–1000 m; the far tunnel is occluded by the outer wall beyond
  ~120–150 m in such curves.
* **Grades / pitch.** The rail-level line has slopes up to −2.4 % relative to the sensor
  (`doubleT_obstacle`), i.e. a combination of sensor pitch and track grade.

## Time scales

Three time scales exist: SQLite/bag time (2 September 2026 UTC), `header.stamp` (≈ 1 January
2000, sensor clock) and the per-point `timestamp` (same sensor clock). Results carry the input
`header.stamp` for matching and the bag time for offline reports; processing time is measured
with the monotonic clock only. Gaps between messages reach 0.41 s (`doubleT_obstacle`),
0.70 s and 1.10 s (`roundT_squareT_pressureGate_squareT`); they are recording gaps, not detector
drops, and the temporal filter resets on gaps larger than `max_gap_s` (1.5 s) or on time
running backwards (bag loop).

## Point density with range

Angular spacing is ≈ 0.15° horizontally and ≈ 0.125° vertically in the dense band, so returns
are ≈ 0.26 m apart at 100 m and ≈ 0.52 m at 200 m. A person-sized object yields tens of returns
at 50 m, ~6–12 at 100 m and 1–4 at 200 m; the track bed is observed up to ≈ 60–120 m, the
ceiling and walls further. Beyond ~100 m single-ring hits and grazing wall/floor patches
dominate the returns inside the corridor, which is why the far-range rules in `detector.py`
exist.
