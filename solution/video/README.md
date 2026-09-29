# Videos

Recorded on 2026-09-29 with `scripts/record_video.sh` (screen capture of the X11 session,
1920×1080, 15 fps). Everything visible runs inside the Humble container `metro-lidar:final`:
the detector node, RViz2 (software OpenGL) and `ros2 bag play --rate 1.0` (real time).

| file | bag | what to look at |
|---|---|---|
| `demo_obstacle_doubleT.mp4` (28 s) | `doubleT_obstacle`, topic `/sensing/lidar/hesai128/pointcloud` | from ≈ 6.5 s of the video the status turns `OBSTACLE 55.x m`: red returns of the object, a red cube with the label, the green track centre-line and the blue clearance edges; the burnt-in lines at the bottom are `status_monitor` output (frame, status, distance, processing time, nearest object size / support / confidence). Later the object moves beside the track and the status returns to `CLEAR`. |
| `demo_clear_roundT_doubleT.mp4` (32 s) | `roundT_doubleT`, topic `/lidar_points` | a round tunnel with a right-hand curve and a transition into a double-track section; the corridor follows the curve and the status stays `CLEAR`. |

The videos were recorded with the image `sha256:68a2bea3…` (before the corrections of
2026-09-29). The detection output of the final image `sha256:69acee4e…` is identical frame
by frame on all 2488 frames (`results/recheck/offline_v3_vs_v2.json`); the RViz fixed frame is
now read from the bag instead of being guessed from the topic name (same result for these two
bags, `results/recheck/demo_obstacle_screen.png`).

The top-left caption shows the bag, topic and playback rate. Processing times in the videos
(≈ 60–90 ms) are higher than in the headless measurements (25–47 ms) because RViz renders in
software on the same CPU.
