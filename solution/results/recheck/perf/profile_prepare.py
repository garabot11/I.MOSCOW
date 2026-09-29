"""Micro-profile of cloud_to_xyz and the prepare stage on a real cloud (median of N repetitions)."""
import sqlite3, sys, time
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from metro_obstacle_detector.cloud_io import cloud_to_xyz, pointcloud2_to_struct
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector

bag = sys.argv[1]
db = next(Path('/data', bag).glob('*.db3'))
c = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
raw = c.execute('SELECT data FROM messages ORDER BY timestamp LIMIT 1 OFFSET 20').fetchone()[0]
def T(f, n=15):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1e3)
    return round(float(np.median(ts)), 2)
msg = deserialize_message(raw, PointCloud2)
print('deserialize', T(lambda: deserialize_message(raw, PointCloud2), 5))
print('struct view', T(lambda: pointcloud2_to_struct(msg)))
print('cloud_to_xyz', T(lambda: cloud_to_xyz(msg, intensity=False)))
xyz = cloud_to_xyz(msg, intensity=False)[0]
x0, y0, z0 = xyz[:, 0], xyz[:, 1], xyz[:, 2]
print('finite+zero', T(lambda: ((np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)), ((x0 == 0) & (y0 == 0) & (z0 == 0)))))
valid = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0) & ~((x0 == 0) & (y0 == 0) & (z0 == 0))
print('xyz[valid] (N,3 gather)', T(lambda: xyz[valid]))
vxyz = xyz[valid]
print('column extract*sign x3', T(lambda: (np.float32(-1) * vxyz[:, 1], vxyz[:, 0] * np.float32(1), vxyz[:, 2] * np.float32(1))))
det = ObstacleDetector(DetectorConfig())
r = det.process(xyz)
print('detector timings', {k: round(v, 1) for k, v in r.timings_ms.items()})
print('process total', T(lambda: ObstacleDetector(DetectorConfig()).process(xyz), 7))
