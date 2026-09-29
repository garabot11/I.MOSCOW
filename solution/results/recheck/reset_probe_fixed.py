"""Audit scenario (audit/2026-09-29/reset_probe.py) through the corrected per-frame pipeline:
first cloud of roundT_doubleT, then the first cloud of the earlier doubleT_obstacle (time jump back)."""
import json
import sqlite3
from pathlib import Path

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from metro_obstacle_detector.cloud_io import cloud_to_xyz
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from metro_obstacle_detector.pipeline import DetectionPipeline


def cloud(bag):
    db = next(Path('/data', bag).glob('*.db3'))
    c = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    raw = c.execute('SELECT data FROM messages ORDER BY timestamp LIMIT 1').fetchone()[0]
    c.close()
    xyz, _, info = cloud_to_xyz(deserialize_message(raw, PointCloud2), intensity=False)
    return xyz, info.stamp_ns


def geometry(res):
    d = res.to_dict()
    d.pop('timings_ms', None)
    d['candidates'] = [{k: v for k, v in c.items() if k not in ('track_id', 'confirmed')} for c in d['candidates']]
    return d


cfg = DetectorConfig()
p = DetectionPipeline(cfg)
a, ta = cloud('roundT_doubleT')
b, tb = cloud('doubleT_obstacle')
p.step(a, ta)
r2, d2 = p.step(b, tb)
fresh = ObstacleDetector(cfg).process(b)
g2, gf = geometry(r2), geometry(fresh)
print(json.dumps({
    'jump_reset_triggered': d2.reset,
    'identical_to_fresh_detector': g2 == gf,
    'n_box': [g2['n_box'], gf['n_box']],
    'curve_max_d_absolute': [g2['curve']['max_d_absolute'], gf['curve']['max_d_absolute']],
    'status': [d2.status, fresh.status],
}, indent=1, default=str))
