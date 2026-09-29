from pathlib import Path
import json, sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from metro_obstacle_detector.cloud_io import cloud_to_xyz
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
from metro_obstacle_detector.temporal import TemporalFilter

def cloud(bag):
    db=next(Path('/data',bag).glob('*.db3'))
    c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    raw=c.execute('SELECT data FROM messages ORDER BY timestamp LIMIT 1').fetchone()[0];c.close()
    msg=deserialize_message(raw,PointCloud2)
    xyz,_,info=cloud_to_xyz(msg,intensity=False)
    return xyz,info.stamp_ns
cfg=DetectorConfig();det=ObstacleDetector(cfg);temporal=TemporalFilter(cfg)
a,ta=cloud('roundT_doubleT');b,tb=cloud('doubleT_obstacle')
r1=det.process(a);temporal.update(r1,ta)
r2=det.process(b);d2=temporal.update(r2,tb)
fresh=ObstacleDetector(cfg).process(b)
print(json.dumps({'jump_reset_triggered':d2.reset,'result_with_old_curve':r2.to_dict(),'fresh_result':fresh.to_dict()},indent=2))
