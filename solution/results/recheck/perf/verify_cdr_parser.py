"""Every PointCloud2 of every bag: parse_pointcloud2_cdr(raw) must give the same cloud as deserialize_message."""
import json, sys, time
sys.path.insert(0, '/src/metro_obstacle_detector')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from metro_obstacle_detector.cloud_io import cloud_to_xyz, parse_pointcloud2_cdr

report = {}
for bag in sys.argv[1:]:
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri='/data/' + bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    n = bad = 0; t_des = []; t_raw = []
    while rd.has_next():
        topic, raw, _ = rd.read_next()
        t = time.perf_counter(); m1 = deserialize_message(raw, PointCloud2); x1, _, i1 = cloud_to_xyz(m1, intensity=False); t_des.append(time.perf_counter() - t)
        t = time.perf_counter(); m2 = parse_pointcloud2_cdr(raw); x2, _, i2 = cloud_to_xyz(m2, intensity=False); t_raw.append(time.perf_counter() - t)
        same = (np.array_equal(x1, x2, equal_nan=True) and i1 == i2 and m1.is_dense == m2.is_dense
                and [(f.name, f.offset, f.datatype, f.count) for f in m1.fields] == [(f.name, f.offset, f.datatype, f.count) for f in m2.fields])
        n += 1; bad += (not same)
    report[bag] = {'messages': n, 'different': bad, 'decode_ms_median_deserialize': round(float(np.median(t_des)) * 1e3, 2),
                   'decode_ms_median_raw_parser': round(float(np.median(t_raw)) * 1e3, 2)}
    print(bag, report[bag], flush=True)
json.dump(report, open('/out/cdr_parser_equality.json', 'w'), indent=1)
