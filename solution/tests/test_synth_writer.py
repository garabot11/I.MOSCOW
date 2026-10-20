"""Synthetic-data writer: PointCloud2 CDR round trip through the project's parser, MCAP bag + Humble metadata."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth"))

from metro_obstacle_detector.cloud_io import parse_pointcloud2_cdr, pointcloud2_to_struct  # noqa: E402
from synth.writer import FALCON_DTYPE, HESAI_DTYPE, serialize_pointcloud2  # noqa: E402


def _hesai_points(n=128 * 20, seed=0):
    rs = np.random.default_rng(seed)
    p = np.zeros(n, dtype=HESAI_DTYPE)
    p["x"], p["y"], p["z"] = rs.normal(size=(3, n)).astype(np.float32)
    p["x"][::3] = p["y"][::3] = p["z"][::3] = 0.0
    p["intensity"] = rs.integers(0, 255, n)
    p["ring"] = np.arange(n) % 128
    p["timestamp"] = 946692931.6334 + np.arange(n) * 1e-6
    return p


@pytest.mark.parametrize("dtype", ["hesai", "falcon"])
def test_cdr_roundtrip(dtype):
    if dtype == "hesai":
        p = _hesai_points()
        frame = "hesai_lidar"
    else:
        rs = np.random.default_rng(1)
        p = np.zeros(1000, dtype=FALCON_DTYPE)
        p["x"], p["y"], p["z"] = rs.normal(size=(3, 1000)).astype(np.float32)
        p["timestamp"] = 3115.3 + np.arange(1000) * 1e-6
        p["flags"], p["scan_id"], p["scan_idx"] = 9, rs.integers(0, 38, 1000), rs.integers(0, 1270, 1000)
        p["is_2nd_return"][::50] = 1
        frame = "seyond"
    raw = serialize_pointcloud2(p, 946692931_633400000, frame, is_dense=(dtype == "falcon"))
    assert len(raw) % 4 == 0
    msg = parse_pointcloud2_cdr(raw)
    assert msg.header.frame_id == frame and msg.height == 1 and msg.width == len(p)
    assert msg.point_step == p.dtype.itemsize and msg.header.stamp.sec == 946692931 and msg.header.stamp.nanosec == 633400000
    back = pointcloud2_to_struct(msg)
    for name in p.dtype.names:
        assert np.array_equal(back[name], p[name]), name
    assert [f.offset for f in msg.fields] == [p.dtype.fields[n][1] for n in p.dtype.names]


def test_mcap_bag_and_humble_metadata(tmp_path):
    pytest.importorskip("mcap")
    yaml = pytest.importorskip("yaml")
    from mcap.reader import make_reader

    from synth.writer import BagWriter

    w = BagWriter(str(tmp_path / "bag"), "/lidar_points", "mcap")
    pts = _hesai_points()
    for i in range(3):
        w.write(serialize_pointcloud2(pts, 1_800_000_000_000_000_000 + i * 100_000_000, "hesai_lidar"),
                1_800_000_000_000_000_000 + i * 100_000_000)
    w.close()
    meta = yaml.safe_load(open(tmp_path / "bag" / "metadata.yaml"))["rosbag2_bagfile_information"]
    assert meta["version"] == 5 and meta["storage_identifier"] == "mcap" and meta["message_count"] == 3
    assert meta["topics_with_message_count"][0]["topic_metadata"]["type"] == "sensor_msgs/msg/PointCloud2"
    assert meta["duration"]["nanoseconds"] == 200_000_000
    f = os.path.join(tmp_path, "bag", meta["relative_file_paths"][0])
    with open(f, "rb") as fh:
        msgs = list(make_reader(fh).iter_messages())
    assert len(msgs) == 3
    sc, ch, m = msgs[0]
    assert sc.name == "sensor_msgs/msg/PointCloud2" and sc.encoding == "ros2msg" and ch.message_encoding == "cdr"
    back = pointcloud2_to_struct(parse_pointcloud2_cdr(m.data))
    assert np.array_equal(back["x"], pts["x"])
