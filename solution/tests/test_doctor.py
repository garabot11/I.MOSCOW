"""doctor: never raises, bag discovery under several dataset folders, broken bags, two cloud topics,
MCAP, SQLite COUNT(*) vs metadata, DDS / display items."""
import json
import os
import sqlite3

import pytest

from metro_obstacle_detector import doctor


def _sqlite_bag(path, topics_counts, meta_counts=None, storage="sqlite3"):
    """A minimal rosbag2 SQLite directory (schema of rosbag2_storage_default_plugins) with dummy blobs."""
    os.makedirs(path, exist_ok=True)
    db = os.path.join(path, os.path.basename(path) + "_0.db3")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    for i, (name, n) in enumerate(topics_counts.items(), 1):
        con.execute("INSERT INTO topics VALUES (?,?,?,?,?)", (i, name, "sensor_msgs/msg/PointCloud2", "cdr", ""))
        for k in range(n):
            con.execute("INSERT INTO messages(topic_id, timestamp, data) VALUES (?,?,?)", (i, k, b"\0" * 8))
    con.commit()
    con.close()
    meta_counts = meta_counts or topics_counts
    lines = ["rosbag2_bagfile_information:", "  version: 5", "  storage_identifier: %s" % storage,
             "  duration:", "    nanoseconds: 2000000000", "  message_count: %d" % sum(meta_counts.values()),
             "  topics_with_message_count:"]
    for name, n in meta_counts.items():
        lines += ["    - topic_metadata:", "        name: %s" % name, "        type: sensor_msgs/msg/PointCloud2",
                  "        serialization_format: cdr", "        offered_qos_profiles: ''", "      message_count: %d" % n]
    with open(os.path.join(path, "metadata.yaml"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return db


def test_find_bags_several_dataset_folders(tmp_path):
    _sqlite_bag(str(tmp_path / "for_hackathon" / "a"), {"/lidar_points": 2})
    _sqlite_bag(str(tmp_path / "falcon" / "b"), {"/iv_points": 2})
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "x.zst").write_bytes(b"x")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "c_0.db3").write_bytes(b"not a database")
    found = doctor.find_bags([str(tmp_path)])
    assert [os.path.relpath(p, tmp_path) for p in found] == ["broken", "falcon/b", "for_hackathon/a"]
    assert doctor.find_bags([str(tmp_path / "falcon" / "b")]) == [str(tmp_path / "falcon" / "b")]
    assert doctor.find_bags([str(tmp_path / "missing")]) == []


def test_inspect_missing_metadata_and_garbage(tmp_path):
    (tmp_path / "nometa").mkdir()
    (tmp_path / "nometa" / "x_0.db3").write_bytes(b"\0")
    info = doctor.inspect_bag(str(tmp_path / "nometa"))
    assert any("no metadata.yaml" in p and "ros2 bag reindex" in p for p in info["problems"])
    (tmp_path / "garbage").mkdir()
    (tmp_path / "garbage" / "metadata.yaml").write_text(": : [ not yaml")
    info = doctor.inspect_bag(str(tmp_path / "garbage"))
    assert any("metadata.yaml unreadable" in p for p in info["problems"])
    assert any("no .db3 / .mcap" in p for p in info["problems"])


def test_sqlite_count_is_the_real_count(tmp_path):
    _sqlite_bag(str(tmp_path / "b"), {"/lidar_points": 7, "/other": 3})
    assert doctor._sqlite_count(str(tmp_path / "b"), "/lidar_points") == 7


def test_resolve_bag(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "LOGS", str(tmp_path / "logs"))
    _sqlite_bag(str(tmp_path / "d1" / "same"), {"/a": 1})
    _sqlite_bag(str(tmp_path / "d2" / "same"), {"/a": 1})
    r = doctor.resolve_bag("nothing_like_this", [str(tmp_path)])
    assert r["ok"] is False and "not found" in r["error"]
    r = doctor.resolve_bag("same", [str(tmp_path)])
    assert r["ok"] is False and "several" in r["error"]
    r = doctor.resolve_bag(str(tmp_path / "d1"), [str(tmp_path)])
    assert r["ok"] is False and "not a rosbag2 directory" in r["error"]


def test_display_and_dds_items(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    it = doctor.check_display()
    assert it[0]["status"] == doctor.WARN and it[0]["data"]["gl_mode"] == "none"
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.delenv("FASTRTPS_DEFAULT_PROFILES_FILE", raising=False)
    assert doctor.check_dds()[0]["status"] == doctor.WARN
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0")
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/nonexistent.xml")
    assert doctor.check_dds()[0]["status"] == doctor.FAIL


def test_cpu_uses_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    it = doctor.check_cpu()[0]
    assert it["data"]["cpus"] == 8 and it["message"].startswith("8 CPUs")


def test_main_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "LOGS", str(tmp_path / "logs"))
    monkeypatch.setattr(doctor, "RESULTS", str(tmp_path))
    monkeypatch.setattr(doctor, "check_gpu", lambda prefer="auto": [doctor._item("gpu", doctor.OK, "fake", "", mode="gpu")])

    def boom():
        raise RuntimeError("broken check")
    monkeypatch.setattr(doctor, "check_ram", boom)
    out = tmp_path / "rep.json"
    assert doctor.main(["--bag-root", str(tmp_path / "nope"), "--json", str(out), "--no-color"]) == 0
    rep = json.loads(out.read_text())
    assert (tmp_path / "logs" / "doctor.json").exists()
    st = {i["check"]: i["status"] for i in rep["items"]}
    assert st["ram"] == doctor.WARN and st["bags"] == doctor.WARN and st["gpu"] == doctor.OK
    assert doctor.main(["--no-such-option"]) == 0
    text = capsys.readouterr().out
    assert "Traceback" not in text and "broken check" in text


# ------------------------------------------------------------------ with ROS 2 (inside the image)
def _write_ros_bag(path, storage, topics, n=3):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name in topics:
        w.create_topic(rosbag2_py.TopicMetadata(name=name, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))
    for k in range(n):
        for name in topics:
            m = PointCloud2()
            m.header.frame_id = "test_frame"
            m.header.stamp.sec = 100 + k
            m.height, m.width, m.point_step = 1, 2, 12
            m.row_step = 24
            m.fields = [PointField(name=c, offset=4 * i, datatype=7, count=1) for i, c in enumerate("xyz")]
            m.data = bytes(24)
            w.write(name, serialize_message(m), (100 + k) * 10**9)
    del w


@pytest.mark.parametrize("storage", ["sqlite3", "mcap"])
def test_real_bag_two_cloud_topics(tmp_path, storage):
    path = str(tmp_path / ("bag_" + storage))
    try:
        _write_ros_bag(path, storage, ["/lidar_points", "/second_cloud"])
    except Exception as exc:  # pragma: no cover
        pytest.skip("cannot write a %s bag here: %s" % (storage, exc))
    info = doctor.inspect_bag(path)
    assert not info["problems"], info["problems"]
    assert info["storage"] == storage and info["frame_id"] == "test_frame" and info["real_count"] == 3
    assert sorted(info["cloud_topics"]) == ["/lidar_points", "/second_cloud"]
    assert any("2 PointCloud2 topics" in n for n in info["notes"])
    assert info["first_cloud"]["points"] == 2 and info["first_cloud"]["point_step"] == 12
    full = doctor.inspect_bag(path, full=True)
    assert full["real_count"] == 3 and full["count_method"].startswith("bag_tools")


def test_truncated_db3_is_a_problem_not_a_crash(tmp_path):
    path = str(tmp_path / "trunc")
    try:
        _write_ros_bag(path, "sqlite3", ["/lidar_points"], n=50)
    except Exception as exc:  # pragma: no cover
        pytest.skip(str(exc))
    db = [f for f in os.listdir(path) if f.endswith(".db3")][0]
    p = os.path.join(path, db)
    data = open(p, "rb").read()
    with open(p, "wb") as fh:
        fh.write(data[: len(data) // 3])
    info = doctor._inspect_in_child(path, False, timeout=60)
    assert info["problems"], info


def test_synthetic_scene_layout(tmp_path, monkeypatch):
    """Synthetic sets keep each bag in <set>/<scene>/bag: listed and played as <set>/<scene>."""
    monkeypatch.setattr(doctor, "LOGS", str(tmp_path / "logs"))
    root = tmp_path / "synthetic"
    _sqlite_bag(str(root / "val_hesai_insert" / "val_hesai_insert_001_pos" / "bag"), {"/lidar_points": 2})
    _sqlite_bag(str(root / "val_hesai_insert" / "val_hesai_insert_002_pos" / "bag"), {"/lidar_points": 2})
    found = doctor.find_bags([str(tmp_path)])
    assert len(found) == 2 and all(p.endswith("/bag") for p in found)
    assert doctor._rel(found[0], [str(tmp_path)]) == "synthetic/val_hesai_insert/val_hesai_insert_001_pos"
    for name in ("val_hesai_insert/val_hesai_insert_001_pos", "val_hesai_insert_001_pos",
                 str(root / "val_hesai_insert" / "val_hesai_insert_001_pos")):
        r = doctor.resolve_bag(name, [str(tmp_path)])
        assert r["path"].endswith("/val_hesai_insert_001_pos/bag"), (name, r)
        assert r["rel"] == "synthetic/val_hesai_insert/val_hesai_insert_001_pos"
    r = doctor.resolve_bag("val_hesai_insert", [str(tmp_path)])
    assert r["ok"] is False and ("several" in r["error"] or "not a rosbag2" in r["error"])
