"""Tests of the Д3 training pipeline against interface B.

The scenes built here are TEST FIXTURES (a handful of synthetic Hesai-layout clouds with one labelled
box, written as rosbag2 sqlite3 / MCAP with rosbags) that exercise the reader, the cache, the
dataset, the metrics and the CLI; nothing here is training data and no metric from it is reported.
Run:  python -m pytest -q solution/train/test_train_pipeline.py   (needs torch + rosbags)
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch = pytest.importorskip("torch")
pytest.importorskip("rosbags")

import _paths  # noqa: E402,F401
import d3_data  # noqa: E402
import d3_metrics  # noqa: E402
import d3_preprocess  # noqa: E402
import d3_scenes  # noqa: E402
import train_d3  # noqa: E402
from metro_obstacle_detector import nn_input  # noqa: E402

EL = np.concatenate([np.linspace(14.4, 2.45, 25), np.linspace(1.97, -6.07, 64), np.linspace(-6.16, -25.1, 39)])
DT_HESAI = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                     "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"], "offsets": [0, 4, 8, 12, 16, 18],
                     "itemsize": 26})
DT_FALCON = np.dtype({"names": ["x", "y", "z", "intensity", "timestamp"],
                      "formats": ["<f4", "<f4", "<f4", "<f4", "<f8"], "offsets": [0, 4, 8, 12, 16], "itemsize": 24})
NF = 400  # Hesai fixture firings: azimuth about -20 .. +20 deg
FALCON_BAG = "/home/gag/i.mos/dataset/falcon/falcon_test_second_dark"
HESAI_BAG = "/home/gag/i.mos/dataset/for_hackathon/roundT_doubleT"


def tunnel_cast(dirs, sensor_h, box_range):
    """Ranges along unit directions (analysis frame d, l, h) in a straight box tunnel: floor at -sensor_h,
    ceiling 4.5 m above it, walls at |l| = 2.6 m, no returns beyond 180 m; a 0.9 x 0.9 m box face at
    d = box_range standing on the floor.  Returns (range or inf, hit-the-box mask)."""
    dd, ll, hh = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    inf = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(hh < -1e-6, -sensor_h / hh, inf)
        tc = np.where(hh > 1e-6, (4.5 - sensor_h) / hh, inf)
        tw = np.where(np.abs(ll) > 1e-6, 2.6 / np.abs(ll), inf)
        tb = np.where(dd > 1e-6, box_range / dd, inf)
    hb, lb = tb * hh, tb * ll
    tb = np.where((np.abs(lb) <= 0.45) & (hb >= -sensor_h) & (hb <= -sensor_h + 0.9), tb, inf)
    t = np.minimum(np.minimum(tf, tc), np.minimum(tw, tb))
    box = np.isfinite(tb) & (tb <= t + 1e-9)
    t = np.where(t > 180.0, inf, t)
    return t, box & np.isfinite(t)


def fixture_cloud(rng, box_range=40.0):
    """Hesai layout (forward -y, up +z, l = +x; 128 rings, dual-return column pairs, (0,0,0) no-return
    slots) of the tunnel with a box; returns (struct, box slots incl. both returns)."""
    R = 128
    icpt = 20.0 + rng.choice([0.0, 0.3, 0.6], size=R)
    f = np.arange(NF)[:, None]
    az = np.radians(icpt[None, :] - 0.1 * f)
    el = np.radians(EL)[None, :] * np.ones((NF, 1))
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    r, box = tunnel_cast(dirs, 1.3, box_range)
    ret = np.isfinite(r) & ((rng.random(r.shape) < 0.95) | box)
    r = np.where(ret, r + rng.normal(0, 0.01, r.shape), 0.0)
    st = np.zeros(NF * 2 * R, dtype=DT_HESAI)
    slots = []
    for k in (0, 1):
        idx = (2 * f + k) * R + np.arange(R)[None, :]
        d, l, h = (r[..., None] * dirs).transpose(2, 0, 1)
        st["x"][idx], st["y"][idx], st["z"][idx] = l, -d, h
        st["intensity"][idx] = np.where(ret, 12.0, 0)
        slots.append(idx[box & ret])
    st["ring"] = np.arange(len(st)) % R
    return st, np.sort(np.concatenate(slots))


def falcon_cloud(rng, box_range=40.0):
    """Falcon-like unorganised cloud ('ros' mode: x forward, y left, z up; 2.0 m above the rails), returned
    points only, 0.18 x 0.24 deg raster over 120 x 25 deg; returns (struct, box point indices)."""
    el, az = np.meshgrid(np.radians(np.arange(-13.0, 12.0, 0.24)), np.radians(np.arange(-60.0, 60.0, 0.18)),
                         indexing="ij")
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
    r, box = tunnel_cast(dirs, 2.0, box_range)
    keep = np.isfinite(r) & ((rng.random(r.shape) < 0.95) | box)
    p = (r[keep, None] + rng.normal(0, 0.02, (keep.sum(), 1))) * dirs[keep]
    st = np.zeros(len(p), dtype=DT_FALCON)
    st["x"], st["y"], st["z"] = p[:, 0], p[:, 1], p[:, 2]
    st["intensity"] = 10.0
    return st, np.flatnonzero(box[keep])


def write_scene(root, set_name, sid, n_frames=4, storage="sqlite3", sensor="hesai128", seed=0):
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    PC2, PF = ts.types["sensor_msgs/msg/PointCloud2"], ts.types["sensor_msgs/msg/PointField"]
    Header, Time = ts.types["std_msgs/msg/Header"], ts.types["builtin_interfaces/msg/Time"]
    falcon = sensor == "falcon_prime"
    dt = DT_FALCON if falcon else DT_HESAI
    codes = {"<f4": 7, "<f8": 8, "<u2": 4}
    fields = [PF(name=n, offset=dt.fields[n][1], datatype=codes[dt.fields[n][0].str], count=1) for n in dt.names]
    topic, frame_id = ("/iv_points", "seyond") if falcon else ("/lidar_points", "hesai_lidar")
    sdir = os.path.join(root, set_name, sid)
    os.makedirs(sdir)
    rng = np.random.default_rng(seed)
    labels, fr, sl, ob = [], [], [], []
    plugin = StoragePlugin.MCAP if storage == "mcap" else StoragePlugin.SQLITE3
    with Writer(os.path.join(sdir, "bag"), version=8, storage_plugin=plugin) as w:
        conn = w.add_connection(topic, "sensor_msgs/msg/PointCloud2", typestore=ts)
        for i in range(n_frames):
            st, slots = (falcon_cloud if falcon else fixture_cloud)(rng, 40.0 - 0.5 * i)
            stamp = 1_700_000_000_000_000_000 + i * 100_000_000
            msg = PC2(header=Header(stamp=Time(sec=stamp // 10**9, nanosec=stamp % 10**9), frame_id=frame_id),
                      height=1, width=len(st), fields=fields, is_bigendian=False, point_step=dt.itemsize,
                      row_step=dt.itemsize * len(st), data=np.frombuffer(st.tobytes(), np.uint8), is_dense=falcon)
            w.write(conn, stamp, ts.serialize_cdr(msg, "sensor_msgs/msg/PointCloud2"))
            labels.append({"frame": i, "stamp_ns": stamp, "sight_limit_m": None, "objects": [
                {"id": 7, "class": "box", "is_obstacle": True, "in_clearance": True, "visible": True,
                 "distance_m": 40.0 - 0.5 * i, "n_hits": int(len(slots) // (1 if falcon else 2))}]})
            fr.append(np.full(len(slots), i))
            sl.append(slots)
            ob.append(np.full(len(slots), 7))
    with open(os.path.join(sdir, "labels.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in labels) + "\n")
    np.savez(os.path.join(sdir, "slot_labels.npz"), frame=np.concatenate(fr).astype(np.int32),
             slot=np.concatenate(sl).astype(np.int32), obj_id=np.concatenate(ob).astype(np.int32))
    with open(os.path.join(sdir, "scene.json"), "w") as fh:
        json.dump({"sensor": sensor, "source": "parametric", "topic": topic, "frame_id": frame_id,
                   "coordinate_mode": "ros" if falcon else None, "seed": seed, "generator_version": "test-fixture"}, fh)
    return sdir


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("synthetic_fixture"))
    write_scene(root, "train_fix", "s000", storage="sqlite3", seed=1)
    write_scene(root, "train_fix", "s001", storage="mcap", seed=2)
    write_scene(root, "train_fix", "s002", storage="mcap", seed=3)
    write_scene(root, "val_hesai_fix", "v000", storage="mcap", seed=4)
    write_scene(root, "train_falcon", "f000", sensor="falcon_prime", storage="mcap", seed=5)
    write_scene(root, "train_falcon", "f001", sensor="falcon_prime", storage="sqlite3", seed=6)
    return root


def test_reader_interface_b(data_root):
    scenes = d3_scenes.discover(data_root, ["train_*"])
    assert [s.key for s in scenes] == ["train_falcon/f000", "train_falcon/f001", "train_fix/s000", "train_fix/s001",
                                       "train_fix/s002"]
    s = scenes[3]
    labels = d3_scenes.read_labels(s)
    sl = d3_scenes.read_slot_labels(s)
    clouds = list(d3_scenes.iter_clouds(s.bag_dir, "/lidar_points"))
    assert len(clouds) == len(labels) == 4 and sorted(sl) == [0, 1, 2, 3]
    assert [c[1] for c in clouds] == [r["stamp_ns"] for r in labels]
    slots, ids = sl[0]
    got = d3_scenes.lookup_obj(np.array([slots[0], slots[-1], -1, slots[0] + 1 if slots[0] + 1 not in slots else -5]), sl[0])
    assert got[:3].tolist() == [7, 7, -1]


def test_preprocess_targets_follow_slot_labels(data_root, tmp_path):
    s = d3_scenes.discover(data_root, ["train_fix"])[0]
    opts = d3_preprocess.PreprocessOptions(aux="none")
    meta = d3_preprocess.preprocess_scene(s, str(tmp_path), opts)
    assert meta["n_frames"] == 4 and meta["frames_matched_by_stamp"] == 4 and not meta["cached"]
    c = d3_data.SceneCache(d3_preprocess.cache_dir_for(str(tmp_path), s))
    assert c.feats.shape == (4, 7, 64, 256 + 64) and c.feats.dtype == np.float16
    assert c.target.shape == (4, 64, 320) and c.objid.shape == (4, 64, 320)
    for t in range(4):
        m = c.target[t] == 1
        assert m.sum() > 10 and np.array_equal(m, c.objid[t] == 7)
        rng = c.feats[t, 0].astype(np.float32)[m] * 200
        assert np.allclose(rng, 40.0 - 0.5 * t, atol=0.1)           # labelled pixels carry the box return (face: slant range)
        assert (c.feats[t, 3][m] == 1).all()
        assert c.objects[t]["objects"][0]["px_in_crop"] == int(m.sum())
    assert d3_preprocess.preprocess_scene(s, str(tmp_path), opts)["cached"]   # cache hit
    opts2 = d3_preprocess.PreprocessOptions(aux="none", margin_cols=16)
    assert not d3_preprocess.is_cached(str(tmp_path), s, opts2)               # options change -> rebuild
    assert meta["range_image_builder"] == nn_input.RANGE_IMAGE_BUILDER_ID and "organised=True" in meta["sensor_profile"]
    with pytest.warns(RuntimeWarning, match="STAND-IN"):   # the fallback warns loudly and is Hesai-only
        with pytest.raises(d3_preprocess.UnsupportedScene):
            d3_preprocess.preprocess_scene(d3_scenes.discover(data_root, ["train_falcon"])[0], str(tmp_path),
                                           d3_preprocess.PreprocessOptions(aux="none", builder="standin"))
    assert "--builder" not in train_d3.build_parser().format_help()          # not exposed by the CLI


def test_preprocess_falcon_scene_binned(data_root, tmp_path):
    """falcon_prime scenes go through the angle-binned range image (organized=False) -- not skipped."""
    s = d3_scenes.discover(data_root, ["train_falcon"])[0]
    meta = d3_preprocess.preprocess_scene(s, str(tmp_path), d3_preprocess.PreprocessOptions(aux="d1"))
    assert meta["n_frames"] == 4 and "organised=False" in meta["sensor_profile"]
    assert "+x/+z" in meta["sensor_profile"]                  # axes of the 'ros' convention detected
    c = d3_data.SceneCache(d3_preprocess.cache_dir_for(str(tmp_path), s))
    assert c.feats.shape == (4, 7, 64, 320)
    for t in range(4):
        m = c.target[t] == 1
        assert m.sum() >= 5 and np.array_equal(m, c.objid[t] == 7)
        assert np.allclose(c.feats[t, 0].astype(np.float32)[m] * 200, 40.0 - 0.5 * t, atol=0.15)
    assert meta["d1_unknown_frames"] < 4 and (c.feats[:, 4] != 0).any()   # Д1 track model -> aux channels


def test_preprocess_with_d1_aux_runs(data_root, tmp_path):
    s = d3_scenes.discover(data_root, ["train_fix"])[1]
    meta = d3_preprocess.preprocess_scene(s, str(tmp_path), d3_preprocess.PreprocessOptions(aux="d1"))
    assert meta["n_frames"] == 4


def test_target_class_definitions():
    tc = d3_preprocess.target_class
    assert tc({"class": "dust"}, "object") == 2
    assert tc({"class": "box", "is_obstacle": False}, "object") == 1
    assert tc({"class": "box", "is_obstacle": False}, "is_obstacle") == 0
    assert tc({"class": "person_lying", "in_clearance": True}, "in_clearance") == 1


def _cache(T=5, h=64, w=320, seed=0):
    rng = np.random.default_rng(seed)
    f = np.zeros((T, 7, h, w), np.float16)
    f[:, 0] = rng.uniform(0.05, 0.5, (T, h, w))
    f[:, 3] = 1
    f[:, 2] = 0.1
    f[:, 5] = 0.25
    y = np.zeros((T, h, w), np.uint8)
    o = np.full((T, h, w), -1, np.int32)
    y[:, 20:30, 150:160] = 1
    o[:, 20:30, 150:160] = 3
    f[:, 0, 20:30, 150:160] = 60 / 200
    return d3_data.SceneCache.from_arrays(f, y, o)


def test_dataset_shapes_and_augmentation_invariants():
    c = _cache()
    ev = d3_data.D3Dataset([c], 256, train=False)
    s = ev[2]
    assert s["x"].shape == (21, 64, 256) and s["y"].shape == (64, 256) and s["x"].dtype == torch.float32
    assert torch.equal(s["y"], torch.from_numpy(c.target[2, :, 32:288].astype(np.int64)))   # centre crop
    tr = d3_data.D3Dataset([c], 256, train=True, aug=d3_data.AugConfig(max_drop=0.3, partial_obj=1.0), seed=1)
    for k in range(len(tr)):
        for ep in range(3):
            tr.set_epoch(ep)
            s = tr[k]
            x, y = s["x"].numpy(), s["y"].numpy()
            assert (x[3][y == 1] == 1).all()        # labels only where the current frame has a return
            assert (s["obj"].numpy()[y == 1] == 3).all()
            rows = np.flatnonzero((y == 1).any(axis=1))
            assert len(rows) and rows.min() == 20   # partial objects keep the upper part
            assert set(np.unique(x[5])) <= {0.0, 0.25, -0.25}   # lateral: kept, negated (flip) or aux-dropped
    t1 = d3_data.D3Dataset([c], 256, train=True, seed=5)
    t2 = d3_data.D3Dataset([c], 256, train=True, seed=5)
    assert torch.equal(t1[3]["x"], t2[3]["x"])     # deterministic per (seed, epoch, index)


def test_history_order_in_dataset():
    c = _cache()
    for t in range(5):
        c.feats[t, 1] = t / 10
    ev = d3_data.D3Dataset([c], 256, train=False)
    x = ev[3]["x"].numpy()
    assert [round(float(x[k * 7 + 1, 0, 0]), 3) for k in range(3)] == [0.3, 0.2, 0.1]
    x0 = ev[0]["x"].numpy()
    assert [round(float(x0[k * 7 + 1, 0, 0]), 3) for k in range(3)] == [0.0, 0.0, 0.0]


def test_losses():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 16)
    y = torch.randint(0, 3, (2, 8, 16))
    w = [1.0, 4.0, 2.0]
    ce = torch.nn.functional.cross_entropy(logits, y, weight=torch.tensor(w), reduction="none").mean()
    assert torch.allclose(d3_data.focal_loss(logits, y, w, gamma=0.0), ce, atol=1e-6)
    assert d3_data.focal_loss(logits, y, w, gamma=2.0) < ce
    perfect = torch.nn.functional.one_hot(y, 3).permute(0, 3, 1, 2).float() * 50
    assert float(d3_data.soft_dice_loss(perfect, y, 1)) < 1e-3
    assert float(d3_data.d3_loss(logits, y, torch.full(y.shape, 50.0), w)) > 0


def test_metrics_per_band():
    acc = d3_metrics.MetricAccumulator(0.5)
    r = np.zeros((1, 10, 40), np.float32)
    y = np.zeros((1, 10, 40), np.uint8)
    o = np.full((1, 10, 40), -1)
    p = np.zeros((1, 10, 40), np.float32)
    r[0, 2:4, 2:4] = 60.0
    y[0, 2:4, 2:4] = 1
    o[0, 2:4, 2:4] = 1          # object 1: 4 px at 60 m, 2 predicted
    p[0, 2, 2:4] = 0.9
    r[0, 6, 20:23] = 120.0       # false-positive component at 120 m
    p[0, 6, 20:23] = 0.8
    r[0, 8, 30] = 25.0
    y[0, 8, 30] = 1
    o[0, 8, 30] = 2              # object 2 at 25 m, missed
    r[0, 0, 35:37] = 10.0
    y[0, 0, 35:37] = 2           # dust, one pixel predicted as obstacle -> FP on dust
    p[0, 0, 35] = 0.7
    acc.update(p, y, o, r)
    res = acc.result()
    b = res["bands"]
    assert b["50-100"]["objects"] == 1 and b["50-100"]["object_recall"] == 1.0
    assert b["50-100"]["pixel_precision"] == 1.0 and b["50-100"]["pixel_recall"] == 0.5
    assert b["20-50"]["objects"] == 1 and b["20-50"]["object_recall"] == 0.0
    assert b["100-150"]["fp_components"] == 1 and b["0-20"]["fp_components"] == 1
    assert res["fp_components_on_dust"] == 1 and res["dust_pixels_as_obstacle_fraction"] == 0.5
    assert res["fp_components_per_frame"] == 2.0 and res["frames_with_fp_fraction"] == 1.0


def test_split_by_scene_is_seeded():
    keys = [f"train_a/s{i:03d}" for i in range(20)]
    a = train_d3.split_scenes(keys, 0.15, 0)
    b = train_d3.split_scenes(list(reversed(keys)), 0.15, 0)
    c = train_d3.split_scenes(keys, 0.15, 1)
    assert a == b and a != c and len(a[1]) == 3 and not set(a[0]) & set(a[1])


def test_train_cli_refuses_without_synthetic_data(tmp_path):
    assert train_d3.main(["--data", str(tmp_path / "nothing"), "--work", str(tmp_path / "w")]) == 2


def test_train_cli_end_to_end_and_export_loads(data_root, tmp_path):
    from metro_obstacle_detector import det_nn

    export = str(tmp_path / "models" / "d3_unet.pt")
    rc = train_d3.main(["--data", data_root, "--train-sets", "train_*", "--val-sets", "val_hesai_*",
                        "--work", str(tmp_path / "work"), "--run-name", "t", "--epochs", "2", "--max-steps", "3",
                        "--batch", "2", "--workers", "0", "--preprocess-workers", "1", "--aux", "none",
                        "--device", "cpu", "--export", export])
    assert rc == 0
    run = tmp_path / "work" / "runs" / "t"
    for f in ("train_log.jsonl", "last.pt", "final_metrics.json", "split.json", "train.log"):
        assert (run / f).exists(), f
    split = json.loads((run / "split.json").read_text())
    assert len(split["train"]) + len(split["val"]) == 5 and len(split["val"]) == 1 and "val_hesai_fix" in split["heldout"]
    assert "skipped" not in (run / "train.log").read_text()      # falcon_prime scenes are used (binned images)
    assert os.path.exists(export) and os.path.exists(export.replace(".pt", ".state.pt"))
    card = (tmp_path / "models" / "model_card.md").read_text()
    assert "Д3 model card" in card and "SMOKE" not in card

    class DM:
        device, mode = torch.device("cpu"), "cpu_forced"

        def report_failure(self, exc):
            raise AssertionError("no GPU here")

    det = det_nn.NNDetector(DM(), export)
    assert det.meta["smoke_test"] is False and det.meta["aux_channels_trained"] == []
    assert det.meta["range_image_builder"] == nn_input.RANGE_IMAGE_BUILDER_ID
    assert det.meta["sensors"] == ["falcon_prime", "hesai128"]
    assert det._zero_idx and len(det._zero_idx) == 9           # aux channels zeroed: trained with --aux none


def _random_model(tmp_path):
    from metro_obstacle_detector.nn_model import RangeUNet, export_d3

    torch.manual_seed(0)
    return export_d3(RangeUNet(), str(tmp_path / "d3_unet.pt"), {"crop": nn_input.CropConfig().to_dict()})["torchscript"]


class _CpuDM:
    device, mode = torch.device("cpu"), "cpu_forced"

    def report_failure(self, exc):
        raise AssertionError(exc)


@pytest.mark.skipif(not os.path.isdir(FALCON_BAG), reason="real Falcon bag not available")
def test_real_falcon_frame_binned_crop_and_inference(tmp_path):
    """One REAL Falcon Prime frame (fetched by id, read-only) through the training path and det_nn."""
    stamp, msg = d3_scenes.sqlite_message(FALCON_BAG, "/iv_points", 0)
    opts = d3_preprocess.PreprocessOptions(aux="d1")
    proc = d3_preprocess.FrameProcessor(opts)
    frame = proc.frame(msg)
    ri = proc.range_image(frame)
    assert not ri.organized and np.all(ri.ring_of_row == -1) and np.all(np.diff(ri.az_deg) < 0)
    aux = proc.aux(frame, ri)
    f, t, o, objs, stray = proc.process(ri, aux, None, None)
    assert f.shape == (7, 64, 320) and (t == 0).all() and (o == -1).all()
    wide = opts.wide_crop()
    cm = nn_input.compute_crop_map(ri, wide)
    assert not cm.native_rows and not cm.native_cols      # 0.08 / 0.09 deg bins -> resampled onto 0.128 / 0.1
    assert cm.inside.all()
    tel, taz = wide.target_el(), wide.target_az()
    assert np.abs(ri.el_deg[cm.src_row] - tel[:, None]).max() <= 0.08 + 1e-4
    assert np.abs(ri.az_deg[cm.src_col] - taz[None, :]).max() <= 0.1 + 1e-4
    valid = f[3] > 0
    assert valid.mean() > 0.2 and (f[0][valid] > 0).all()
    if proc.last_status != "unknown":
        assert (f[4][valid] != 0).mean() > 0.5               # Д1 rail level of the same frame -> h_above_rail
    det = det_nn_cpu(tmp_path)
    cands = det.step(ri, aux)
    assert isinstance(cands, list) and det.last_map.shape == (64, 256)


def det_nn_cpu(tmp_path):
    from metro_obstacle_detector import det_nn

    return det_nn.NNDetector(_CpuDM(), _random_model(tmp_path))


@pytest.mark.skipif(not os.path.isdir(HESAI_BAG), reason="real Hesai bag not available")
def test_real_hesai_frame_range_image_native_crop():
    _, _, msg = next(d3_scenes.iter_messages(HESAI_BAG))
    proc = d3_preprocess.FrameProcessor(d3_preprocess.PreprocessOptions(aux="none"))
    ri = proc.range_image(proc.frame(msg))
    assert ri.organized and ri.sensor == "hesai128"
    cm = nn_input.compute_crop_map(ri, nn_input.CropConfig())
    assert cm.native_rows and cm.native_cols
    assert np.array_equal(ri.ring_of_row[cm.src_row[:, 0]], np.arange(25, 89))
    assert np.abs(ri.az_deg[cm.src_col[0]] - nn_input.CropConfig().target_az()).max() <= 0.051


def test_split_by_background_keeps_backgrounds_apart():
    def mk(k, rb):
        return d3_scenes.Scene("train_x", k, "/nonexistent", {"real_bag": rb} if rb else {})
    scenes = [mk(f"a{i}", "bagA") for i in range(6)] + [mk(f"b{i}", "bagB") for i in range(4)] + \
        [mk(f"c{i}", "bagC") for i in range(3)] + [mk(f"p{i}", None) for i in range(7)]
    tr, va = train_d3.split_by_background(scenes, 0.15, 0, holdout=["bagC"])
    bg = {s.key: train_d3.background_of(s) for s in scenes}
    assert not ({bg[k] for k in tr} & {bg[k] for k in va})         # no background on both sides
    assert {f"train_x/c{i}" for i in range(3)} <= set(va)            # forced hold-out
    assert len(tr) + len(va) == len(scenes) and len(va) >= 3
    assert train_d3.split_by_background(scenes, 0.15, 0, ["bagC"]) == (tr, va)   # seeded


REALNEG_BAG = "/home/gag/i.mos/dataset/for_hackathon/roundT_pressureGate_roundT"


@pytest.mark.skipif(not os.path.isdir(REALNEG_BAG), reason="real bag not available")
def test_real_negative_scene_is_all_background(tmp_path):
    sc = d3_scenes.real_negative_scenes(f"{REALNEG_BAG}:2:5")[0]
    assert sc.key == "real_negatives/roundT_pressureGate_roundT_2_5" and sc.missing() == []
    assert train_d3.background_of(sc) == "real:roundT_pressureGate_roundT"
    meta = d3_preprocess.preprocess_scene(sc, str(tmp_path), d3_preprocess.PreprocessOptions(aux="d1"))
    assert meta["n_frames"] == 3 and meta["pixels"]["obstacle"] == 0 and "organised=True" in meta["sensor_profile"]
    c = d3_data.SceneCache(d3_preprocess.cache_dir_for(str(tmp_path), sc))
    assert (c.target == 0).all() and (c.objid == -1).all() and (c.feats[:, 3] > 0).mean() > 0.5
    assert [o["frame"] for o in c.objects] == [2, 3, 4]
