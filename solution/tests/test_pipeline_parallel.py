"""WP-S execution: D1 beside the range image and D3 beside D2 (worker thread), GPU binning with a hot
fallback - the frame results must be identical to the sequential execution."""
import numpy as np
import pytest

from metro_obstacle_detector import range_image as RI
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.pipeline import DetectionPipeline

from synthetic import make_pointcloud2, tunnel_cloud

T0 = 946_687_297_199_933_052


def _result(res, dec):
    d = res.to_dict()
    out = {k: v for k, v in d.items() if "timing" not in k and not k.endswith("_ms")}
    return out, (dec.status, dec.distance_m, dec.forward_m, dec.reset)


_CLOUDS = {}


def _cloud(k):
    if k not in _CLOUDS:
        _CLOUDS[k] = tunnel_cloud(obstacle=(60.0 - 3.0 * k, 0.1, 0.6, 1.2), rng=np.random.default_rng(100 + k),
                                  n_ring=160, n_az=400)
    return _CLOUDS[k]


def xyzi16_message(xyz, stamp_ns):
    """Like the organisers' edited bag: x, y, z, intensity float32 only (point_step 16), height 1."""
    from sensor_msgs.msg import PointCloud2, PointField
    arr = np.zeros((len(xyz), 4), np.float32)
    arr[:, :3] = xyz
    arr[:, 3] = 7.0
    msg = PointCloud2()
    msg.header.frame_id = "hesai_lidar"
    msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000
    for off, name in ((0, "x"), (4, "y"), (8, "z"), (12, "intensity")):
        pf = PointField()
        pf.name, pf.offset, pf.datatype, pf.count = name, off, PointField.FLOAT32, 1
        msg.fields.append(pf)
    msg.height, msg.width, msg.point_step, msg.row_step = 1, len(xyz), 16, 16 * len(xyz)
    msg.is_bigendian, msg.is_dense = False, False
    msg.data = arr.tobytes()
    return msg


def _sequence(unorganised: bool, n=8):
    """A train approaching a box, then standing (static scene: frozen model path), as PointCloud2 messages."""
    msgs = []
    for k in range(n):
        xyz = _cloud(min(k, n // 2))                     # moves, then stands still
        if unorganised:
            xyz = xyz[(xyz != 0).any(axis=1)]
            xyz = xyz[np.random.default_rng(k).permutation(len(xyz))][: len(xyz) - 7 * k]   # count varies per frame
        msg = make_pointcloud2(xyz, drop=("ring",) if unorganised else None)
        st = T0 + k * 100_000_000
        msg.header.stamp.sec, msg.header.stamp.nanosec = st // 1_000_000_000, st % 1_000_000_000
        msgs.append(msg)
    return msgs


def _run(msgs, parallel, dm=None, cfg=None):
    cfg = cfg or DetectorConfig(d3_enabled=False)
    p = DetectionPipeline(cfg, device_manager=dm)
    p.parallel = parallel
    out = []
    for m in msgs:
        fr = p.build_frame(m)
        out.append(_result(*p.step_frame(fr)))
    p.close()
    return out


@pytest.mark.parametrize("unorganised", [False, True])
def test_parallel_execution_gives_identical_results(unorganised):
    msgs = _sequence(unorganised)
    seq = _run(msgs, parallel=False)
    par = _run(msgs, parallel=True)
    assert seq == par
    assert any(r[1][0] != "unknown" for r in seq)


class _FakeDM:
    """Device manager stand-in: GPU mode with a device whose binning raises (hot fallback test)."""

    def __init__(self):
        self.mode = "gpu"
        self.failures = []

        class _Dev:
            type = "cuda"
        self.device = _Dev()

    def report_failure(self, exc):
        self.failures.append(exc)
        self.mode = "cpu_emergency"


def test_gpu_binning_failure_switches_to_the_cpu_with_identical_results(monkeypatch):
    msgs = _sequence(True, n=4)
    ref = _run(msgs, parallel=False)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(RI, "_plan_torch", boom)
    dm = _FakeDM()
    got = _run(msgs, parallel=True, dm=dm)
    assert got == ref
    assert len(calls) == 1 and len(dm.failures) == 1 and dm.mode == "cpu_emergency"


def test_edited_style_stream_with_a_307200_point_frame():
    """The organisers' edited bag: no ring field, point_step 16, the point count varies and some edited frames have
    exactly 307 200 points (= the organised count).  Every frame is binned (never treated as organised because of
    its count), consecutive frames switch between counts, and parallel == sequential."""
    base = _cloud(0)
    base = base[(base != 0).any(axis=1)]
    counts = [307_187, 307_200, 306_950, 307_200, 307_174]
    msgs = []
    for k, n in enumerate(counts):
        xyz = np.resize(base[np.random.default_rng(k).permutation(len(base))], (n, 3))   # duplicates = range ties
        msgs.append(xyzi16_message(xyz, T0 + k * 100_000_000))
    p = DetectionPipeline(DetectorConfig(d3_enabled=False))
    for m in msgs:
        fr = p.build_frame(m)
        assert not fr.profile.organized and fr.n == m.width
        ri = p._range_image(fr)
        assert not ri.organized
    p.close()
    seq = _run(msgs, parallel=False)
    assert seq == _run(msgs, parallel=True)
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if cuda:                                    # GPU binning in the pipeline: same results, no failure reported
        dm = _FakeDM()
        dm.device = torch.device("cuda:0")
        before = RI.BIN_STATS["gpu"]
        assert _run(msgs, parallel=True, dm=dm) == seq
        assert not dm.failures and RI.BIN_STATS["gpu"] > before


@pytest.mark.parametrize("parallel", ["1", "0"])
def test_d3_warm_up_runs_in_the_thread_that_runs_d3(monkeypatch, tmp_path, parallel):
    """The first convolutions cost seconds in every thread that runs them: with the worker thread on, a warm-up in
    the calling thread left 4 s on the first frame (launch check of v2), so D3 is warmed up where it runs."""
    import threading

    from metro_obstacle_detector import det_nn, pipeline as PL

    seen = []

    class _StubNN:
        def __init__(self, dm, model, logger=None, range_image_params=None):
            pass

        def warmup(self):
            seen.append(threading.current_thread().name)
            return "gpu"

    model = tmp_path / "d3.pt"
    model.write_bytes(b"")
    monkeypatch.setenv("METRO_PARALLEL", parallel)
    monkeypatch.setattr(PL, "_find_d3_model", lambda cfg: str(model))
    monkeypatch.setattr(det_nn, "NNDetector", _StubNN)
    p = DetectionPipeline(DetectorConfig(d3_enabled=True), device_manager=_FakeDM())
    try:
        assert len(seen) == 1 and p.nn_note.startswith("D3 on")
        assert seen[0].startswith("metro-det") == (parallel == "1")
    finally:
        p.close()
