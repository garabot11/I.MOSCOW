"""Д3 latency benchmark (batch 1, input (21, 64, 256)) through the real NNDetector code paths.

    python solution/train/bench_latency.py [--model solution/models/d3_unet.pt] [--bag DIR] [--n 300] [--json out]

Timed per call, from the NumPy input to NumPy class probabilities (host->device copy via a pinned
buffer, forward, softmax, device->host copy, synchronised):
  gpu_fp32, gpu_fp16 (model.half()), cpu_8_threads, cpu_4_threads (torch.set_num_threads).
With ``--bag`` the end-to-end ``NNDetector.step`` (range image from range_image.build_range_image +
Д1 aux -> input -> forward -> candidates) is timed on real frames too; building the frame, Д1 and the
range image is NOT included (main session's modules, shared with Д1/Д2).  Without ``--model`` a randomly initialised RangeUNet is used: latency does not
depend on the weights, but the candidate post-processing time does (random weights produce
arbitrary components), so ``post`` is only indicative then.
To approximate the judges' 8-core i7-9700E on this 6P+4E laptop run it under
``taskset -c 0,2,4,6,8,10,12,13`` (8 distinct physical cores); the stand's clocks are lower, so treat
the result as an optimistic estimate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "metro_obstacle_detector"))


class _DM:
    """Minimal interface-C object for the benchmark (the real DeviceManager lives in device.py)."""

    def __init__(self, torch, gpu: bool):
        self.device = torch.device("cuda:0" if gpu else "cpu")
        self.mode = "gpu" if gpu else "cpu_forced"
        self.failures = []

    def report_failure(self, exc):
        self.failures.append(repr(exc))
        self.mode = "cpu_emergency"


def stats(ts):
    a = np.asarray(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "mean_ms": round(float(a.mean()), 3), "max_ms": round(float(a.max()), 3), "n": int(len(a))}


def timeit(fn, n, warm):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return ts


def read_range_images(bag, k):
    """Range images of the first ``k`` clouds, built as in the node (build_frame + build_range_image),
    with Д1's rail/centre line of the same frame as aux."""
    from rosbags.rosbag2 import Reader
    from metro_obstacle_detector import cloud_io, nn_input
    from metro_obstacle_detector.config import DetectorConfig
    from metro_obstacle_detector.pipeline import DetectionPipeline
    from metro_obstacle_detector.range_image import build_range_image

    pipe = DetectionPipeline(DetectorConfig())
    out = []
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        for _, _, raw in r.messages(connections=conns[:1]):
            frame = pipe.build_frame(cloud_io.parse_pointcloud2_cdr(bytes(raw)))
            res, _ = pipe.step_frame(frame)
            ri = build_range_image(frame)
            out.append((ri, nn_input.aux_from_track(ri, res.rail, res.curve), frame.n_msg))
            if len(out) >= k:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="TorchScript d3_unet.pt (default: random-init RangeUNet)")
    ap.add_argument("--bag", help="rosbag2 dir for the end-to-end step() timing on real frames")
    ap.add_argument("--falcon-bag", help="sqlite3 Falcon bag (/iv_points): end-to-end step() and infer() on "
                                         "binned range images, messages fetched by id")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", default="8,4")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    import torch
    from metro_obstacle_detector import det_nn, nn_input
    from metro_obstacle_detector.nn_model import RangeUNet, count_macs, count_parameters, export_d3

    res = {"torch": torch.__version__, "input_shape": [1, 21, 64, 256], "n": args.n, "warmup": args.warmup,
           "affinity": sorted(os.sched_getaffinity(0)), "cpu_count": os.cpu_count()}
    try:
        res["gpu_processes"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        res["gpu_util_before"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm",
                                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:
        res["gpu_processes"] = f"nvidia-smi unavailable: {exc}"
    tmp = None
    model_path = args.model
    if not model_path:
        torch.manual_seed(0)
        m = RangeUNet()
        res["model"] = "random-init RangeUNet"
        res["params"] = count_parameters(m)
        res["macs"] = count_macs(m)
        tmp = tempfile.mkdtemp(prefix="d3bench_")
        model_path = export_d3(m, os.path.join(tmp, "d3_unet.pt"), {"crop": nn_input.CropConfig().to_dict()})["torchscript"]
    else:
        res["model"] = model_path
    x = np.random.default_rng(0).random((21, 64, 256)).astype(np.float32)
    x[3] = (x[3] > 0.4)

    ref = None
    for threads in [int(t) for t in args.threads.split(",") if t]:
        torch.set_num_threads(threads)
        det = det_nn.NNDetector(_DM(torch, False), model_path, cpu_threads=threads)
        ts = timeit(lambda: det.predict_proba(x), args.n, args.warmup)
        res[f"cpu_{threads}_threads"] = stats(ts)
        if ref is None:
            ref = det.predict_proba(x)
        print(f"cpu {threads} threads: {res[f'cpu_{threads}_threads']}", flush=True)
    torch.set_num_threads(max(int(t) for t in args.threads.split(",")))
    if torch.cuda.is_available():
        res["gpu_name"] = torch.cuda.get_device_name(0)
        for fp16 in (False, True):
            dm = _DM(torch, True)
            det = det_nn.NNDetector(dm, model_path, gpu_fp16=fp16)
            det.warmup()  # the GPU copy is created lazily
            if not det.on_gpu:
                res[f"gpu_{'fp16' if fp16 else 'fp32'}"] = {"error": dm.failures}
                continue
            ts = timeit(lambda: det.predict_proba(x), args.n, args.warmup)
            key = f"gpu_{'fp16' if fp16 else 'fp32'}"
            res[key] = stats(ts)
            p = det.predict_proba(x)
            res[key]["max_abs_dprob_vs_cpu_fp32"] = float(np.abs(p - ref).max())
            # forward only (no host<->device copies), for reference
            xg = torch.from_numpy(x[None]).cuda()
            if fp16:
                xg = xg.half()

            def fwd():
                with torch.inference_mode():
                    det._gpu_model(xg)
                torch.cuda.synchronize()
            res[key]["forward_only"] = stats(timeit(fwd, args.n, args.warmup))
            print(f"{key}: {res[key]}", flush=True)
    else:
        res["gpu"] = "torch.cuda.is_available() is False"
    if args.bag:
        ris = read_range_images(args.bag, 6)
        for name, dm in (("gpu_fp16", _DM(torch, torch.cuda.is_available())), ("cpu_8_threads", _DM(torch, False))):
            det = det_nn.NNDetector(dm, model_path, cpu_threads=8)
            det.warmup()
            i = [0]
            parts = {"input": [], "forward": [], "post": [], "total": []}

            def run():
                ri, aux, _ = ris[i[0] % len(ris)]
                det.step(ri, aux)
                i[0] += 1
                for k in parts:
                    parts[k].append(det.last_timings_ms[k])
            timeit(run, min(args.n, 100), 10)
            res[f"e2e_{name}"] = {k: stats(v[-min(args.n, 100):]) for k, v in parts.items()}
            res[f"e2e_{name}"]["bag"] = os.path.basename(os.path.normpath(args.bag))
            res[f"e2e_{name}"]["points_per_frame"] = int(ris[0][2])
            res[f"e2e_{name}"]["range_image_shape"] = list(ris[0][0].rng.shape)
            print(f"e2e {name}: total {res[f'e2e_{name}']['total']}", flush=True)
    if args.falcon_bag:
        import _paths  # noqa: F401
        import d3_preprocess
        import d3_scenes

        proc = d3_preprocess.FrameProcessor(d3_preprocess.PreprocessOptions(aux="d1"))
        fr = []
        for k in range(6):
            _, msg = d3_scenes.sqlite_message(args.falcon_bag, "/iv_points", k)
            frame = proc.frame(msg)
            ri = proc.range_image(frame)
            fr.append((ri, proc.aux(frame, ri)))
        dm = _DM(torch, torch.cuda.is_available())
        for api in ("step", "infer"):
            det = det_nn.NNDetector(dm, model_path, cpu_threads=8)
            det.warmup()
            i = [0]
            parts = {"input": [], "forward": [], "post": [], "total": []}

            def run():
                k = i[0]
                if api == "step":
                    det.step(*fr[k % len(fr)])
                else:  # sliding window of the last three frames, as a stateless caller would pass them
                    win = [fr[j % len(fr)] for j in range(max(k - 2, 0), k + 1)]
                    det.infer([w[0] for w in win], [w[1] for w in win])
                i[0] += 1
                for key in parts:
                    parts[key].append(det.last_timings_ms[key])
            timeit(run, min(args.n, 100), 10)
            key = f"e2e_falcon_{api}_{det.device_name}"
            res[key] = {k2: stats(v[-min(args.n, 100):]) for k2, v in parts.items()}
            res[key]["range_image_shape"] = list(fr[0][0].rng.shape)
            print(f"{key}: total {res[key]['total']}, input {res[key]['input']}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
