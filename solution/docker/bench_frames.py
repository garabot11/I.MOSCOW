"""Per-frame timing of the detection pipeline on a slice of a recording (WP-S A/B measurements).

Runs inside the image (``metro-lidar:gpu-dev``), e.g.

    docker run --rm --name wps-bench --cpuset-cpus 0-7 --gpus all -v <dataset>:/data:ro \
        -v <this dir>:/bench:ro [-v <src>:<site-packages>/metro_obstacle_detector:ro] metro-lidar:gpu-dev \
        python3 /bench/bench_frames.py --bag /data/extra/cloud_with_fake_obj --start 650 --n 120 --device gpu

The frames start..start+n are read into memory first (bag I/O is not timed), then every frame goes
through ``build_frame`` + ``step_frame`` like in the offline runner; the first ``--warm`` frames are not
counted.  Prints one JSON line: median / p95 / max of the total and the median of every
``timings_ms`` part.  ``--cprofile OUT`` writes a cProfile of the counted frames.
"""
import argparse
import json
import sys
import time

import numpy as np

from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.offline import _device_manager, iter_pointclouds
from metro_obstacle_detector.pipeline import DetectionPipeline


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--cprofile")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    msgs = []
    for i, _name, msg, _t in iter_pointclouds(a.bag):
        if i < a.start:
            continue
        if i >= a.start + a.n:
            break
        msg.data = bytes(msg.data)
        msgs.append(msg)
    cfg = DetectorConfig()
    pipe = DetectionPipeline(cfg, device_manager=_device_manager(cfg, a.device))
    print(pipe.nn_note, file=sys.stderr)
    prof = None
    if a.cprofile:
        import cProfile
        prof = cProfile.Profile()
    tot, parts = [], {}
    for k, msg in enumerate(msgs):
        if prof is not None and k == a.warm:
            prof.enable()
        t0 = time.perf_counter()
        fr = pipe.build_frame(msg)
        t1 = time.perf_counter()
        res, _dec = pipe.step_frame(fr)
        t2 = time.perf_counter()
        if k < a.warm:
            continue
        tot.append((t2 - t0) * 1e3)
        parts.setdefault("build_frame", []).append((t1 - t0) * 1e3)
        for key, v in res.timings_ms.items():
            parts.setdefault(key, []).append(v)
    if prof is not None:
        prof.disable()
        prof.dump_stats(a.cprofile)
    tot = np.array(tot) if tot else np.zeros(1)
    out = {"label": a.label, "bag": a.bag, "start": a.start, "n": len(tot), "device": a.device,
           "median": round(float(np.median(tot)), 1), "p95": round(float(np.percentile(tot, 95)), 1),
           "max": round(float(tot.max()), 1),
           "over_100ms": int((tot > 100.0).sum()),          # frames a 10 Hz stream would replace (freshest frame)
           "d3_frames": len(parts.get("d3", [])),            # frames on which D3 ran (budget / CPU stride)
           "parts_median": {k: round(float(np.median(v)), 1) for k, v in parts.items()},
           "parts_p95": {k: round(float(np.percentile(v, 95)), 1) for k, v in parts.items()}}
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
