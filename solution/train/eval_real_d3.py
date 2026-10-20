"""Д3 on REAL bags (no synthetic labels): detections of the annotated real object and candidates on
real tunnels, through the node's frame path (build_frame -> Д1 step_frame -> build_range_image ->
NNDetector.step).

    python solution/train/eval_real_d3.py --model solution/models/d3_unet.pt \\
        --bag /home/gag/i.mos/dataset/for_hackathon/doubleT_obstacle [--frames 0:201] [--json out.json]

Per frame: Д1 status, Д3 obstacle candidates (distance, n_support, score, lateral offset / height
above rail from Д1's track model) and whether they lie in the clearance gauge (|l - l_c| <= 1.4 m,
0.1 m <= h_above_rail <= 3.4 m; unknown when Д1 has no track model).  For bags listed in
``solution/annotations/events.yaml`` the frames labelled ``obstacle`` are scored: detected when an
in-gauge Д3 candidate lies within ``--match-m`` of the reference distance range; frames labelled
``uncertain`` are excluded; candidates in all other frames count as false positives.

Huge sqlite bags (e.g. the Falcon recording): ``--sqlite-topic /iv_points --frames 3800:4400`` fetches the
messages by row id instead of reading the bag from the start.  A moving reference object without an
annotation file: ``--ref-track 68,3840,0.092`` = expected distance 68 - (frame - 3840) * 0.092 m; frames
with an in-gauge candidate (n_support >= ``--min-n``) within ``--ref-tol`` m of it are counted as hits,
other in-gauge candidates as false.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import _paths
import d3_preprocess
import d3_scenes

GAUGE_HALF_M = 1.4
GAUGE_H = (0.1, 3.4)


def in_gauge(c) -> object:
    if c.lateral_offset is None or c.h_above_rail is None:
        return None
    return abs(c.lateral_offset) <= GAUGE_HALF_M and GAUGE_H[0] <= c.h_above_rail <= GAUGE_H[1]


def load_events(bag_name: str):
    import yaml

    path = os.path.join(_paths.SOLUTION_DIR, "annotations", "events.yaml")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return ((doc.get("bags") or {}).get(bag_name) or {}).get("events") or []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--frames", default="", help="a:b frame range (default: all)")
    ap.add_argument("--match-m", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sqlite-topic", help="fetch messages of this topic by id (sqlite3 bags; needs --frames)")
    ap.add_argument("--min-n", type=int, default=1, help="count only candidates with at least this n_support")
    ap.add_argument("--ref-track", help="d0,f0,slope: reference distance d0 - (frame - f0) * slope (m)")
    ap.add_argument("--ref-tol", type=float, default=3.0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    import torch

    from metro_obstacle_detector import det_nn

    class DM:  # minimal interface-C object for an offline evaluation
        def __init__(self):
            gpu = args.device in ("auto", "gpu") and torch.cuda.is_available()
            self.device = torch.device("cuda:0" if gpu else "cpu")
            self.mode = "gpu" if gpu else "cpu_forced"

        def report_failure(self, exc):
            self.mode = "cpu_emergency"
            self.device = torch.device("cpu")

    det = det_nn.NNDetector(DM(), args.model, threshold=args.threshold,
                            range_image_params={"az_step_deg": 0.1, "binned_res_deg": (0.09, 0.08), "az_limit_deg": None})
    det.warmup()
    proc = d3_preprocess.FrameProcessor(d3_preprocess.PreprocessOptions(aux="d1"))
    lo, hi = 0, None
    if args.frames:
        a, b = args.frames.split(":")
        lo, hi = int(a or 0), (int(b) if b else None)
    name = os.path.basename(os.path.normpath(args.bag))
    events = load_events(name)
    label_of = {}
    for ev in events:
        f0, f1 = ev["frame_range"]
        for f in range(int(f0), int(f1) + 1):
            label_of[f] = ev
    rows = []
    t0 = time.time()

    def messages():
        if args.sqlite_topic:
            if hi is None:
                raise SystemExit("--sqlite-topic needs --frames a:b")
            for k in range(lo, hi):
                stamp_k, msg_k = d3_scenes.sqlite_message(args.bag, args.sqlite_topic, k)
                yield k, stamp_k, msg_k
            return
        for k, stamp_k, msg_k in d3_scenes.iter_messages(args.bag):
            if k < lo:
                continue
            if hi is not None and k >= hi:
                break
            yield k, stamp_k, msg_k

    for i, stamp, msg in messages():
        frame = proc.frame(msg)
        ri = proc.range_image(frame)
        aux = proc.aux(frame, ri)
        cands = [c for c in det.step(ri, aux) if c.cls == "obstacle" and c.n_support >= args.min_n]
        rows.append({"frame": i, "stamp_ns": stamp, "d1": proc.last_status,
                     "label": (label_of.get(i) or {}).get("label", "clear" if events else None),
                     "ms": det.last_timings_ms.get("total"),
                     "cands": [{"distance_m": round(c.distance_m, 2), "n": c.n_support, "score": round(c.score, 3),
                                "lat_off": None if c.lateral_offset is None else round(c.lateral_offset, 2),
                                "h_rail": None if c.h_above_rail is None else round(c.h_above_rail, 2),
                                "in_gauge": in_gauge(c)} for c in cands]})
    n = len(rows)
    in_g = [[c for c in r["cands"] if c["in_gauge"]] for r in rows]
    out = {"bag": name, "model": args.model, "frames": n, "seconds": round(time.time() - t0, 1),
           "device": det.device_name, "threshold": det.threshold, "input_warnings": det.input_warnings,
           "frames_with_any_candidate": sum(1 for r in rows if r["cands"]),
           "frames_with_in_gauge_candidate": sum(1 for g in in_g if g),
           "in_gauge_candidates_per_frame": round(sum(len(g) for g in in_g) / n, 3) if n else None,
           "d1_unknown_frames": sum(1 for r in rows if r["d1"] == "unknown"),
           "d3_ms_p50_p95": [round(float(np.percentile([r["ms"] for r in rows], q)), 2) for q in (50, 95)] if n else None}
    out["min_n"] = args.min_n
    profs = list(proc.pipeline.frames.profiles.values())
    out["sensor_profile"] = profs[0].summary() if profs else None
    if args.ref_track:
        d0, f0, slope = (float(v) for v in args.ref_track.split(","))
        hits = false_frames = 0
        errs = []
        for r, g in zip(rows, in_g):
            ref = d0 - (r["frame"] - f0) * slope
            near = [c for c in g if abs(c["distance_m"] - ref) <= args.ref_tol]
            if near:
                hits += 1
                errs.append(min(c["distance_m"] for c in near) - ref)
            false_frames += int(any(abs(c["distance_m"] - ref) > args.ref_tol for c in g))
        out["ref_track"] = {"formula": f"{d0} - (frame - {f0}) * {slope}", "tol_m": args.ref_tol,
                            "frames_with_candidate_near_ref": hits, "frames": n,
                            "frames_with_other_in_gauge_candidate": false_frames,
                            "distance_error_m_p50": round(float(np.median(errs)), 2) if errs else None,
                            "ref_distance_range_m": [round(d0 - (rows[0]["frame"] - f0) * slope, 1),
                                                     round(d0 - (rows[-1]["frame"] - f0) * slope, 1)] if rows else None}
    if events:
        tp = fn = fp_frames = 0
        dists = []
        for r, g in zip(rows, in_g):
            ev = label_of.get(r["frame"])
            if ev is not None and ev["label"] == "uncertain":
                continue
            if ev is not None and ev["label"] == "obstacle":
                d0, d1_ = ev["distance_reference_m"]
                hit = [c for c in g if d0 - args.match_m <= c["distance_m"] <= d1_ + args.match_m]
                if hit:
                    tp += 1
                    dists.append(min(c["distance_m"] for c in hit))
                else:
                    fn += 1
                others = [c for c in g if c not in hit]
                fp_frames += int(bool(others))
            else:
                fp_frames += int(bool(g))
        counted = sum(1 for r in rows if (label_of.get(r["frame"]) or {}).get("label") != "uncertain")
        out["annotated"] = {"obstacle_frames_detected": tp, "obstacle_frames_missed": fn,
                            "frames_counted": counted, "frames_with_false_in_gauge_candidate": fp_frames,
                            "detected_distance_m_min_max": [min(dists), max(dists)] if dists else None}
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dict(out, rows=rows), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
