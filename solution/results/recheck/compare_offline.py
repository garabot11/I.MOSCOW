#!/usr/bin/env python3
"""Frame-by-frame comparison of two offline result directories (JSONL per bag).

usage: compare_offline.py <reference_dir> <new_dir> [--out report.json]
Compared per frame: frame_index, header_time_ns, status, distance_m, forward_m, reset and the whole
detector output except the timing fields.  Processing times naturally differ and are only summarised.
"""
import argparse
import glob
import json
import os


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def strip(det):
    return {k: v for k, v in det.items() if k != "timings_ms"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--out")
    a = ap.parse_args()
    report = {"reference": a.ref, "new": a.new, "bags": {}, "total_frames": 0, "total_differences": 0}
    for ref_path in sorted(glob.glob(os.path.join(a.ref, "*.jsonl"))):
        bag = os.path.basename(ref_path)[:-6]
        new_path = os.path.join(a.new, bag + ".jsonl")
        ref = load(ref_path)
        new = load(new_path) if os.path.exists(new_path) else []
        diffs = []
        if len(ref) != len(new):
            diffs.append({"what": "frame count", "ref": len(ref), "new": len(new)})
        for r, n in zip(ref, new):
            for key in ("frame_index", "header_time_ns", "status", "distance_m", "forward_m", "reset"):
                if r.get(key) != n.get(key):
                    diffs.append({"frame": r.get("frame_index"), "field": key, "ref": r.get(key), "new": n.get(key)})
            if strip(r["detector"]) != strip(n["detector"]):
                diffs.append({"frame": r.get("frame_index"), "field": "detector"})
        report["bags"][bag] = {"frames": len(ref), "differences": len(diffs), "first_differences": diffs[:10],
                               "processing_ms_median_ref": sorted(x["processing_ms"] for x in ref)[len(ref) // 2] if ref else None,
                               "processing_ms_median_new": sorted(x["processing_ms"] for x in new)[len(new) // 2] if new else None}
        report["total_frames"] += len(ref)
        report["total_differences"] += len(diffs)
    text = json.dumps(report, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "bags"}))
    for bag, r in report["bags"].items():
        print(f"{bag}: {r['frames']} frames, {r['differences']} differences")
    return 0 if report["total_differences"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
