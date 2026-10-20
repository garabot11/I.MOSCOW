#!/usr/bin/env python3
"""Compare offline results (JSONL per bag) with annotations/events.yaml and print / save metrics.

Usage: evaluate.py <results_dir> [--annotations annotations/events.yaml] [--out metrics.json]

Rules (fixed before tuning, see docs/experiments.md):
* frame-level: a frame is positive if it lies in an 'obstacle' interval, negative if in a 'clear'
  interval; 'uncertain' frames are excluded from TP/FP/FN/TN;
* detector 'unknown' on a positive frame counts as a miss (FN); on a negative frame it is neither
  TN nor FP but reported separately;
* an alarm event = a maximal run of consecutive processed frames with status 'detected'
  (gaps of one processed frame are bridged); an alarm on negative frames is one false-alarm event;
* an obstacle event is detected if at least one alarm frame falls inside its interval; the first
  detection distance/time is the first such frame.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml


def load_results(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows.sort(key=lambda r: r["frame_index"])
    return rows


def label_of(frame, events):
    lab = "unlabelled"
    for ev in events:
        a, b = ev["frame_range"]
        if a <= frame <= b:
            if ev["label"] == "uncertain":
                return "uncertain", ev["event_id"]
            lab = (ev["label"], ev["event_id"])
    return lab if isinstance(lab, tuple) else (lab, None)


def _minmax(values):
    vals = [v for v in values if v is not None]
    return [min(vals), max(vals)] if vals else None


def alarm_runs(rows):
    runs, cur = [], None
    for i, r in enumerate(rows):
        if r["status"] == "detected":
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        elif cur is not None and i - cur[1] > 1:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return runs


def evaluate_bag(rows, events):
    tp = fp = fn = tn = 0
    unk_pos = unk_neg = unc = 0
    for r in rows:
        lab, _ = label_of(r["frame_index"], events)
        st = r["status"]
        if lab == "obstacle":
            if st == "detected":
                tp += 1
            elif st == "unknown":
                fn += 1
                unk_pos += 1
            else:
                fn += 1
        elif lab == "clear":
            if st == "detected":
                fp += 1
            elif st == "unknown":
                unk_neg += 1
            else:
                tn += 1
        else:
            unc += 1
    runs = alarm_runs(rows)
    false_alarm_events, uncertain_alarm_events = [], []
    for a, b in runs:
        labs = {label_of(rows[i]["frame_index"], events)[0] for i in range(a, b + 1)}
        if labs <= {"clear"}:
            false_alarm_events.append({"frames": [rows[a]["frame_index"], rows[b]["frame_index"]],
                                       "time_s": [rows[a]["relative_time_s"], rows[b]["relative_time_s"]],
                                       "distance_m": _minmax([rows[i]["distance_m"] for i in range(a, b + 1)]),
                                       "n_frames": b - a + 1})
        elif "uncertain" in labs and "obstacle" not in labs:
            uncertain_alarm_events.append({"frames": [rows[a]["frame_index"], rows[b]["frame_index"]], "n_frames": b - a + 1})
    obstacle_events = []
    for ev in events:
        if ev["label"] != "obstacle":
            continue
        a, b = ev["frame_range"]
        inside = [r for r in rows if a <= r["frame_index"] <= b]
        hits = [r for r in inside if r["status"] == "detected"]
        first = hits[0] if hits else None
        obstacle_events.append({
            "event_id": ev["event_id"], "frames_processed": len(inside), "frames_detected": len(hits),
            "detected": bool(hits),
            "first_detection": None if first is None else {"frame": first["frame_index"], "time_s": first["relative_time_s"], "distance_m": first["distance_m"]},
            "distance_reported_m": _minmax([h["distance_m"] for h in hits]),
            "distance_reference_m": ev.get("distance_reference_m"),
        })
    proc = np.array([r["processing_ms"] for r in rows])
    return {
        "frames": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "uncertain_frames": unc,
        "unknown_on_positive": unk_pos, "unknown_on_negative": unk_neg,
        "false_alarm_events": false_alarm_events, "uncertain_alarm_events": uncertain_alarm_events,
        "obstacle_events": obstacle_events,
        "processing_ms": {"median": round(float(np.median(proc)), 1), "p95": round(float(np.percentile(proc, 95)), 1),
                          "p99": round(float(np.percentile(proc, 99)), 1), "max": round(float(proc.max()), 1)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--annotations", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotations", "events.yaml"))
    ap.add_argument("--out")
    args = ap.parse_args()
    ann = yaml.safe_load(open(args.annotations, encoding="utf-8"))
    report = {"results_dir": args.results_dir, "bags": {}}
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "false_alarm_events": 0, "uncertain_frames": 0, "frames": 0}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        if name not in ann["bags"]:
            continue
        rows = load_results(path)
        m = evaluate_bag(rows, ann["bags"][name]["events"])
        report["bags"][name] = m
        for k in ("tp", "fp", "fn", "tn", "uncertain_frames", "frames"):
            tot[k] += m[k]
        tot["false_alarm_events"] += len(m["false_alarm_events"])
        print(f"== {name}: frames={m['frames']} TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} uncertain={m['uncertain_frames']} "
              f"unknown(pos/neg)={m['unknown_on_positive']}/{m['unknown_on_negative']} false-alarm events={len(m['false_alarm_events'])} "
              f"proc ms median/p95/max={m['processing_ms']['median']}/{m['processing_ms']['p95']}/{m['processing_ms']['max']}")
        for ev in m["obstacle_events"]:
            print(f"   obstacle {ev['event_id']}: detected={ev['detected']} frames {ev['frames_detected']}/{ev['frames_processed']} first={ev['first_detection']} reported dist={ev['distance_reported_m']} ref={ev['distance_reference_m']}")
        for fa in m["false_alarm_events"]:
            print(f"   FALSE ALARM frames {fa['frames']} t={fa['time_s']} dist={fa['distance_m']} ({fa['n_frames']} frames)")
        for ua in m["uncertain_alarm_events"]:
            print(f"   alarm on uncertain frames {ua['frames']} ({ua['n_frames']} frames) - not counted")
    prec = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else None
    rec = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    report["totals"] = dict(tot, precision=prec, recall=rec, f1=f1)
    print(f"TOTAL frames={tot['frames']} TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} TN={tot['tn']} precision={prec} recall={rec} f1={f1} false-alarm events={tot['false_alarm_events']}")
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
