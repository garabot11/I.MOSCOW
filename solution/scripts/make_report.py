#!/usr/bin/env python3
"""Markdown tables for docs/experiments.md from evaluate.py outputs.

usage: make_report.py <results_dir> [<results_dir> ...]   (each dir is evaluated on the fly)
"""
import os
import subprocess
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(d):
    out = os.path.join(d, "metrics.json")
    subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), d, "--out", out], check=True, capture_output=True)
    return json.load(open(out))


def main():
    dirs = sys.argv[1:]
    print("| variant | frames | TP | FP | FN | TN | precision | recall | false-alarm events | obstacle detected (first frame / distance) | proc ms median (max of bags) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for d in dirs:
        m = evaluate(d)
        t = m["totals"]
        ob = m["bags"].get("doubleT_obstacle", {}).get("obstacle_events", [])
        first = ", ".join(f"{e['event_id']}: {'yes' if e['detected'] else 'NO'} ({e['first_detection']['frame'] if e['first_detection'] else '-'} / {e['first_detection']['distance_m'] if e['first_detection'] else '-'} m)" for e in ob)
        fa = sum(len(b["false_alarm_events"]) for b in m["bags"].values())
        pm = max(b["processing_ms"]["median"] for b in m["bags"].values())
        prec = "n/a" if t["precision"] is None else f"{t['precision']:.3f}"
        rec = "n/a" if t["recall"] is None else f"{t['recall']:.3f}"
        print(f"| {os.path.basename(d.rstrip('/'))} | {t['frames']} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} | {prec} | {rec} | {fa} | {first} | {pm:.0f} |")
    print()
    for d in dirs:
        m = json.load(open(os.path.join(d, "metrics.json")))
        print(f"### {os.path.basename(d.rstrip('/'))}")
        print("| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |")
        print("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for name, b in m["bags"].items():
            fas = "; ".join(f"{fa['frames']} t={fa['time_s'][0]:.1f}-{fa['time_s'][1]:.1f}s {fa['distance_m']}" for fa in b["false_alarm_events"]) or "-"
            p = b["processing_ms"]
            print(f"| {name} | {b['frames']} | {b['tp']} | {b['fp']} | {b['fn']} | {b['tn']} | {b['uncertain_frames']} | {fas} | {p['median']} / {p['p95']} / {p['max']} |")
        print()


if __name__ == "__main__":
    main()
