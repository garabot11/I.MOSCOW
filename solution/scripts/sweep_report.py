#!/usr/bin/env python3
"""Report of the distance / object-count sweep (``synth/configs/val_sweep.yaml``).

    sweep_report.py --set /path/synthetic/val_sweep --results work/eval/<tag>/synth/val_sweep \
        --config solution/synth/configs/val_sweep.yaml [--out report.md]

Every labelled obstacle in every frame (inside the clearance, top >= 0.3 m above the rail) gets one outcome:

* ``detected``     - a confirmed candidate of the frame matches it (the scoreboard's match: 2 m + 2 % in range,
                     1.2 m laterally);
* ``beyond``       - not detected and the object lies beyond the frame's ``verified_to_m`` (the corridor is not built
                     there, nothing is reported by design);
* ``seen``         - not detected, but D1 (accepted or rejected cluster) or D2 (obstacle group) produced a candidate at
                     the object (the most frequent D1 rejection reason is listed);
* ``few points``   - the renderer gave the object fewer than 3 returns (below every detector's minimum);
* ``missed``       - points exist, no candidate at the object.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

MIN_TOP = 0.3


def near(d_c, l_c, obj):
    tol = 2.0 + 0.02 * obj["distance_m"]
    if d_c is None or abs(d_c - obj["distance_m"]) > tol:
        return False
    return l_c is None or abs(l_c - obj["lateral_m"]) <= 1.2


def tags_from_config(path):
    tags = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"#\s+(\d+)\s+(\S+)\s*$", line)
            if m:
                tags[int(m.group(1))] = m.group(2)
    return tags


def outcome(obj, det, fr):
    if any(c.get("confirmed") and near(c["distance_m"], c.get("lateral_offset_from_track"), obj)
           for c in det.get("candidates") or []):
        return "detected", None
    v = det.get("verified_to_m")
    if v is not None and obj["distance_m"] > float(v) + 2.0:
        return "beyond", None
    reason = None
    seen = False
    for c in det.get("candidates") or []:
        if near(c["distance_m"], c.get("lateral_offset_from_track"), obj):
            seen, reason = True, "D1 accepted, not confirmed"
    for c in det.get("rejected") or []:
        if near(c["distance_m"], c.get("lateral_offset_from_track"), obj):
            seen = True
            reason = reason or ("D1 rejected: " + str(c.get("reason", ""))[:60])
    for g in det.get("d2") or []:
        if g.get("cls", "obstacle") == "obstacle" and near(g.get("distance_m"), g.get("lateral_offset"), obj):
            seen = True
            reason = reason or "D2 only"
    if seen:
        return "seen", reason
    if obj.get("n_hits", 0) < 3:
        return "few points", None
    return "missed", None


def scene_report(set_dir, res_dir, scene):
    labels = [json.loads(x) for x in open(os.path.join(set_dir, scene, "labels.jsonl"), encoding="utf-8") if x.strip()]
    path = os.path.join(res_dir, scene + ".jsonl")
    if not os.path.exists(path):
        return None
    rows = {}
    for x in open(path, encoding="utf-8"):
        if x.strip():
            r = json.loads(x)
            rows[r["frame_index"]] = r
    per_obj = defaultdict(lambda: {"cls": None, "out": Counter(), "reasons": Counter(), "first": None, "d": [],
                                   "hits": [], "first_seen": None})
    fp = 0
    beside = 0
    ms = []
    verified = []
    for lab in labels:
        r = rows.get(lab["frame"])
        if r is None:
            continue
        det = r.get("detector") or {}
        ms.append(float(r.get("processing_ms") or 0.0))
        if det.get("verified_to_m") is not None:
            verified.append(float(det["verified_to_m"]))
        conf = [c for c in det.get("candidates") or [] if c.get("confirmed")]
        if r["status"] == "detected" and conf:
            if not any(near(c["distance_m"], c.get("lateral_offset_from_track"), o) for c in conf for o in lab["objects"]):
                fp += 1
            elif not any(near(c["distance_m"], c.get("lateral_offset_from_track"), o) for c in conf
                         for o in lab["objects"] if o.get("in_clearance")):
                beside += 1
        for o in lab["objects"]:
            if not (o.get("is_obstacle") and o.get("in_clearance") and o.get("top_above_rail_m", 0.0) >= MIN_TOP):
                continue
            po = per_obj[o["id"]]
            po["cls"] = o["class"]
            out, why = outcome(o, det, r)
            po["out"][out] += 1
            if why:
                po["reasons"][why] += 1
            po["d"].append(o["distance_m"])
            po["hits"].append(o.get("n_hits", 0))
            if out == "detected" and po["first"] is None:
                po["first"] = (lab["frame"], o["distance_m"])
            if out in ("detected", "seen") and po["first_seen"] is None:
                po["first_seen"] = (lab["frame"], o["distance_m"])
    ms.sort()
    q = (lambda p: ms[min(len(ms) - 1, int(p * len(ms)))] if ms else float("nan"))
    verified.sort()
    return {"objects": per_obj, "fp_frames": fp, "beside_frames": beside, "frames": len(ms),
            "ms": (q(0.5), q(0.95), ms[-1] if ms else float("nan")), "over_100": sum(1 for x in ms if x > 100.0),
            "verified_med": verified[len(verified) // 2] if verified else None}


def fmt_out(c: Counter, n: int) -> str:
    keys = ["detected", "beyond", "seen", "few points", "missed"]
    return " / ".join(str(c.get(k, 0)) for k in keys) + f" of {n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    tags = tags_from_config(a.config)
    scenes = sorted(d for d in os.listdir(a.set) if os.path.isdir(os.path.join(a.set, d)))
    rep = {}
    for sc in scenes:
        idx = int(re.search(r"_(\d{3})_", sc).group(1))
        r = scene_report(a.set, a.results, sc)
        if r is not None:
            rep[tags.get(idx, sc)] = r
    L = ["# Sweep report (val_sweep)", "",
         "Outcome per object-frame: detected / beyond verified range / seen by D1 or D2, not confirmed / "
         "< 3 rendered points / missed (points, no candidate).", ""]
    # A) distance sweep
    L += ["## A. One object at a fixed distance, standing train", "",
          "| class | " + " | ".join(f"{d} m" for d in (5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200, 230, 270, 300)) + " |",
          "|---|" + "---:|" * 14]
    cells = defaultdict(dict)
    detail = []
    for tag, r in rep.items():
        m = re.match(r"dist_(\w+?)_(\d+)$", tag)
        if not m:
            continue
        cls, d = m.group(1), int(m.group(2))
        objs = list(r["objects"].values())
        if not objs:
            cells[cls][d] = "n/p"     # not a positive (outside the clearance / below the design floor)
            continue
        o = objs[0]
        n = sum(o["out"].values())
        det = o["out"].get("detected", 0)
        top = max((k for k in o["out"] if k != "detected"), key=lambda k: o["out"][k], default=None)
        hits = sorted(o["hits"])[len(o["hits"]) // 2]
        cells[cls][d] = f"{det}/{n}" + ("" if det == n or top is None else f" ({top})") + f" [{hits} pts]"
        detail.append((cls, d, fmt_out(o["out"], n), dict(o["reasons"].most_common(2)), r["verified_med"],
                       r["fp_frames"], r["ms"]))
    for cls in ("person_standing", "box", "cube", "wall"):
        L.append(f"| {cls} | " + " | ".join(cells[cls].get(d, "-") for d in (5, 10, 20, 40, 60, 80, 100, 125, 150, 175,
                                                                               200, 230, 270, 300)) + " |")
    L += ["", "Cell: detected frames / object-frames (most frequent other outcome) [median rendered points].", "",
          "| class | d, m | detected / beyond / seen / few / missed | reasons | verified med, m | FP frames | ms p50/p95/max |",
          "|---|---:|---|---|---:|---:|---|"]
    for cls, d, o, why, v, fp, ms in sorted(detail):
        L.append(f"| {cls} | {d} | {o} | {why or ''} | {v} | {fp} | {ms[0]:.0f} / {ms[1]:.0f} / {ms[2]:.0f} |")
    # B) approach
    L += ["", "## B. Approach from 300 m, moving train", "",
          "| scene | class | range covered, m | first seen (D1/D2), m | first confirmed, m | detected / beyond / seen / few / missed | FP frames |",
          "|---|---|---|---:|---:|---|---:|"]
    for tag, r in rep.items():
        if not tag.startswith("appr_"):
            continue
        for oid, o in r["objects"].items():
            n = sum(o["out"].values())
            fs = f"{o['first_seen'][1]:.1f}" if o["first_seen"] else "-"
            fc = f"{o['first'][1]:.1f}" if o["first"] else "-"
            L.append(f"| {tag} | {o['cls']} | {max(o['d']):.0f} -> {min(o['d']):.0f} | {fs} | {fc} | {fmt_out(o['out'], n)} | {r['fp_frames']} |")
    # C) count
    L += ["", "## C. Number of objects per frame", "",
          "| scene | frames | positives (object-frames) | detected | recall | FP frames | beside-track frames | ms p50 / p95 / max | frames > 100 ms |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for tag, r in rep.items():
        if not tag.startswith("count_"):
            continue
        pos = sum(sum(o["out"].values()) for o in r["objects"].values())
        det = sum(o["out"].get("detected", 0) for o in r["objects"].values())
        L.append(f"| {tag} | {r['frames']} | {pos} | {det} | {det / pos if pos else 0:.2f} | {r['fp_frames']} | "
                 f"{r['beside_frames']} | {r['ms'][0]:.0f} / {r['ms'][1]:.0f} / {r['ms'][2]:.0f} | {r['over_100']} |")
    L += ["", "Per object of the count scenes (detected / beyond / seen / few / missed):", ""]
    for tag, r in rep.items():
        if not tag.startswith("count_") or tag.endswith(("_1", "_2")):
            continue
        parts = []
        for oid, o in sorted(r["objects"].items()):
            parts.append(f"{o['cls']}@{min(o['d']):.0f}-{max(o['d']):.0f}m {o['out'].get('detected', 0)}/{sum(o['out'].values())}"
                         + (f" ({o['reasons'].most_common(1)[0][0]})" if o['reasons'] and o['out'].get('detected', 0) < sum(o['out'].values()) else ""))
        L.append(f"- {tag}: " + "; ".join(parts))
    text = "\n".join(L) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({t: {"fp_frames": r["fp_frames"], "ms": r["ms"], "over_100": r["over_100"],
                           "objects": {str(k): {"cls": o["cls"], "out": dict(o["out"]), "reasons": dict(o["reasons"]),
                                                "first": o["first"], "first_seen": o["first_seen"]}
                                       for k, o in r["objects"].items()}} for t, r in rep.items()}, fh, indent=1)
    print(text)


if __name__ == "__main__":
    main()
