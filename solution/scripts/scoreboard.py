#!/usr/bin/env python3
"""Scoreboard: detector results against synthetic labels (interface B) per range band.

    scoreboard.py --set /path/synthetic/val_hesai_insert --results /path/results_dir [--out report.md] [--json report.json]

``results`` holds one offline JSONL per scene (``<scene_dir_name>.jsonl``, written by
``ros2 run metro_obstacle_detector offline --bag <scene>/bag --out ...``).  Standard library only.

An object counts as a positive in a frame when it is labelled an obstacle (inside the clearance),
visible, has at least ``--min-hits`` returns and its top is at least ``--min-top`` above the rail
(the detector's design floor).  It is detected when a confirmed candidate of that frame lies within
``2 m + 2 %`` in Euclidean range and 1.2 m laterally.  A frame is a false positive when its status
is ``detected`` and none of its confirmed candidates matches any labelled object (objects outside
the clearance are reported separately as "beside-track detections", not as false positives).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional

BANDS = [(0, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 300)]


def band_of(d: float) -> Optional[str]:
    for lo, hi in BANDS:
        if lo <= d < hi:
            return f"{lo}-{hi}"
    return None


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def match(cand: dict, obj: dict) -> bool:
    tol = 2.0 + 0.02 * obj["distance_m"]
    if abs(cand["distance_m"] - obj["distance_m"]) > tol:
        return False
    lat_c = cand.get("lateral_offset_from_track")
    if lat_c is None:
        return True
    return abs(lat_c - obj["lateral_m"]) <= 1.2


def quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def evaluate(set_dir: str, results_dir: str, min_hits: int, min_top: float) -> dict:
    scenes = sorted(d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d)))
    band_pos = Counter(); band_det = Counter()
    cls_pos = Counter(); cls_det = Counter()
    first_det = {}             # (scene, id) -> dict
    fp_frames = []; beside = 0; unknown = 0; n_frames = 0; det_frames = 0
    proc = []; corridor = {"rail_ok": 0, "curve_ok": 0, "max_d_abs": [], "jitter50": [], "jitter100": [], "jitter150": []}
    missing = []
    per_scene = {}
    for sc in scenes:
        lab_path = os.path.join(set_dir, sc, "labels.jsonl")
        res_path = os.path.join(results_dir, sc + ".jsonl")
        if not os.path.exists(lab_path):
            continue
        if not os.path.exists(res_path):
            missing.append(sc)
            continue
        labels = load_jsonl(lab_path)
        results = load_jsonl(res_path)
        by_frame = {r["frame_index"]: r for r in results}
        s_pos = s_det = s_fp = 0
        prev_lc = None
        obj_seen = defaultdict(int); obj_det_after = defaultdict(int)
        for lab in labels:
            fi = lab["frame"]
            r = by_frame.get(fi)
            if r is None:
                continue
            n_frames += 1
            det = r.get("detector", {})
            if r["status"] == "unknown":
                unknown += 1
            if r["status"] == "detected":
                det_frames += 1
            proc.append(r.get("processing_ms", 0.0))
            rail = det.get("rail"); curve = det.get("curve")
            if rail and rail.get("ok"):
                corridor["rail_ok"] += 1
            if curve and curve.get("ok"):
                corridor["curve_ok"] += 1
                corridor["max_d_abs"].append(curve.get("max_d_absolute", 0.0))
                lc = [curve["l0"] + curve["a"] * D + curve["b"] * D * D for D in (50, 100, 150)]
                if prev_lc is not None:
                    for k, key in enumerate(("jitter50", "jitter100", "jitter150")):
                        corridor[key].append(abs(lc[k] - prev_lc[k]))
                prev_lc = lc
            else:
                prev_lc = None
            cands = [c for c in det.get("candidates", []) if c.get("confirmed")]
            matched_cands = set()
            for obj in lab["objects"]:
                positive = (obj["is_obstacle"] and obj.get("visible", True) and obj.get("n_hits", 0) >= min_hits
                            and obj.get("top_above_rail_m", 9.9) >= min_top)
                hit = [k for k, c in enumerate(cands) if match(c, obj)]
                if hit:
                    matched_cands.update(hit)
                if positive:
                    key = (sc, obj["id"]); b = band_of(obj["distance_m"])
                    band_pos[b] += 1; cls_pos[obj["class"]] += 1; s_pos += 1
                    obj_seen[key] += 1
                    if hit:
                        band_det[b] += 1; cls_det[obj["class"]] += 1; s_det += 1
                        if key not in first_det:
                            first_det[key] = {"scene": sc, "id": obj["id"], "class": obj["class"], "frame": fi,
                                              "distance_m": obj["distance_m"], "frames_before": obj_seen[key] - 1}
                    if key in first_det:
                        obj_det_after[key] += 1 if hit else 0
                elif hit and not obj["is_obstacle"]:
                    beside += 1
            if r["status"] == "detected" and cands and len(matched_cands) < len(cands) and not any(
                    match(c, o) for c in cands for o in lab["objects"]):
                fp_frames.append((sc, fi, r.get("distance_m")))
                s_fp += 1
        for key, n in obj_seen.items():
            if key in first_det:
                after = n - first_det[key]["frames_before"]
                first_det[key]["stability"] = obj_det_after[key] / after if after else float("nan")
        per_scene[sc] = {"positives": s_pos, "detected": s_det, "fp_frames": s_fp, "frames": len(labels)}
    # FP episodes
    episodes = 0; last = None
    for sc, fi, _ in fp_frames:
        if last is None or last[0] != sc or fi - last[1] > 2:
            episodes += 1
        last = (sc, fi)
    bands = {f"{lo}-{hi}": {"positives": band_pos[f"{lo}-{hi}"], "detected": band_det[f"{lo}-{hi}"],
                            "recall": (band_det[f"{lo}-{hi}"] / band_pos[f"{lo}-{hi}"]) if band_pos[f"{lo}-{hi}"] else None}
             for lo, hi in BANDS}
    classes = {c: {"positives": cls_pos[c], "detected": cls_det[c], "recall": cls_det[c] / cls_pos[c]} for c in cls_pos}
    return {
        "set": os.path.basename(os.path.normpath(set_dir)), "results": results_dir, "scenes": len(per_scene), "missing_results": missing,
        "frames": n_frames, "detected_frames": det_frames, "unknown_frames": unknown,
        "bands": bands, "classes": classes,
        "first_detection": sorted(first_det.values(), key=lambda x: (x["scene"], x["id"])),
        "objects_never_detected": [{"scene": k[0], "id": k[1]} for k in set(k for k in obj_seen_keys(set_dir, scenes, min_hits, min_top)) if k not in first_det],
        "fp_frames": len(fp_frames), "fp_episodes": episodes, "fp_examples": fp_frames[:10], "beside_track_detections": beside,
        "processing_ms": {"median": statistics.median(proc) if proc else None, "p95": quantile(proc, 0.95), "max": max(proc) if proc else None},
        "corridor": {"rail_ok_frac": corridor["rail_ok"] / n_frames if n_frames else None,
                     "curve_ok_frac": corridor["curve_ok"] / n_frames if n_frames else None,
                     "max_d_absolute_median": statistics.median(corridor["max_d_abs"]) if corridor["max_d_abs"] else None,
                     "max_d_absolute_p10": quantile(corridor["max_d_abs"], 0.10),
                     "lc_jitter_m": {k: {"median": statistics.median(v) if v else None, "p95": quantile(v, 0.95)} for k, v in
                                     (("50", corridor["jitter50"]), ("100", corridor["jitter100"]), ("150", corridor["jitter150"]))}},
        "per_scene": per_scene,
    }


def obj_seen_keys(set_dir, scenes, min_hits, min_top):
    for sc in scenes:
        p = os.path.join(set_dir, sc, "labels.jsonl")
        if not os.path.exists(p):
            continue
        for lab in load_jsonl(p):
            for obj in lab["objects"]:
                if (obj["is_obstacle"] and obj.get("visible", True) and obj.get("n_hits", 0) >= min_hits
                        and obj.get("top_above_rail_m", 9.9) >= min_top):
                    yield (sc, obj["id"])


def to_markdown(rep: dict) -> str:
    out = [f"# Scoreboard: {rep['set']}", "",
           f"scenes {rep['scenes']}, frames {rep['frames']}, detected frames {rep['detected_frames']}, unknown {rep['unknown_frames']}"
           + (f", **missing results for {rep['missing_results']}**" if rep["missing_results"] else ""), "",
           "## Recall per range band (object-frames)", "", "| band, m | positives | detected | recall |", "|---|---:|---:|---:|"]
    for b, v in rep["bands"].items():
        out.append(f"| {b} | {v['positives']} | {v['detected']} | {'' if v['recall'] is None else '%.3f' % v['recall']} |")
    out += ["", "## Recall per class", "", "| class | positives | detected | recall |", "|---|---:|---:|---:|"]
    for c, v in sorted(rep["classes"].items()):
        out.append(f"| {c} | {v['positives']} | {v['detected']} | {v['recall']:.3f} |")
    out += ["", "## First detection per object", "", "| scene | id | class | distance, m | frames before | stability after |", "|---|---:|---|---:|---:|---:|"]
    for f in rep["first_detection"]:
        out.append(f"| {f['scene']} | {f['id']} | {f['class']} | {f['distance_m']:.1f} | {f['frames_before']} | {f.get('stability', float('nan')):.2f} |")
    if rep["objects_never_detected"]:
        out.append(f"\nnever detected: {rep['objects_never_detected']}")
    pm = rep["processing_ms"]; co = rep["corridor"]
    out += ["", "## False positives", "", f"FP frames {rep['fp_frames']}, episodes {rep['fp_episodes']}, beside-track detections {rep['beside_track_detections']}; examples {rep['fp_examples'][:5]}",
            "", "## Corridor and timing", "",
            f"rail ok {co['rail_ok_frac']:.3f}, curve ok {co['curve_ok_frac']:.3f}, max_d_absolute median {co['max_d_absolute_median']} (p10 {co['max_d_absolute_p10']}); "
            f"l_c jitter m (median/p95): 50 m {co['lc_jitter_m']['50']['median']}/{co['lc_jitter_m']['50']['p95']}, 100 m {co['lc_jitter_m']['100']['median']}/{co['lc_jitter_m']['100']['p95']}, 150 m {co['lc_jitter_m']['150']['median']}/{co['lc_jitter_m']['150']['p95']}",
            f"processing ms median {pm['median']}, p95 {pm['p95']}, max {pm['max']}", "", "## Per scene", "", "| scene | frames | positives | detected | FP frames |", "|---|---:|---:|---:|---:|"]
    for sc, v in rep["per_scene"].items():
        out.append(f"| {sc} | {v['frames']} | {v['positives']} | {v['detected']} | {v['fp_frames']} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--out"); ap.add_argument("--json"); ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--min-top", type=float, default=0.3, help="objects whose top is lower above the rail are not counted (detector floor)")
    a = ap.parse_args()
    rep = evaluate(a.set, a.results, a.min_hits, a.min_top)
    md = to_markdown(rep)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False)
    print(md)


if __name__ == "__main__":
    main()
