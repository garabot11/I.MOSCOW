"""Scoreboard of a fake_style set split by insertion style (fake_a / in_place / physical).

    python solution/synth/tools/style_scoreboard.py --set /home/gag/i.mos/synthetic/val_fake_style \
        --results /home/gag/i.mos/work/eval/<tag>/synth/val_fake_style --out /home/gag/i.mos/work/eval/<tag>/style_scoreboard.md

Uses ``solution/scripts/scoreboard.py`` unchanged (same positive definition: obstacle, visible, n_hits >= min_hits,
top >= min_top above the rail; same match rule; same FP-frame rule) on a per-style view of the set (symlinks), and
adds a class x band recall matrix per style computed with the scoreboard's own ``match`` / ``band_of``.
The style of a scene is read from its ``scene.json`` (``style``), not from the name.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "scripts"))
sys.path.insert(0, SCRIPTS)
import scoreboard as sb  # noqa: E402

STYLES = ("fake_a", "in_place", "physical")


def scene_style(sdir: str):
    p = os.path.join(sdir, "scene.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get("style")


def class_band(set_dir, results_dir, scenes, min_hits, min_top):
    pos, det = Counter(), Counter()
    for sc in scenes:
        res = os.path.join(results_dir, sc + ".jsonl")
        lab = os.path.join(set_dir, sc, "labels.jsonl")
        if not (os.path.exists(res) and os.path.exists(lab)):
            continue
        by_frame = {r["frame_index"]: r for r in sb.load_jsonl(res)}
        for l in sb.load_jsonl(lab):
            r = by_frame.get(l["frame"])
            if r is None:
                continue
            cands = [c for c in r.get("detector", {}).get("candidates", []) if c.get("confirmed")]
            for o in l["objects"]:
                if not (o["is_obstacle"] and o.get("visible", True) and o.get("n_hits", 0) >= min_hits
                        and o.get("top_above_rail_m", 9.9) >= min_top):
                    continue
                k = (o["class"], sb.band_of(o["distance_m"]))
                pos[k] += 1
                if any(sb.match(c, o) for c in cands):
                    det[k] += 1
    return pos, det


def fmt_matrix(pos, det):
    bands = [f"{lo}-{hi}" for lo, hi in sb.BANDS]
    classes = sorted({c for c, _ in pos})
    out = ["| class | " + " | ".join(bands) + " | all |", "|---|" + "---:|" * (len(bands) + 1)]
    for c in classes:
        cells = []
        for b in bands:
            n = pos[(c, b)]
            cells.append(f"{det[(c, b)]}/{n} ({det[(c, b)] / n:.2f})" if n else "-")
        n = sum(pos[(c, b)] for b in bands); d = sum(det[(c, b)] for b in bands)
        cells.append(f"{d}/{n} ({d / n:.2f})" if n else "-")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="markdown report; <out>.json and the per-style views next to it")
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--min-top", type=float, default=0.3)
    a = ap.parse_args(argv)
    set_dir, res_dir = os.path.abspath(a.set), os.path.abspath(a.results)
    scenes = sorted(d for d in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, d)))
    by_style = {s: [sc for sc in scenes if scene_style(os.path.join(set_dir, sc)) == s] for s in STYLES}
    view_root = os.path.splitext(os.path.abspath(a.out))[0] + "_views"
    md = [f"# Scoreboard per insertion style: {os.path.basename(set_dir)}", "",
          f"results: `{res_dir}`; positive = obstacle, visible, n_hits >= {a.min_hits}, top >= {a.min_top} m above rail; "
          "rules of `scripts/scoreboard.py`.", ""]
    js = {}
    summary = ["| style | scenes | frames | object-frames | detected | recall | FP frames | FP episodes | ms median |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in STYLES + ("all",):
        scs = scenes if s == "all" else by_style[s]
        if not scs:
            continue
        vset, vres = os.path.join(view_root, s, "set"), os.path.join(view_root, s, "results")
        for d in (vset, vres):
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.unlink(os.path.join(d, f))
        for sc in scs:
            os.symlink(os.path.join(set_dir, sc), os.path.join(vset, sc))
            rp = os.path.join(res_dir, sc + ".jsonl")
            if os.path.exists(rp):
                os.symlink(rp, os.path.join(vres, sc + ".jsonl"))
        rep = sb.evaluate(vset, vres, a.min_hits, a.min_top)
        pos, det = class_band(set_dir, res_dir, scs, a.min_hits, a.min_top)
        npos, ndet = sum(pos.values()), sum(det.values())
        nfr = sum(v["frames"] for v in rep["per_scene"].values())
        summary.append(f"| {s} | {len(scs)} | {nfr} | {npos} | {ndet} | {ndet / npos if npos else float('nan'):.3f} | "
                       f"{rep['fp_frames']} | {rep['fp_episodes']} | {rep['processing_ms']['median']} |")
        js[s] = {"scenes": scs, "scoreboard": rep,
                 "class_band": {f"{c}|{b}": [det[(c, b)], pos[(c, b)]] for (c, b) in sorted(pos)}}
        md += [f"## {s}", "", "Recall per band (object-frames):", "", "| band, m | positives | detected | recall |",
               "|---|---:|---:|---:|"]
        for b, v in rep["bands"].items():
            r = "-" if v["recall"] is None else f"{v['recall']:.3f}"
            md.append(f"| {b} | {v['positives']} | {v['detected']} | {r} |")
        md += ["", "Recall per class x band (detected/positives):", ""] + fmt_matrix(pos, det)
        md += ["", f"FP frames {rep['fp_frames']}, episodes {rep['fp_episodes']}, beside-track detections "
                   f"{rep['beside_track_detections']}, missing results {rep['missing_results']}", "",
               "| scene | frames | positives | detected | FP frames |", "|---|---:|---:|---:|---:|"]
        for sc, v in rep["per_scene"].items():
            md.append(f"| {sc} | {v['frames']} | {v['positives']} | {v['detected']} | {v['fp_frames']} |")
        md.append("")
    md[4:4] = ["## Summary", ""] + summary + [""]
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=1)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
