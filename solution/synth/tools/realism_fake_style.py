"""Realism self-check of a fake_style set against the organisers' cloud_with_fake_obj (work/specs/11_fake_style.md).

    python solution/synth/tools/realism_fake_style.py --set /home/gag/i.mos/synthetic/val_fake_style \
        --org /home/gag/i.mos/synthetic/labels_org/cloud_with_fake_obj [--style fake_a] [--md out.md]

Compares, per object class and range band, the points per object (median n_hits) and the size-normalised
density n_hits * r^2 / (frontal area) [points per m^2 at 1 m] of our inserted objects with the organisers'
objects (labels_org, appended block per frame clustered into objects).  Frontal area = lateral x height extent
of the label (extent_dlh[1] * extent_dlh[2]); for the organisers' far objects the extent is that of the points,
so the density there is an upper bound.  Reads only labels.jsonl files (no ROS).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

BANDS = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 300)]
# organisers' classes (labels_org) -> comparable classes of ours
GROUPS = {
    "wall": ("wall",), "plate": ("plate",), "cube": ("cube", "small_object"), "side_box": ("side_box",),
    "overhead_box": ("overhead_box",), "pole": ("pole",), "person": ("person_standing", "person_crouching", "person_lying"),
    "box": ("box",)}


def band_of(r):
    for lo, hi in BANDS:
        if lo <= r < hi:
            return f"{lo}-{hi}"
    return None


# nominal frontal area (lateral x height, m^2) of the organisers' objects (11_fake_style.md section 4); their label
# extents come from the hit points and collapse to 1-2 points beyond ~75 m
ORG_AREA = {"wall": 2.0 * 2.0, "plate": 2.0 * 0.25, "cube": 0.33 * 0.33, "side_box": 2.4 * 2.0,
            "overhead_box": 2.3 * 2.0, "pole": 0.07 * 2.0}


def collect(label_files, class_map, style=None, nominal_area=None):
    hits = defaultdict(list)       # (group, band) -> n_hits
    dens = defaultdict(list)       # (group, band) -> n_hits * r^2 / area
    for lf in label_files:
        with open(lf, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                for ob in rec.get("objects", []):
                    if style and ob.get("style", style) != style:
                        continue
                    g = class_map.get(ob["class"])
                    if g is None:
                        continue
                    r = float(ob["distance_m"])
                    b = band_of(r)
                    if b is None:
                        continue
                    n = int(ob["n_hits"])
                    hits[(g, b)].append(n)
                    if nominal_area is not None:
                        area = nominal_area.get(g, 0.0)
                    else:
                        ext = ob.get("extent_dlh") or [0, 0, 0]   # ours: true mesh extents
                        area = float(ext[1]) * float(ext[2])
                    if n > 0 and area > 1e-3:
                        dens[(g, b)].append(n * r * r / area)
    return hits, dens


def fmt(v, digits=0):
    if not v:
        return "-"
    a = np.asarray(v, dtype=np.float64)
    q = np.percentile(a, [50, 10, 90])
    if digits == 0:
        return f"{q[0]:.0f} ({q[1]:.0f}-{q[2]:.0f}) n={len(a)}"
    return f"{q[0]:.{digits}f} ({q[1]:.{digits}f}-{q[2]:.{digits}f}) n={len(a)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--org", default="/home/gag/i.mos/synthetic/labels_org/cloud_with_fake_obj")
    ap.add_argument("--style", default="fake_a")
    ap.add_argument("--md")
    a = ap.parse_args()
    our_map = {c: g for g, cs in GROUPS.items() for c in cs}
    org_map = {"wall": "wall", "plate": "plate", "cube": "cube", "side_box": "side_box", "overhead_box": "overhead_box",
               "pole": "pole", "box": "box", "person": "person", "person_standing": "person"}
    ours = sorted(glob.glob(os.path.join(a.set, "*", "labels.jsonl")))
    h_o, d_o = collect(ours, our_map, a.style)
    h_g, d_g = collect([os.path.join(a.org, "labels.jsonl")], org_map, nominal_area=ORG_AREA)
    lines = [f"# Realism of `{os.path.basename(a.set)}` (style {a.style}) vs organisers' `{os.path.basename(a.org)}`", "",
             "Per class and range band: median (p10-p90) of points per object per frame, and of the size-normalised density "
             "n_hits * r^2 / frontal area [pts per m^2 at 1 m]; frontal area = lateral x height: nominal catalogue sizes "
             "for theirs (wall 2x2, plate 2x0.25, cube 0.33, side box 2.4x2, overhead 2.3x2, pole 0.07x2), true mesh "
             "extents for ours; n = object-frames (density: frames with hits).", "",
             "| class | band m | theirs pts | ours pts | theirs density | ours density |", "|---|---|---|---|---|---|"]
    for g in GROUPS:
        for lo, hi in BANDS:
            b = f"{lo}-{hi}"
            if not h_g.get((g, b)) and not h_o.get((g, b)):
                continue
            lines.append(f"| {g} | {b} | {fmt(h_g.get((g, b)))} | {fmt(h_o.get((g, b)))} | "
                         f"{fmt(d_g.get((g, b)))} | {fmt(d_o.get((g, b)))} |")
    # visibility: fraction of object-frames with 0 hits per band
    lines += ["", "| class | band m | theirs frames with 0 hits | ours frames with 0 hits |", "|---|---|---|---|"]
    for g in GROUPS:
        for lo, hi in BANDS:
            b = f"{lo}-{hi}"
            tg, to = h_g.get((g, b)), h_o.get((g, b))
            if not tg and not to:
                continue
            z = lambda v: "-" if not v else f"{np.mean(np.asarray(v) == 0):.2f} (n={len(v)})"
            lines.append(f"| {g} | {b} | {z(tg)} | {z(to)} |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
