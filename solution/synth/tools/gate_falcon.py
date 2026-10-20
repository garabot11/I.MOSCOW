"""Falcon gate: boxes inserted into a real Falcon Prime frame vs geometric expectation and vs a real person.

    python solution/synth/tools/gate_falcon.py [--bg 2000]

Scene (falcon_test_second_dark): train stationary for 444 s in a straight double-track tunnel; a person in
dark clothing walks along the track axis towards the sensor from ~75 m (frame ~3830) to ~12 m (frame 4440).
Frames before 3800 are empty backgrounds.

* Real person: first returns inside a corridor around the track axis whose 3D region is empty in the
  background frame; counted per frame together with the person's distance, split into upper body
  (bed + 0.9 .. 2.1 m) and legs (bed + 0.05 .. 0.9 m).
* Synthetic: every slot of the Falcon table (incl. this frame's no-return slots, with the frame's
  per-line azimuth shift) is ray-cast against a box; a slot is on target when the box is nearer than the
  background return or the slot had no return.
* Geometric expectation: (a) the datasheet ROI raster 0.09 x 0.08 deg / non-ROI 0.18 x 0.24 deg times the
  box's angular size, (b) the number of table slots whose median direction falls inside the box outline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.realbag import RealBag, xyz_of  # noqa: E402
from synth.geometry import ray_obb  # noqa: E402
from synth.objects import make_object  # noqa: E402
from synth.raycast import Caster  # noqa: E402
from metro_obstacle_detector.beam_model import FalconBeamModel  # noqa: E402

BAG = "/home/gag/i.mos/dataset/falcon/falcon_test_second_dark"
TABLE = os.path.join(os.path.dirname(HERE), "data", "falcon_prime_seyond.npz")
AXES = np.eye(3)  # box axes = sensor axes (x forward along the straight track, y left, z up)


def fit_bed(xyz, d_lo=5.0, d_hi=70.0):
    """Robust bed line z = a + b d near the track axis (|y| < 1): lowest returns per 5 m bin, then a line."""
    x, y, z = xyz.T
    near = (np.abs(y) < 1.0) & (x > d_lo) & (x < 35.0) & (z < -1.0)
    b, a = np.polyfit(x[near], z[near], 1)
    ds, zs = [], []
    for d0 in np.arange(d_lo, d_hi, 5.0):
        pred = a + b * (d0 + 2.5)
        m = (np.abs(y) < 1.0) & (x > d0) & (x < d0 + 5) & (np.abs(z - pred) < 0.4 + 0.005 * d0)
        if m.sum() >= 3:
            ds.append(d0 + 2.5)
            zs.append(np.percentile(z[m], 50))
            b, a = np.polyfit(np.array(ds), np.array(zs), 1) if len(ds) >= 3 else (b, a)
    ds, zs = np.array(ds), np.array(zs)
    return float(a), float(b), {"bins_d": ds.tolist(), "bins_z": zs.round(3).tolist()}


def person_counts(bag, frames, bg_xyz, bed, corridor=(-0.9, 1.1)):
    a, b = bed
    bx, by, bz = bg_xyz.T
    out = []
    for fi in frames:
        f = bag.frame(fi)
        c = f.cloud[f.cloud["is_2nd_return"] == 0]
        p = xyz_of(c).astype(np.float64)
        x, y, z = p.T
        hb = z - (a + b * x)
        m = (x > 8) & (x < 90) & (y > corridor[0]) & (y < corridor[1]) & (hb > 0.05) & (hb < 2.2)
        if not m.any():
            continue
        h, e = np.histogram(x[m], bins=np.arange(8, 91, 0.5))
        j = int(np.argmax(np.convolve(h, np.ones(3), "same")))
        dc = e[j] + 0.25
        sel = m & (np.abs(x - dc) < 0.8)
        # background check: the same 3D region in the background frame
        hbb = bz - (a + b * bx)
        bgm = (np.abs(bx - dc) < 0.8) & (by > corridor[0]) & (by < corridor[1]) & (hbb > 0.05) & (hbb < 2.2)
        up = sel & (hb >= 0.9)
        lo = sel & (hb < 0.9)
        rr = np.linalg.norm(p[sel], axis=1)
        out.append({"frame": int(fi), "d_m": round(float(np.median(x[sel])), 2), "range_min_m": round(float(rr.min()), 2),
                    "points": int(sel.sum()), "upper_body": int(up.sum()), "legs": int(lo.sum()),
                    "bg_points_same_region": int(bgm.sum()),
                    "y_p5_p95": [round(float(np.percentile(y[sel], 5)), 2), round(float(np.percentile(y[sel], 95)), 2)],
                    "h_above_bed_p5_p95": [round(float(np.percentile(hb[sel], 5)), 2), round(float(np.percentile(hb[sel], 95)), 2)],
                    "intensity_median": float(np.median(c["intensity"][sel]))})
    return out


def box_on_target(model, cloud, centre, size):
    first = cloud["is_2nd_return"] == 0
    cf = cloud[first]
    j = model.slot_of_cloud(cf)
    ok = j >= 0
    r_bg = np.full(model.n_slots, np.inf)
    r_bg[j[ok]] = np.linalg.norm(xyz_of(cf[ok]).astype(np.float64), axis=1)
    dirs = model.directions_for_frame(cloud).astype(np.float64)
    t = ray_obb(dirs, centre, AXES, np.asarray(size) / 2)
    hit = np.isfinite(t) & (t < r_bg)
    return hit, np.isfinite(r_bg)


def table_inside_outline(model, centre, size):
    """Slots whose median direction falls inside the angular outline of the box's front face."""
    c = np.asarray(centre, dtype=np.float64)
    hx, hy, hz = np.asarray(size) / 2
    front = c[0] - hx
    az_lo, az_hi = np.degrees(np.arctan2(c[1] - hy, front)), np.degrees(np.arctan2(c[1] + hy, front))
    el_lo = np.degrees(np.arctan2(c[2] - hz, np.hypot(front, c[1])))
    el_hi = np.degrees(np.arctan2(c[2] + hz, np.hypot(front, c[1])))
    m = (model.az_deg >= az_lo) & (model.az_deg <= az_hi) & (model.el_deg >= el_lo) & (model.el_deg <= el_hi)
    return int(m.sum()), (az_hi - az_lo), (el_hi - el_lo), (az_lo + az_hi) / 2, (el_lo + el_hi) / 2


def datasheet_expectation(daz, delv, az_c, el_c, roi_centre=(10.8, -0.6), roi_size=(40.0, 4.8)):
    in_roi = abs(az_c - roi_centre[0]) < roi_size[0] / 2 and abs(el_c - roi_centre[1]) < roi_size[1] / 2
    res = (0.09, 0.08) if in_roi else (0.18, 0.24)
    return float(daz / res[0] * delv / res[1]), bool(in_roi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "gate_falcon.json"))
    args = ap.parse_args(argv)
    bag = RealBag(BAG)
    model = FalconBeamModel.load(TABLE)
    bgf = bag.frame(args.bg)
    bg = bgf.cloud
    bg_xyz = xyz_of(bg[bg["is_2nd_return"] == 0]).astype(np.float64)
    a, b, bed_info = fit_bed(bg_xyz)
    person = person_counts(bag, list(range(3830, 4441, 15)), bg_xyz, (a, b))
    y_axis = float(np.median([np.mean(p["y_p5_p95"]) for p in person if 45 < p["d_m"] < 80]))
    # ROI location from the table: the lines with only ROI slots (11, 13, ..., 23)
    roi = np.isin(model.scan_id, np.arange(11, 24, 2))
    roi_c = (float(np.median(model.az_deg[roi])), float(np.median(model.el_deg[roi])))
    roi_sz = (float(np.ptp(model.az_deg[roi])), float(np.ptp(model.el_deg[roi])))
    rows = []
    shapes = {"box 0.9 x 0.9 x 0.85 (spec)": (0.9, 0.9, 0.85, 0.0),
              "upper body 0.5 x 0.8 x 0.95 (bed+0.9..1.85)": (0.5, 0.8, 0.95, 0.9)}
    for name, (sx, sy, sz, lift) in shapes.items():
        for dfront in (50.0, 55.0, 60.0, 65.0):
            cx = dfront + sx / 2
            bed_z = a + b * cx
            centre = np.array([cx, y_axis, bed_z + lift + sz / 2])
            hit, had = box_on_target(model, bg, centre, (sx, sy, sz))
            n_tab, daz, delv, az_c, el_c = table_inside_outline(model, centre, (sx, sy, sz))
            exp_ds, in_roi = datasheet_expectation(daz, delv, az_c, el_c, roi_c, roi_sz)
            rows.append({"shape": name, "d_front_m": dfront, "centre": centre.round(3).tolist(),
                         "on_target_slots": int(hit.sum()), "of_which_no_return_in_bg": int((hit & ~had).sum()),
                         "table_slots_inside_outline": n_tab, "datasheet_expectation": round(exp_ds, 1),
                         "in_roi": in_roi, "angular_size_deg": [round(daz, 3), round(delv, 3)],
                         "ratio_on_target_to_datasheet": round(hit.sum() / exp_ds, 3) if exp_ds else None})
            print(f"{name:45s} d {dfront:4.0f}  on target {hit.sum():4d} (no-ret {int((hit & ~had).sum()):3d})  "
                  f"table-in-outline {n_tab:4d}  datasheet {exp_ds:6.1f} roi={in_roi}")
    # the real person vs the upper-body box at matching distances
    comp = []
    for p in person:
        if 48 <= p["d_m"] <= 76:
            sx, sy, sz, lift = shapes["upper body 0.5 x 0.8 x 0.95 (bed+0.9..1.85)"]
            cx = p["d_m"] + 0.1
            centre = np.array([cx, y_axis, a + b * cx + lift + sz / 2])
            hit, _ = box_on_target(model, bg, centre, (sx, sy, sz))
            comp.append({"frame": p["frame"], "d_m": p["d_m"], "real_upper_body": p["upper_body"],
                         "real_all": p["points"], "synthetic_upper_body_box": int(hit.sum()),
                         "ratio_real_to_synth": round(p["upper_body"] / max(1, hit.sum()), 3)})
    # the real person vs the procedural standing human (1.75 m, walking pose), every hit slot returning
    caster = Caster()
    human = make_object("person_standing", np.random.default_rng(3), height=1.75, walking=0.3, girth=1.15)
    first = bg["is_2nd_return"] == 0
    jb = model.slot_of_cloud(bg[first])
    r_bg = np.full(model.n_slots, np.inf)
    r_bg[jb[jb >= 0]] = np.linalg.norm(xyz_of(bg[first][jb >= 0]).astype(np.float64), axis=1)
    dirs_bg = model.directions_for_frame(bg).astype(np.float64)
    # beam divergence 0.1 deg (datasheet): centre ray + 6 sub-rays on a ring of radius 0.035 deg
    sub = [np.zeros(2)] + [0.035 * np.array([np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)]) for k in range(6)]
    up = np.array([0.0, 0.0, 1.0])
    e_az = np.cross(up, dirs_bg)
    e_az /= np.linalg.norm(e_az, axis=1, keepdims=True)
    e_el = np.cross(dirs_bg, e_az)
    for q in comp:
        cx = q["d_m"]
        V = human.V + np.array([cx, y_axis, a + b * cx])
        mh = caster.mesh(V, human.F)
        n_sub = np.zeros(model.n_slots, dtype=int)
        t_c = None
        for k, (u, v) in enumerate(sub):
            dsub = dirs_bg + np.radians(u) * e_az + np.radians(v) * e_el
            dsub /= np.linalg.norm(dsub, axis=1, keepdims=True)
            t, _, _ = caster.cast(mh, np.zeros(3), dsub, 400.0)
            n_sub += np.isfinite(t) & (t < r_bg)
            if k == 0:
                t_c = t
        hit = np.isfinite(t_c) & (t_c < r_bg)
        hb = (dirs_bg[hit] * t_c[hit, None])[:, 2] - (a + b * cx)
        q["synthetic_human_all"] = int(hit.sum())
        q["synthetic_human_upper_body"] = int((hb >= 0.9).sum())
        q["ratio_real_to_human_all"] = round(q["real_all"] / max(1, hit.sum()), 3)
        q["ratio_real_to_human_upper"] = round(q["real_upper_body"] / max(1, (hb >= 0.9).sum()), 3)
        for need in (1, 2, 4):
            q[f"synthetic_human_divergence_ge{need}of7"] = int((n_sub >= need).sum())
            q[f"ratio_real_to_divergence_ge{need}of7"] = round(q["real_all"] / max(1, (n_sub >= need).sum()), 3)
    rep = {"bag": os.path.basename(BAG), "background_frame": args.bg, "bed_line": {"a": a, "b": b, **bed_info},
           "track_axis_y": y_axis, "roi_from_table": {"centre_az_el": roi_c, "size_az_el": roi_sz},
           "boxes": rows, "real_person": person, "person_vs_box": comp,
           "note": "on-target counts assume every slot hit by the box returns (P~1 at <= 65 m for rho >= 10 %); "
                   "the real person wears dark clothing, so real/synthetic < 1 measures its return fraction"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("bed line z = %.3f + %.4f d; track axis y = %.2f; ROI (table) centre az/el %s size %s" % (a, b, y_axis, np.round(roi_c, 2), np.round(roi_sz, 2)))
    for q in comp:
        print(f"  person frame {q['frame']} d {q['d_m']:5.1f}: real upper body {q['real_upper_body']:4d} (all {q['real_all']:4d})  "
              f"synthetic upper-body box {q['synthetic_upper_body_box']:4d}  ratio {q['ratio_real_to_synth']} | human 1.75 m: "
              f"all {q['synthetic_human_all']:4d} upper {q['synthetic_human_upper_body']:4d}  real/human all {q['ratio_real_to_human_all']} "
              f"upper {q['ratio_real_to_human_upper']} | divergence >=1/7 {q['synthetic_human_divergence_ge1of7']} "
              f"(real/syn {q['ratio_real_to_divergence_ge1of7']}) >=2/7 {q['synthetic_human_divergence_ge2of7']} "
              f"(real/syn {q['ratio_real_to_divergence_ge2of7']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
