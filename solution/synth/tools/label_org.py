"""Point and object labels for the organisers' edited bags (``synthetic/labels_org/<bag>/``).

    env -u PYTHONPATH python tools/label_org.py fake  --out /home/gag/i.mos/synthetic/labels_org/cloud_with_fake_obj \
        [--motion s.npy]
    env -u PYTHONPATH python tools/label_org.py platform --out /home/gag/i.mos/synthetic/labels_org/synthetic_platform_cloud_only

``fake`` (cloud_with_fake_obj, style "fake_a"): the inserted points are the off-grid tail block of every frame
(``fake_style_probe.split``).  They are grouped into objects per frame (connectivity on the organisers' 128 x 2400
grid: |drow| <= 2, |dcol| <= 2, |dr| < 1 m), linked over frames into static world objects (predicted range
d - ds, ds = ego step from ``--motion`` or the median step of the other objects), and each track is typed from its
geometry (wall, plate, cube, side_box, overhead_box, pole; else stray).

``platform`` (synthetic_platform_cloud_only frames 0-314, style "in_place"): inserted slots = slots that differ from
the pristine frames 1195-1509 of cloud_with_fake_obj (identical header stamps).

Outputs: ``objects.jsonl`` (one line per frame: frame_index, stamp_ns, objects[{id, type, distance_m, lateral_m,
bbox_min, bbox_max (sensor frame), n_points, in_gauge, ...}]), ``labels.jsonl`` (interface B style),
``slot_labels.npz`` (frame, slot = message point index, obj_id), ``tracks.json`` (per object summary).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fake_style_probe as probe  # noqa: E402
from synth.fake_style import OT128_EL_DEG  # noqa: E402
from synth.realbag import RealBag  # noqa: E402

PLATFORM = "/home/gag/i.mos/dataset/org_synth/synthetic_platform_cloud_only"
V1 = "/home/gag/i.mos/work/eval/v1-gpu/extra/cloud_with_fake_obj.jsonl"
SIX = ("wall", "plate", "cube", "side_box", "overhead_box", "pole")


def components(xyz: np.ndarray) -> np.ndarray:
    """Labels of connected components of points on the organisers' grid."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(xyz)
    if n == 0:
        return np.zeros(0, np.int64)
    r = np.linalg.norm(xyz, axis=1)
    el = np.degrees(np.arcsin(xyz[:, 2] / r)); az = np.degrees(np.arctan2(xyz[:, 0], -xyz[:, 1]))
    row = np.abs(el[:, None] - OT128_EL_DEG[None, :]).argmin(1)
    col = np.rint(az / 0.15).astype(np.int64)
    key = {}
    for i in range(n):
        key.setdefault((row[i], col[i]), []).append(i)
    I, J = [], []
    for i in range(n):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                for j in key.get((row[i] + dr, col[i] + dc), ()):
                    if j > i and abs(r[i] - r[j]) < 1.0:
                        I.append(i); J.append(j)
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    return connected_components(A, directed=False)[1]


def comp_record(xyz):
    lo, hi = xyz.min(0), xyz.max(0)
    r = np.linalg.norm(xyz, axis=1)
    c = 0.5 * (lo + hi)
    # far off the track corridor (beyond a curve, through rock): x or z more than 6 m off the sensor axis
    return {"n": len(xyz), "lo": lo, "hi": hi, "c": c, "rmin": float(r.min()), "rmed": float(np.median(r)),
            "stray": bool(abs(c[0]) > 6.0 or abs(c[2]) > 6.0)}


def estimate_steps(per_frame, n):
    """Per-frame approach step (m) from the inserted objects themselves: median decrease of the range of
    component pairs of consecutive frames that line up laterally/vertically; gaps filled, 5-frame median."""
    steps = np.full(n, np.nan)
    for f in range(1, n):
        a, b = per_frame.get(f - 1), per_frame.get(f)
        if not a or not b:
            continue
        cand = []
        for ca in a:
            for cb in b:
                if ca["n"] < 3 or cb["n"] < 3:
                    continue
                dd = cb["c"][1] - ca["c"][1]          # y increases when the object comes nearer
                if -0.2 <= dd <= 3.0 and abs(cb["c"][0] - ca["c"][0]) < 0.3 and abs(cb["c"][2] - ca["c"][2]) < 0.3:
                    cand.append(dd)
        if cand:
            steps[f] = float(np.median(cand))
    ok = np.isfinite(steps)
    if ok.any():
        idx = np.arange(n)
        steps = np.interp(idx, idx[ok], steps[ok])
        steps = np.array([np.median(steps[max(0, k - 2):k + 3]) for k in range(n)])
    else:
        steps = np.zeros(n)
    return steps


def link(per_frame, steps, max_gap=25):
    """Greedy linking of per-frame components into static objects.  per_frame: {frame: [comp]}; steps[f] = ego
    displacement between f-1 and f (m)."""
    tracks = []          # list of dict(frames=[], comps=[], last_f, last_c)
    frames = sorted(per_frame)
    for f in frames:
        comps = sorted(per_frame[f], key=lambda c: c["rmin"])
        used = set()
        cands = []
        for ti, t in enumerate(tracks):
            g = f - t["last_f"]
            if g < 1 or g > max_gap:
                continue
            ds = float(np.sum(steps[t["last_f"] + 1:f + 1]))
            pc = t["last_c"].copy()
            pc[1] += ds                       # sensor y = -d: d decreases by ds -> y increases by ds
            for ci, c in enumerate(comps):
                dd = abs(c["c"][1] - pc[1]); dx = abs(c["c"][0] - pc[0]); dz = abs(c["c"][2] - pc[2])
                dist = max(5.0, -pc[1])
                if c["stray"] != t["comps"][-1]["stray"]:
                    continue
                if dd < 1.5 + 0.01 * dist + 0.3 * g and dx < 0.6 + 0.006 * dist and dz < 0.6 + 0.005 * dist:
                    cands.append((dd + dx + dz + 0.05 * g, ti, ci))
        cands.sort()
        taken_t = set()
        for _, ti, ci in cands:
            if ti in taken_t or ci in used:
                continue
            t = tracks[ti]
            t["frames"].append(f); t["comps"].append(comps[ci]); t["last_f"] = f; t["last_c"] = comps[ci]["c"]
            taken_t.add(ti); used.add(ci)
        for ci, c in enumerate(comps):
            if ci not in used:
                tracks.append({"frames": [f], "comps": [c], "last_f": f, "last_c": c["c"]})
    return tracks


def type_of(t) -> str:
    """Type of a track from its best-resolved observation (most points)."""
    k = int(np.argmax([c["n"] for c in t["comps"]]))
    c = t["comps"][k]
    ext = c["hi"] - c["lo"]
    xc, zc = c["c"][0], c["c"][2]
    f = t["frames"][k]
    if c["rmin"] > 200 and c["n"] < 4:
        far = True
    else:
        far = False
    if c["stray"]:
        return "stray"
    if f <= 231 and ext[1] < 0.1 and ext[0] > 1.0 and abs(xc) < 0.6:
        return "wall"
    if ext[0] < 0.2 and ext[2] > 0.3 and c["lo"][2] > 1.0 and abs(xc) < 0.6:
        return "pole"
    if c["hi"][2] < -0.8 and ext[0] > 0.8 and abs(xc) < 0.8:
        return "plate"
    if c["lo"][2] > 1.2 and (ext[0] > 0.8 or ext[2] > 0.8):
        return "overhead_box"
    if max(ext) < 0.5 and c["n"] >= 3 and not far and abs(xc) < 1.8:
        return "cube"
    if (c["lo"][0] > 0.5 or c["hi"][0] < -0.5) and (ext[2] > 1.0 or ext[0] > 1.0):
        return "side_box"
    if max(ext) < 0.5 and not far and abs(xc) < 1.8:
        return "cube"
    if abs(xc) >= 1.8 and c["hi"][2] < 1.5:
        return "side_box"
    return "unknown"


def v1_tracks():
    out = {}
    if not os.path.exists(V1):
        return out
    for line in open(V1, encoding="utf-8"):
        r = json.loads(line)
        det = r.get("detector", {})
        out[r["frame_index"]] = (det.get("rail") or {}, det.get("curve") or {})
    return out


def lateral_of(V1T, f, c):
    rail, cur = V1T.get(f, ({}, {}))
    d = -float(c[1])
    if cur.get("ok") and d <= cur.get("max_d_supported", 0) + 30:
        return float(c[0] - (cur["l0"] + cur["a"] * d + cur["b"] * d * d))
    return float(c[0])


def write_outputs(out, frames_all, stamps, per_frame_objs, slots, tracks_summary, style, bag):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "objects.jsonl"), "w") as fo, open(os.path.join(out, "labels.jsonl"), "w") as fl:
        for f in frames_all:
            objs = per_frame_objs.get(f, [])
            fo.write(json.dumps({"frame_index": f, "stamp_ns": stamps[f], "objects": objs}) + "\n")
            lab = [{"id": o["id"], "class": o["type"], "is_obstacle": o["in_gauge"], "center_sensor": o["center_sensor"],
                    "extent_dlh": o["extent_dlh"], "distance_m": o["distance_m"], "lateral_m": o["lateral_m"],
                    "n_hits": o["n_points"], "visible": o["n_points"] > 0, "in_clearance": o["in_gauge"],
                    "reflectivity": None, "d_along_m": o["d_along_m"], "height_above_rail_m": None,
                    "top_above_rail_m": None} for o in objs]
            fl.write(json.dumps({"frame": f, "stamp_ns": stamps[f], "sight_limit_m": None, "objects": lab}) + "\n")
    fr, sl, oi = slots
    np.savez_compressed(os.path.join(out, "slot_labels.npz"), frame=np.asarray(fr, np.int32), slot=np.asarray(sl, np.int32),
                        obj_id=np.asarray(oi, np.int32))
    json.dump({"bag": bag, "style": style, "tracks": tracks_summary,
               "format": {"objects.jsonl": "one line per frame: frame_index, stamp_ns, objects[{id, type, distance_m "
                          "(nearest labelled point), lateral_m (from the v1 centre-line where supported, else sensor x), "
                          "d_along_m (-y of the bbox centre), bbox_min, bbox_max (sensor frame, labelled points), "
                          "center_sensor, extent_dlh, n_points, in_gauge}]",
                          "slot_labels.npz": "frame (bag index), slot (message point index), obj_id"}},
              open(os.path.join(out, "tracks.json"), "w"), indent=1)


def run_fake(out, motion_path=None):
    bag = RealBag(probe.FAKE_BAG)
    per_frame, stamps, pts_of = {}, {}, {}
    for f in range(len(bag)):
        fr = bag.frame(f)
        stamps[f] = fr.stamp_ns
        c = fr.cloud
        r, ring, fi, v, off, m = probe.split(c)
        if m == len(c):
            continue
        xyz = np.stack([c["x"][m:], c["y"][m:], c["z"][m:]], 1).astype(np.float64)
        lab = components(xyz)
        comps = []
        for k in np.unique(lab):
            sel = np.nonzero(lab == k)[0]
            rec = comp_record(xyz[sel]); rec["idx"] = m + sel
            comps.append(rec)
        per_frame[f] = comps
    n = len(bag)
    if motion_path and os.path.exists(motion_path):
        s = np.load(motion_path)
        steps = np.zeros(n); steps[1:len(s)] = np.diff(s)
    else:
        steps = estimate_steps(per_frame, n)
    tracks = link(per_frame, steps)
    V1T = v1_tracks()
    per_frame_objs, fr_l, sl_l, id_l, summ = {}, [], [], [], []
    for tid, t in enumerate(sorted(tracks, key=lambda t: (t["frames"][0], t["comps"][0]["rmin"])), start=1):
        typ = type_of(t)
        in_gauge = typ in SIX
        rmins = [c["rmin"] for c in t["comps"]]
        summ.append({"id": tid, "type": typ, "frames": [t["frames"][0], t["frames"][-1]], "n_frames": len(t["frames"]),
                     "range_m": [round(rmins[0], 2), round(rmins[-1], 2)], "points_max": int(max(c["n"] for c in t["comps"])),
                     "bbox_at_max": [np.round(t["comps"][int(np.argmax([c['n'] for c in t['comps']]))]["lo"], 2).tolist(),
                                     np.round(t["comps"][int(np.argmax([c['n'] for c in t['comps']]))]["hi"], 2).tolist()],
                     "in_gauge": in_gauge})
        for f, c in zip(t["frames"], t["comps"]):
            ext = c["hi"] - c["lo"]
            per_frame_objs.setdefault(f, []).append({
                "id": tid, "type": typ, "distance_m": round(c["rmin"], 3), "lateral_m": round(lateral_of(V1T, f, c["c"]), 3),
                "d_along_m": round(-float(c["c"][1]), 3), "bbox_min": np.round(c["lo"], 3).tolist(),
                "bbox_max": np.round(c["hi"], 3).tolist(), "center_sensor": np.round(c["c"], 3).tolist(),
                "extent_dlh": [round(float(ext[1]), 3), round(float(ext[0]), 3), round(float(ext[2]), 3)],
                "n_points": int(c["n"]), "in_gauge": in_gauge})
            fr_l += [f] * c["n"]; sl_l += list(c["idx"]); id_l += [tid] * c["n"]
    write_outputs(out, list(range(n)), stamps, per_frame_objs, (fr_l, sl_l, id_l), summ, "fake_a", "cloud_with_fake_obj")
    return summ


def run_platform(out):
    plat = RealBag(PLATFORM)
    fake = RealBag(probe.FAKE_BAG)
    per_frame, stamps = {}, {}
    for j in range(len(plat)):
        fp = plat.frame(j)
        stamps[j] = fp.stamp_ns
        i = 1195 + j
        if i >= len(fake):
            continue
        ff = fake.frame(i)
        if ff.stamp_ns != fp.stamp_ns or len(ff.cloud) != len(fp.cloud):
            continue
        p, a = fp.cloud, ff.cloud
        d = (p["x"] != a["x"]) | (p["y"] != a["y"]) | (p["z"] != a["z"]) | (p["intensity"] != a["intensity"])
        idx = np.nonzero(d)[0]
        if not len(idx):
            continue
        xyz = np.stack([p["x"][idx], p["y"][idx], p["z"][idx]], 1).astype(np.float64)
        # in-place style: points sit on the real beam grid; group by 3-D proximity
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        r = np.linalg.norm(xyz, axis=1)
        tree = cKDTree(xyz)
        pairs = tree.query_pairs(r=max(0.8, 0.006 * float(np.median(r))), output_type="ndarray")
        from scipy.sparse import coo_matrix

        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz))) if len(pairs) else \
            coo_matrix((len(xyz), len(xyz)))
        lab = connected_components(A, directed=False)[1]
        comps = []
        for k in np.unique(lab):
            sel = np.nonzero(lab == k)[0]
            rec = comp_record(xyz[sel]); rec["idx"] = idx[sel]
            comps.append(rec)
        per_frame[j] = comps
    tracks = link(per_frame, estimate_steps(per_frame, len(plat)))
    per_frame_objs, fr_l, sl_l, id_l, summ = {}, [], [], [], []
    for tid, t in enumerate(sorted(tracks, key=lambda t: (t["frames"][0], t["comps"][0]["rmin"])), start=1):
        k = int(np.argmax([c["n"] for c in t["comps"]]))
        ext = t["comps"][k]["hi"] - t["comps"][k]["lo"]
        typ = "person_like" if ext[2] > 1.2 else "box"
        summ.append({"id": tid, "type": typ, "frames": [t["frames"][0], t["frames"][-1]], "n_frames": len(t["frames"]),
                     "range_m": [round(t["comps"][0]["rmin"], 2), round(t["comps"][-1]["rmin"], 2)],
                     "points_max": int(max(c["n"] for c in t["comps"])),
                     "bbox_at_max": [np.round(t["comps"][k]["lo"], 2).tolist(), np.round(t["comps"][k]["hi"], 2).tolist()]})
        for f, c in zip(t["frames"], t["comps"]):
            ext = c["hi"] - c["lo"]
            per_frame_objs.setdefault(f, []).append({
                "id": tid, "type": typ, "distance_m": round(c["rmin"], 3), "lateral_m": round(float(c["c"][0]), 3),
                "d_along_m": round(-float(c["c"][1]), 3), "bbox_min": np.round(c["lo"], 3).tolist(),
                "bbox_max": np.round(c["hi"], 3).tolist(), "center_sensor": np.round(c["c"], 3).tolist(),
                "extent_dlh": [round(float(ext[1]), 3), round(float(ext[0]), 3), round(float(ext[2]), 3)],
                "n_points": int(c["n"]), "in_gauge": bool(abs(float(c["c"][0])) < 2.0)})
            fr_l += [f] * c["n"]; sl_l += list(c["idx"]); id_l += [tid] * c["n"]
    write_outputs(out, sorted(stamps), stamps, per_frame_objs, (fr_l, sl_l, id_l), summ, "in_place",
                  "synthetic_platform_cloud_only (frames 0-314 labelled; later frames have no pristine reference)")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["fake", "platform"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion")
    a = ap.parse_args(argv)
    summ = run_fake(a.out, a.motion) if a.which == "fake" else run_platform(a.out)
    for s in summ:
        if s["points_max"] >= 3 or s["n_frames"] >= 3:
            print(json.dumps(s))
    print(len(summ), "tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
