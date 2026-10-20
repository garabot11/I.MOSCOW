"""Gate test: a synthetic 0.9 x 0.9 x 0.85 m box at the real object's position vs the real object.

    python solution/synth/tools/gate_hesai.py [--out solution/synth/results/gate_hesai.json]

Real count (independent of the detector): ``doubleT_obstacle`` is the only real positive and the train is
stationary.  In frames 30..45 the object stands at x ~ -1.7 m (sensor frame), 55.5-56.5 m ahead; in frames
>= 100 it stands at x ~ +0.95 m, > 2 m away.  A slot of frame k is an object return when it is valid, lies in
a loose region around the object and differs from *every* background frame of the same parity
(background zero or |r_k - r_bg| > 0.3 m).  Requiring a difference to all backgrounds removes flicker.

Synthetic count: the box is ray-cast along the beam-model direction of every fired slot of a background
frame; a slot is replaced (both echoes) when the box is nearer than the background return or the slot had
no return.  The composite is measured with exactly the same differencing procedure against the *other*
background frames.  Placement: lateral centre and front face from the real object, bottom on the local bed
measured in the background frames.  Sensitivity: bottom +-0.1 m, lateral +-0.2 m, and a box fitted to the
measured extent of the real object.
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
from metro_obstacle_detector.beam_model import hesai128  # noqa: E402

REAL_FRAMES = list(range(30, 46))
BG = {0: [100, 120, 140, 160, 180, 200], 1: [101, 121, 141, 161, 181, 199]}
DIFF_M = 0.3


def load(bag, i):
    c = bag.frame(i).cloud
    xyz = xyz_of(c).astype(np.float64)
    return xyz, np.linalg.norm(xyz, axis=1)


def object_slots(xyz, r, bgs, region):
    """Slots of a frame that are object returns (see module doc)."""
    d, x, z = -xyz[:, 1], xyz[:, 0], xyz[:, 2]
    (d0, d1), (x0, x1), (z0, z1) = region
    m = (r > 0) & (d > d0) & (d < d1) & (x > x0) & (x < x1) & (z > z0) & (z < z1)
    for _, rb in bgs:
        m &= (rb == 0) | (np.abs(r - rb) > DIFF_M)
    return np.nonzero(m)[0]


def stats(slots, xyz):
    if len(slots) == 0:
        return {"slots": 0, "beams": 0, "rings": 0, "per_ring": {}}
    d, x, z = -xyz[slots, 1], xyz[slots, 0], xyz[slots, 2]
    return {"slots": int(len(slots)), "near_echo_slots": int(np.sum((slots // 128) % 2 == 1)),
            "beams": int(len(np.unique((slots // 256) * 128 + slots % 128))), "rings": int(len(np.unique(slots % 128))),
            "per_ring": {int(k): int(v) for k, v in zip(*np.unique(slots % 128, return_counts=True))},
            "ring_min": int((slots % 128).min()), "ring_max": int((slots % 128).max()),
            "d_p10": float(np.percentile(d, 10)), "x_min": float(np.percentile(x, 5)), "x_max": float(np.percentile(x, 95)),
            "x_mid": float((np.percentile(x, 5) + np.percentile(x, 95)) / 2),
            "z_min": float(z.min()), "z_p5": float(np.percentile(z, 5)), "z_max": float(z.max())}


def insert_box(xyz_bg, r_bg, bm, fired, centre, half, rng, sigma=0.02, quant=0.004):
    """Composite frame: box along every fired slot direction (both echoes become the box)."""
    dirs = bm.directions().astype(np.float64)
    axes = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T  # columns: d=-y, l=+x, h=+z
    t = np.full(len(dirs), np.inf)
    t[fired] = ray_obb(dirs[fired], centre, axes, half)
    hit = np.isfinite(t) & ((r_bg == 0) | (t < r_bg))
    # dual return: a solid object replaces both echoes of a firing (whichever column the box hit)
    fir_hit = np.zeros(len(dirs) // 256, dtype=bool)
    fir_hit[np.nonzero(hit)[0] // 256] = True
    t_fir = np.full(len(fir_hit), np.inf)
    np.minimum.at(t_fir, np.nonzero(hit)[0] // 256, t[hit])
    out = xyz_bg.copy()
    sl = np.nonzero(np.repeat(fir_hit, 256) & fired)[0]
    # both columns of a firing share one direction per ring: recompute t per slot
    tt = t[sl]
    ok = np.isfinite(tt)
    sl = sl[ok]
    rr = np.round((t[sl] + rng.normal(0, sigma, len(sl))) / quant) * quant
    out[sl] = dirs[sl] * rr[:, None]
    return out, np.linalg.norm(out, axis=1), sl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "gate_hesai.json"))
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    bag = RealBag("doubleT_obstacle")
    bm = hesai128(7200)
    fired = bm.fired_mask()
    rng = np.random.default_rng(args.seed)
    bgs = {p: [load(bag, i) for i in idx] for p, idx in BG.items()}
    # ---------------- real object
    region = ((54.5, 58.5), (-3.0, -0.4), (-3.4, -1.2))
    real = {}
    for k in REAL_FRAMES:
        xyz, r = load(bag, k)
        s = object_slots(xyz, r, bgs[k % 2], region)
        real[k] = stats(s, xyz)
    n_real = np.array([real[k]["slots"] for k in REAL_FRAMES])
    f_real = np.array([real[k]["beams"] for k in REAL_FRAMES])
    x_mid = float(np.median([real[k]["x_mid"] for k in REAL_FRAMES]))
    d_front = float(np.median([real[k]["d_p10"] for k in REAL_FRAMES]))
    z_top = float(np.median([real[k]["z_max"] for k in REAL_FRAMES]))
    z_low = float(np.median([real[k]["z_p5"] for k in REAL_FRAMES]))
    x_w = float(np.median([real[k]["x_max"] - real[k]["x_min"] for k in REAL_FRAMES]))
    # ---------------- local bed under the object (background frames, both parities)
    zs = []
    for p in (0, 1):
        for xyz, r in bgs[p]:
            d, x, z = -xyz[:, 1], xyz[:, 0], xyz[:, 2]
            m = (r > 0) & (d > d_front - 1.0) & (d < d_front + 2.0) & (np.abs(x - x_mid) < 0.45) & (z < z_low + 0.4)
            zs.append(z[m])
    zs = np.concatenate(zs)
    bed = {"n": int(len(zs)), "p10": float(np.percentile(zs, 10)), "p50": float(np.percentile(zs, 50)),
           "p90": float(np.percentile(zs, 90))}
    z_bed = bed["p50"]

    # ---------------- synthetic
    def run(name, cx, bottom, size):
        half = np.array([size[0], size[1], size[2]]) / 2
        centre = np.array([cx, -(d_front + half[0]), bottom + half[2]])
        res = []
        for k in REAL_FRAMES:
            p = k % 2
            host_i = (k // 2) % len(BG[p])
            xyz_bg, r_bg = bgs[p][host_i]
            comp, rc, _ = insert_box(xyz_bg, r_bg, bm, fired, centre, half, rng)
            others = [b for j, b in enumerate(bgs[p]) if j != host_i]
            s = object_slots(comp, rc, others, region)
            res.append(stats(s, comp))
        n = np.array([q["slots"] for q in res])
        f = np.array([q["beams"] for q in res])
        rings = sorted({r for q in res for r in q["per_ring"]})
        prof = {r: float(np.mean([q["per_ring"].get(r, 0) for q in res])) for r in rings}
        return {"name": name, "centre_sensor": centre.round(3).tolist(), "size_dlh": list(size),
                "bottom_z": round(bottom, 3), "slots_mean": float(n.mean()), "slots_min": int(n.min()), "slots_max": int(n.max()),
                "beams_mean": float(f.mean()), "rings_median": float(np.median([q["rings"] for q in res])),
                "per_ring_mean": prof,
                "ratio_to_real_slots": float(n.mean() / n_real.mean()), "ratio_to_real_beams": float(f.mean() / f_real.mean())}

    spec = (0.9, 0.9, 0.85)
    fitted = (0.6, round(x_w + 0.1, 3), round(z_top - z_low + 0.12, 3))
    variants = [run("spec box, bottom on local bed", x_mid, z_bed, spec),
                run("spec box, bottom +0.1 m", x_mid, z_bed + 0.1, spec),
                run("spec box, bottom -0.1 m", x_mid, z_bed - 0.1, spec),
                run("spec box, lateral +0.2 m", x_mid + 0.2, z_bed, spec),
                run("spec box, lateral -0.2 m", x_mid - 0.2, z_bed, spec),
                run("spec box, bottom at lowest object return (p5)", x_mid, z_low, spec),
                run("box fitted to the measured extent (p5-p95)", x_mid, z_low - 0.06, fitted)]
    out = {
        "bag": "doubleT_obstacle", "layout": "921 600-point (7200 cols); the 307 200-point layout fires the same slots "
                                              "for an object at azimuth ~ -1.7 deg (inside the +-50 deg crop)",
        "real_frames": REAL_FRAMES, "background_frames": BG, "diff_threshold_m": DIFF_M, "region_dxz": region,
        "real": {"slots_mean": float(n_real.mean()), "slots_min": int(n_real.min()), "slots_max": int(n_real.max()),
                 "beams_mean": float(f_real.mean()), "per_frame": real,
                 "per_ring_mean": {r: float(np.mean([real[k]["per_ring"].get(r, 0) for k in REAL_FRAMES]))
                                   for r in sorted({r for k in REAL_FRAMES for r in real[k]["per_ring"]})},
                 "x_mid": x_mid, "d_front": d_front, "z_top": z_top, "z_low_p5": z_low, "x_width": x_w},
        "local_bed_z": bed, "synthetic": variants,
        "target": "synthetic spec box within +-20 % of the real slot count",
    }
    main_v = variants[0]
    out["verdict"] = {"ratio": main_v["ratio_to_real_slots"],
                      "pass_pm20": bool(abs(main_v["ratio_to_real_slots"] - 1) <= 0.2)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"real: slots mean {n_real.mean():.1f} (min {n_real.min()}, max {n_real.max()}), beams {f_real.mean():.1f}; "
          f"object x_mid {x_mid:.2f} d_front {d_front:.2f} z {z_low:.2f}..{z_top:.2f} width {x_w:.2f}; local bed z {bed}")
    for v in variants:
        print(f"  {v['name']:48s} slots {v['slots_mean']:6.1f} ({v['slots_min']}-{v['slots_max']})  beams {v['beams_mean']:5.1f} "
              f"rings {v['rings_median']:4.1f}  ratio {v['ratio_to_real_slots']:.3f}  size {v['size_dlh']}")
    print("real per-ring mean slots", {k: round(v, 1) for k, v in out["real"]["per_ring_mean"].items()})
    print("spec per-ring mean slots", {k: round(v, 1) for k, v in variants[0]["per_ring_mean"].items()})
    print("verdict", out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
