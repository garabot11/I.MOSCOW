"""Realism numbers of parametric Hesai scenes vs the real bags (and the return-model calibration).

    python solution/synth/tools/realism_hesai.py [--set k=v ...] [--out solution/synth/results/realism_hesai.json]

Compared on fired slots of the 307 200-point layout:
* no-return fraction per ring band (rings 0-24 up, 25-88 dense band, 89-127 down) and per ring;
* returns per surface class (floor h < -1.1 m, wall |l| > 1.6 m & -1.1 < h < 2.0, ceiling h > 2.0 relative to
  the sensor) in range bands, and their intensity quantiles;
* the farthest return along the track axis (|l| < 3 m).
Real: frames of roundT_squareT_pressureGate_squareT (mostly straight; round and rectangular sections).
Synthetic: straight circular and rectangular tunnels rendered with the same beam model, mount 1.3 m.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src", "metro_obstacle_detector")))
from metro_obstacle_detector.beam_model import hesai128  # noqa: E402

BANDS = {"up": (0, 25), "dense": (25, 89), "down": (89, 128)}
RBINS = [(2, 10), (10, 30), (30, 80), (80, 210)]


def frame_stats(xyz, inten, bm, fired):
    R = bm.axis_matrix()
    dlh = xyz.astype(np.float64) @ R.T
    r = np.linalg.norm(xyz, axis=1)
    valid = r > 0
    ring = np.arange(len(r)) % 128
    out = {"noret_ring": [], "cls": {}}
    for k in range(128):
        m = fired & (ring == k)
        out["noret_ring"].append(float(1 - valid[m].mean()) if m.any() else np.nan)
    d, l, h = dlh.T
    cls = {"floor": valid & (h < -1.1), "wall": valid & (np.abs(l) > 1.6) & (h >= -1.1) & (h < 2.0),
           "ceiling": valid & (h >= 2.0)}
    for name, m in cls.items():
        for lo, hi in RBINS:
            mm = m & (r >= lo) & (r < hi)
            out["cls"][f"{name}_{lo}_{hi}"] = {"n": int(mm.sum()), "inten": inten[mm].tolist() if mm.sum() < 200000 else []}
    ax = valid & (np.abs(l) < 3.0) & (d > 0)
    out["far_d"] = float(np.percentile(d[ax], 99.9)) if ax.any() else 0.0
    out["valid_frac_fired"] = float(valid[fired].mean())
    rr = r.reshape(-1, 2, 128)
    both = (rr[:, 0] > 0) & (rr[:, 1] > 0)
    out["dual_diff_frac"] = float((both & (np.abs(rr[:, 0] - rr[:, 1]) > 0.01)).sum() / max(1, (rr[:, 1] > 0).sum()))
    return out


def summarise(stats_list):
    ring = np.nanmean(np.array([s["noret_ring"] for s in stats_list]), axis=0)
    res = {"noret_band": {b: float(np.nanmean(ring[a:c])) for b, (a, c) in BANDS.items()},
           "noret_ring": ring.round(4).tolist(),
           "valid_frac_fired": float(np.mean([s["valid_frac_fired"] for s in stats_list])),
           "dual_diff_frac": float(np.mean([s["dual_diff_frac"] for s in stats_list])),
           "far_d_median": float(np.median([s["far_d"] for s in stats_list])), "classes": {}}
    for key in stats_list[0]["cls"]:
        n = np.mean([s["cls"][key]["n"] for s in stats_list])
        it = np.concatenate([np.asarray(s["cls"][key]["inten"]) for s in stats_list])
        q = np.percentile(it, [10, 50, 90]).round(1).tolist() if len(it) else None
        res["classes"][key] = {"n_per_frame": round(float(n), 1), "intensity_p10_50_90": q}
    return res


def real_stats(bm, fired, frames=(20, 80, 140, 200, 300, 380, 450, 520)):
    from synth.realbag import RealBag

    b = RealBag("roundT_squareT_pressureGate_squareT")
    out = []
    for i in frames:
        c = b.frame(i).cloud
        xyz = np.stack([c["x"], c["y"], c["z"]], 1)
        out.append(frame_stats(xyz, np.asarray(c["intensity"]), bm, fired))
    return out


def synth_stats(bm, fired, overrides, kinds=("circular", "rectangular"), frames_per=4, seed=5):
    from synth.parametric import ParametricScene
    from synth.raycast import Caster

    caster = Caster()
    out = []
    for kind in kinds:
        cfg = {"sensor": "hesai128", "layout": 2400, "speed_mps": [10.0, 10.0], "p_stationary": 0.0,
               "edge_mix_p": overrides.get("edge_mix_p", 0.35),
               "tunnel": {"kinds": [kind], "p_curve": 0.0, "p_platform": 0.0, "length_m": 400, "grade": False},
               "return_model": overrides.get("return_model", {})}
        rng = np.random.default_rng(seed)
        sc = ParametricScene(cfg, rng, caster, frames_per)
        if "rho_scale" in overrides:
            sc.rho = sc.rho * overrides["rho_scale"] / sc.spec.rho_scale
        rr = np.random.default_rng(seed + 1)
        for j in range(frames_per):
            pts, *_ = sc.render(j, 0, rr)
            xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1)
            out.append(frame_stats(xyz, np.asarray(pts["intensity"]), bm, fired))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="return_model key=value or rho_scale=value")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "realism_hesai.json"))
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args(argv)
    ov = {"return_model": {}}
    for kv in args.set:
        k, v = kv.split("=")
        if k in ("rho_scale", "edge_mix_p"):
            ov[k] = float(v)
        else:
            ov["return_model"][k] = float(v)
    bm = hesai128(2400)
    fired = bm.fired_mask()
    rep = {"overrides": ov}
    cache = os.path.join(os.path.dirname(HERE), "results", "realism_hesai_real.json")
    if os.path.exists(cache):
        rep["real"] = json.load(open(cache))
    elif not args.no_real:
        rep["real"] = summarise(real_stats(bm, fired))
        json.dump(rep["real"], open(cache, "w"), indent=1)
    rep["synthetic"] = summarise(synth_stats(bm, fired, ov))
    json.dump(rep, open(args.out, "w"), indent=1)
    R, S = rep.get("real"), rep["synthetic"]
    print("no-return fraction of fired slots (real / synthetic):")
    for b in BANDS:
        print(f"  {b:6s} {R['noret_band'][b]:.3f} / {S['noret_band'][b]:.3f}")
    print(f"  valid fraction of fired slots {R['valid_frac_fired']:.3f} / {S['valid_frac_fired']:.3f};  far d (p99.9 on axis) "
          f"{R['far_d_median']:.1f} / {S['far_d_median']:.1f}")
    print(f"  beams with two different echoes {R['dual_diff_frac']:.4f} / {S['dual_diff_frac']:.4f}")
    print("returns per frame and intensity p10/50/90 (real | synthetic):")
    for key in R["classes"]:
        a, b = R["classes"][key], S["classes"][key]
        print(f"  {key:18s} n {a['n_per_frame']:8.1f} | {b['n_per_frame']:8.1f}   I {a['intensity_p10_50_90']} | {b['intensity_p10_50_90']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
