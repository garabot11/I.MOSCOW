"""Realism numbers of parametric Falcon Prime scenes vs the real recording (and the return-model calibration).

    python solution/synth/tools/realism_falcon.py [--set key=value ...]

Real: frames 1000, 2000, 3000 of falcon_test_second_dark (train stationary, straight double-track tunnel, no
person yet).  Synthetic: straight double-track tunnel mirrored like the recording (near wall 3.3 m left,
column row 1.8 m right, far wall 6.6 m right), mount 1.98 m above rail (2.03 m above the bed), pitch 2.33 deg.
Points are classified in a pitch-corrected frame (h' = z + 0.0407 d): floor h' < -1.7, ceiling h' > 2.0,
left wall y > 2.5, right side y < -1.2 (columns and far wall); counts, intensity and range quantiles per class
and range band, the fraction of table slots without a return, second-return fraction.
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

BAG = "/home/gag/i.mos/dataset/falcon/falcon_test_second_dark"
RBINS = [(1.5, 10), (10, 30), (30, 80), (80, 200), (200, 500)]
SLOPE = 0.0407


def classify(x, y, z):
    hp = z + SLOPE * x
    return {"floor": hp < -1.7, "ceiling": hp > 2.0, "wall_left": (y > 2.5) & (hp >= -1.7) & (hp <= 2.0),
            "right_side": (y < -1.2) & (hp >= -1.7) & (hp <= 2.0)}


def stats(pts, n_slots):
    x, y, z = (pts[k].astype(np.float64) for k in ("x", "y", "z"))
    r = np.sqrt(x * x + y * y + z * z)
    first = pts["is_2nd_return"] == 0
    out = {"points": int(len(pts)), "first": int(first.sum()), "second_frac": float((~first).mean()),
           "noret_frac": float(1 - first.sum() / n_slots), "cls": {}}
    cls = classify(x, y, z)
    for name, m in cls.items():
        for lo, hi in RBINS:
            mm = m & first & (r >= lo) & (r < hi)
            out["cls"][f"{name}_{int(lo)}_{hi}"] = {"n": int(mm.sum()), "inten": pts["intensity"][mm].tolist()}
    ax = first & (np.abs(y) < 3.0)
    out["corridor_beyond"] = {str(d): int((ax & (x > d)).sum()) for d in (100, 150, 200)}
    out["corridor_max_x"] = float(x[ax].max()) if ax.any() else 0.0
    return out


def summarise(rows):
    res = {k: float(np.mean([r[k] for r in rows])) for k in ("points", "first", "second_frac", "noret_frac", "corridor_max_x")}
    res["corridor_beyond"] = {d: float(np.mean([r["corridor_beyond"][d] for r in rows])) for d in rows[0]["corridor_beyond"]}
    res["classes"] = {}
    for key in rows[0]["cls"]:
        it = np.concatenate([np.asarray(r["cls"][key]["inten"]) for r in rows])
        res["classes"][key] = {"n_per_frame": round(float(np.mean([r["cls"][key]["n"] for r in rows])), 1),
                               "intensity_p10_50_90": np.percentile(it, [10, 50, 90]).round(1).tolist() if len(it) else None}
    return res


def main(argv=None):
    from metro_obstacle_detector.beam_model import FalconBeamModel
    from synth.parametric import ParametricScene
    from synth.raycast import Caster
    from synth.realbag import RealBag

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "realism_falcon.json"))
    args = ap.parse_args(argv)
    rm = {}
    extra = {}
    for kv in args.set:
        k, v = kv.split("=")
        (extra if k in ("rho_scale", "mount_height_m", "pitch_deg", "divergence_deg", "edge_mix_p") else rm)[k] = float(v)
    tab = FalconBeamModel.load(os.path.join(os.path.dirname(HERE), "data", "falcon_prime_seyond.npz"))
    cache = os.path.join(os.path.dirname(HERE), "results", "realism_falcon_real.json")
    if os.path.exists(cache):
        real = json.load(open(cache))
    else:
        b = RealBag(BAG)
        real = summarise([stats(b.frame(i).cloud, tab.n_slots) for i in (1000, 2000, 3000)])
        json.dump(real, open(cache, "w"), indent=1)
    cfg = {"sensor": "falcon_prime", "speed_mps": [0, 0], "p_stationary": 1.0,
           "mount_height_m": extra.get("mount_height_m", 1.98), "pitch_deg": extra.get("pitch_deg", 2.33),
           "divergence_deg": extra.get("divergence_deg", 0.1), "edge_mix_p": extra.get("edge_mix_p", 0.35),
           "tunnel": {"kinds": ["double"], "p_curve": 0.0, "p_platform": 0.0, "length_m": 600, "grade": False},
           "return_model": rm}
    caster = Caster()
    rng = np.random.default_rng(11)
    sc = ParametricScene(cfg, rng, caster, 3)
    # rebuild the tunnel mirrored as in the recording
    from synth.tunnel import build_tunnel

    sc.spec.double_side, sc.spec.double_wall_m, sc.spec.double_column_m = -1, 3.3, 1.8
    if "rho_scale" in extra:
        sc.spec.rho_scale = extra["rho_scale"]
    sc.cl, V, F, sc.surf, sc.rho, sc.tinfo = build_tunnel(sc.spec)
    sc.mesh = caster.mesh(V, F)
    rows = []
    rr = np.random.default_rng(3)
    for j in range(3):
        pts, *_ = sc.render(j, 0, rr)
        rows.append(stats(pts, tab.n_slots))
    syn = summarise(rows)
    rep = {"overrides": {"return_model": rm, **extra}, "real": real, "synthetic": syn}
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"points/frame {real['points']:.0f} | {syn['points']:.0f};  no-return frac of table slots {real['noret_frac']:.3f} | "
          f"{syn['noret_frac']:.3f};  2nd-return frac {real['second_frac']:.3f} | {syn['second_frac']:.3f}")
    print(f"corridor |y|<3: points beyond 100/150/200 m {list(real['corridor_beyond'].values())} | {list(syn['corridor_beyond'].values())};"
          f" max x {real['corridor_max_x']:.0f} | {syn['corridor_max_x']:.0f}")
    for key in real["classes"]:
        a, b = real["classes"][key], syn["classes"][key]
        print(f"  {key:22s} n {a['n_per_frame']:8.1f} | {b['n_per_frame']:8.1f}   I {a['intensity_p10_50_90']} | {b['intensity_p10_50_90']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
