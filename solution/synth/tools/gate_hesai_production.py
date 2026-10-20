"""Gate through the production renderer: the spec box of ``gate_hesai.py`` rendered with ``overlay_objects`` and the
calibrated ``ReturnModel.hesai()`` (detection probability, 2.5 % dropout, noise, dual-return echoes) into the same
background frames, converted with ``hesai_points`` and counted with the same differencing procedure.

    python solution/synth/tools/gate_hesai_production.py   (needs results/gate_hesai.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gate_hesai as G  # noqa: E402
from synth.objects import mesh_box  # noqa: E402
from synth.raycast import Caster  # noqa: E402
from synth.realbag import RealBag  # noqa: E402
from synth.render import SceneObject, hesai_beam_dirs, hesai_beam_fired, hesai_echoes_from_cloud, hesai_points, overlay_objects  # noqa: E402
from synth.returns import ReturnModel  # noqa: E402
from metro_obstacle_detector.beam_model import hesai128  # noqa: E402


def main():
    res_dir = os.path.join(os.path.dirname(HERE), "results")
    gate = json.load(open(os.path.join(res_dir, "gate_hesai.json")))
    spec = gate["synthetic"][0]
    centre = np.array(spec["centre_sensor"])
    half = np.array(spec["size_dlh"]) / 2           # d, l, h -> sensor: l = +x, d = -y, h = +z
    V, F = mesh_box(centre, [half[1], half[0], half[2]])
    bag = RealBag("doubleT_obstacle")
    bm = hesai128(7200)
    caster = Caster()
    ret = ReturnModel.hesai()
    dirs = hesai_beam_dirs(bm)
    beams = np.nonzero(hesai_beam_fired(bm))[0]
    bgs = {p: [G.load(bag, i) for i in idx] for p, idx in G.BG.items()}
    region = tuple(tuple(x) for x in gate["region_dxz"])
    rows = []
    rng = np.random.default_rng(5)
    for k in G.REAL_FRAMES:
        p = k % 2
        host = (k // 2) % len(G.BG[p])
        f = bag.frame(G.BG[p][host])
        e = hesai_echoes_from_cloud(f.cloud, bm)
        overlay_objects(e, [SceneObject(1, "box", V, F, 0.3)], np.zeros(3), dirs, caster, ret, rng, beams=beams)
        pts, src = hesai_points(e, bm, f.stamp_ns, timestamps=f.cloud["timestamp"])
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        others = [b for j, b in enumerate(bgs[p]) if j != host]
        s = G.object_slots(xyz, r, others, region)
        rows.append({"frame": k, "measured_slots": int(len(s)), "labelled_object_slots": int((src > 0).sum())})
    n = np.array([r_["measured_slots"] for r_ in rows])
    lab = np.array([r_["labelled_object_slots"] for r_ in rows])
    out = {"real_slots_mean": gate["real"]["slots_mean"], "production_measured_slots_mean": float(n.mean()),
           "production_labelled_slots_mean": float(lab.mean()), "ratio_to_real": float(n.mean() / gate["real"]["slots_mean"]),
           "reflectivity": 0.3, "return_model": ret.to_dict(), "rows": rows}
    json.dump(out, open(os.path.join(res_dir, "gate_hesai_production.json"), "w"), indent=1)
    print(f"real {gate['real']['slots_mean']:.1f} slots/frame; production renderer: measured {n.mean():.1f} "
          f"(labelled n_hits {lab.mean():.1f}) -> ratio {n.mean() / gate['real']['slots_mean']:.3f}")


if __name__ == "__main__":
    main()
