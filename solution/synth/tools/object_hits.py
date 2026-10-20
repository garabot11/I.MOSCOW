"""Per-scene, per-object statistics of labelled hits (slots per frame) for one or more sets.

    python solution/synth/tools/object_hits.py val_hesai_far_dark val_hesai_far_dark_div \
        --out solution/synth/results/far_dark.json
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.environ.get("IMOS_SYNTHETIC", "/home/gag/i.mos/synthetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    for s in args.sets:
        for p in sorted(glob.glob(os.path.join(ROOT, s, "*", "labels.jsonl"))):
            sdir = os.path.dirname(p)
            sj = json.load(open(os.path.join(sdir, "scene.json")))
            per = {}
            for line in open(p):
                for o in json.loads(line)["objects"]:
                    per.setdefault(o["id"], []).append(o)
            for oid, obs in per.items():
                n = np.array([o["n_hits"] for o in obs])
                rows.append({"set": s, "scene": os.path.basename(sdir), "tunnel": sj["tunnel"]["type"],
                             "divergence_deg": sj.get("divergence_deg"), "object_id": oid, "class": obs[0]["class"],
                             "d_along_m": round(float(np.median([o["d_along_m"] for o in obs])), 2),
                             "distance_m": round(float(np.median([o["distance_m"] for o in obs])), 2),
                             "reflectivity": round(obs[0]["reflectivity"], 3), "top_above_rail_m": round(obs[0]["top_above_rail_m"], 2),
                             "frames": len(obs), "slots_per_frame_mean": round(float(n.mean()), 2),
                             "slots_per_frame_median": float(np.median(n)), "slots_min": int(n.min()), "slots_max": int(n.max()),
                             "frac_frames_ge1": round(float(np.mean(n >= 1)), 3), "frac_frames_ge3": round(float(np.mean(n >= 3)), 3)})
                r = rows[-1]
                print(f"{r['scene']:32s} {r['class']:16s} d {r['d_along_m']:6.1f} rho {r['reflectivity']:.3f} div {r['divergence_deg']}"
                      f"  slots/frame mean {r['slots_per_frame_mean']:5.2f} median {r['slots_per_frame_median']:4.1f} [{r['slots_min']}-{r['slots_max']}]"
                      f"  >=1: {r['frac_frames_ge1']:.2f}  >=3: {r['frac_frames_ge3']:.2f}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
