"""Quick look: labelled obstacles vs the offline detector run of container_check.sh (not the official scoreboard).

    python solution/synth/tools/quick_baseline.py /home/gag/i.mos/synthetic/val_hesai_param

Per scene: frames with a visible in-clearance obstacle (n_hits >= 3), the nearest such obstacle's distance band,
and the fraction of those frames the detector reported "detected"; frames without obstacles reported detected.
"""
import glob
import json
import os
import sys

BANDS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 1000)]


def main(set_dir):
    name = os.path.basename(os.path.normpath(set_dir))
    chk = os.path.join(os.path.dirname(os.path.normpath(set_dir)), "_checks", name)
    tot = {b: [0, 0] for b in BANDS}
    neg = [0, 0]
    for sdir in sorted(glob.glob(os.path.join(set_dir, "*", "labels.jsonl"))):
        scene = os.path.basename(os.path.dirname(sdir))
        det_path = os.path.join(chk, scene + ".jsonl")
        if not os.path.exists(det_path):
            continue
        labels = [json.loads(l) for l in open(sdir)]
        det = [json.loads(l) for l in open(det_path)]
        per = {b: [0, 0] for b in BANDS}
        n_neg, fp = 0, 0
        for L, D in zip(labels, det):
            obs = [o for o in L["objects"] if o["is_obstacle"] and o["n_hits"] >= 3]
            hit = D["status"] == "detected"
            if obs:
                d = min(o["distance_m"] for o in obs)
                b = next(b for b in BANDS if b[0] <= d < b[1])
                per[b][0] += 1
                per[b][1] += hit
            else:
                n_neg += 1
                fp += hit
        for b in BANDS:
            tot[b][0] += per[b][0]
            tot[b][1] += per[b][1]
        neg[0] += n_neg
        neg[1] += fp
        s = " ".join(f"{b[0]}-{b[1]}:{v[1]}/{v[0]}" for b, v in per.items() if v[0])
        print(f"{scene:34s} obstacle frames (band: detected/frames) {s or '-'} | frames without obstacle: {fp}/{n_neg} detected")
    print("TOTAL", " ".join(f"{b[0]}-{b[1]} m: {v[1]}/{v[0]}" for b, v in tot.items()), f"| no-obstacle frames detected {neg[1]}/{neg[0]}")


if __name__ == "__main__":
    main(sys.argv[1])
