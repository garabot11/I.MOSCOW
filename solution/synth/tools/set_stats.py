"""Content statistics of generated sets (markdown table for the README / reports).

    python solution/synth/tools/set_stats.py /home/gag/i.mos/synthetic [--json out.json]
"""
import argparse
import glob
import json
import os
from collections import Counter

BANDS = [(0, 50), (50, 100), (100, 150), (150, 210), (210, 1000)]


def set_stats(set_dir):
    st = {"set": os.path.basename(set_dir), "scenes": 0, "frames": 0, "pos": 0, "neg": 0, "GB": 0.0,
          "class_frames": Counter(), "visible_obstacle_frames_by_band": Counter(), "dust_scenes": 0,
          "single_return_scenes": 0, "tunnels": Counter(), "curves": 0, "moving": 0, "sensor": set()}
    for sj_path in sorted(glob.glob(os.path.join(set_dir, "*", "scene.json"))):
        sdir = os.path.dirname(sj_path)
        sj = json.load(open(sj_path))
        st["scenes"] += 1
        st["sensor"].add(sj["sensor"])
        st["pos" if "_pos" in os.path.basename(sdir) else "neg"] += 1
        st["GB"] += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(sdir) for f in fs) / 1e9
        st["tunnels"][sj["tunnel"]["type"] if sj["source"] == "parametric" else sj["real_bag"]] += 1
        if sj["source"] == "parametric":
            st["curves"] += sj["tunnel"].get("curve_radius_m") is not None
            st["moving"] += sj["speed_mps"] > 0
        st["dust_scenes"] += bool(sj.get("dust"))
        st["single_return_scenes"] += bool(sj.get("single_return"))
        for line in open(os.path.join(sdir, "labels.jsonl")):
            L = json.loads(line)
            st["frames"] += 1
            for o in L["objects"]:
                st["class_frames"][o["class"]] += 1
                if o["is_obstacle"] and o["n_hits"] >= 3:
                    b = next(b for b in BANDS if b[0] <= o["distance_m"] < b[1])
                    st["visible_obstacle_frames_by_band"][f"{b[0]}-{b[1]}"] += 1
    st["sensor"] = sorted(st["sensor"])
    st["GB"] = round(st["GB"], 2)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [set_stats(d) for d in sorted(glob.glob(os.path.join(args.root, "*")))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "scene.json")) and not os.path.basename(d).startswith(("_", "infra"))]
    print("| set | sensor | scenes (pos/neg) | frames | GB | object-frames by class | visible obstacle-frames by distance (m) | other |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        cls = ", ".join(f"{k} {v}" for k, v in sorted(r["class_frames"].items()))
        band = ", ".join(f"{b[0]}-{b[1]}: {r['visible_obstacle_frames_by_band'].get(f'{b[0]}-{b[1]}', 0)}" for b in BANDS)
        other = (f"backgrounds/tunnels {dict(r['tunnels'])}; curves {r['curves']}, moving {r['moving']}; "
                 f"dust scenes {r['dust_scenes']} (single-return {r['single_return_scenes']})")
        print(f"| {r['set']} | {'/'.join(r['sensor'])} | {r['scenes']} ({r['pos']}/{r['neg']}) | {r['frames']} | {r['GB']} | {cls} | {band} | {other} |")
    if args.json:
        json.dump([dict(r, class_frames=dict(r["class_frames"]), tunnels=dict(r["tunnels"]),
                        visible_obstacle_frames_by_band=dict(r["visible_obstacle_frames_by_band"])) for r in rows],
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
