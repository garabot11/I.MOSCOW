"""Precompute the ego-motion caches (fake_scenes.ego_motion) of the moving scenes of a fake_style config, so that
several motion estimates (1 CPU core, ~200 MB, ~10 min per 150-frame scene) run in parallel before the single
GPU rendering process starts.  CPU only (no Warp).

    python solution/synth/tools/precompute_motion.py --config configs/val_fake_style.yaml --list
    python solution/synth/tools/precompute_motion.py --config configs/val_fake_style.yaml --scene 1
    python solution/synth/tools/precompute_motion.py --config ... --list | xargs -P 3 -I{} python ... --scene {}

A cached scene is skipped (same file name as the generator: results/fake_style/motion_<base>_<a>_<b>.npy).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.fake_scenes import BASES, CACHE, ego_motion  # noqa: E402
from synth.realbag import RealBag  # noqa: E402


def scenes_of(cfg):
    sc = cfg["scenes"]
    if "list" not in sc:
        raise SystemExit("only configs with scenes.list (random plans draw their windows at generation time)")
    return sc["list"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--list", action="store_true", help="print the indices of the moving scenes without a cache")
    ap.add_argument("--scene", type=int)
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lst = scenes_of(cfg)
    if a.list:
        for i, spec in enumerate(lst):
            if spec.get("standing"):
                continue
            f0, f1 = int(spec["frames"][0]), int(spec["frames"][1])
            if not os.path.exists(os.path.join(CACHE, f"motion_{spec['base']}_{f0}_{f1}.npy")):
                print(i)
        return 0
    spec = lst[a.scene]
    if spec.get("standing"):
        print(f"scene {a.scene}: standing, no motion cache needed")
        return 0
    f0, f1 = int(spec["frames"][0]), int(spec["frames"][1])
    frames = list(range(f0, f1 + 1))
    t0 = time.time()
    s = ego_motion(RealBag(BASES[spec["base"]][0]), frames, spec["base"], False)
    st = np.diff(s)
    print(f"scene {a.scene} {spec['base']} {f0}-{f1}: total {s[-1]:.2f} m, step min/max/mean {st.min():.3f}/{st.max():.3f}/"
          f"{st.mean():.3f} m, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
