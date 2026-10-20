"""Evaluate a Д3 model on synthetic held-out sets (interface B) -- per range band, as in training.

    python solution/train/eval_cached_d3.py --model solution/models/d3_unet.pt \\
        --sets val_hesai_insert,val_hesai_insert_tunnel,val_falcon_param [--thresholds 0.5,0.3,0.7] [--json out]

Scenes are preprocessed into the training cache if needed (same options as training: range_image
builder, Д1 aux of the same frame, wide crop; falcon_prime scenes through the binned range image).
The model sees the central crop of the cache, exactly as in ``train_d3.evaluate``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import _paths
import d3_data
import d3_metrics
import d3_preprocess
import d3_scenes
import train_d3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True, help="comma-separated set names / fnmatch patterns")
    ap.add_argument("--data", default=_paths.SYNTHETIC_ROOT)
    ap.add_argument("--work", default=_paths.WORK_DIR)
    ap.add_argument("--thresholds", default="0.5,0.3,0.7")
    ap.add_argument("--preprocess-workers", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    import torch

    from metro_obstacle_detector import det_nn
    from metro_obstacle_detector.nn_input import CropConfig

    model, meta, kind = det_nn.load_model_cpu(args.model)
    crop = CropConfig.from_dict(meta.get("crop"))
    rp = meta.get("range_image_params") or {}
    opts = d3_preprocess.PreprocessOptions(crop=crop.to_dict(), aux="d1" if meta.get("aux_channels_trained") else "none",
                                           obstacle_def=meta.get("obstacle_definition", "object"),
                                           az_step_deg=rp.get("az_step_deg", 0.1),
                                           binned_res_deg=tuple(rp.get("binned_res_deg", (0.09, 0.08))),
                                           az_limit_deg=rp.get("az_limit_deg"))
    cache_root = os.path.join(args.work, "cache")
    scenes = d3_scenes.discover(args.data, [s for s in args.sets.split(",") if s])
    metas = d3_preprocess.preprocess_all(scenes, cache_root, opts, args.preprocess_workers)
    by_set = {}
    for s in scenes:
        if s.key in metas:
            by_set.setdefault(s.set_name, []).append(d3_data.SceneCache(d3_preprocess.cache_dir_for(cache_root, s)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    thr = [float(t) for t in args.thresholds.split(",")]
    out = {"model": args.model, "model_kind": kind, "trained_on": meta.get("trained_on"), "sets": {}}
    for name, caches in sorted(by_set.items()):
        ds = d3_data.D3Dataset(caches, crop.n_cols, train=False)
        res = train_d3.evaluate(model, ds, device, thr[0], 32, 0, thresholds=tuple(thr[1:]))
        out["sets"][name] = {"scenes": len(caches), "frames": len(ds), "metrics": res}
        print(f"### {name} ({len(caches)} scenes, {len(ds)} frames) @ {thr[0]}\n" + d3_metrics.format_table(res[str(thr[0])]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
