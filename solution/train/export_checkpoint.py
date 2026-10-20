"""Export a training checkpoint (``last.pt`` / ``best.pt`` of a run) as a Д3 model, with the meta of an
already exported model of the same run (crop, channels, builder, aux, training data).

    python solution/train/export_checkpoint.py --checkpoint <run>/last.pt --like <run>/export/d3_unet.pt \\
        --out <run>/export_last/d3_unet.pt [--note "last epoch"]
"""
from __future__ import annotations

import argparse
import sys

import _paths  # noqa: F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--like", required=True, help="exported model whose meta is reused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    import torch

    from metro_obstacle_detector import det_nn
    from metro_obstacle_detector.nn_model import build_model, export_d3

    _, meta, _ = det_nn.load_model_cpu(args.like)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ck.get("format") != det_nn.STATE_FORMAT:
        raise SystemExit(f"{args.checkpoint} is not a Д3 training checkpoint")
    model = build_model(ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    meta = dict(meta, checkpoint=args.checkpoint, checkpoint_epoch=ck.get("epoch"), note=args.note)
    for k in ("model_config", "n_parameters", "torch_version"):
        meta.pop(k, None)
    print(export_d3(model, args.out, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
