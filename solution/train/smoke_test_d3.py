"""SMOKE TEST of the Д3 pipeline -- proves that it runs end to end; its numbers are NOT quality metrics.

    python solution/train/smoke_test_d3.py [--bag /home/gag/i.mos/dataset/for_hackathon/roundT_doubleT]
                                           [--frames 12] [--steps 60] [--out /home/gag/i.mos/work/ml_d3/smoke]

1. reads a few REAL frames (read-only) and builds their range images as the node does
   (DetectionPipeline.build_frame + range_image.build_range_image);
2. stamps synthetic boxes into them (pixels of existing slots get a closer return at the box range;
   labels = those slots) -- a tiny in-memory sample, NOT a dataset;
3. runs the real preprocessing (Д1 aux, wide crop, slot targets), dataset + augmentations, the training
   loop (GPU if available, AMP) for a few steps, the metrics, the TorchScript export (meta
   ``smoke_test: true``) into ``--out`` (never into solution/models);
4. loads the export with ``det_nn.NNDetector`` (GPU, with a stub device manager) and runs ``step()``
   on the stamped frames, printing the candidates next to the stamped boxes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

import _paths
import d3_data
import d3_metrics
import d3_preprocess
import d3_scenes
import train_d3

BANNER = "SMOKE TEST (pipeline check on boxes stamped into real frames; not a quality figure)"


def stamp_boxes(ri, rng, n_boxes, R, next_id):
    """Stamp boxes into the near/far returns of ``ri`` (in place). Returns objects and (slots, ids)."""
    objs, slots_all, ids_all, writes = [], [], [], []
    el = ri.el_deg
    az = ri.az_deg
    for _ in range(n_boxes):
        r = float(rng.uniform(20.0, 150.0))
        w, h = float(rng.uniform(0.4, 1.0)), float(rng.uniform(0.4, 1.8))
        az_c, el_c = float(rng.uniform(-8.0, 8.0)), float(rng.uniform(-1.5, 0.3))  # near the horizon: real returns behind
        daz, delv = math.degrees(w / r) / 2, math.degrees(h / r) / 2
        rows = np.flatnonzero(np.abs(el - el_c) <= max(delv, 0.06))
        cols = np.flatnonzero(np.abs(az - az_c) <= max(daz, 0.05))
        if len(rows) == 0 or len(cols) == 0:
            continue
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        ok = (ri.slot_of_pixel[0, rr, cc] >= 0) & (~ri.valid[0, rr, cc] | (ri.rng[0, rr, cc] > r + 0.5))
        rr, cc = rr[ok], cc[ok]
        if len(rr) == 0:
            continue
        oid = next_id + len(objs)
        e, a = np.radians(el[rr]), np.radians(az[cc])
        rng_px = r + rng.normal(0.0, 0.02, len(rr))
        dlh = np.stack([rng_px * np.cos(e) * np.cos(a), rng_px * np.cos(e) * np.sin(a), rng_px * np.sin(e)], 1)
        xyz = (dlh @ R).astype(np.float32)  # analysis -> sensor (R is orthonormal: xyz = R^T dlh)
        for k in (0, 1):
            writes.append((ri.slot_of_pixel[k, rr, cc].copy(), xyz))
            ri.rng[k, rr, cc] = rng_px
            ri.dlh[k, rr, cc] = dlh
            ri.xyz[k, rr, cc] = xyz
            ri.valid[k, rr, cc] = True
            ri.intensity[k, rr, cc] = 20.0
            slots_all.append(ri.slot_of_pixel[k, rr, cc])
            ids_all.append(np.full(len(rr), oid))
        objs.append({"id": oid, "class": "box", "is_obstacle": True, "in_clearance": True, "visible": True,
                     "distance_m": float(rng_px.min()), "n_hits": int(len(rr)), "az_deg": az_c, "el_deg": el_c,
                     "size_m": [w, h]})
    if slots_all:
        s = np.concatenate(slots_all).astype(np.int64)
        o = np.concatenate(ids_all).astype(np.int64)
        order = np.argsort(s)
        return objs, (s[order], o[order]), writes
    return objs, None, writes


def write_back(frame, writes) -> None:
    """Put the stamped returns into the Frame too, so that Д1 (aux channels) sees the boxes."""
    for slots, xyz in writes:
        pos = np.searchsorted(frame.slot, slots) if frame.slot is not None else slots
        frame.xyz[pos] = xyz


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("--bag", default="/home/gag/i.mos/dataset/for_hackathon/roundT_doubleT")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--boxes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(_paths.WORK_DIR, "smoke"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true", help="disable augmentations (pure fit/capacity check)")
    args = ap.parse_args(argv)
    print("=" * 100 + f"\n{BANNER}\n" + "=" * 100, flush=True)
    import torch

    from metro_obstacle_detector import det_nn, nn_input
    from metro_obstacle_detector.nn_model import CLASSES, build_model, export_d3

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    from metro_obstacle_detector.sensor_profile import axis_rotation

    proc = d3_preprocess.FrameProcessor(d3_preprocess.PreprocessOptions(aux="d1"))
    feats, target, objid, objects, ris, stamped = [], [], [], [], [], []
    t0 = time.time()
    for i, stamp, msg in d3_scenes.iter_messages(args.bag):
        if i >= args.frames:
            break
        frame = proc.frame(msg)
        ri = proc.range_image(frame)
        R = axis_rotation(frame.profile.forward_axis, frame.profile.up_axis)
        objs, fslots, writes = stamp_boxes(ri, rng, args.boxes, R, next_id=100 * i)
        write_back(frame, writes)
        aux = proc.aux(frame, ri)
        f, t, o, ob, _ = proc.process(ri, aux, {"frame": i, "objects": objs}, fslots)
        feats.append(f)
        target.append(t)
        objid.append(o)
        objects.append({"frame": i, "objects": ob})
        ris.append((ri, aux))
        stamped.append(objs)
    print(f"[smoke] {len(feats)} real frames from {args.bag}, {sum(len(s) for s in stamped)} stamped boxes, "
          f"{int(sum((t == 1).sum() for t in target))} obstacle pixels in the wide crop, {time.time() - t0:.1f} s")
    cache = d3_data.SceneCache.from_arrays(np.stack(feats), np.stack(target), np.stack(objid), objects,
                                           {"smoke_test": True})
    aug = d3_data.AugConfig(shift=False, flip=0.0, max_drop=0.0, intensity=False, aux_drop=0.0, hist_drop=0.0,
                            partial_obj=0.0, dark_obj=0.0) if args.no_aug else None
    train_ds = d3_data.D3Dataset([cache], 256, train=True, aug=aug, seed=args.seed)
    val_ds = d3_data.D3Dataset([cache], 256, train=False)
    targs = train_d3.build_parser().parse_args([])
    targs.epochs = 10_000
    targs.max_steps = args.steps
    targs.batch = 4
    targs.workers = 0
    targs.eval_every = max(1, args.steps // max(len(train_ds) // 4, 1))
    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(args.seed)
    model = build_model().to(device)
    t1 = time.time()
    best, history = train_d3.fit(model, train_ds, val_ds, targs, device, lambda m: print("[smoke]", m, flush=True), None)
    print(f"[smoke] {history[-1]['step']} optimizer steps on {device} in {time.time() - t1:.1f} s; "
          f"loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")
    model.load_state_dict(best["state"])
    res = train_d3.evaluate(model, val_ds, device, 0.5, 4)["0.5"]
    print("[smoke] metrics on the SAME stamped frames (SMOKE TEST, not a quality metric):\n" + d3_metrics.format_table(res))
    meta = {"crop": nn_input.CropConfig().to_dict(), "channels": list(nn_input.CHANNELS), "history": 3,
            "classes": list(CLASSES), "range_image_builder": nn_input.RANGE_IMAGE_BUILDER_ID, "obstacle_definition": "object",
            "aux_channels_trained": ["h_above_rail", "lateral_from_centreline"], "thresholds": {"obstacle": 0.5, "dust": 0.5},
            "smoke_test": True, "trained_on": {"smoke_test": f"{len(feats)} frames of {os.path.basename(args.bag)} with stamped boxes"}}
    cpu_model = build_model(model.config)
    cpu_model.load_state_dict(best["state"])
    out_pt = os.path.join(args.out, "d3_unet_SMOKE.pt")
    paths = export_d3(cpu_model, out_pt, meta)
    train_d3.write_model_card(os.path.join(args.out, "model_card_SMOKE.md"),
                              dict(meta, model_config=cpu_model.config, n_parameters=sum(p.numel() for p in cpu_model.parameters()),
                                   torch_version=torch.__version__), {"smoke (same frames)": res})
    print(f"[smoke] exported {paths}")

    class StubDM:  # interface-C stand-in for this smoke test only
        def __init__(self):
            self.device = device
            self.mode = "gpu" if device.type == "cuda" else "cpu_forced"
            self.failures = []

        def report_failure(self, exc):
            self.failures.append(repr(exc))
            self.mode = "cpu_emergency"

    dm = StubDM()
    det = det_nn.NNDetector(dm, out_pt)
    det.warmup()  # GPU copy + warm-up before the first frame
    print(f"[smoke] NNDetector: {det.status_dict()}")
    hits = total = 0
    for (ri, aux), objs in zip(ris, stamped):
        cands = det.step(ri, aux)   # Д1 aux of the same frame, as at runtime
        obst = [c for c in cands if c.cls == "obstacle"]
        for o in objs:
            total += 1
            hit = any(abs(c.distance_m - o["distance_m"]) < 1.0 for c in obst)
            hits += int(hit)
    print(f"[smoke] det_nn.step on the stamped frames: {hits}/{total} stamped boxes matched by a candidate "
          f"within 1 m (SMOKE TEST), last timings {det.last_timings_ms}, device {det.device_name}, "
          f"GPU failures {dm.failures}")
    print(f"[smoke] last frame candidates: {[(c.cls, round(c.distance_m, 2), c.n_support, round(c.score, 3)) for c in cands][:8]}")
    print(f"[smoke] last frame stamped boxes: {[(round(o['distance_m'], 2), o['n_hits']) for o in stamped[-1]]}")
    with open(os.path.join(args.out, "smoke_result.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": BANNER, "frames": len(feats), "steps": history[-1]["step"], "device": str(device),
                   "loss_first": history[0]["loss"], "loss_last": history[-1]["loss"], "metrics_same_frames": res,
                   "det_nn_boxes_matched": [hits, total], "export": paths}, fh, indent=1)
    print("=" * 100 + f"\n{BANNER}: finished\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
