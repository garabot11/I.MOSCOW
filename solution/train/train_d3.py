"""Train the Д3 range-image U-Net on the synthetic dataset (interface B) and export it.

Real training -- one command, refuses to run (exit 2) while no ``<data>/train_*`` scene exists:

    /home/gag/miniconda3/envs/imos-ml/bin/python solution/train/train_d3.py \\
        --data /home/gag/i.mos/synthetic --export solution/models/d3_unet.pt

Defaults: train sets ``train_*`` (scenes split by a seeded permutation, 15 % of the scenes for
validation), held-out sets ``val_*`` evaluated separately, cache + runs under
``/home/gag/i.mos/work/ml_d3`` (outside git), 40 epochs, batch 32, AdamW + one-cycle LR, AMP on the
GPU, loss = focal (gamma 2, class weights 1/4/2) + soft Dice of obstacle (1.0) and dust (0.5), far
obstacle pixels up-weighted (w = 1 + r / 100 m).
Outputs: ``<work>/runs/<name>/{train_log.jsonl, last.pt, best.pt, final_metrics.json}``; with
``--export``: TorchScript ``d3_unet.pt`` + ``d3_unet.state.pt`` + ``model_card.md`` next to it.
Range images are built exactly as at runtime (DetectionPipeline.build_frame + range_image.build_range_image;
organised Hesai and angle-binned falcon_prime scenes); incomplete scenes are skipped and listed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import _paths
import d3_data
import d3_metrics
import d3_preprocess
import d3_scenes


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", _paths.SOLUTION_DIR, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the Д3 U-Net on the synthetic dataset.")
    ap.add_argument("--data", default=_paths.SYNTHETIC_ROOT)
    ap.add_argument("--train-sets", default="train_*", help="comma-separated fnmatch patterns of set names")
    ap.add_argument("--val-sets", default="val_*", help="held-out sets, evaluated separately ('' = none)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of TRAIN scenes (or backgrounds) held out")
    ap.add_argument("--split-by", choices=("scene", "background"), default="scene",
                    help="hold out whole scenes, or whole backgrounds (insert scenes grouped by real_bag)")
    ap.add_argument("--holdout-backgrounds", default="",
                    help="comma-separated real_bag names never used for training (their train scenes go to the "
                         "validation split; held-out sets on them then measure UNSEEN backgrounds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default=_paths.WORK_DIR)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = off)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers")
    ap.add_argument("--preprocess-workers", type=int, default=6)
    ap.add_argument("--aux", choices=("d1", "none"), default="d1", help="aux channels from Д1's track model")
    ap.add_argument("--obstacle-def", choices=d3_preprocess.OBSTACLE_DEFS, default="object")
    ap.add_argument("--margin-cols", type=int, default=32, help="azimuth-shift augmentation margin (columns)")
    ap.add_argument("--az-step", type=float, default=0.1, help="build_range_image az_step_deg (match the runtime)")
    ap.add_argument("--binned-res", default="0.09,0.08", help="build_range_image binned_res_deg (az, el)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--class-weights", default="1,4,2")
    ap.add_argument("--far-weight", type=float, default=1.0)
    ap.add_argument("--dice-obstacle", type=float, default=1.0, help="weight of the soft-Dice term (obstacle)")
    ap.add_argument("--dice-dust", type=float, default=0.5, help="weight of the soft-Dice term (dust)")
    ap.add_argument("--widths", default="24,48,96,160")
    ap.add_argument("--decoder", default="add", choices=("add", "concat"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--export", default=None, help="TorchScript output path, e.g. solution/models/d3_unet.pt")
    ap.add_argument("--max-scenes", type=int, default=0, help="debug: use at most this many train scenes")
    ap.add_argument("--real-negatives", default="",
                    help="comma-separated /path/to/real_bag[:start:stop]: REAL frames of bags annotated as obstacle-free, "
                         "trained as pure negatives (all background)")
    ap.add_argument("--force-preprocess", action="store_true")
    return ap


class Logger:
    def __init__(self, path: Optional[str]):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def background_of(scene) -> str:
    """Background identity: the real bag of insert scenes; parametric / other scenes are their own background."""
    rb = scene.meta.get("real_bag")
    return f"real:{rb}" if rb else f"scene:{scene.key}"


def split_by_background(scenes, val_frac: float, seed: int, holdout: Sequence[str] = ()):
    """(train keys, val keys): whole backgrounds go to validation; ``holdout`` real bags always do."""
    groups: Dict[str, List[str]] = {}
    for s in scenes:
        groups.setdefault(background_of(s), []).append(s.key)
    forced = {f"real:{h}" for h in holdout if h}
    free = sorted(g for g in groups if g not in forced)
    rng = np.random.default_rng(seed)
    perm = [free[i] for i in rng.permutation(len(free))]
    n_total = sum(len(v) for v in groups.values())
    val_groups = [g for g in groups if g in forced]
    n_val = sum(len(groups[g]) for g in val_groups)
    for g in perm:  # add backgrounds until the validation share is reached (keep at least one for training)
        if n_val >= val_frac * n_total or len(val_groups) >= len(groups) - 1:
            break
        val_groups.append(g)
        n_val += len(groups[g])
    va = sorted(k for g in val_groups for k in groups[g])
    tr = sorted(k for g in groups if g not in val_groups for k in groups[g])
    return tr, va


def split_scenes(keys: Sequence[str], val_frac: float, seed: int):
    keys = sorted(keys)
    rng = np.random.default_rng(seed)
    perm = [keys[i] for i in rng.permutation(len(keys))]
    n_val = int(round(val_frac * len(keys)))
    if len(keys) >= 2:
        n_val = min(max(n_val, 1), len(keys) - 1)
    else:
        n_val = 0
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def evaluate(model, ds, device, threshold: float, batch: int = 32, workers: int = 0,
             thresholds: Sequence[float] = ()) -> Dict[str, dict]:
    import torch

    accs = {float(t): d3_metrics.MetricAccumulator(t) for t in (threshold, *thresholds)}
    if len(ds) == 0:
        return {str(t): a.result() for t, a in accs.items()}
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    model.eval()
    with torch.inference_mode():
        for b in dl:
            x = b["x"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            y, o, r = b["y"].numpy(), b["obj"].numpy(), b["rng"].numpy()
            for a in accs.values():
                a.update(p, y, o, r)
    return {str(t): a.result() for t, a in accs.items()}


def fit(model, train_ds, val_ds, args, device, log, run_dir: Optional[str]):
    """Training loop with AMP, one-cycle LR, per-epoch validation, last/best checkpoints."""
    import torch

    from metro_obstacle_detector.det_nn import STATE_FORMAT

    cw = [float(v) for v in args.class_weights.split(",")]
    amp = device.type == "cuda" and not args.no_amp
    dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) > args.batch,
                                     num_workers=args.workers, pin_memory=device.type == "cuda",
                                     generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = max(len(dl), 1)
    total = args.epochs * steps_per_epoch
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 1),
                                                pct_start=0.05 if total > 100 else 0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = {"score": -1e9, "epoch": -1, "state": None, "metrics": None}
    step = 0
    history = []
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        losses = []
        for b in dl:
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            r = b["rng"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = d3_data.d3_loss(logits, y, r, cw, args.focal_gamma, args.far_weight, args.dice_obstacle, args.dice_dust)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if step >= total:
                break
        entry = {"epoch": epoch, "step": step, "loss": float(np.mean(losses)) if losses else None,
                 "lr": float(sched.get_last_lr()[0]), "seconds": round(time.time() - t0, 1)}
        if val_ds is not None and len(val_ds) and ((epoch + 1) % args.eval_every == 0 or step >= total
                                                   or epoch == args.epochs - 1):
            res = evaluate(model, val_ds, device, args.threshold, args.batch, min(args.workers, 4))[str(float(args.threshold))]
            entry["val"] = {"score": d3_metrics.selection_score(res),
                            "mean_object_recall_20_300": res["mean_object_recall_20_300"],
                            "pixel_iou_20_300": res["pixel_iou_20_300"],
                            "fp_components_per_frame": res["fp_components_per_frame"]}
            if entry["val"]["score"] > best["score"]:
                best = {"score": entry["val"]["score"], "epoch": epoch, "metrics": res,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append(entry)
        log(f"epoch {epoch} step {step}: {json.dumps(entry)}")
        if run_dir:
            with open(os.path.join(run_dir, "train_log.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            ck = {"format": STATE_FORMAT, "model_config": dict(model.config), "epoch": epoch,
                  "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": {}}
            torch.save(ck, os.path.join(run_dir, "last.pt"))
            if best["epoch"] == epoch:
                torch.save(dict(ck, meta={"val_score": best["score"]}), os.path.join(run_dir, "best.pt"))
        if step >= total:
            break
    if best["state"] is None:  # no validation data: the last weights
        best = {"score": None, "epoch": history[-1]["epoch"] if history else -1, "metrics": None,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    return best, history


def write_model_card(path: str, meta: dict, metrics: Dict[str, dict], latency: Optional[dict] = None) -> None:
    lines = ["# Д3 model card — range-image U-Net", ""]
    if meta.get("smoke_test"):
        lines += ["**SMOKE TEST MODEL — trained on a few boxes stamped into real frames only to prove that the "
                  "pipeline runs. Not a detector; the numbers below are not quality metrics.**", ""]
    cfg = meta.get("model_config", {})
    lines += [
        f"* created (UTC): {meta.get('created_utc')}, git {meta.get('git_commit')}, torch {meta.get('torch_version')}",
        f"* architecture: RangeUNet widths {cfg.get('widths')}, decoder `{cfg.get('decoder')}`, "
        f"{meta.get('n_parameters')} parameters; input (21, {meta['crop']['n_rows']}, {meta['crop']['n_cols']}) = "
        f"current frame + 2 previous × channels {meta.get('channels')}; output logits {meta.get('classes')}",
        f"* crop: {json.dumps(meta['crop'])}",
        f"* range-image builder used for training: `{meta.get('range_image_builder')}` "
        f"{json.dumps(meta.get('range_image_params'))} — inference must use range images built the same way "
        "(det_nn warns otherwise); sensors: " + str(meta.get("sensors")),
        f"* aux channels populated in training: {meta.get('aux_channels_trained')} (whole-channel dropout "
        f"p = {meta.get('aux_dropout')}); other aux channels are zeroed by det_nn",
        f"* obstacle class definition: `{meta.get('obstacle_definition')}` (see d3_preprocess.target_class); "
        "labels from the near-return slot of each pixel",
        f"* threshold: {meta.get('thresholds')}",
        f"* training data: {json.dumps(meta.get('trained_on'))}",
        f"* hyper-parameters: {json.dumps(meta.get('hyper_parameters'))}",
        "", "## Validation metrics", ""]
    for name, res in metrics.items():
        lines += [f"### {name}", "", d3_metrics.format_table(res), ""]
    if latency:
        lines += ["## Latency", "", "```", json.dumps(latency, indent=1), "```", ""]
    lines += ["## Limitations", "",
              "* trained on synthetic data only (interface B); real-data quality is unknown until checked on the "
              "real bags (the only real positive is the box at 56 m in doubleT_obstacle);",
              "* the network segments foreign objects; whether they are inside the clearance gauge is decided "
              "downstream (fusion);",
              "* residual channel (r_expected) is not populated in training.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    from metro_obstacle_detector import nn_input
    from metro_obstacle_detector.det_nn import MODEL_FORMAT
    from metro_obstacle_detector.nn_model import CLASSES, build_model, count_parameters, export_d3

    train_pats = [p for p in args.train_sets.split(",") if p]
    val_pats = [p for p in args.val_sets.split(",") if p]
    train_scenes = d3_scenes.discover(args.data, train_pats)
    if not train_scenes:
        print(f"ERROR: no training scenes under {args.data} matching {train_pats}. Real Д3 training waits for "
              "the synthetic dataset (interface B); nothing was trained.", file=sys.stderr)
        return 2
    if args.max_scenes:
        train_scenes = train_scenes[: args.max_scenes]
    train_scenes = train_scenes + d3_scenes.real_negative_scenes(args.real_negatives)
    train_keys = {s.key for s in train_scenes}
    val_scenes = [s for s in d3_scenes.discover(args.data, val_pats) if s.key not in train_keys] if val_pats else []
    name = args.run_name or _dt.datetime.now().strftime("d3_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.work, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "train.log"))
    log(f"run {run_dir}; {len(train_scenes)} train scenes, {len(val_scenes)} held-out scenes; args {vars(args)}")

    opts = d3_preprocess.PreprocessOptions(margin_cols=args.margin_cols, aux=args.aux, obstacle_def=args.obstacle_def,
                                           builder="range_image", az_step_deg=args.az_step,
                                           binned_res_deg=tuple(float(v) for v in args.binned_res.split(",")))
    cache_root = os.path.join(args.work, "cache")
    metas = d3_preprocess.preprocess_all(train_scenes + val_scenes, cache_root, opts, args.preprocess_workers,
                                         args.force_preprocess, log)
    usable_train = [s.key for s in train_scenes if s.key in metas]
    if not usable_train:
        print("ERROR: none of the training scenes could be used (see the log above); nothing was trained.", file=sys.stderr)
        return 2
    by_key = {s.key: s for s in train_scenes + val_scenes}
    holdout = [h for h in args.holdout_backgrounds.split(",") if h]
    if args.split_by == "background" or holdout:
        tr_keys, va_keys = split_by_background([by_key[k] for k in usable_train], args.val_frac, args.seed, holdout)
    else:
        tr_keys, va_keys = split_scenes(usable_train, args.val_frac, args.seed)
    train_bgs = sorted({background_of(by_key[k]) for k in tr_keys if by_key[k].meta.get("real_bag")})
    for s in val_scenes:   # say which held-out scenes share a real background with the training scenes
        s.meta["_background_seen_in_training"] = background_of(s) in train_bgs

    def caches(keys):
        return [d3_data.SceneCache(d3_preprocess.cache_dir_for(cache_root, by_key[k])) for k in keys]

    crop = nn_input.CropConfig.from_dict(opts.crop)
    train_ds = d3_data.D3Dataset(caches(tr_keys), crop.n_cols, train=True, seed=args.seed)
    val_ds = d3_data.D3Dataset(caches(va_keys), crop.n_cols, train=False) if va_keys else None
    heldout: Dict[str, d3_data.D3Dataset] = {}
    for s in val_scenes:
        if s.key in metas:
            heldout.setdefault(s.set_name, []).append(s.key)
    heldout = {k: d3_data.D3Dataset(caches(v), crop.n_cols, train=False) for k, v in heldout.items()}
    log(f"split (seed {args.seed}): train {len(tr_keys)} scenes / {len(train_ds)} frames, "
        f"val {len(va_keys)} scenes / {len(val_ds) if val_ds else 0} frames; held-out sets "
        f"{ {k: len(v) for k, v in heldout.items()} }")
    seen = {k: sorted({s.key for s in val_scenes if s.set_name == k and s.meta.get("_background_seen_in_training")})
            for k in heldout}
    log(f"split by {args.split_by}; held-out backgrounds {holdout}; training real backgrounds {train_bgs}; "
        f"held-out scenes on a training background: { {k: len(v) for k, v in seen.items()} }")
    with open(os.path.join(run_dir, "split.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "split_by": args.split_by, "holdout_backgrounds": holdout,
                   "training_real_backgrounds": train_bgs, "heldout_scenes_on_training_background": seen,
                   "train": tr_keys, "val": va_keys,
                   "heldout": {k: [c.path for c in v.scenes] for k, v in heldout.items()}}, fh, indent=1)

    device = torch.device("cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model({"widths": [int(w) for w in args.widths.split(",")], "decoder": args.decoder}).to(device)
    log(f"device {device}, model {count_parameters(model)} parameters")
    best, history = fit(model, train_ds, val_ds, args, device, log, run_dir)
    model.load_state_dict(best["state"])
    final = {}
    if val_ds is not None:
        final["val_split"] = evaluate(model, val_ds, device, args.threshold, args.batch, 0, thresholds=(0.3, 0.7))
    for k, ds in heldout.items():
        final[k] = evaluate(model, ds, device, args.threshold, args.batch, 0, thresholds=(0.3, 0.7))
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_epoch": best["epoch"], "best_score": best["score"], "metrics": final}, fh, indent=1)
    for k, res in final.items():
        log(f"{k} @ {args.threshold}:\n" + d3_metrics.format_table(res[str(float(args.threshold))]))

    if args.export:
        frames = {"train": len(train_ds), "val_split": len(val_ds) if val_ds else 0,
                  **{k: len(v) for k, v in heldout.items()}}
        gens = sorted({str(by_key[k].meta.get("generator_version")) for k in usable_train})
        meta = {"format": MODEL_FORMAT, "crop": crop.to_dict(), "channels": list(nn_input.CHANNELS),
                "history": nn_input.HISTORY, "classes": list(CLASSES),
                "range_image_builder": opts.builder_id(), "range_image_params": opts.builder_params(),
                "sensors": sorted({str(by_key[k].meta.get("sensor", "hesai128")) for k in usable_train}),
                "obstacle_definition": args.obstacle_def,
                "aux_channels_trained": ["h_above_rail", "lateral_from_centreline"] if args.aux == "d1" else [],
                "aux_dropout": d3_data.AugConfig().aux_drop, "thresholds": {"obstacle": args.threshold, "dust": args.threshold},
                "smoke_test": False,
                "trained_on": {"data": args.data, "train_sets": train_pats, "heldout_sets": list(heldout),
                               "split_by": args.split_by, "holdout_backgrounds": holdout,
                               "real_negatives": [x for x in args.real_negatives.split(",") if x],
                               "training_real_backgrounds": train_bgs,
                               "heldout_scenes_on_training_background": {k: len(v) for k, v in seen.items()},
                               "scenes": {"train": len(tr_keys), "val_split": len(va_keys)}, "frames": frames,
                               "generator_versions": gens, "split_seed": args.seed},
                "hyper_parameters": {k: getattr(args, k) for k in ("epochs", "batch", "lr", "wd", "focal_gamma",
                                                                  "class_weights", "far_weight", "dice_obstacle", "dice_dust",
                                                                  "margin_cols", "aux")},
                "best_epoch": best["epoch"], "run_dir": run_dir,
                "created_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": _git_commit()}
        model_cpu = build_model(model.config)
        model_cpu.load_state_dict(best["state"])
        paths = export_d3(model_cpu, args.export, meta)
        meta_full = dict(meta, model_config=model_cpu.config, n_parameters=count_parameters(model_cpu),
                         torch_version=torch.__version__)
        card = os.path.join(os.path.dirname(os.path.abspath(args.export)), "model_card.md")
        write_model_card(card, meta_full, {k: v[str(float(args.threshold))] for k, v in final.items()})
        log(f"exported {paths} and {card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
