"""Generate a labelled synthetic set (interface B).

    cd solution/synth && python -m synth.generate --config configs/val_hesai_insert.yaml \
        --out /home/gag/i.mos/synthetic/val_hesai_insert [--only 0,3] [--max-frames N]

Every scene gets ``bag/`` (rosbag2, MCAP+zstd by default), ``scene.json``, ``labels.jsonl`` and
``slot_labels.npz``.  Scene i of a set is generated from ``numpy.random.default_rng([seed, i])`` only, so a
scene can be regenerated alone with the same result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.normpath(os.path.join(HERE, "..", "..", "src", "metro_obstacle_detector"))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

GENERATOR_VERSION = "synth-1.0"


def _scene_dir(out, set_name, i, tag):
    return os.path.join(out, f"{set_name}_{i:03d}_{tag}")


def run_insert_scene(cfg, i, out, caster, max_frames=None):
    from metro_obstacle_detector.beam_model import hesai128

    from .insert import InsertScene, plan_objects, stationary_segments
    from .realbag import RealBag
    from .returns import ReturnModel
    from .writer import BagWriter, LabelWriter, serialize_pointcloud2, write_scene_json

    rng = np.random.default_rng([int(cfg["seed"]), i])
    sc = cfg["scenes"]
    bags = sc["bags"]
    bag_name = bags[i % len(bags)]
    from .clean_source import FRAMES as CLEAN_FRAMES, NAME as CLEAN_NAME

    segs = [list(CLEAN_FRAMES)] if bag_name == CLEAN_NAME else stationary_segments(RealBag(bag_name))
    segs = [s for s in segs if s[1] - s[0] + 1 >= sc.get("min_segment", 40)]
    if not segs:
        raise RuntimeError(f"no stationary segment in {bag_name}")
    seg = segs[int(rng.integers(len(segs)))]
    n = min(int(sc.get("frames", 120)), seg[1] - seg[0] + 1, 300)
    if max_frames:
        n = min(n, max_frames)
    i0 = int(rng.integers(seg[0], seg[1] - n + 2))
    frames = list(range(i0, i0 + n))
    bm = hesai128(int(cfg.get("layout", 2400)))
    ret = ReturnModel.hesai(**cfg.get("return_model", {}))
    negative = i < int(sc.get("negatives", 0))
    scene = InsertScene(bag_name, frames, [], int(rng.integers(1 << 31)), caster, ret, bm,
                        divergence_deg=float(cfg.get("divergence_deg", 0.0)), n_sub=int(cfg.get("sub_rays", 7)))
    d_vis = min((scene.sight or 200.0) - 5.0, scene.track.max_d_curve)
    occ_max = float(cfg.get("objects", {}).get("max_occluded_frac", 0.25))
    rejected = []
    if not negative:
        want = plan_objects(rng, cfg.get("objects", {}), scene.track, scene.xyz_dlh0, d_vis)
        for p in want:
            # up to 12 placements; an object that is not visible at all (occ is None, beyond sight) is kept and
            # labelled visible = false; after 12 rejected placements the object is dropped (already popped)
            for attempt in range(12):
                scene.add_plans([p])
                occ = scene.first_frame_occlusion(len(scene.models) - 1)
                if p["crossing_speed"] != 0.0 or occ is None or occ <= occ_max:
                    break
                rejected.append({"d": p["d"], "lat": p["lat"], "occluded_frac": occ})
                scene.models.pop()
                p = dict(p, d=float(rng.uniform(20.0, max(21.0, d_vis))), lat=float(rng.uniform(-0.8, 0.8)))
    tag = "neg" if negative else "pos"
    sdir = _scene_dir(out, cfg["set"], i, tag)
    os.makedirs(sdir, exist_ok=True)
    topic = scene.bag.topic if bm.n_cols == 7200 else "/lidar_points"
    frame_id = cfg.get("frame_id", "hesai_lidar")
    bw = BagWriter(os.path.join(sdir, "bag"), topic, cfg.get("storage", "mcap"))
    lw = LabelWriter(sdir)
    rng_r = np.random.default_rng([int(cfg["seed"]), i, 7])
    t0 = time.perf_counter()
    for j in range(len(frames)):
        f, pts, slots, ids, labels = scene.render_frame(j, rng_r)
        bw.write(serialize_pointcloud2(pts, f.stamp_ns, frame_id), f.bag_time_ns)
        lw.frame(j, f.stamp_ns, labels, scene.sight, slots, ids, extra={"real_frame": int(f.index)})
    bw.close()
    lw.close()
    dt = time.perf_counter() - t0
    tr = scene.track
    info = {"sensor": "hesai128", "source": "insert_real", "real_bag": bag_name, "real_frames": [frames[0], frames[-1]],
            "tunnel": {"type": "real", "radius_m": None, "length_m": None, "description": "real development bag"},
            "speed_mps": 0.0, "seed": [int(cfg["seed"]), i], "generator_version": GENERATOR_VERSION,
            "coordinate_mode": "hesai (forward -y, up +z)", "frame_id": frame_id, "topic": topic,
            "layout": {"n_cols": bm.n_cols, "points": bm.n_slots, "dual_return": True},
            "motion_model": "stationary (segment detected from data; moving segments are not used)",
            "background_note": ("doubleT_obstacle frames with the real object removed (its angular window replaced by a "
                                "same-parity frame where it stands 2.6 m away), cropped from 921 600 to the 307 200-point "
                                "layout") if bag_name == CLEAN_NAME else "real frames unchanged",
            "placement_method": "rail level + centre-line: medians of ObstacleDetector on 6 frames; bottom on the local "
                                "floor when >= 5 returns around the footprint, else rail model + floor_offset_m (median "
                                "local floor - rail model on the centre-line, -0.05 m when unknown)",
            "floor_offset_m": scene.floor_offset_m, "floor_offset_samples": scene.floor_offset_samples,
            "track": {"z0": tr.z0, "grade": tr.grade, "curv": tr.curv, "max_d_rail_observed": tr.max_d_rail,
                      "l0": tr.l0, "a": tr.a, "b": tr.b, "max_d_curve_supported": tr.max_d_curve},
            "sight_limit_m": scene.sight, "objects": [dict(m["plan"], bottom_h=m["bottom"], placement=m["placement"])
                                                       for m in scene.models],
            "rejected_placements": rejected, "max_occluded_frac": occ_max, "d_max_placement_m": d_vis,
            "return_model": ret.to_dict(), "divergence_deg": scene.div, "clearance_gauge": __import__("synth.insert", fromlist=["GAUGE"]).GAUGE,
            "frames": len(frames), "render_s": round(dt, 2)}
    write_scene_json(sdir, info)
    return sdir, len(frames), dt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated scene indices")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--workers", type=int, default=1, help="scenes generated in parallel processes")
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    from .raycast import Caster

    caster = Caster(cfg.get("raycast", "auto"))
    os.makedirs(args.out, exist_ok=True)
    count = int(cfg["scenes"]["count"]) if "count" in cfg["scenes"] else len(cfg["scenes"]["list"])
    idx = [int(x) for x in args.only.split(",")] if args.only else list(range(count))
    runner = {"insert_real": run_insert_scene}
    if cfg["source"] == "parametric":
        from .parametric import run_parametric_scene

        runner["parametric"] = run_parametric_scene
    if cfg["source"] == "fake_style":
        from .fake_scenes import run_fake_scene

        runner["fake_style"] = run_fake_scene
    total_frames, total_s = 0, 0.0
    summary = []
    wall0 = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = pool.starmap(_worker, [(args.config, i, args.out, args.max_frames) for i in idx])
    else:
        results = []
        for i in idx:
            results.append(runner[cfg["source"]](cfg, i, args.out, caster, args.max_frames))
            _print_row(results[-1])
    for sdir, n, dt in results:
        total_frames += n
        total_s += dt
        summary.append(_row(sdir, n, dt))
        if args.workers > 1:
            print(json.dumps(summary[-1]), flush=True)
    wall = time.perf_counter() - wall0
    with open(os.path.join(args.out, "set.json"), "w", encoding="utf-8") as fh:
        json.dump({"set": cfg["set"], "config": cfg, "generator_version": GENERATOR_VERSION, "backend": caster.backend,
                   "scenes": summary, "frames": total_frames, "workers": args.workers,
                   "frames_per_min_per_process": round(60 * total_frames / max(total_s, 1e-9), 1),
                   "frames_per_min_wall": round(60 * total_frames / max(wall, 1e-9), 1)}, fh, indent=1)
    print(f"done: {total_frames} frames, {60 * total_frames / max(wall, 1e-9):.0f} frames/min wall "
          f"({args.workers} worker(s), {caster.backend})")
    return 0


def _row(sdir, n, dt):
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(sdir) for f in fs)
    return {"scene": os.path.basename(sdir), "frames": n, "seconds": round(dt, 1), "MB": round(size / 1e6, 1)}


def _print_row(res):
    print(json.dumps(_row(*res)), flush=True)


def _worker(config, i, out, max_frames):
    with open(config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    from .raycast import Caster

    caster = Caster(cfg.get("raycast", "auto"))
    if cfg["source"] == "parametric":
        from .parametric import run_parametric_scene as run
    elif cfg["source"] == "fake_style":
        from .fake_scenes import run_fake_scene as run
    else:
        run = run_insert_scene
    return run(cfg, i, out, caster, max_frames)


if __name__ == "__main__":
    sys.exit(main())
