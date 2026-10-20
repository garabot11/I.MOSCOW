"""Per-scene cache for training: NN channels on a WIDE crop + per-pixel targets (via slot_of_pixel).

Every frame goes through the same path as in the running node:
  parsed PointCloud2 -> ``DetectionPipeline.build_frame`` (sensor profile, forward crop, axes)
        -> ``DetectionPipeline.step_frame`` = Д1 of the CURRENT frame (aux ``--aux d1``): rail level and
           centre line -> h_above_rail / lateral channels (zeros when Д1 has no track model, as at runtime)
        -> ``range_image.build_range_image(frame, az_step_deg, binned_res_deg, az_limit_deg)``: organised
           Hesai image, or the angle-binned image of unorganised sensors (falcon_prime)
        -> crop map on the wide crop (n_cols + 2 * margin columns, same centre and pitch as the model crop,
           so the central n_cols columns ARE the inference crop; the margin feeds the azimuth-shift augmentation)
        -> feats (7, rows, wide_cols) float16, target (rows, wide_cols) uint8 {0 bg, 1 obstacle, 2 dust},
           objid (rows, wide_cols) int32 (-1 none) from the NEAR-return slot of each pixel and slot_labels.npz.
``builder="standin"`` (NOT exposed by the training CLI) uses nn_input.HesaiStandinBuilder instead:
it differs from range_image by one column on 14-23 % of the crop pixels of real frames, so a model
trained on it does not match the runtime images -- tests only.
Real negatives (``d3_scenes.real_negative_scenes``): frames of real bags annotated as obstacle-free go
through the same path with every pixel labelled background.
Cache: ``<cache>/<set>/<scene_id>/{feats,target,objid}.npy + objects.json + meta.json``; rebuilt when the
options or CACHE_VERSION change.  Unusable scenes raise ``UnsupportedScene`` and are skipped by the
caller with the reason logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import _paths
import d3_scenes
from metro_obstacle_detector import nn_input
from metro_obstacle_detector.nn_input import CropConfig

CACHE_VERSION = 2
TARGET_BG, TARGET_OBSTACLE, TARGET_DUST = 0, 1, 2
OBSTACLE_DEFS = ("object", "is_obstacle", "in_clearance")


class UnsupportedScene(Exception):
    pass


@dataclass
class PreprocessOptions:
    crop: dict = field(default_factory=lambda: CropConfig().to_dict())
    margin_cols: int = 32
    aux: str = "d1"                    # "d1" (Д1 rail/centre line of the current frame) | "none"
    obstacle_def: str = "object"       # which labelled objects are class "obstacle" (see target_class)
    detector_yaml: Optional[str] = None
    builder: str = "range_image"       # "range_image" (reference) | "standin" (fallback, Hesai only)
    az_step_deg: float = 0.1           # build_range_image parameters (must match the runtime)
    binned_res_deg: Tuple[float, float] = (0.09, 0.08)
    az_limit_deg: Optional[float] = None

    def wide_crop(self) -> CropConfig:
        c = CropConfig.from_dict(self.crop)
        c.n_cols = c.n_cols + 2 * int(self.margin_cols)
        return c

    def builder_id(self) -> str:
        return nn_input.STANDIN_BUILDER_ID if self.builder == "standin" else nn_input.RANGE_IMAGE_BUILDER_ID

    def builder_params(self) -> dict:
        return {"az_step_deg": self.az_step_deg, "binned_res_deg": list(self.binned_res_deg),
                "az_limit_deg": self.az_limit_deg}

    def digest(self) -> str:
        d = asdict(self)
        d["binned_res_deg"] = list(self.binned_res_deg)
        d["cache_version"] = CACHE_VERSION
        d["builder_id"] = self.builder_id()
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def target_class(obj: dict, definition: str) -> int:
    """dust -> 2; foreign objects -> 1 according to ``definition``; else background.

    "object": every non-dust object (the network segments foreign objects; the clearance decision is
    left to fusion, which knows the gauge); "is_obstacle" / "in_clearance": only objects with that flag.
    """
    if obj.get("class") == "dust":
        return TARGET_DUST
    if definition == "object":
        return TARGET_OBSTACLE
    if definition in ("is_obstacle", "in_clearance"):
        return TARGET_OBSTACLE if obj.get(definition) else TARGET_BG
    raise ValueError(f"unknown obstacle definition {definition!r}")


def load_detector_config(path: Optional[str]):
    from metro_obstacle_detector.config import DetectorConfig

    if path is None:
        cand = os.path.join(_paths.SOLUTION_DIR, "config", "detector.yaml")
        path = cand if os.path.exists(cand) else None
    return DetectorConfig.from_yaml(path) if path else DetectorConfig()


class FrameProcessor:
    """One scene's frames -> cache arrays, through the runtime frame path (see the module docstring)."""

    def __init__(self, opts: PreprocessOptions):
        from metro_obstacle_detector.pipeline import DetectionPipeline

        if opts.aux not in ("d1", "none"):
            raise ValueError(f"unknown aux mode {opts.aux!r}")
        if opts.builder not in ("range_image", "standin"):
            raise ValueError(f"unknown range-image builder {opts.builder!r}")
        self.opts = opts
        self.crop = opts.wide_crop()
        self._cache: dict = {}
        self.pipeline = DetectionPipeline(load_detector_config(opts.detector_yaml))
        self.standin = None
        if opts.builder == "standin":
            import warnings

            warnings.warn("Д3 preprocessing with the STAND-IN range-image builder: it differs from "
                          "range_image.build_range_image by one column on 14-23 % of the crop pixels of real "
                          "frames; a model trained on it does not match the runtime images (tests only)",
                          RuntimeWarning, stacklevel=2)
            self.standin = nn_input.HesaiStandinBuilder()
        self.last_status: Optional[str] = None

    def reset(self) -> None:
        self.pipeline.reset()

    def frame(self, msg):
        return self.pipeline.build_frame(msg)

    def range_image(self, frame):
        if self.standin is not None:
            if not frame.profile.organized or frame.profile.sensor != "hesai128":
                raise UnsupportedScene("the stand-in builder handles the organised Hesai layout only")
            return self.standin.build(frame.struct)
        from metro_obstacle_detector.range_image import build_range_image

        o = self.opts
        return build_range_image(frame, az_step_deg=o.az_step_deg, binned_res_deg=tuple(o.binned_res_deg),
                                 az_limit_deg=o.az_limit_deg)

    def aux(self, frame, ri) -> Optional[dict]:
        """Д1 on the current frame (as in the node: Д1 first, Д3 after it); None with --aux none."""
        if self.opts.aux != "d1":
            return None
        res, _ = self.pipeline.step_frame(frame)
        self.last_status = res.status
        return nn_input.aux_from_track(ri, res.rail, res.curve)

    def process_msg(self, msg, frame_label: Optional[dict], frame_slots):
        frame = self.frame(msg)
        ri = self.range_image(frame)
        return self.process(ri, self.aux(frame, ri), frame_label, frame_slots), frame

    def process(self, ri, aux: Optional[dict], frame_label: Optional[dict],
                frame_slots: Optional[Tuple[np.ndarray, np.ndarray]]):
        cm = nn_input.compute_crop_map(ri, self.crop, self._cache)
        feats = nn_input.frame_channels(ri, cm, aux)
        valid = cm.gather(np.asarray(ri.valid[0]), fill=False) & cm.inside
        slots = cm.gather(np.asarray(ri.slot_of_pixel[0]), fill=-1)
        objid = d3_scenes.lookup_obj(np.where(valid, slots, -1), frame_slots).astype(np.int32)
        target = np.zeros(objid.shape, np.uint8)
        objects = []
        for obj in (frame_label or {}).get("objects", []):
            oid = int(obj["id"])
            tc = target_class(obj, self.opts.obstacle_def)
            m = objid == oid
            target[m] = tc
            objects.append({"id": oid, "class": obj.get("class"), "target_class": tc,
                            "distance_m": obj.get("distance_m"), "n_hits": obj.get("n_hits"),
                            "visible": obj.get("visible"), "in_clearance": obj.get("in_clearance"),
                            "is_obstacle": obj.get("is_obstacle"), "px_in_crop": int(m.sum())})
        known = {o["id"] for o in objects}
        stray = set(np.unique(objid[objid >= 0]).tolist()) - known
        if stray:  # slots labelled with an id that labels.jsonl does not list for this frame
            objid[np.isin(objid, list(stray))] = -1
        return feats.astype(np.float16), target, objid, objects, len(stray)


def cache_dir_for(cache_root: str, scene: d3_scenes.Scene) -> str:
    return os.path.join(cache_root, scene.set_name, scene.scene_id)


def is_cached(cache_root: str, scene: d3_scenes.Scene, opts: PreprocessOptions) -> bool:
    mp = os.path.join(cache_dir_for(cache_root, scene), "meta.json")
    if not os.path.exists(mp):
        return False
    try:
        with open(mp, encoding="utf-8") as fh:
            return json.load(fh).get("digest") == opts.digest()
    except Exception:
        return False


def write_cache(out_dir: str, feats, target, objid, objects, meta) -> None:
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    np.save(os.path.join(tmp, "feats.npy"), np.stack(feats))
    np.save(os.path.join(tmp, "target.npy"), np.stack(target))
    np.save(os.path.join(tmp, "objid.npy"), np.stack(objid))
    with open(os.path.join(tmp, "objects.json"), "w", encoding="utf-8") as fh:
        json.dump(objects, fh)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.replace(tmp, out_dir)


def preprocess_scene(scene: d3_scenes.Scene, cache_root: str, opts: PreprocessOptions, force: bool = False) -> dict:
    out_dir = cache_dir_for(cache_root, scene)
    if not force and is_cached(cache_root, scene, opts):
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
            return dict(json.load(fh), cached=True)
    missing = scene.missing()
    if missing:
        raise UnsupportedScene(f"incomplete scene (missing {missing})")
    proc = FrameProcessor(opts)
    real_neg = d3_scenes.is_real_negative(scene)
    f_lo, f_hi = (scene.meta.get("frames") or [0, None]) if real_neg else (0, None)
    labels = [] if real_neg else d3_scenes.read_labels(scene)
    by_stamp = {int(r["stamp_ns"]): r for r in labels if r.get("stamp_ns") is not None}
    slot_labels = {} if real_neg else d3_scenes.read_slot_labels(scene)
    feats, target, objid, objects = [], [], [], []
    n_stamp_match = n_stray = n_unknown = 0
    profile = None
    t0 = time.time()
    for i, stamp, msg in d3_scenes.iter_messages(scene.bag_dir, scene.meta.get("topic")):
        if real_neg and (i < f_lo or (f_hi is not None and i >= f_hi)):
            if f_hi is not None and i >= f_hi:
                break
            continue
        lab = by_stamp.get(stamp)
        if lab is not None:
            n_stamp_match += 1
        elif i < len(labels):
            lab = labels[i]
        frame_no = int(lab["frame"]) if lab is not None else i
        if real_neg:
            lab = {"frame": i, "objects": []}       # annotated obstacle-free real frames: all background
        (f, t, o, objs, stray), frame = proc.process_msg(msg, lab, slot_labels.get(frame_no))
        if profile is None:
            profile = frame.profile.summary()
        n_unknown += int(proc.last_status == "unknown")
        n_stray += stray
        feats.append(f)
        target.append(t)
        objid.append(o)
        objects.append({"frame": frame_no, "stamp_ns": stamp, "objects": objs})
    if not feats:
        raise UnsupportedScene("bag has no clouds")
    meta = {"digest": opts.digest(), "cache_version": CACHE_VERSION, "options": asdict(opts),
            "scene": scene.key, "scene_meta": scene.meta, "range_image_builder": opts.builder_id(),
            "range_image_params": opts.builder_params(), "sensor_profile": profile, "d1_unknown_frames": n_unknown,
            "n_frames": len(feats), "labels_rows": len(labels), "frames_matched_by_stamp": n_stamp_match,
            "stray_slot_label_frames": n_stray, "wide_crop": opts.wide_crop().to_dict(),
            "seconds": round(time.time() - t0, 1),
            "pixels": {"obstacle": int(sum((t == TARGET_OBSTACLE).sum() for t in target)),
                       "dust": int(sum((t == TARGET_DUST).sum() for t in target))}}
    write_cache(out_dir, feats, target, objid, objects, meta)
    return dict(meta, cached=False)


def _worker(args):
    scene, cache_root, opts, force = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return scene.key, preprocess_scene(scene, cache_root, opts, force), None
    except UnsupportedScene as exc:
        return scene.key, None, f"skipped: {exc}"
    except Exception as exc:  # report and continue with the other scenes
        return scene.key, None, f"FAILED: {type(exc).__name__}: {exc}"


def preprocess_all(scenes, cache_root: str, opts: PreprocessOptions, workers: int = 4, force: bool = False,
                   log=print) -> Dict[str, dict]:
    """Cache every scene (parallel); returns {scene key: meta} for the usable ones, logs the others."""
    done: Dict[str, dict] = {}
    jobs = [(s, cache_root, opts, force) for s in scenes]
    if workers <= 1:
        results = map(_worker, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        results = ex.map(_worker, jobs)
    for key, meta, err in results:
        if err:
            log(f"[preprocess] {key}: {err}")
        else:
            done[key] = meta
            if not meta.get("cached"):
                log(f"[preprocess] {key}: {meta['n_frames']} frames in {meta['seconds']} s, "
                    f"obstacle px {meta['pixels']['obstacle']}, dust px {meta['pixels']['dust']}")
    if workers > 1:
        ex.shutdown()
    return done
