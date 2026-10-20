"""Д3 validation metrics per range band.

For prediction ``P(obstacle) > threshold`` on the current frame of every sample:
  * pixel level, per band of the pixel's near range: TP / FP / FN, precision, recall, IoU (valid pixels);
  * object level, per band of the object's distance (= minimum range of its labelled pixels in the
    crop, the same definition as labels.jsonl ``distance_m`` restricted to the crop): recall = objects
    with >= 1 predicted pixel / objects with >= 1 labelled obstacle pixel in the crop;
  * false-positive components: 8-connected predicted components (1-px gaps merged, as in det_nn) that
    touch no obstacle pixel, per band
    of their minimum range, reported per frame; ``fp_on_dust`` = such components touching dust pixels;
  * dust pixels predicted as obstacle (fraction).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

BANDS: Tuple[Tuple[float, float], ...] = ((0, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 300))


def band_name(b) -> str:
    return f"{int(b[0])}-{int(b[1])}"


def _band_index(r: np.ndarray) -> np.ndarray:
    idx = np.full(np.shape(r), -1, np.int64)
    for i, (lo, hi) in enumerate(BANDS):
        idx[(r >= lo) & (r < hi)] = i
    return idx


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        nb = len(BANDS)
        self.tp = np.zeros(nb, np.int64)
        self.fp = np.zeros(nb, np.int64)
        self.fn = np.zeros(nb, np.int64)
        self.obj_total = np.zeros(nb, np.int64)
        self.obj_hit = np.zeros(nb, np.int64)
        self.fp_comp = np.zeros(nb, np.int64)
        self.fp_on_dust = 0
        self.dust_px = 0
        self.dust_as_obst = 0
        self.frames = 0
        self.frames_with_fp = 0

    def update(self, p_obst: np.ndarray, target: np.ndarray, objid: np.ndarray, rng: np.ndarray) -> None:
        """Arrays (B, h, w): obstacle probability, target class, object id (-1 none), near range in metres."""
        from scipy import ndimage

        struct = np.ones((3, 3), bool)
        for p, y, o, r in zip(p_obst, target, objid, rng):
            self.frames += 1
            valid = r > 0
            pred = (p > self.threshold) & valid
            gt = (y == 1) & valid
            b = _band_index(r)
            for i in range(len(BANDS)):
                m = b == i
                self.tp[i] += int((pred & gt & m).sum())
                self.fp[i] += int((pred & ~gt & m).sum())
                self.fn[i] += int((~pred & gt & m).sum())
            dust = (y == 2) & valid
            self.dust_px += int(dust.sum())
            self.dust_as_obst += int((dust & pred).sum())
            ids = np.unique(o[gt])
            for oid in ids:
                om = gt & (o == oid)
                bi = _band_index(np.array([r[om].min()]))[0]
                if bi < 0:
                    continue
                self.obj_total[bi] += 1
                self.obj_hit[bi] += int((pred & om).any())
            lab, n = ndimage.label(ndimage.binary_dilation(pred, structure=struct) if pred.any() else pred,
                                   structure=struct)  # components with 1-px gaps merged, as det_nn does
            lab = np.where(pred, lab, 0)
            any_fp = False
            if n:
                for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                    if sl is None:
                        continue
                    cm = lab[sl] == k
                    if (gt[sl] & cm).any():
                        continue
                    any_fp = True
                    bi = _band_index(np.array([r[sl][cm].min()]))[0]
                    if bi >= 0:
                        self.fp_comp[bi] += 1
                    if (dust[sl] & cm).any():
                        self.fp_on_dust += 1
            self.frames_with_fp += int(any_fp)

    def result(self) -> Dict[str, object]:
        out: Dict[str, object] = {"threshold": self.threshold, "frames": self.frames, "bands": {}}
        for i, b in enumerate(BANDS):
            tp, fp, fn = int(self.tp[i]), int(self.fp[i]), int(self.fn[i])
            ot, oh = int(self.obj_total[i]), int(self.obj_hit[i])
            out["bands"][band_name(b)] = {
                "pixel_precision": tp / (tp + fp) if tp + fp else None,
                "pixel_recall": tp / (tp + fn) if tp + fn else None,
                "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                "objects": ot, "objects_detected": oh, "object_recall": oh / ot if ot else None,
                "fp_components": int(self.fp_comp[i]),
                "fp_components_per_frame": self.fp_comp[i] / self.frames if self.frames else None}
        out["fp_components_per_frame"] = float(self.fp_comp.sum() / self.frames) if self.frames else None
        out["frames_with_fp_fraction"] = self.frames_with_fp / self.frames if self.frames else None
        out["fp_components_on_dust"] = int(self.fp_on_dust)
        out["dust_pixels"] = int(self.dust_px)
        out["dust_pixels_as_obstacle_fraction"] = self.dust_as_obst / self.dust_px if self.dust_px else None
        rec = [v["object_recall"] for k, v in out["bands"].items() if k != "0-20" and v["objects"] >= 5]
        out["mean_object_recall_20_300"] = float(np.mean(rec)) if rec else None
        tp, fp, fn = int(self.tp[1:].sum()), int(self.fp[1:].sum()), int(self.fn[1:].sum())
        out["pixel_iou_20_300"] = tp / (tp + fp + fn) if tp + fp + fn else None
        return out


def selection_score(res: Dict[str, object]) -> float:
    """Checkpoint selection: mean object recall (bands 20-300 m with >= 5 objects) minus FP components
    per frame (capped at 1); falls back to the pixel IoU when no band has enough objects."""
    rec = res.get("mean_object_recall_20_300")
    fp = res.get("fp_components_per_frame") or 0.0
    if rec is None:
        return float(res.get("pixel_iou_20_300") or 0.0) - min(1.0, fp)
    return float(rec) - min(1.0, float(fp))


def format_table(res: Dict[str, object]) -> str:
    lines = ["| band, m | objects | object recall | pixel P | pixel R | pixel IoU | FP comp./frame |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    def f(v):
        return "–" if v is None else f"{v:.3f}"
    for k, v in res["bands"].items():
        lines.append(f"| {k} | {v['objects']} | {f(v['object_recall'])} | {f(v['pixel_precision'])} | "
                     f"{f(v['pixel_recall'])} | {f(v['pixel_iou'])} | {f(v['fp_components_per_frame'])} |")
    lines.append(f"\nframes {res['frames']}, FP components per frame {f(res['fp_components_per_frame'])}, "
                 f"frames with FP {f(res['frames_with_fp_fraction'])}, dust px -> obstacle "
                 f"{f(res['dust_pixels_as_obstacle_fraction'])}, FP comps on dust {res['fp_components_on_dust']}")
    return "\n".join(lines)
