"""Torch dataset over the per-scene caches, augmentations and the class-imbalance loss.

Sample = (scene, frame t): channels of frames t, t-1, t-2 (current first; earlier frames clipped at the
scene start = the oldest available is repeated, as at inference), cropped from the wide cache crop.
Augmentations (training only):
  * azimuth shift: random window of ``n_cols`` inside the wide crop (+-margin columns, i.e. +-3.2 deg);
  * mirror in azimuth (p 0.5): columns flipped, lateral channel negated;
  * dropout of returns: per frame a random fraction U(0, max_drop) of the valid pixels loses its
    return (all channels 0); dropped pixels of the current frame become background (no return, no object);
  * intensity jitter: x U(0.7, 1.3) + N(0, 0.01) on valid pixels, clipped to [0, 1];
  * aux-channel dropout (p 0.3): h_above_rail / lateral zeroed in all frames, so that the model also
    works when the track model is unavailable at inference;
  * history dropout (p 0.05): previous frames replaced by the current one (start-up / after a reset);
  * partial objects (p 0.3 per sample): for every obstacle object the lower 30-70 % of its rows lose
    their returns (and become background) -- e.g. a person in dark clothing at 40-75 m on the Falcon
    returns only 8-45 points from the upper body, legs invisible: the class must not assume a
    full-height silhouette;
  * dark objects (p 0.3): intensity of obstacle pixels x U(0.15, 0.6).
  The two object augmentations act on the object's pixel footprint in all three frames (pixels whose
  range is within 1.5 m of the object's median range), so that the history stays consistent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import _paths  # noqa: F401
from metro_obstacle_detector import nn_input

NF = nn_input.N_FRAME_CHANNELS
CH_RNG_N, CH_RNG_F, CH_INT, CH_VALID, CH_H, CH_LAT, CH_RES = range(7)


class SceneCache:
    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        mode = "r" if mmap else None
        self.feats = np.load(os.path.join(path, "feats.npy"), mmap_mode=mode)    # (T, 7, h, Wc) f16
        self.target = np.load(os.path.join(path, "target.npy"), mmap_mode=mode)  # (T, h, Wc) u8
        self.objid = np.load(os.path.join(path, "objid.npy"), mmap_mode=mode)    # (T, h, Wc) i32
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        with open(os.path.join(path, "objects.json"), encoding="utf-8") as fh:
            self.objects = json.load(fh)

    @classmethod
    def from_arrays(cls, feats, target, objid, objects=None, meta=None) -> "SceneCache":
        self = cls.__new__(cls)
        self.path = "<memory>"
        self.feats, self.target, self.objid = feats, target, objid
        self.objects = objects or [{"frame": i, "objects": []} for i in range(len(feats))]
        self.meta = meta or {}
        return self

    def __len__(self) -> int:
        return int(self.feats.shape[0])


@dataclass
class AugConfig:
    shift: bool = True
    flip: float = 0.5
    max_drop: float = 0.10
    intensity: bool = True
    aux_drop: float = 0.3
    hist_drop: float = 0.05
    partial_obj: float = 0.3
    dark_obj: float = 0.3


class D3Dataset(torch.utils.data.Dataset):
    def __init__(self, scenes: Sequence[SceneCache], n_cols: int = 256, train: bool = True,
                 aug: Optional[AugConfig] = None, seed: int = 0):
        self.scenes = list(scenes)
        self.n_cols = int(n_cols)
        self.train = bool(train)
        self.aug = aug or AugConfig()
        self.seed = int(seed)
        self.index = [(si, t) for si, s in enumerate(self.scenes) for t in range(len(s))]
        for s in self.scenes:
            if s.feats.shape[-1] < self.n_cols:
                raise ValueError(f"cache {s.path} is narrower ({s.feats.shape[-1]}) than the model crop {self.n_cols}")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def _object_aug(self, frames, y, obj, rng, partial: bool, dark: bool) -> None:
        cur = frames[0]
        for oid in np.unique(obj[y == 1]):
            om = (obj == oid) & (y == 1)
            r_med = float(np.median(cur[CH_RNG_N][om]) * nn_input.RANGE_SCALE_M)
            if partial:
                rows = np.flatnonzero(om.any(axis=1))
                if len(rows) >= 2:
                    keep = max(1, int(round(len(rows) * rng.uniform(0.3, 0.7))))
                    low = om.copy()
                    low[: rows[0] + keep] = False      # row 0 = highest: keep the upper part
                    # previous frames: the rectangle under the cut, restricted to the object's range
                    fp = np.outer(low.any(axis=1), om.any(axis=0))
                    masks = [low] + _footprint(frames[1:], fp, r_med)
                    for f, m in zip(frames, masks):
                        f[:, m] = 0.0
                    y[low] = 0
                    obj[low] = -1
                    om = om & ~low
            if dark and om.any():
                scale = rng.uniform(0.15, 0.6)
                box = np.outer(om.any(axis=1), om.any(axis=0))
                masks = [om] + _footprint(frames[1:], box, r_med)
                for f, m in zip(frames, masks):
                    f[CH_INT][m] *= scale

    def __getitem__(self, k: int):
        si, t = self.index[k]
        s = self.scenes[si]
        wc = s.feats.shape[-1]
        margin = (wc - self.n_cols) // 2
        rng = np.random.default_rng([self.seed, self._epoch, k]) if self.train else None
        a = self.aug
        off = margin
        if self.train and a.shift:
            off = int(rng.integers(0, wc - self.n_cols + 1))
        ts = [t, max(t - 1, 0), max(t - 2, 0)]
        if self.train and rng.random() < a.hist_drop:
            ts = [t, t, t]
        cols = slice(off, off + self.n_cols)
        frames = [np.asarray(s.feats[tt, :, :, cols], dtype=np.float32) for tt in ts]
        y = np.asarray(s.target[t, :, cols], dtype=np.int64)
        obj = np.asarray(s.objid[t, :, cols], dtype=np.int64)
        if self.train:
            if rng.random() < a.flip:
                frames = [f[:, :, ::-1].copy() for f in frames]
                for f in frames:
                    f[CH_LAT] = -f[CH_LAT]
                y = y[:, ::-1].copy()
                obj = obj[:, ::-1].copy()
            do_partial = rng.random() < a.partial_obj
            do_dark = rng.random() < a.dark_obj
            if do_partial or do_dark:
                self._object_aug(frames, y, obj, rng, do_partial, do_dark)
            for j, f in enumerate(frames):
                if a.max_drop > 0:
                    rate = rng.uniform(0.0, a.max_drop)
                    drop = (f[CH_VALID] > 0) & (rng.random(f[CH_VALID].shape) < rate)
                    f[:, drop] = 0.0
                    if j == 0:
                        y[drop] = 0
                        obj[drop] = -1
                if a.intensity:
                    v = f[CH_VALID] > 0
                    f[CH_INT] = np.where(v, np.clip(f[CH_INT] * rng.uniform(0.7, 1.3)
                                                    + rng.normal(0.0, 0.01, f[CH_INT].shape), 0.0, 1.0), 0.0)
            if rng.random() < a.aux_drop:
                for f in frames:
                    f[CH_H] = 0.0
                    f[CH_LAT] = 0.0
        x = np.concatenate(frames, axis=0)
        return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                "y": torch.from_numpy(np.ascontiguousarray(y)),
                "obj": torch.from_numpy(np.ascontiguousarray(obj)),
                "rng": torch.from_numpy(np.ascontiguousarray(frames[0][CH_RNG_N] * nn_input.RANGE_SCALE_M))}


def _footprint(frames, om: np.ndarray, r_med: float):
    """Per frame: pixels of the object's footprint ``om`` whose near range is within 1.5 m of ``r_med``."""
    return [om & (np.abs(f[CH_RNG_N] * nn_input.RANGE_SCALE_M - r_med) < 1.5) & (f[CH_VALID] > 0) for f in frames]


def focal_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: Sequence[float], gamma: float = 2.0,
               rng: Optional[torch.Tensor] = None, far_weight: float = 0.0) -> torch.Tensor:
    """Per-pixel focal loss (gamma = 0 -> weighted cross-entropy), alpha = class weights.

    ``far_weight`` up-weights obstacle pixels with range: w *= 1 + far_weight * r / 100 m
    (far objects have few pixels).  Normalised by the number of pixels.
    """
    logp = F.log_softmax(logits.float(), dim=1)
    logp_t = logp.gather(1, y[:, None]).squeeze(1)
    w = torch.as_tensor(class_weights, dtype=logp.dtype, device=logp.device)[y]
    if far_weight and rng is not None:
        w = w * torch.where(y == 1, 1.0 + far_weight * rng / 100.0, torch.ones_like(rng))
    if gamma:
        w = w * (1.0 - logp_t.exp()).clamp_min(0.0).pow(gamma)
    return -(w * logp_t).mean()


def soft_dice_loss(logits: torch.Tensor, y: torch.Tensor, cls: int, rng: Optional[torch.Tensor] = None,
                   far_weight: float = 0.0) -> torch.Tensor:
    """1 - soft Dice of class ``cls`` over the whole batch (insensitive to how rare the class is).
    Pixels are weighted like the focal loss (far obstacle pixels count more).  0 if the class is absent
    in both the target and the prediction."""
    p = torch.softmax(logits.float(), dim=1)[:, cls]
    g = (y == cls).to(p.dtype)
    w = torch.ones_like(p)
    if far_weight and rng is not None:
        w = 1.0 + far_weight * rng / 100.0
    inter = (w * p * g).sum()
    denom = (w * p).sum() + (w * g).sum()
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def d3_loss(logits, y, rng, class_weights, gamma=2.0, far_weight=1.0, dice_obstacle=1.0, dice_dust=0.5):
    """focal (pixel-mean, class-weighted) + dice_obstacle * Dice(obstacle) + dice_dust * Dice(dust)."""
    loss = focal_loss(logits, y, class_weights, gamma, rng, far_weight)
    if dice_obstacle:
        loss = loss + dice_obstacle * soft_dice_loss(logits, y, 1, rng, far_weight)
    if dice_dust:
        loss = loss + dice_dust * soft_dice_loss(logits, y, 2)
    return loss
