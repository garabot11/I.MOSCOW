"""Д3 neural detector: range-image U-Net inference with GPU -> CPU fallback (owner: ml-builder).

    det = NNDetector(device_manager, "models/d3_unet.pt")   # crop from the model's meta
    det.warmup()                                            # optional: GPU copy + warm-up now, not on frame 1
    cands = det.step(ri, aux)        # PREFERRED in the node: keeps the 3-frame history; det.reset() on jumps
    # stateless alternative: det.infer([ri_t2, ri_t1, ri_t0], auxs=[...])  CHRONOLOGICAL: frames[-1] = current
    # (per-frame channels of the last frames are memoised by object identity, so a sliding window is cheap)

* ``device_manager`` follows interface C (``device``, ``mode``, ``report_failure(exc)``).  The model is
  loaded on the CPU first and kept there.  The GPU copy (fp16 by default) is created lazily -- by
  ``warmup()`` or by the first inference while the device manager is in GPU mode -- so the device
  manager may be probed after the detector is constructed.  When
  the GPU path raises, the same input is rerun on the CPU: if the CPU succeeds the failure is
  reported (``device_manager.report_failure(exc)`` -> emergency mode) and the detector stays on the
  CPU; if the CPU fails too it is a bug, not a GPU fault, and the CPU exception is raised without
  switching the system to emergency mode.  The GPU copy is never touched again after a failure
  (a faulted CUDA context cannot be trusted, so nothing is moved back from it).  If the device
  manager leaves GPU mode for another reason (another component failed), the detector follows.
* Output: ``DetCandidate`` (interface E) per connected component (8-connected; components separated
  by gaps of up to ``merge_gap_px`` pixels are merged) of pixels with
  P(obstacle) > ``threshold`` (and, if ``report_dust``, of P(dust) > ``dust_threshold`` with
  ``cls="dust"``), split along the range axis where the sorted ranges jump by more than
  max(``split_gap_m``, ``split_gap_rel`` * r).  Geometry comes from the near return of the source
  pixels (``RangeImage.dlh[0]`` / ``rng[0]``) through the crop map (resampled crops never count a
  source pixel twice): centre = middle of the d/l extent, ``extent_dlh`` = max - min,
  ``distance_m`` = minimum range (10th percentile from 10 pixels on, like Д1), ``h_above_rail`` and
  ``lateral_offset`` (l - l_c(d)) from the aux arrays when given (``None`` otherwise), ``score`` = max
  probability, ``slots`` = message slots, ``info`` = p_max, crop pixels, source rows/cols, device.
* Aux channels that the model was not trained with (``meta["aux_channels_trained"]``) are zeroed
  before the forward pass, so passing e.g. ``r_expected`` to a model trained without it is harmless.
* At most ``max_candidates`` candidates per class (highest scores) are returned.
* Input consistency: the model meta records the range-image builder and its parameters
  (``range_image_params``: az_step_deg, binned_res_deg, az_limit_deg).  They are compared once per
  image kind (organised / binned) with the parameters passed as ``range_image_params`` (exact) and
  with what the image itself shows (column / row pitch, azimuth extent); mismatches are logged as
  warnings, because a model only works on images built the way it was trained on.
* ``torch.backends.cudnn.benchmark`` is not touched (measured: < 0.1 ms difference for this model).
* Model file: TorchScript ``d3_unet.pt`` with ``meta.json`` in its extra files; if it cannot be
  loaded (e.g. a torch version mismatch) the state-dict checkpoint ``<stem>.state.pt`` next to it is
  used to rebuild ``nn_model.RangeUNet``.
torch is imported lazily: importing this module does not require torch.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import nn_input
from .nn_input import CropConfig, CropMap, NNInputBuilder

try:  # pragma: no cover - the main session's module (interface E)
    from .candidates import DetCandidate  # type: ignore
    DET_CANDIDATE_SOURCE = "candidates"
except Exception:
    DET_CANDIDATE_SOURCE = "det_nn fallback"

    @dataclass
    class DetCandidate:  # FALLBACK: field-for-field copy of candidates.DetCandidate
        detector: str                 # "d1" | "d2" | "d3"
        d: float                      # centre along the track (analysis frame)
        l: float                      # centre lateral
        h_above_rail: Optional[float]  # centre height above the rail level (None = unknown)
        extent_dlh: tuple             # (d, l, h) metres
        distance_m: float             # Euclidean range to the nearest reliable surface
        n_support: int                # points / pixels
        score: float                  # detector-specific confidence 0..1 (Д3: max obstacle probability)
        cls: str = "obstacle"         # "obstacle" | "dust" | "structure"
        lateral_offset: Optional[float] = 0.0   # l - l_c(d) (None = unknown)
        slots: Optional[np.ndarray] = None
        info: Dict[str, Any] = field(default_factory=dict)


MODEL_FORMAT = "d3_unet/v1"
_STRUCT8 = np.ones((3, 3), bool)
STATE_FORMAT = "d3_unet_state/v1"


def label_components(mask: np.ndarray, merge_gap_px: int = 1) -> np.ndarray:
    """8-connected component labels of ``mask``; components separated by gaps of up to ``merge_gap_px``
    pixels are merged (labels of a dilated mask, read on the original pixels only)."""
    from scipy import ndimage

    grown = ndimage.binary_dilation(mask, structure=_STRUCT8, iterations=merge_gap_px) if merge_gap_px > 0 else mask
    lab, _ = ndimage.label(grown, structure=_STRUCT8)
    return np.where(mask, lab, 0)


def _params_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a, b = list(np.atleast_1d(a)), list(np.atleast_1d(b))
        return len(a) == len(b) and all(abs(float(x) - float(y)) <= 1e-6 for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= 1e-6


def _opt(v) -> Optional[float]:
    """float, or None when unknown (NaN) -- JSON writers cannot take NaN."""
    v = float(v)
    return v if np.isfinite(v) else None


def state_path_for(model_path: str) -> str:
    stem, _ = os.path.splitext(model_path)
    return stem + ".state.pt"


def load_model_cpu(model_path: str):
    """(module on CPU in eval mode, meta dict, kind "torchscript" | "state_dict")."""
    import torch

    errors = []
    if os.path.exists(model_path):
        extra = {"meta.json": ""}
        try:
            m = torch.jit.load(model_path, map_location="cpu", _extra_files=extra)
            raw = extra["meta.json"]
            meta = json.loads(raw.decode() if isinstance(raw, bytes) else (raw or "{}"))
            return m.eval(), meta, "torchscript"
        except Exception as exc:
            errors.append(f"{model_path}: {type(exc).__name__}: {exc}")
    for cand in (state_path_for(model_path), model_path):
        if not os.path.exists(cand):
            continue
        try:
            ck = torch.load(cand, map_location="cpu", weights_only=True)
            if not isinstance(ck, dict) or ck.get("format") != STATE_FORMAT:
                raise ValueError("not a d3_unet state checkpoint")
            from .nn_model import build_model

            m = build_model(ck["model_config"])
            m.load_state_dict(ck["state_dict"])
            return m.eval(), dict(ck.get("meta") or {}), "state_dict"
        except Exception as exc:
            errors.append(f"{cand}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError("cannot load the Д3 model: " + ("; ".join(errors) or f"{model_path} does not exist"))


def _optimize(torch, module, example: "torch.Tensor"):
    """Freeze (folds BatchNorm into the convolutions); returns the module unchanged if that fails."""
    try:
        if not isinstance(module, torch.jit.ScriptModule):
            module = torch.jit.trace(module, example)
        frozen = torch.jit.freeze(module.eval())
        if example.device.type == "cpu":
            frozen = torch.jit.optimize_for_inference(frozen)
        with torch.inference_mode():
            frozen(example)
        return frozen
    except Exception:
        return module


class NNDetector:
    def __init__(self, device_manager, model_path: str, crop_cfg: Optional[CropConfig] = None, *,
                 threshold: Optional[float] = None, dust_threshold: Optional[float] = None, report_dust: bool = True,
                 min_pixels: int = 1, max_candidates: int = 64, merge_gap_px: int = 1,
                 split_gap_m: float = 1.0, split_gap_rel: float = 0.05,
                 gpu_fp16: bool = True, cpu_threads: Optional[int] = None, warmup: int = 3, optimize: bool = True,
                 range_image_params: Optional[dict] = None, logger: Optional[logging.Logger] = None):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"Д3 needs PyTorch: {exc}") from exc
        self._torch = torch
        self.log = logger or logging.getLogger("metro_obstacle_detector.det_nn")
        self.dm = device_manager
        self.model_path = model_path
        cpu_model, self.meta, self.model_kind = load_model_cpu(model_path)
        if self.meta.get("smoke_test"):
            self.log.warning("Д3 model %s is a SMOKE-TEST model (not trained on the synthetic dataset)" % model_path)
        meta_crop = CropConfig.from_dict(self.meta["crop"]) if self.meta.get("crop") else CropConfig()
        if crop_cfg is not None and crop_cfg != meta_crop:
            self.log.warning("Д3 crop differs from the one the model was trained with: %s vs %s" % (crop_cfg, meta_crop))
        self.crop_cfg = crop_cfg or meta_crop
        thr = self.meta.get("thresholds") or {}
        self.threshold = float(threshold if threshold is not None else thr.get("obstacle", 0.5))
        self.dust_threshold = float(dust_threshold if dust_threshold is not None else thr.get("dust", 0.5))
        self.report_dust = bool(report_dust)
        self.min_pixels = int(min_pixels)
        self.max_candidates = int(max_candidates)
        self.merge_gap_px = int(merge_gap_px)
        self.split_gap_m = float(split_gap_m)
        self.split_gap_rel = float(split_gap_rel)
        self.gpu_fp16 = bool(gpu_fp16)
        self.cpu_threads = cpu_threads
        self._builder = NNInputBuilder(self.crop_cfg)
        # aux channels the model never saw populated in training are zeroed (meta "aux_channels_trained")
        trained = self.meta.get("aux_channels_trained")
        self._zero_idx: List[int] = []
        if trained is not None:
            zero = [nn_input.CHANNELS.index(c) for c in ("h_above_rail", "lateral_from_centreline", "residual")
                    if c not in trained]
            self._zero_idx = [k * nn_input.N_FRAME_CHANNELS + c for k in range(nn_input.HISTORY) for c in zero]
        self._checked_kinds: set = set()
        self.range_image_params = dict(range_image_params) if range_image_params else None
        self.input_warnings: List[str] = []
        self._memo: List[tuple] = []        # infer(): (ri, aux, channels, crop map) of the last frames
        self._axes_cache: dict = {}
        self.last_timings_ms: Dict[str, float] = {}
        self.last_probs: Optional[np.ndarray] = None
        self.last_map: Optional[CropMap] = None
        self.gpu_failures = 0

        self._example_shape = (1, nn_input.N_INPUT_CHANNELS, self.crop_cfg.n_rows, self.crop_cfg.n_cols)
        example = torch.zeros(self._example_shape)
        self._optimize_models = bool(optimize)
        self._warmup_n = max(int(warmup), 0)
        self._cpu_model = _optimize(torch, cpu_model, example) if optimize else cpu_model
        self._cpu_threads_set = False
        self._gpu_model = None
        self._gpu_dev = None
        self._gpu_disabled = False          # set after a GPU failure: never touch the GPU copy again
        self._pinned: Dict[Tuple[int, ...], "torch.Tensor"] = {}

    # ------------------------------------------------------------------ device handling
    def _dm_wants_gpu(self) -> bool:
        dev = getattr(self.dm, "device", None)
        return (dev is not None and getattr(dev, "type", "") == "cuda"
                and getattr(self.dm, "mode", "gpu") == "gpu")

    def _ensure_gpu(self) -> None:
        """Create the GPU copy (once) if the device manager is in GPU mode; failures -> CPU + report."""
        if self._gpu_model is not None or self._gpu_disabled or not self._dm_wants_gpu():
            return
        torch = self._torch
        try:
            self._gpu_dev = self.dm.device
            gm, _, _ = load_model_cpu(self.model_path)
            gm = gm.to(self._gpu_dev)
            if self.gpu_fp16:
                gm = gm.half()
            ex = torch.zeros(self._example_shape, device=self._gpu_dev,
                             dtype=torch.float16 if self.gpu_fp16 else torch.float32)
            self._gpu_model = _optimize(torch, gm, ex) if self._optimize_models else gm
            for _ in range(self._warmup_n):
                self._run_gpu(np.zeros(self._example_shape, np.float32))
            self._say("info", "Д3: model on %s (%s)" % (self._gpu_dev, "fp16" if self.gpu_fp16 else "fp32"))
        except Exception as exc:
            self._gpu_failed(exc, during="initialisation")

    def warmup(self) -> str:
        """Create the GPU copy now (if in GPU mode) and run one forward pass; returns "gpu" or "cpu"."""
        self._ensure_gpu()
        self.predict_proba(np.zeros(self._example_shape, np.float32))
        return self.device_name

    @property
    def on_gpu(self) -> bool:
        return self._gpu_model is not None and self._dm_wants_gpu()

    @property
    def device_name(self) -> str:
        return "gpu" if self.on_gpu else "cpu"

    def _gpu_failed(self, exc: BaseException, during: str) -> None:
        self.gpu_failures += 1
        self._gpu_model = None  # never touch the faulted context again
        self._gpu_disabled = True
        self._pinned.clear()
        self._say("error", "Д3: GPU failure during %s (%s: %s) -> CPU" % (during, type(exc).__name__, exc))
        try:
            self.dm.report_failure(exc)
        except Exception as rep_exc:  # pragma: no cover - device manager must not take the detector down
            self._say("error", "Д3: device_manager.report_failure raised %s" % rep_exc)

    def _say(self, level: str, text: str) -> None:
        """Log one pre-formatted line (rclpy loggers take a single string and key the severity by
        call site, so each level gets its own line here); never raises."""
        try:
            if level == "error":
                self.log.error(text)
            elif level == "warning":
                self.log.warning(text)
            else:
                self.log.info(text)
        except Exception:  # pragma: no cover - a logger must not take the detector down
            pass

    def _run_gpu(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = np.ascontiguousarray(x, dtype=np.float32)
        pin = self._pinned.get(x.shape)
        if pin is None:
            pin = torch.empty(x.shape, dtype=torch.float32).pin_memory()
            self._pinned[x.shape] = pin
        pin.copy_(torch.from_numpy(x))
        with torch.inference_mode():
            xg = pin.to(self._gpu_dev, non_blocking=True)
            if self.gpu_fp16:
                xg = xg.half()
            probs = torch.softmax(self._gpu_model(xg).float(), dim=1)
            out = probs.cpu()  # synchronises
        return out.numpy()

    def _run_cpu(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        if self.cpu_threads and not self._cpu_threads_set:
            torch.set_num_threads(int(self.cpu_threads))
            self._cpu_threads_set = True
        with torch.inference_mode():
            logits = self._cpu_model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """(21, h, w) or (N, 21, h, w) float32 -> class probabilities (N, 3, h, w) float32."""
        if x.ndim == 3:
            x = x[None]
        self._ensure_gpu()
        if self.on_gpu:
            try:
                return self._run_gpu(x)
            except Exception as gpu_exc:
                out = self._run_cpu(x)  # raises (without flipping the system to emergency) if it is a bug
                self._gpu_failed(gpu_exc, during="inference")
                return out
        return self._run_cpu(x)

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        self._builder.reset()
        self._memo.clear()

    def step(self, ri, aux: Optional[dict] = None) -> List[DetCandidate]:
        """Stateful: push the current frame into the 3-frame history and detect."""
        t0 = time.perf_counter()
        self._check_builder(ri)
        x, cmap = self._builder.push(ri, aux)
        return self._detect(x, cmap, ri, aux, t0)

    def infer(self, frames: Sequence, auxs: Optional[Sequence[Optional[dict]]] = None) -> List[DetCandidate]:
        """Stateless: ``frames`` in chronological order (``frames[-1]`` = current), up to 3 used."""
        t0 = time.perf_counter()
        if not frames:
            return []
        auxs = list(auxs) if auxs is not None else [None] * len(frames)
        if len(auxs) != len(frames):
            raise ValueError("auxs must match frames")
        self._check_builder(frames[-1])
        recent = list(zip(frames, auxs))[-nn_input.HISTORY:][::-1]   # current first
        chans, cur_map = [], None
        for ri, aux in recent:
            hit = next((m for m in self._memo if m[0] is ri and m[1] is aux), None)
            if hit is None:
                cm = nn_input.compute_crop_map(ri, self.crop_cfg, self._axes_cache)
                hit = (ri, aux, nn_input.frame_channels(ri, cm, aux), cm)
                self._memo.append(hit)
            chans.append(hit[2])
            if cur_map is None:
                cur_map = hit[3]
        keep = [m for m in self._memo if any(m[0] is ri and m[1] is aux for ri, aux in recent)]
        self._memo = keep[-(nn_input.HISTORY + 1):]
        x = nn_input.stack_history(chans)
        return self._detect(x, cur_map, frames[-1], auxs[-1], t0)

    # ------------------------------------------------------------------ internals
    def _warn_input(self, msg: str, *args) -> None:
        text = msg % args if args else msg
        self.input_warnings.append(text)
        self.log.warning("Д3 input: " + text)

    def _check_builder(self, ri) -> None:
        """Once per image kind: builder id and range-image parameters vs the model meta."""
        kind = "organized" if bool(getattr(ri, "organized", True)) else "binned"
        if kind in self._checked_kinds:
            return
        self._checked_kinds.add(kind)
        want = self.meta.get("range_image_builder")
        have = nn_input.range_image_builder_id(ri)
        if want and want != have:
            self._warn_input("model trained on range images from %s, input comes from %s", want, have)
        mp = self.meta.get("range_image_params")
        if not mp:
            return
        if self.range_image_params is not None:
            for k in ("az_step_deg", "binned_res_deg", "az_limit_deg"):
                a, b = mp.get(k), self.range_image_params.get(k)
                if k == "az_limit_deg" and a is None:
                    continue          # trained on full images: any sector that contains the crop is fine (checked below)
                if not _params_equal(a, b):
                    self._warn_input("range_image parameter %s = %s at runtime, %s in training", k, b, a)
        # what the image itself shows
        az = np.asarray(ri.az_deg, dtype=np.float64)
        el = np.asarray(ri.el_deg, dtype=np.float64)
        az_pitch = float(np.median(np.abs(np.diff(np.unwrap(np.radians(az)))))) * 180.0 / np.pi if len(az) > 1 else None
        if kind == "organized":
            want_step = mp.get("az_step_deg")
            if want_step and az_pitch and abs(az_pitch - want_step) > 0.02 * want_step:
                self._warn_input("organised image column pitch %.4f deg, model trained with az_step_deg %s", az_pitch, want_step)
        else:
            res = mp.get("binned_res_deg")
            el_pitch = float(np.median(np.abs(np.diff(el)))) if len(el) > 1 else None
            if res and az_pitch and abs(az_pitch - res[0]) > 0.02 * res[0]:
                self._warn_input("binned image column pitch %.4f deg, model trained with %s", az_pitch, res[0])
            if res and el_pitch and abs(el_pitch - res[1]) > 0.02 * res[1]:
                self._warn_input("binned image row pitch %.4f deg, model trained with %s", el_pitch, res[1])
        lim = mp.get("az_limit_deg")
        if lim is not None and len(az) and np.abs(az).max() > float(lim) + (az_pitch or 0.1):
            self._warn_input("image spans |az| up to %.1f deg, model trained with az_limit_deg %s", np.abs(az).max(), lim)
        cm = nn_input.compute_crop_map(ri, self.crop_cfg, self._axes_cache)
        if not cm.inside.all():
            self._warn_input("%.0f %% of the crop lies outside this %s image (az %.1f..%.1f deg)",
                             100.0 * (1 - cm.inside.mean()), kind, float(az.min()), float(az.max()))

    def _detect(self, x, cmap: CropMap, ri, aux, t0) -> List[DetCandidate]:
        if self._zero_idx:
            x = x.copy()
            x[self._zero_idx] = 0.0
        t1 = time.perf_counter()
        probs = self.predict_proba(x)[0]
        t2 = time.perf_counter()
        cands = self.candidates_from_probs(probs, ri, cmap, aux)
        t3 = time.perf_counter()
        self.last_probs, self.last_map = probs, cmap
        self.last_timings_ms = {"input": (t1 - t0) * 1e3, "forward": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3,
                                "total": (t3 - t0) * 1e3}
        return cands

    def candidates_from_probs(self, probs: np.ndarray, ri, cmap: CropMap, aux: Optional[dict] = None) -> List[DetCandidate]:
        """Connected components -> range split -> DetCandidate list (sorted by distance), vectorised."""
        valid_t = cmap.gather(np.asarray(ri.valid[0]), fill=False) & cmap.inside
        hr_full = nn_input._aux2d(aux, "h_above_rail")
        lat_full = nn_input._aux2d(aux, "lateral_from_centreline")
        W = int(np.asarray(ri.rng).shape[-1])
        rng0 = np.asarray(ri.rng[0])
        dlh0 = np.asarray(ri.dlh[0])
        slot0 = np.asarray(ri.slot_of_pixel[0])
        jobs = [(1, "obstacle", self.threshold)]
        if self.report_dust:
            jobs.append((2, "dust", self.dust_threshold))
        out: List[DetCandidate] = []
        for ci, cls, thr in jobs:
            p = probs[ci]
            mask = (p > thr) & valid_t
            if not mask.any():
                continue
            lab = label_components(mask, self.merge_gap_px)
            idx = np.flatnonzero(mask)
            comp = lab.ravel()[idx].astype(np.int64)
            src = cmap.src_row.ravel()[idx].astype(np.int64) * W + cmap.src_col.ravel()[idx]
            pv = p.ravel()[idx].astype(np.float64)
            # one entry per (component, source pixel), keeping the best probability of its target pixels
            o = np.lexsort((-pv, src, comp))
            comp, src, pv = comp[o], src[o], pv[o]
            first = np.ones(len(src), bool)
            first[1:] = (comp[1:] != comp[:-1]) | (src[1:] != src[:-1])
            n_tp = np.diff(np.append(np.flatnonzero(first), len(first)))   # target pixels per source pixel
            comp, src, pv = comp[first], src[first], pv[first]
            r_, c_ = src // W, src % W
            rr = rng0[r_, c_].astype(np.float64)
            # sort by (component, range) and cut where the range jumps -> clusters
            o = np.lexsort((rr, comp))
            comp, pv, rr, r_, c_, n_tp = comp[o], pv[o], rr[o], r_[o], c_[o], n_tp[o]
            new = np.ones(len(rr), bool)
            if len(rr) > 1:
                gap = np.maximum(self.split_gap_m, self.split_gap_rel * rr[:-1])
                new[1:] = (comp[1:] != comp[:-1]) | ((rr[1:] - rr[:-1]) > gap)
            starts = np.flatnonzero(new)
            counts = np.diff(np.append(starts, len(rr)))
            score = np.maximum.reduceat(pv, starts)
            keep = np.flatnonzero(counts >= self.min_pixels)
            if len(keep) > self.max_candidates:
                keep = keep[np.argsort(-score[keep], kind="stable")[: self.max_candidates]]
            if len(keep) == 0:
                continue
            d = dlh0[r_, c_].astype(np.float64)
            lo = np.minimum.reduceat(d, starts, axis=0)
            hi = np.maximum.reduceat(d, starts, axis=0)
            # distance: minimum below 10 pixels, else the 10th percentile (linear, as np.percentile)
            pos = 0.1 * (counts - 1)
            i0 = np.floor(pos).astype(np.int64)
            i1 = np.minimum(i0 + 1, counts - 1)
            p10 = rr[starts + i0] + (pos - i0) * (rr[starts + i1] - rr[starts + i0])
            dist = np.where(counts < 10, rr[starts], p10)
            h_above = self._group_mid(hr_full, r_, c_, starts)
            lat_off = self._group_mid(lat_full, r_, c_, starts)
            n_tgt = np.add.reduceat(n_tp, starts)
            slots = slot0[r_, c_].astype(np.int32)
            centre = 0.5 * (lo + hi)
            ext = hi - lo
            for j in keep:
                s0, s1 = starts[j], starts[j] + counts[j]
                sl = slots[s0:s1]
                out.append(DetCandidate(detector="d3", d=float(centre[j, 0]), l=float(centre[j, 1]),
                                        h_above_rail=_opt(h_above[j]), extent_dlh=tuple(float(v) for v in ext[j]),
                                        distance_m=float(dist[j]), n_support=int(counts[j]), score=float(score[j]),
                                        cls=cls, lateral_offset=_opt(lat_off[j]), slots=sl[sl >= 0],
                                        info={"p_max": float(score[j]), "n_crop_px": int(n_tgt[j]),
                                              "rows": (int(r_[s0:s1].min()), int(r_[s0:s1].max())),
                                              "cols": (int(c_[s0:s1].min()), int(c_[s0:s1].max())),
                                              "device": self.device_name}))
        out.sort(key=lambda c: c.distance_m)
        return out

    @staticmethod
    def _group_mid(full: Optional[np.ndarray], r_, c_, starts) -> np.ndarray:
        """Middle of the [min, max] of a per-pixel aux array per group (NaN if unknown everywhere)."""
        if full is None:
            return np.full(len(starts), np.nan)
        v = full[r_, c_].astype(np.float64)
        fin = np.isfinite(v)
        lo = np.minimum.reduceat(np.where(fin, v, np.inf), starts)
        hi = np.maximum.reduceat(np.where(fin, v, -np.inf), starts)
        return np.where(np.isfinite(lo), 0.5 * (lo + hi), np.nan)

    def status_dict(self) -> dict:
        return {"device": self.device_name, "model": os.path.basename(self.model_path), "model_kind": self.model_kind,
                "gpu_fp16": self.gpu_fp16 and self.on_gpu, "gpu_failures": self.gpu_failures,
                "smoke_test_model": bool(self.meta.get("smoke_test")), "threshold": self.threshold,
                "range_image_builder": self.meta.get("range_image_builder"),
                "range_image_params": self.meta.get("range_image_params"), "input_warnings": list(self.input_warnings),
                "crop": self.crop_cfg.to_dict(), "det_candidate_source": DET_CANDIDATE_SOURCE,
                "range_image_source": nn_input.RANGE_IMAGE_SOURCE}
