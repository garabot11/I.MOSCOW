"""Per-frame state handling shared by the ROS node and the offline evaluator.

Order for every frame: (1) check the header stamp for a backward jump / gap,
(2) on a discontinuity reset *both* the temporal state (tracks) and the
geometric state of the detector (previous track curve, held model),
(3) D1 geometry, (4) range image + D2 on the same track model, (5) fusion of
the detectors over time.  Doing the check first guarantees that the first
frame after a jump never uses the model of the previous sequence, and using
this one class in both adapters guarantees identical behaviour online and
offline.

When the rail-level fit fails (an object right in front of the train blocks
the near track bed) the last good rail/track model is kept for
``model_hold_s`` so that D2 and the fusion can go on; the status can then be
``detected`` but never ``clear``.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import DetectorConfig
from .det_beam import BeamDetector
from .detector import FrameResult, ObstacleDetector
from .frame import Frame, FrameBuilder
from .fusion import FusionFilter
from .range_image import build_range_image
from .track_model import corridor_support, extension_weight, median_model
from .temporal import TemporalDecision, TemporalFilter


def header_stamp_ns(msg) -> Optional[int]:
    """Stamp of a message header in ns (None if it has no header)."""
    try:
        st = msg.header.stamp
        return int(st.sec) * 1_000_000_000 + int(st.nanosec)
    except AttributeError:
        return None


def _find_d3_model(cfg: DetectorConfig) -> str:
    cands = [cfg.d3_model_path, os.environ.get("METRO_D3_MODEL", ""), "/workspace/models/d3_unet.pt",
             os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "d3_unet.pt"))]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


class DetectionPipeline:
    def __init__(self, cfg: DetectorConfig, device_manager=None, logger=None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("metro_obstacle_detector.pipeline")
        self.dm = device_manager
        self.nn = None            # D3 (NNDetector) when PyTorch and a model are available
        self.nn_note = ""
        self._nn_failed = 0
        self._nn_slow = 0
        # execution: one worker thread runs D1 beside the range image and D3 beside D2 (identical results;
        # METRO_PARALLEL=0 runs everything in the calling thread); GPU binning of unorganised clouds
        # (set before D3 is attached: its warm-up also runs in the worker thread)
        self.parallel = os.environ.get("METRO_PARALLEL", "1").strip() not in ("0", "false", "no", "off")
        self._pool = None
        self._pool_lock = threading.Lock()
        self._bin_gpu_off = os.environ.get("METRO_GPU_BINNING", "1").strip() in ("0", "false", "no", "off")
        if cfg.d3_enabled:
            self.attach_device_manager(device_manager)
        else:
            self.nn_note = "D3 off: disabled (d3_enabled=false)"
            self.log.info(self.nn_note)
        self.detector = ObstacleDetector(cfg)
        self.temporal = TemporalFilter(cfg)
        self.fusion = FusionFilter(cfg)
        self.beam = BeamDetector(cfg)
        self.frames = FrameBuilder(cfg)
        self._held = None   # (rail, curve, stamp_ns) of the last frame with a good model
        self._prev_rng = None   # near-range layer of the previous range image (static-scene test)
        self._static_models = []   # per-frame models collected while the scene is static (warm-up)
        self._frozen = None        # (rail, curve) frozen after the warm-up of a static scene
        self._support_hist = []    # corridor support of the last frames (D2 reports up to their median + margin)
        self._bend_hist = []       # wall-sweep limits of the last frames (inf = none)
        self._quasi_fits = []
        self._reset_corridor()

    def _reset_corridor(self) -> None:
        self._b_long = []          # |b| of the per-frame fits while the scene is quasi-static (extension decision)
        self._ext_prev = False     # the straight-section extension was engaged in the previous frame (hysteresis)

    # ------------------------------------------------------------------ execution helpers (WP-S)
    def _worker(self):
        if not self.parallel:
            return None
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metro-det")
            return self._pool

    def close(self) -> None:
        """Stop the worker thread (optional; it is idle between frames)."""
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

    def _bin_device(self):
        """Device for binning unorganised clouds: the GPU while the device manager is in GPU mode, else None."""
        dm = self.dm
        if self._bin_gpu_off or dm is None or getattr(dm, "mode", "") != "gpu":
            return None
        dev = getattr(dm, "device", None)
        return dev if dev is not None and getattr(dev, "type", "") == "cuda" else None

    def _bin_failed(self, exc: BaseException) -> None:
        """GPU binning raised: this frame was binned on the CPU; the GPU is reported as failed (hot switch)."""
        self._bin_gpu_off = True
        self.log.warning("range image: GPU binning failed (%s) - binning on the CPU from now on" % exc)
        try:
            if self.dm is not None and hasattr(self.dm, "report_failure"):
                self.dm.report_failure(exc)
        except Exception as rep_exc:  # never let the fallback itself fail the frame
            self.log.warning("range image: report_failure raised %s" % rep_exc)

    def _range_image(self, frame: Frame):
        return build_range_image(frame, az_limit_deg=self.cfg.d2_az_limit_deg, device=self._bin_device(),
                                 on_device_error=self._bin_failed)

    def _run_d3(self, ri, rail, curve):
        """D3 on this frame (worker thread or inline): (candidates, ms spent in the network call)."""
        t = time.perf_counter()
        from .nn_input import aux_from_track
        out = self.nn.step(ri, aux_from_track(ri, rail, curve))
        return out, (time.perf_counter() - t) * 1e3

    def attach_device_manager(self, dm) -> None:
        """Create D3 on the given device manager (or on one of our own); never raises."""
        if not self.cfg.d3_enabled:
            self.nn_note = "D3 off: disabled (d3_enabled=false)"
            self.log.info(self.nn_note)
            return
        try:
            model = _find_d3_model(self.cfg)
            if not model:
                self.nn_note = "D3 off: no model file"
                self.log.info(self.nn_note)
                return
            if dm is None:
                from .device import DeviceManager
                dm = DeviceManager(prefer="auto", logger=self.log)   # probes in its constructor
            self.dm = dm
            from .det_nn import NNDetector
            self.nn = NNDetector(dm, model, logger=self.log,
                                 range_image_params={"az_step_deg": 0.1, "binned_res_deg": (0.09, 0.08),
                                                     "az_limit_deg": self.cfg.d2_az_limit_deg})
            t0 = time.perf_counter()
            try:
                # the first convolutions take seconds (kernel setup) in every thread that runs them: warm up in
                # the thread D3 runs in (the worker; a main-thread warm-up left 4 s on the first frame)
                pool = self._worker()
                if pool is not None:
                    pool.submit(self.nn.warmup).result()
                else:
                    self.nn.warmup()
            except Exception as exc:
                self.log.warning("D3 warm-up failed: %s" % exc)
            self.nn_note = "D3 on: %s (%s), warm-up %.1f s" % (os.path.basename(model), getattr(dm, "mode", "?"),
                                                              time.perf_counter() - t0)
        except Exception as exc:  # torch missing, model incompatible, ...: the detector runs without D3
            self.nn = None
            self.nn_note = "D3 off: %s" % exc
        if not self.nn_note:
            self.nn_note = "D3 off"
        self.log.info(self.nn_note)

    def reset(self) -> None:
        """Forget all state (stale input, restart)."""
        self.detector.reset()
        self.temporal.reset()
        self.fusion.reset()
        self._held = None
        self._prev_rng = None
        self._static_models = []
        self._quasi_fits = []
        self._frozen = None
        self._support_hist = []
        self._bend_hist = []
        self._reset_corridor()

    def build_frame(self, msg) -> Frame:
        """Parse a PointCloud2 (or the raw-CDR stand-in) into a Frame with the stream's sensor profile."""
        return self.frames.build(msg)

    def _ensure_axes(self, frame: Frame) -> None:
        """Use the axes of the frame's sensor profile (auto-detected for another lidar convention)."""
        p = frame.profile
        c = self.detector.cfg
        if (p.forward_axis, p.up_axis) != (c.forward_axis, c.up_axis):
            self.detector = ObstacleDetector(dataclasses.replace(self.cfg, forward_axis=p.forward_axis, up_axis=p.up_axis))
            self.beam = BeamDetector(self.detector.cfg)

    def _begin(self, stamp_ns: Optional[int]) -> bool:
        reset = self.temporal.check_time(stamp_ns)
        if reset:
            self.detector.reset()
            self.fusion.reset()
            self._held = None
            self._prev_rng = None
            self._static_models = []
            self._quasi_fits = []
            self._frozen = None
            self._support_hist = []
            self._bend_hist = []
            self._reset_corridor()
        return reset

    def _median_model(self, models):
        return median_model(models, straight_keep_abs_b=self.cfg.straight_keep_abs_b)

    def _static_fraction(self, ri) -> float:
        """Fraction of pixels (valid in both frames) whose range did not change since the previous frame."""
        prev, cur = self._prev_rng, ri.rng[0]
        self._prev_rng = cur
        if prev is None or prev.shape != cur.shape:
            return 0.0
        # the scan pattern of a raster lidar (Falcon Prime) jitters by a fraction of a pixel from frame
        # to frame, so a pixel is compared with the 3x3 neighbourhood of the previous image
        pv = prev > 0
        lo = ndimage.minimum_filter(np.where(pv, prev, np.inf), size=3)
        hi = ndimage.maximum_filter(np.where(pv, prev, -np.inf), size=3)
        both = (cur > 0) & np.isfinite(lo)
        n = int(both.sum())
        if n < 1000:
            return 0.0
        tol = self.cfg.static_tol_m + 0.002 * cur[both]
        ok = (cur[both] >= lo[both] - tol) & (cur[both] <= hi[both] + tol)
        return float(ok.sum()) / n

    # ---- corridor support (verified range) ------------------------------------------------------------------
    def _track_curvature(self, res: FrameResult, static_frac: float) -> None:
        """Collect |b| of the frame's own fit while the scene is quasi-static (cleared as soon as it moves)."""
        cfg = self.cfg
        fit = res.fit[1] if (res.fit is not None and res.fit[1] is not None) else None
        if cfg.static_hold and static_frac >= cfg.quasi_static_frac and cfg.corridor_static_b_window > 0:
            if fit is not None and fit.ok:
                self._b_long = (self._b_long + [abs(float(fit.b))])[-cfg.corridor_static_b_window:]
        else:
            self._b_long = []

    def _corridor_support(self, rail, curve) -> float:
        """This frame's corridor support for the verified range: the wall/ceiling-centre support extended towards
        the ceiling reach by a weight that fades with |b| (track_model.extension_weight).  While the train stands
        the curvature is the median of the frame fits collected so far (the tunnel cannot bend while standing:
        Falcon's per-frame |b| of 0.7-2.2e-4 around the 1.0/1.5e-4 thresholds flipped the support 81 <-> 141 m),
        unless this frame's own model is clearly curved."""
        cfg = self.cfg
        b_now = abs(float(curve.b))
        b_eff = b_now
        if len(self._b_long) >= cfg.corridor_static_b_min_frames and b_now <= cfg.corridor_static_b_max:
            b_eff = float(np.median(self._b_long))
        w = extension_weight(b_eff, self._ext_prev, cfg)
        self._ext_prev = w >= 0.5
        return corridor_support(curve, rail, w)

    def step(self, xyz: np.ndarray, stamp_ns: Optional[int]) -> Tuple[FrameResult, TemporalDecision]:
        """Legacy entry (points only): D1 + the D1 temporal filter."""
        reset = self._begin(stamp_ns)
        res = self.detector.process(xyz)
        return res, self.temporal.update(res, reset=reset)

    def step_frame(self, frame: Frame) -> Tuple[FrameResult, TemporalDecision]:
        cfg = self.cfg
        reset = self._begin(frame.stamp_ns)
        self._ensure_axes(frame)
        if not cfg.fusion_enabled:
            res = self.detector.process(frame.xyz)
            return res, self.temporal.update(res, reset=reset)
        t0 = time.perf_counter()
        # D1's inputs depend on the static-scene test (range image) only when a frozen model or >= 2
        # quasi-static fits exist; otherwise D1 = process(xyz) runs in the worker beside the range image
        pool = self._worker()
        d1_job = None
        if pool is not None and self._frozen is None and len(self._quasi_fits) < 2:
            d1_job = pool.submit(self.detector.process, frame.xyz, fixed_model=None, smooth_with=None)
        d1_wait = None
        try:
            ri = self._range_image(frame)
            static_frac = self._static_fraction(ri)
            t_ri = (time.perf_counter() - t0) * 1e3          # the range image alone, before the join with D1
        finally:
            if d1_job is not None:
                t_w = time.perf_counter()
                res = d1_job.result()
                d1_wait = (time.perf_counter() - t_w) * 1e3  # how long the frame waited for D1 after the range image
        static = cfg.static_hold and static_frac >= cfg.static_min_frac
        if not static:
            self._static_models, self._frozen = [], None
        # nothing moved: the track model cannot have changed.  The first static_warmup frames are
        # still estimated and their element-wise median is frozen afterwards (a single bad frame
        # must not become the model for the whole stop)
        fixed = self._frozen if static else None
        quasi = (fixed is None and cfg.static_hold and static_frac >= cfg.quasi_static_frac and len(self._quasi_fits) >= 2)
        if d1_job is None:
            res = self.detector.process(frame.xyz, fixed_model=fixed, smooth_with=self._quasi_fits if quasi else None)
        if (fixed is None and res.fit is not None and res.fit[0] is not None and res.fit[0].ok
                and res.fit[1] is not None and res.fit[1].ok):
            if static_frac >= cfg.quasi_static_frac:
                self._quasi_fits = (self._quasi_fits + [res.fit])[-max(1, cfg.quasi_static_frames - 1):]
            else:
                self._quasi_fits = []
        elif fixed is not None:
            self._quasi_fits = []        # a frozen stop: the fits before it must not smooth the frames after it
        self._track_curvature(res, static_frac)
        res.timings_ms["range_image"] = t_ri
        if d1_wait is not None:
            res.timings_ms["d1_wait"] = d1_wait
        res.static_frac = static_frac
        t0 = time.perf_counter()
        rail, curve = res.rail, res.curve
        model_ok = rail is not None and rail.ok and curve is not None
        if model_ok and static and self._frozen is None and curve.ok:
            self._static_models.append((rail, curve))
            if len(self._static_models) >= cfg.static_warmup:
                self._frozen = self._median_model(self._static_models)
                rail, curve = self._frozen
                res.rail, res.curve = rail, curve
        if model_ok:
            self._held = (rail, curve, frame.stamp_ns)
        elif self._held is not None and frame.stamp_ns is not None and self._held[2] is not None \
                and 0.0 <= (frame.stamp_ns - self._held[2]) / 1e9 <= cfg.model_hold_s:
            rail, curve = self._held[0], self._held[1]
            res.held_model = True
            model_ok = True
        # D3 decision first: it runs in the worker while D2 runs here (the verified range only filters its output)
        self._frame_no = getattr(self, "_frame_no", 0) + 1
        on_cpu = self.dm is not None and getattr(self.dm, "mode", "gpu") != "gpu"
        run_d3 = self.nn is not None and model_ok and self._nn_slow < 3
        if run_d3 and on_cpu and self._frame_no % cfg.d3_cpu_stride:
            run_d3 = False            # emergency CPU mode: the network runs every d3_cpu_stride-th frame
        d3_job = None
        if run_d3 and pool is not None:
            t3 = time.perf_counter()
            d3_job = pool.submit(self._run_d3, ri, rail, curve)
        d2 = []
        if model_ok and cfg.d2_enabled:
            t1 = time.perf_counter()
            if curve.ok:
                self._support_hist = (self._support_hist + [self._corridor_support(rail, curve)])[-5:]
            self._bend_hist = (self._bend_hist + [float(res.bend_limit_m)])[-5:]
            # the support of the track model jumps from frame to frame (87 -> 202 m in a curve transition);
            # D2 reports up to the median of the last frames, not up to the optimistic frame
            max_d_report = (float(np.median(self._support_hist)) + cfg.report_beyond_support_m) if self._support_hist else cfg.d_max
            # the corridor is "verified" up to the track-model support (median of recent frames), the far rail
            # validity (ceiling extension or observed bed + margin) and the analysed range
            rail_reach = getattr(rail, "far_max_d", rail.max_d_observed) + cfg.report_beyond_support_m
            res.verified_to_m = float(min(cfg.d_max, max_d_report, rail_reach))
            # a wall swept into the corridor: hard cap (no margin beyond), median of recent frames for stability
            bend_cap = float(np.median(self._bend_hist)) if self._bend_hist else float("inf")
            if math.isfinite(bend_cap) and bend_cap < res.verified_to_m:
                res.verified_to_m = bend_cap
            d2 = self.beam.process(ri, rail, curve, max_d_report=res.verified_to_m)
            # D1 candidates beyond the verified range are not reported either (its own limit is per frame)
            keep, drop = [], []
            for c in res.candidates:
                (keep if c.forward_m <= res.verified_to_m else drop).append(c)
            for c in drop:
                c.rejected_reason = "beyond the verified range (%.0f m)" % res.verified_to_m
            res.candidates, res.rejected = keep, res.rejected + drop
            res.timings_ms["d2"] = (time.perf_counter() - t1) * 1e3
        res.d2 = d2
        d3 = []
        if run_d3:
            if d3_job is None:
                t3 = time.perf_counter()
            d3_own = None
            try:
                if d3_job is not None:
                    d3, d3_net_ms = d3_job.result()
                    t_filter = time.perf_counter()
                else:
                    d3, d3_net_ms = self._run_d3(ri, rail, curve)
                    t_filter = None
                # the network sees the whole crop: keep only components inside the clearance corridor
                hw = cfg.gauge_halfwidth_body + cfg.d3_corridor_margin_m
                if math.isfinite(res.verified_to_m):
                    d3 = [c for c in d3 if c.d <= res.verified_to_m]
                # a component longer along the track than any obstacle is floor / wall texture (parametric
                # synthetic floors gave 10-20 m blobs that dragged real tracks away from their object)
                d3 = [c for c in d3 if not (c.extent_dlh and c.extent_dlh[0] > cfg.cluster_max_extent_d)]
                d3 = [c for c in d3 if c.lateral_offset is None or (abs(c.lateral_offset) <= hw and
                      -cfg.d3_corridor_margin_m <= (c.h_above_rail if c.h_above_rail is not None else 1.0) <= cfg.box_top_above_rail + cfg.d3_corridor_margin_m)]
                # the budget check uses D3's own time (network call + filtering), not the wait for D2
                dt = (time.perf_counter() - t3) * 1e3 if t_filter is None else d3_net_ms + (time.perf_counter() - t_filter) * 1e3
                d3_own = dt
                if dt > cfg.d3_max_ms:
                    self._nn_slow += 1       # three slow frames in a row: give up on D3 (budget first)
                    if self._nn_slow >= 3:
                        self.log.warning("D3 disabled: %.0f ms per frame exceeds the budget (%d ms)" % (dt, cfg.d3_max_ms))
                else:
                    self._nn_slow = 0
                for c in d3:
                    if c.n_support < cfg.d3_min_pixels:
                        c.score *= 0.3           # a few pixels: weak evidence only, no vote
                self._nn_failed = 0
            except Exception as exc:
                self._nn_failed += 1
                if self._nn_failed <= 3 or self._nn_failed % 100 == 0:
                    self.log.warning("D3 failed on this frame (%d): %s" % (self._nn_failed, exc))
                if self._nn_failed >= 20:
                    self.log.error("D3 disabled after repeated failures")
                    self.nn = None
                d3 = []
            res.timings_ms["d3"] = d3_own if d3_own is not None else (time.perf_counter() - t3) * 1e3
        res.d3 = d3
        t2 = time.perf_counter()
        allow_clear = res.status != "unknown"
        dec = self.fusion.update(res, list(d2) + list(d3), rail if model_ok else None, self.detector.R, reset=reset, allow_clear=allow_clear,
                                 max_d=res.verified_to_m if math.isfinite(res.verified_to_m) else None)
        res.timings_ms["fusion"] = (time.perf_counter() - t2) * 1e3
        res.timings_ms["total"] = res.timings_ms.get("total", 0.0) + (time.perf_counter() - t0) * 1e3
        return res, dec

    def step_invalid(self, reason: str, stamp_ns: Optional[int]) -> Tuple[FrameResult, TemporalDecision]:
        """A frame that could not be decoded: UNKNOWN, but its stamp still counts for the gap check."""
        reset = self._begin(stamp_ns)
        res = FrameResult(status="unknown", reason=reason)
        if self.cfg.fusion_enabled:
            return res, self.fusion.update(res, [], None, self.detector.R, reset=reset, allow_clear=False)
        return res, self.temporal.update(res, reset=reset)
