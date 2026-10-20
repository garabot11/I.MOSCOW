"""Д3 input tensor from range images (interface D; owner: ml-builder).  NumPy only (no torch).

Per frame 7 channels (float32), in this order (``CHANNELS``):
    rng_near/200, rng_far/200, intensity_near/255, valid_near, h_above_rail/4 (0 if unknown),
    clip(lateral_from_centreline/4, -1, 1) (0 if unknown),
    residual = clip((r_near - r_expected) / max(1, r_expected), -1, 1) (0 where r_expected is unknown;
    a missing return where one is expected gives r_near = 0, i.e. -1).
The network input stacks the CURRENT frame first, then the previous one, then the one before
(channels 0-6, 7-13, 14-20) -> (21, n_rows, n_cols).  With a shorter history the oldest available
frame is repeated.

Range images come from the main session's ``range_image.build_range_image(frame, ...)`` (organised
Hesai: (2, 128, W) rows = rings sorted by elevation, row 0 highest, columns = true azimuth at
0.1 deg; binned unorganised sensors such as Falcon Prime: ``organized=False``, ``ring_of_row = -1``,
any H x W).  Both have ``el_deg`` and ``az_deg`` DESCENDING (column 0 = largest azimuth = +l side,
like the message order).

Crop (``CropConfig``) is defined by angle VALUES, never by row/column numbers:
  * rows: the elevation window [el_bottom_deg, el_top_deg] mapped onto ``n_rows`` target rows
    (row 0 = top);
  * columns: ``n_cols`` target columns of ``az_pitch_deg`` centred on ``az_centre_deg``, descending
    like the range image (``az_descending``; column n_cols // 2 = the centre azimuth).
Per axis, if the source image has the target resolution inside the window (within
``native_tolerance``), the source rows/columns are taken as they are -- no resampling: for the
Hesai layout the default crop is exactly rings 25..88 (the 64-ring dense band +1.97 .. -6.07 deg)
and 256 columns +12.8 .. -12.7 deg.  Otherwise (e.g. the Falcon image binned at 0.09 x 0.08 deg)
the source pixels are resampled onto the uniform target grid: every source pixel goes to the
target cell containing its angles, the nearest valid return wins when several fall into one cell,
and cells that receive nothing (coarser source) copy the nearest source pixel.  ``CropMap`` keeps,
for every target pixel, the source (row, column) so that the detector (det_nn) and the training
targets (slot_of_pixel) use exactly the same pixels.

Fallbacks, used only when ``range_image`` cannot be imported: a field-for-field copy of the
``RangeImage`` dataclass and ``HesaiStandinBuilder`` (organised Hesai layout, same conventions).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

CHANNELS = ("rng_near", "rng_far", "intensity_near", "valid_near", "h_above_rail",
            "lateral_from_centreline", "residual")
N_FRAME_CHANNELS = len(CHANNELS)
HISTORY = 3
N_INPUT_CHANNELS = N_FRAME_CHANNELS * HISTORY  # 21
RANGE_SCALE_M = 200.0
INTENSITY_SCALE = 255.0
AUX_SCALE_M = 4.0
AUX_KEYS = ("h_above_rail", "lateral_from_centreline", "r_expected")

# ----------------------------------------------------------------------------------------------
# RangeImage (interface D).  Owner: main session (range_image.py).
# ----------------------------------------------------------------------------------------------
try:  # pragma: no cover - depends on the main session's module being present
    from .range_image import RangeImage  # type: ignore
    RANGE_IMAGE_SOURCE = "range_image"
except Exception:  # ImportError, or range_image importing something unavailable
    RANGE_IMAGE_SOURCE = "nn_input stand-in"

    @dataclass
    class RangeImage:  # FALLBACK: field-for-field copy of range_image.RangeImage
        rng: np.ndarray            # (2, H, W) float32 metres; [0] near/first, [1] far/last; 0 = no return
        intensity: np.ndarray      # (2, H, W) float32
        xyz: np.ndarray            # (2, H, W, 3) float32 sensor frame
        dlh: np.ndarray            # (2, H, W, 3) float32 analysis frame (rotation only)
        valid: np.ndarray          # (2, H, W) bool
        el_deg: np.ndarray         # (H,) rows sorted by elevation, row 0 = highest
        az_deg: np.ndarray         # (W,) column azimuth, 0 = forward, + towards +l
        ring_of_row: np.ndarray    # (H,) ring index of each row (-1 for angle-binned images)
        slot_of_pixel: np.ndarray  # (2, H, W) int64 message slot index (-1 = none)
        sensor: str                # "hesai128" | "falcon_prime" | "generic"
        organized: bool


# ----------------------------------------------------------------------------------------------
# Crop
# ----------------------------------------------------------------------------------------------
@dataclass
class CropConfig:
    n_rows: int = 64
    n_cols: int = 256
    el_top_deg: float = 1.97        # elevation window (sensor frame); default = Hesai dense band
    el_bottom_deg: float = -6.07
    az_centre_deg: float = 0.0      # crop centre (0 = forward)
    az_pitch_deg: float = 0.1       # target column pitch (Hesai firing pitch)
    az_descending: bool = True      # target column 0 = largest azimuth (+l side), like range_image
    native_tolerance: float = 0.02  # relative resolution mismatch up to which source rows/cols are used as is

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CropConfig":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise KeyError(f"unknown crop parameters: {unknown}")
        return cls(**d)

    @property
    def el_step_deg(self) -> float:
        return (self.el_top_deg - self.el_bottom_deg) / max(self.n_rows - 1, 1)

    def target_el(self) -> np.ndarray:
        return np.linspace(self.el_top_deg, self.el_bottom_deg, self.n_rows)

    @property
    def az_sign(self) -> float:
        return -1.0 if self.az_descending else 1.0

    def target_az(self) -> np.ndarray:
        return self.az_centre_deg + self.az_sign * (np.arange(self.n_cols) - self.n_cols // 2) * self.az_pitch_deg


@dataclass
class CropMap:
    """For every target pixel the source (row, col) in the RangeImage, -1 = outside the field of view."""

    src_row: np.ndarray            # (n_rows, n_cols) int32
    src_col: np.ndarray            # (n_rows, n_cols) int32
    native_rows: bool
    native_cols: bool

    @property
    def shape(self) -> Tuple[int, int]:
        return self.src_row.shape

    @property
    def inside(self) -> np.ndarray:
        return (self.src_row >= 0) & (self.src_col >= 0)

    def gather(self, arr: np.ndarray, fill=0) -> np.ndarray:
        """``arr`` of shape (..., H, W) -> (..., n_rows, n_cols); ``fill`` outside the field of view."""
        ins = self.inside
        r = np.where(ins, self.src_row, 0)
        c = np.where(ins, self.src_col, 0)
        out = arr[..., r, c]
        if not ins.all():
            out = np.where(ins, out, np.asarray(fill, dtype=out.dtype))
        return out


def _wrap180(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _nearest(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Index of the nearest value of the monotone array ``src`` (either direction) for every target."""
    asc = bool(src[-1] >= src[0])
    v = src if asc else src[::-1]
    pos = np.searchsorted(v, tgt)
    lo = np.clip(pos - 1, 0, len(v) - 1)
    hi = np.clip(pos, 0, len(v) - 1)
    pick = np.where(np.abs(v[lo] - tgt) <= np.abs(v[hi] - tgt), lo, hi)
    return pick if asc else len(v) - 1 - pick


def _monotone(a: np.ndarray) -> bool:
    if len(a) < 2:
        return True
    d = np.diff(a)
    return bool(np.all(d > 0) or np.all(d < 0))


def _row_axis(el: np.ndarray, cfg: CropConfig):
    """Rows: (source -> target map or native block, target -> fill source row, native)."""
    el = np.asarray(el, dtype=np.float64)
    H = len(el)
    step = cfg.el_step_deg
    n = cfg.n_rows
    if H == 0:
        return np.zeros(0, np.int64), np.full(n, -1, np.int64), False
    lo, hi = cfg.el_bottom_deg - 0.5 * step, cfg.el_top_deg + 0.5 * step
    inwin = np.flatnonzero((el >= lo) & (el <= hi))
    if H >= 2 and len(inwin) > 0 and abs(len(inwin) - n) <= max(cfg.native_tolerance * n, 0.5) \
            and np.all(np.diff(el) <= 0):
        # native: n contiguous source rows that best cover the window (by value)
        best, best_cost = 0, np.inf
        for s in range(max(H - n, 0) + 1):
            e = min(s + n, H) - 1
            cost = max(abs(el[s] - cfg.el_top_deg), abs(el[e] - cfg.el_bottom_deg))
            if cost < best_cost - 1e-9:
                best, best_cost = s, cost
        rows = np.arange(best, best + n)
        rows = np.where(rows < H, rows, -1)
        return rows.astype(np.int64), None, True
    t = np.rint((cfg.el_top_deg - el) / step).astype(np.int64)
    smap = np.where((t >= 0) & (t < n), t, -1)
    tel = cfg.target_el()
    near = _nearest(el, tel) if _monotone(el) else np.abs(el[None, :] - tel[:, None]).argmin(axis=1)
    src_step = float(np.median(np.abs(np.diff(el)))) if H > 1 else step
    ok = np.abs(el[near] - tel) <= max(step, src_step)
    return smap, np.where(ok, near, -1).astype(np.int64), False


def _col_axis(az: np.ndarray, cfg: CropConfig):
    """Columns: (source -> target map or native block, target -> fill source column, native)."""
    az = _wrap180(az)
    W = len(az)
    n = cfg.n_cols
    p = cfg.az_pitch_deg
    sgn = cfg.az_sign
    taz = cfg.target_az()
    if W == 0:
        return np.zeros(0, np.int64), np.full(n, -1, np.int64), False
    d = np.diff(np.unwrap(np.radians(az))) if W > 1 else np.array([sgn * np.radians(p)])
    src_pitch = float(np.degrees(np.median(np.abs(d))))
    monotone = bool(np.all(d > 0) or np.all(d < 0))
    full_circle = src_pitch * W > 359.0
    if monotone and abs(src_pitch - p) <= cfg.native_tolerance * p:
        src_sign = 1.0 if d[0] > 0 else -1.0
        c0 = int(np.abs(_wrap180(az - cfg.az_centre_deg)).argmin())
        offs = np.arange(n) - n // 2
        cols = c0 + offs if src_sign == sgn else c0 - offs
        if full_circle:
            cols = cols % W
        else:
            cols = np.where((cols >= 0) & (cols < W), cols, -1)
        return cols.astype(np.int64), None, True
    rel = _wrap180(az - cfg.az_centre_deg)              # target az relative to the centre: sgn*(j - n//2)*p
    t = np.rint(sgn * rel / p).astype(np.int64) + n // 2
    smap = np.where((t >= 0) & (t < n), t, -1)
    trel = taz - cfg.az_centre_deg
    if _monotone(rel):
        near = _nearest(rel, trel)
    else:
        near = np.abs(_wrap180(rel[None, :] - trel[:, None])).argmin(axis=1)
    ok = np.abs(_wrap180(rel[near] - trel)) <= max(p, src_pitch)
    return smap, np.where(ok, near, -1).astype(np.int64), False


def _axes(ri, cfg: CropConfig, cache: Optional[dict]):
    """Per-axis maps, as (source -> target index, target -> fill source index, native) for rows and cols."""
    key = None
    if cache is not None:
        key = (np.asarray(ri.el_deg).tobytes(), np.asarray(ri.az_deg).tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
    rmap, rfill, rnat = _row_axis(ri.el_deg, cfg)
    cmap, cfill, cnat = _col_axis(ri.az_deg, cfg)
    n_r, n_c = cfg.n_rows, cfg.n_cols
    if rnat:  # native block (target -> source) also expressed as a source -> target map
        sel = rmap
        rmap = np.full(len(ri.el_deg), -1, np.int64)
        ok = sel >= 0
        rmap[sel[ok]] = np.arange(n_r)[ok]
        rfill = sel
    if cnat:
        sel = cmap
        cmap = np.full(len(ri.az_deg), -1, np.int64)
        ok = sel >= 0
        cmap[sel[ok]] = np.arange(n_c)[ok]
        cfill = sel
    ax = (rmap, rfill, rnat, cmap, cfill, cnat)
    if cache is not None:
        if len(cache) > 16:
            cache.clear()
        cache[key] = ax
    return ax


def compute_crop_map(ri, cfg: CropConfig, _cache: Optional[dict] = None) -> CropMap:
    """Target-grid -> source-pixel map of one RangeImage (see the module docstring).

    Native on both axes (Hesai): a fixed block of source rows x columns.  Otherwise the map depends on
    the frame's ranges (nearest valid return wins per target cell) and is recomputed every frame.
    """
    rmap, rfill, rnat, cmap, cfill, cnat = _axes(ri, cfg, _cache)
    n_r, n_c = cfg.n_rows, cfg.n_cols
    if rnat and cnat:
        src_row = np.repeat(rfill[:, None], n_c, axis=1).astype(np.int32)
        src_col = np.repeat(cfill[None, :], n_r, axis=0).astype(np.int32)
        out = (src_row < 0) | (src_col < 0)
        src_row[out] = -1
        src_col[out] = -1
        return CropMap(src_row, src_col, True, True)
    # at least one axis is resampled -> pooling over the source pixels of each target cell
    src_row = np.full((n_r, n_c), -1, np.int32)
    src_col = np.full((n_r, n_c), -1, np.int32)
    rs = np.where(rmap >= 0)[0]
    cs = np.where(cmap >= 0)[0]
    if len(rs) and len(cs):
        lin = (rmap[rs][:, None] * n_c + cmap[cs][None, :]).ravel()
        rng = np.asarray(ri.rng[0])[np.ix_(rs, cs)].astype(np.float64)
        val = np.asarray(ri.valid[0])[np.ix_(rs, cs)]
        keyv = np.where(val & (rng > 0), rng, np.inf).ravel()
        order = np.lexsort((keyv, lin))
        ls = lin[order]
        first = np.ones(len(ls), bool)
        first[1:] = ls[1:] != ls[:-1]
        win = order[first]
        tl = ls[first]
        src_row.ravel()[tl] = rs[win // len(cs)]
        src_col.ravel()[tl] = cs[win % len(cs)]
    empty = src_row < 0
    if empty.any():
        fr = np.repeat(rfill[:, None], n_c, axis=1)
        fc = np.repeat(cfill[None, :], n_r, axis=0)
        ok = empty & (fr >= 0) & (fc >= 0)
        src_row[ok] = fr[ok]
        src_col[ok] = fc[ok]
    return CropMap(src_row, src_col, rnat, cnat)


# ----------------------------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------------------------
def aux_from_track(ri, rail=None, curve=None) -> Dict[str, Optional[np.ndarray]]:
    """Per-pixel aux arrays (H, W) for the near return from Д1's track model.

    ``rail``: callable d -> rail-level h (``track_model.RailLevel``); ``curve``: callable d -> centre-line
    l (``track_model.TrackCurve``), both in the analysis frame of ``ri.dlh`` (same as Д1:
    h_above_rail = h - rail(d), lateral = l - curve(d)).  NaN where the near return is missing.
    """
    out: Dict[str, Optional[np.ndarray]] = {"h_above_rail": None, "lateral_from_centreline": None, "r_expected": None}
    dlh = np.asarray(ri.dlh[0], dtype=np.float64)
    v = np.asarray(ri.valid[0], bool)
    d = dlh[..., 0]
    if rail is not None and bool(getattr(rail, "ok", True)):
        out["h_above_rail"] = np.where(v, dlh[..., 2] - rail(d), np.nan).astype(np.float32)
    if curve is not None and bool(getattr(curve, "ok", True)):
        out["lateral_from_centreline"] = np.where(v, dlh[..., 1] - curve(d), np.nan).astype(np.float32)
    return out


def _aux2d(aux: Optional[dict], key: str):
    if not aux:
        return None
    a = aux.get(key)
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:  # (2, H, W) -> near return
        a = a[0]
    return a


def frame_channels(ri, cmap: CropMap, aux: Optional[dict] = None) -> np.ndarray:
    """(7, n_rows, n_cols) float32 channels of one frame on the crop grid."""
    n_r, n_c = cmap.shape
    out = np.zeros((N_FRAME_CHANNELS, n_r, n_c), np.float32)
    valid = cmap.gather(np.asarray(ri.valid), fill=False)          # (2, h, w)
    rng = cmap.gather(np.asarray(ri.rng, dtype=np.float32), 0.0)
    inten = cmap.gather(np.asarray(ri.intensity, dtype=np.float32), 0.0)
    vn = valid[0] & np.isfinite(rng[0]) & (rng[0] > 0)
    vf = valid[1] & np.isfinite(rng[1]) & (rng[1] > 0)
    rn = np.where(vn, rng[0], 0.0)
    out[0] = rn / RANGE_SCALE_M
    out[1] = np.where(vf, rng[1], 0.0) / RANGE_SCALE_M
    out[2] = np.where(vn, inten[0], 0.0) / INTENSITY_SCALE
    out[3] = vn
    h = _aux2d(aux, "h_above_rail")
    if h is not None:
        hv = cmap.gather(h, np.nan)
        out[4] = np.where(vn & np.isfinite(hv), hv / AUX_SCALE_M, 0.0)
    lat = _aux2d(aux, "lateral_from_centreline")
    if lat is not None:
        lv = cmap.gather(lat, np.nan)
        out[5] = np.where(vn & np.isfinite(lv), np.clip(lv / AUX_SCALE_M, -1.0, 1.0), 0.0)
    rexp = _aux2d(aux, "r_expected")
    if rexp is not None:
        ev = cmap.gather(rexp, np.nan)
        known = cmap.inside & np.isfinite(ev) & (ev > 0)
        res = (rn - np.where(known, ev, 0.0)) / np.maximum(1.0, np.where(known, ev, 1.0))
        out[6] = np.where(known, np.clip(res, -1.0, 1.0), 0.0)
    return out


def stack_history(chans: Sequence[np.ndarray]) -> np.ndarray:
    """``chans`` = per-frame (7, h, w) arrays, CURRENT FIRST -> (21, h, w); pads by repeating the oldest."""
    if not chans:
        raise ValueError("need at least one frame")
    seq = list(chans)[:HISTORY]
    while len(seq) < HISTORY:
        seq.append(seq[-1])
    return np.concatenate(seq, axis=0).astype(np.float32, copy=False)


def build_input(frames: Sequence, auxs: Optional[Sequence[Optional[dict]]] = None,
                cfg: Optional[CropConfig] = None) -> Tuple[np.ndarray, CropMap]:
    """Stateless builder.  ``frames`` in CHRONOLOGICAL order (``frames[-1]`` = current frame).

    Returns the (21, n_rows, n_cols) input and the crop map of the current frame.
    """
    cfg = cfg or CropConfig()
    if not frames:
        raise ValueError("need at least one frame")
    auxs = list(auxs) if auxs is not None else [None] * len(frames)
    if len(auxs) != len(frames):
        raise ValueError("auxs must match frames")
    recent = list(zip(frames, auxs))[-HISTORY:][::-1]  # current first
    chans, cur_map = [], None
    for ri, aux in recent:
        cm = compute_crop_map(ri, cfg)
        if cur_map is None:
            cur_map = cm
        chans.append(frame_channels(ri, cm, aux))
    return stack_history(chans), cur_map


class NNInputBuilder:
    """Stateful builder: ``push(ri, aux)`` once per frame -> (21, h, w) input and the current crop map.

    Call ``reset()`` when the stream jumps (bag switch, time gap), like the temporal filter.
    """

    def __init__(self, cfg: Optional[CropConfig] = None):
        self.cfg = cfg or CropConfig()
        self._hist: Deque[np.ndarray] = deque(maxlen=HISTORY)
        self._cache: dict = {}
        self.last_map: Optional[CropMap] = None

    def reset(self) -> None:
        self._hist.clear()
        self.last_map = None

    def push(self, ri, aux: Optional[dict] = None) -> Tuple[np.ndarray, CropMap]:
        cm = compute_crop_map(ri, self.cfg, self._cache)
        ch = frame_channels(ri, cm, aux)
        self._hist.appendleft(ch)
        self.last_map = cm
        return stack_history(list(self._hist)), cm


# ----------------------------------------------------------------------------------------------
# FALLBACK range-image builder for the organised Hesai layout (range_image.py is the reference)
# ----------------------------------------------------------------------------------------------
STANDIN_BUILDER_ID = "nn_input.HesaiStandinBuilder/v2"
RANGE_IMAGE_BUILDER_ID = "range_image.build_range_image"


def _axis_matrix(forward_axis: str, up_axis: str) -> np.ndarray:
    def vec(spec):
        spec = spec.strip().lower()
        sign = -1.0 if spec.startswith("-") else 1.0
        idx = {"x": 0, "y": 1, "z": 2}[spec.lstrip("+-")]
        v = np.zeros(3)
        v[idx] = sign
        return v
    fwd, up = vec(forward_axis), vec(up_axis)
    if abs(float(fwd @ up)) > 1e-6:
        raise ValueError("forward_axis and up_axis must be orthogonal")
    return np.vstack([fwd, np.cross(up, fwd), up])  # same as config.DetectorConfig.axis_matrix


class HesaiStandinBuilder:
    """FALLBACK for range_image.build_range_image: RangeImage of the organised Hesai PointCloud2 layout.

    Layout facts used (measured on the real bags): slot i -> ring i % 128, message column i // 128;
    column pairs (2f, 2f+1) are the two returns of firing f: even = far/last -> index 1,
    odd = near/first -> index 0; per ring the beam azimuth is exactly az = a_r + pitch * f
    (pitch -0.1 deg) and the elevation is constant.  Rows are the rings sorted by elevation (row 0 =
    highest); columns are true azimuth bins of |pitch|, DESCENDING like range_image (column 0 = +l
    side; each ring shifted by its own integer offset, so a vertical object stays vertical; no two
    slots of a ring share a pixel).  The 307k (120 deg)
    and 921k (360 deg, wraps at +-180) layouts are both handled.  a_r / elevations / pitch are
    estimated from the valid returns of the first frame(s) and cached in the builder.
    """

    builder_id = STANDIN_BUILDER_ID

    def __init__(self, n_rings: int = 128, dual_return: bool = True, forward_axis: str = "-y",
                 up_axis: str = "+z", sensor: str = "hesai128"):
        self.n_rings = int(n_rings)
        self.dual_return = bool(dual_return)
        self.R = _axis_matrix(forward_axis, up_axis).astype(np.float32)
        self.sensor = sensor
        self._n = None
        self._pitch: Optional[float] = None
        self._icpt = np.full(self.n_rings, np.nan)
        self._el = np.full(self.n_rings, np.nan)

    # -- geometry estimation ---------------------------------------------------------------
    def _estimate(self, near: np.ndarray, valid: np.ndarray) -> None:
        nf, R = valid.shape
        dlh = near.astype(np.float64) @ self.R.T.astype(np.float64)
        rr = np.linalg.norm(dlh, axis=-1)
        az = np.degrees(np.arctan2(dlh[..., 1], dlh[..., 0]))
        el = np.degrees(np.arcsin(np.clip(dlh[..., 2] / np.maximum(rr, 1e-9), -1.0, 1.0)))
        if self._pitch is None:
            both = valid[1:] & valid[:-1]
            daz = _wrap180(az[1:] - az[:-1])[both]
            if len(daz) < 10:
                return
            self._pitch = float(np.round(np.median(daz), 4))
        f = np.arange(nf, dtype=np.float64)[:, None]
        a = np.radians(az - self._pitch * f)
        for r in range(R):
            m = valid[:, r]
            if m.sum() < 3 or np.isfinite(self._icpt[r]):
                continue
            self._icpt[r] = float(np.degrees(np.arctan2(np.sin(a[m, r]).mean(), np.cos(a[m, r]).mean())))
            self._el[r] = float(np.median(el[m, r]))

    @staticmethod
    def _filled(vals: np.ndarray, angular: bool = False) -> np.ndarray:
        """Fill rings without any valid return (they carry no data) by interpolation over the ring index."""
        v = vals.copy()
        ok = np.isfinite(v)
        if not ok.any():
            return np.zeros_like(v)
        if not ok.all():
            idx = np.arange(len(v))
            known = np.degrees(np.unwrap(np.radians(v[ok]))) if angular else v[ok]
            v[~ok] = np.interp(idx[~ok], idx[ok], known)
        return v

    # -- build ---------------------------------------------------------------------------
    def build(self, cloud, intensity: Optional[np.ndarray] = None) -> RangeImage:
        """``cloud``: structured array in message order (``cloud_io.pointcloud2_to_struct``) or (N, 3) xyz."""
        names = getattr(getattr(cloud, "dtype", None), "names", None)
        if names:
            xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)
            if intensity is None and "intensity" in names:
                intensity = np.asarray(cloud["intensity"], dtype=np.float32)
            if "ring" in names:
                ring = np.asarray(cloud["ring"][: min(len(cloud), 4 * self.n_rings)])
                if not np.array_equal(ring, np.arange(len(ring)) % self.n_rings):
                    raise ValueError("cloud is not in the organised ring-fastest Hesai layout (ring != slot % n_rings)")
        else:
            xyz = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        n = len(xyz)
        R = self.n_rings
        per_firing = R * (2 if self.dual_return else 1)
        if n == 0 or n % per_firing:
            raise ValueError(f"{n} points do not form whole firings of {per_firing} slots")
        if intensity is None:
            intensity = np.zeros(n, np.float32)
        if self._n != n:  # new layout -> re-estimate
            self._n, self._pitch = n, None
            self._icpt[:] = np.nan
            self._el[:] = np.nan
        n_cols = n // R
        X = xyz.reshape(n_cols, R, 3)
        I = np.asarray(intensity, dtype=np.float32).reshape(n_cols, R)
        if self.dual_return:
            near, far, inear, ifar = X[1::2], X[0::2], I[1::2], I[0::2]
            m_near = np.arange(1, n_cols, 2)
            m_far = np.arange(0, n_cols, 2)
        else:
            near = far = X
            inear = ifar = I
            m_near = m_far = np.arange(n_cols)
        nf = near.shape[0]
        vn = np.isfinite(near).all(-1) & ~(near == 0).all(-1)
        vf = np.isfinite(far).all(-1) & ~(far == 0).all(-1)
        if self._pitch is None or not np.isfinite(self._icpt).all():
            self._estimate(near, vn)
        if self._pitch is None:
            raise ValueError("too few valid returns to estimate the firing azimuth pitch")
        pitch = self._pitch
        p = abs(pitch)
        s = 1 if pitch > 0 else -1
        icpt = _wrap180(self._filled(self._icpt, angular=True))
        el_r = self._filled(self._el)
        full = nf * p > 359.0
        f = np.arange(nf)
        if full:
            W = int(round(360.0 / p))
            az0 = -180.0
            K = np.rint((icpt - az0) / p).astype(np.int64)
            cols = (K[None, :] + s * f[:, None]) % W
        else:
            az_lo = np.minimum(icpt, icpt + pitch * (nf - 1))
            az0 = p * np.floor(az_lo.min() / p)
            K = np.rint((icpt - az0) / p).astype(np.int64)
            cols = K[None, :] + s * f[:, None]
            W = int(cols.max()) + 1
        ring_of_row = np.argsort(-el_r, kind="stable")
        row_of_ring = np.empty(R, np.int64)
        row_of_ring[ring_of_row] = np.arange(R)
        rows = np.broadcast_to(row_of_ring[None, :], (nf, R))
        H = R
        xyz2 = np.zeros((2, H, W, 3), np.float32)
        inten2 = np.zeros((2, H, W), np.float32)
        valid2 = np.zeros((2, H, W), bool)
        slot2 = np.full((2, H, W), -1, np.int64)
        rr_idx = np.arange(R)[None, :]
        for k, (pts, ints, v, mcol) in enumerate(((near, inear, vn, m_near), (far, ifar, vf, m_far))):
            xyz2[k, rows, cols] = np.where(v[..., None], pts, 0.0)
            inten2[k, rows, cols] = np.where(v, ints, 0.0)
            valid2[k, rows, cols] = v
            slot2[k, rows, cols] = mcol[:, None] * R + rr_idx
        # descending azimuth (column 0 = largest azimuth), like range_image
        xyz2, inten2, valid2, slot2 = (np.ascontiguousarray(a[:, :, ::-1]) for a in (xyz2, inten2, valid2, slot2))
        rng2 = np.linalg.norm(xyz2, axis=-1).astype(np.float32)
        dlh2 = (xyz2 @ self.R.T).astype(np.float32)
        az_deg = (az0 + np.arange(W) * p).astype(np.float64)[::-1].copy()
        if full:
            az_deg = _wrap180(az_deg)
        ri = RangeImage(rng=rng2, intensity=inten2, xyz=xyz2, dlh=dlh2, valid=valid2,
                        el_deg=el_r[ring_of_row].astype(np.float64), az_deg=az_deg,
                        ring_of_row=ring_of_row.astype(np.int32), slot_of_pixel=slot2,
                        sensor=self.sensor, organized=True)
        try:
            ri.builder_id = STANDIN_BUILDER_ID  # lets det_nn check train/infer builder consistency
        except Exception:  # pragma: no cover - slotted dataclass in range_image.py
            pass
        return ri


def range_image_builder_id(ri) -> str:
    """Identity of the builder that produced ``ri`` (the fallback sets it; otherwise range_image's)."""
    return str(getattr(ri, "builder_id", RANGE_IMAGE_BUILDER_ID))
