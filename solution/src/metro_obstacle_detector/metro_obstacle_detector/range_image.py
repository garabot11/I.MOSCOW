"""Range image of one frame: rows = beams sorted by elevation, columns = true azimuth.

Organised clouds (the development Hesai-like sensor) are mapped by index: the slot of
ring ``k`` in firing column ``c`` goes to row ``rank(elevation_k)`` and to the column of
its true azimuth (column azimuth + the ring's azimuth offset, so that a vertical edge in
the scene is vertical in the image). Dual returns give two layers: [0] = near/first return
(odd message column), [1] = far/last return (even column); a single-echo firing has the
same value in both. Unorganised clouds (e.g. Falcon Prime from a driver without no-return
points) are binned by angle with a configurable resolution; [0] holds the nearest and [1]
the farthest return of a cell.

Every pixel keeps the message slot it came from (``slot_of_pixel``) so that results can be
mapped back to points and synthetic per-slot labels can be mapped onto the image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .frame import Frame
from .sensor_profile import axis_rotation


@dataclass
class RangeImage:
    rng: np.ndarray            # (2, H, W) float32 metres; [0] near/first, [1] far/last; 0 = no return
    intensity: np.ndarray      # (2, H, W) float32
    xyz: np.ndarray            # (2, H, W, 3) float32 sensor frame
    dlh: np.ndarray            # (2, H, W, 3) float32 analysis frame (rotation only)
    valid: np.ndarray          # (2, H, W) bool
    el_deg: np.ndarray         # (H,) row elevation, row 0 = highest
    az_deg: np.ndarray         # (W,) column azimuth, 0 = forward, + towards +l
    ring_of_row: np.ndarray    # (H,) ring of each row (-1 for binned images)
    slot_of_pixel: np.ndarray  # (2, H, W) int64 message slot (-1 = none)
    sensor: str
    organized: bool

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rng.shape[1], self.rng.shape[2]


def _empty(H: int, W: int):
    return (np.zeros((2, H, W), np.float32), np.zeros((2, H, W), np.float32), np.zeros((2, H, W, 3), np.float32),
            np.full((2, H, W), -1, np.int64))


def build_range_image(frame: Frame, az_step_deg: float = 0.1, binned_res_deg: Tuple[float, float] = (0.09, 0.08),
                      az_limit_deg: Optional[float] = None, device=None, on_device_error=None) -> RangeImage:
    """Range image of a frame.  ``device`` (a CUDA ``torch.device``) bins unorganised clouds on the GPU with a
    result identical to the CPU path; if that fails, the frame is binned on the CPU and ``on_device_error(exc)``
    is called (the caller switches to the CPU for the following frames)."""
    prof = frame.profile
    R = _axes_matrix(prof.forward_axis, prof.up_axis)
    if prof.organized and prof.ring_el_deg is not None and np.isfinite(prof.ring_el_deg).all():
        return _organized(frame, R, az_step_deg, az_limit_deg)
    return _binned(frame, R, binned_res_deg, az_limit_deg, device, on_device_error)


_AXES_CACHE: dict = {}


def _axes_matrix(forward_axis: str, up_axis: str) -> np.ndarray:
    """axis_rotation() per stream once (it builds a DetectorConfig on every call)."""
    key = (forward_axis, up_axis)
    R = _AXES_CACHE.get(key)
    if R is None:
        R = _AXES_CACHE[key] = axis_rotation(forward_axis, up_axis)
        R.setflags(write=False)
    return R


def _organized_map(frame: Frame, step: float, az_limit: Optional[float]):
    """Pixel of every point of the (cropped) frame; fixed per profile, so cached on it."""
    prof = frame.profile
    cache = prof.__dict__.setdefault("_ri_maps", {})
    key = (step, az_limit, len(frame.xyz))
    if key in cache:
        return cache[key]
    nr = prof.n_rings
    order = np.argsort(-prof.ring_el_deg)              # row 0 = highest elevation
    row_of_ring = np.empty(nr, np.int64)
    row_of_ring[order] = np.arange(nr)
    cols = prof.keep_cols if prof.keep_cols is not None else np.arange(prof.n_cols)
    col_az = prof.col_az_deg[cols]
    off = prof.ring_az_off_deg
    # true azimuth grid covering every (column, ring) of the kept sector
    lo = float(col_az.min() + off.min()) - step
    hi = float(col_az.max() + off.max()) + step
    if az_limit is not None:
        lo, hi = max(lo, -az_limit), min(hi, az_limit)
    W = int(np.ceil((hi - lo) / step)) + 1
    az_grid = (hi - step * np.arange(W)).astype(np.float32)   # + azimuth on the left, like the message order
    n = len(frame.xyz)
    slot = frame.slot if frame.slot is not None else np.arange(n, dtype=np.int64)
    ring = (slot % nr).astype(np.int64)
    col = slot // nr
    # dual returns: odd message column = near/first echo -> layer 0, even = far/last -> layer 1
    # (verified on the development bags: when the two echoes differ the odd column is always nearer)
    layer = (col % 2 == 0).astype(np.int64) if prof.dual_return else np.zeros(n, np.int64)
    ci = np.rint((hi - (prof.col_az_deg[col] + off[ring])) / step).astype(np.int64)
    ri = row_of_ring[ring]
    inmap = (ci >= 0) & (ci < W)
    flat = layer * (nr * W) + ri * W + ci
    # keep the first slot of a pixel if two slots round to the same pixel
    idx = np.flatnonzero(inmap)
    _, first = np.unique(flat[idx], return_index=True)
    keep = np.zeros(n, bool)
    keep[idx[first]] = True
    pts_idx = np.flatnonzero(keep)
    out = (nr, W, az_grid, order, pts_idx, flat[pts_idx], slot[pts_idx])
    cache[key] = out
    return out


def _signed_perm(R: np.ndarray):
    idx = [int(np.argmax(np.abs(R[i]))) for i in range(3)]
    sign = [np.float32(np.sign(R[i, idx[i]])) for i in range(3)]
    exact = all(abs(abs(R[i, idx[i]]) - 1.0) < 1e-9 for i in range(3))
    return idx, sign, exact


def _to_dlh(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    idx, sign, exact = _signed_perm(R)
    if not exact:
        return xyz @ R.T.astype(np.float32)
    out = np.empty_like(xyz)
    for i in range(3):
        np.multiply(xyz[..., idx[i]], sign[i], out=out[..., i])
    return out


def _organized(frame: Frame, R: np.ndarray, step: float, az_limit: Optional[float]) -> RangeImage:
    prof = frame.profile
    H, W, az_grid, order, pts_idx, flat, slot = _organized_map(frame, step, az_limit)
    p = frame.xyz[pts_idx]
    r = np.sqrt(np.einsum("ij,ij->i", p, p))
    r[~np.isfinite(r)] = 0.0
    n_pix = 2 * H * W
    rng = np.zeros(n_pix, np.float32)
    rng[flat] = r
    xyz = np.zeros((n_pix, 3), np.float32)
    xyz[flat] = p
    inten = np.zeros(n_pix, np.float32)
    if frame.intensity is not None:
        inten[flat] = frame.intensity[pts_idx]
    slots = np.full(n_pix, -1, np.int64)
    slots[flat] = slot                     # also no-return slots: every pixel knows its beam
    rng, inten, slots = rng.reshape(2, H, W), inten.reshape(2, H, W), slots.reshape(2, H, W)
    xyz = xyz.reshape(2, H, W, 3)
    if not prof.dual_return:
        rng[1], xyz[1], inten[1], slots[1] = rng[0], xyz[0], inten[0], slots[0]
    return RangeImage(rng=rng, intensity=inten, xyz=xyz, dlh=_to_dlh(xyz, R), valid=rng > 0,
                      el_deg=prof.ring_el_deg[order].astype(np.float32), az_deg=az_grid,
                      ring_of_row=order.astype(np.int64), slot_of_pixel=slots, sensor=prof.sensor, organized=True)


# ---------------------------------------------------------------------------
# Binned (unorganised) clouds.
#
# The image is defined by the original NumPy formulation (kept verbatim in tests/test_range_image_binned.py
# as the reference): float64 range r = sqrt(sum(p**2)), returns with r > 0.5 m, (d, l, h) = p @ R.T,
# az = degrees(arctan2(l, d)), el = degrees(arcsin(clip(h / r))), |az| <= az_limit; the grid starts at
# the frame's largest azimuth / elevation; per cell layer [0] = the nearest return (lowest point index
# among equal ranges) and layer [1] = the farthest (highest index among equal ranges).
#
# Two implementations of the per-cell winners ("plan") produce exactly that image:
#   * CPU: the same NumPy expressions, applied only to the points that can pass the azimuth limit (a
#     conservative wedge test on the raw coordinates drops the ~2/3 of a 120-degree cloud outside +-25
#     degrees before any square root or trigonometry), and one stable integer sort + segment min/max
#     instead of a two-key lexsort;
#   * GPU (torch, float64): the same arithmetic; the only operations that are not bit-identical between
#     CUDA and glibc are atan2/asin (CUDA <= 2 ulp, glibc 2.35 <= 1 ulp: |az_gpu - az_cpu| < 2e-13 degree
#     for any angle up to 180 degrees, _ERR_DEG).  The grid origin and size (largest / smallest angle) are
#     recomputed with NumPy on the few points near the extremes, so they are the reference values; the GPU
#     plan is only used when no point lies within 10x that error of a bin edge or of the azimuth limit,
#     otherwise the frame is binned on the CPU (``BIN_STATS["gpu_guard"]``, reason in ``GUARD_WHY``).
# Both feed the same fill function, so the images are identical to the reference.
# ---------------------------------------------------------------------------
BIN_STATS = {"cpu": 0, "gpu": 0, "gpu_guard": 0, "gpu_error": 0}
GUARD_WHY = {"limit": 0, "edge": 0, "axes": 0}
_ERR_DEG = 2e-13         # bound of |angle_gpu - angle_cpu| in degrees (3 ulp of 180 degrees + the degree factor)
_GUARD_DEG = 10 * _ERR_DEG   # closest a point may be to the azimuth limit (degrees)
_GUARD_Q = None          # override (bins) for tests; default 10x the error of (hi - angle) / step, see _guard_q
_DEG = 180.0 / np.pi     # the factor of np.degrees


@dataclass
class _BinPlan:
    H: int
    W: int
    az_hi: float
    el_hi: float
    cells: np.ndarray        # (k,) flat cell index (row * W + col) of every occupied cell
    pidx: Tuple[np.ndarray, np.ndarray]   # point index (into frame.xyz) of the winner of every cell, per layer
    rng: Tuple[np.ndarray, np.ndarray]    # float32 range of the winner, per layer


def _rotation_parts(R: np.ndarray):
    """(signed permutation (idx, sign) or None, R.T as float32) - computed once per axis matrix."""
    key = R.tobytes()
    hit = _ROT_CACHE.get(key)
    if hit is None:
        idx, sign, exact = _signed_perm(R)
        hit = ((idx, sign) if exact else None, R.T.astype(np.float32))
        _ROT_CACHE[key] = hit
    return hit


_ROT_CACHE: dict = {}


def _dlh_image(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``xyz @ R.T.astype(np.float32)`` of the image, bit for bit: for a signed permutation the product is a
    copy with signs (exact) and every zero is +0.0 like the result of the BLAS product."""
    perm, RT32 = _rotation_parts(R)
    if perm is None:
        return xyz @ RT32
    idx, sign = perm
    out = np.empty_like(xyz)
    for i in range(3):
        np.multiply(xyz[..., idx[i]], sign[i], out=out[..., i])
    out += np.float32(0.0)               # -0.0 -> +0.0 (x + 0 == x for every other value)
    return out


def _wedge_candidates(xyz: np.ndarray, perm, az_limit: Optional[float]) -> Optional[np.ndarray]:
    """Indices of the points that may lie within +-az_limit of forward (a superset), or None = all points.

    Exact test afterwards; this only has to keep every point the exact test can keep: |az| <= limit < 90 deg
    needs d >= 0 and |l| <= d * tan(limit) (a 1e-5 relative margin covers every rounding)."""
    if az_limit is None or perm is None or not (0.0 <= az_limit < 89.0):
        return None
    idx, sign = perm
    d = xyz[:, idx[0]] * sign[0]
    l = xyz[:, idx[1]]
    t = np.float32(np.tan(np.radians(az_limit)) * (1.0 + 1e-5) + 1e-6)
    with np.errstate(invalid="ignore"):
        return np.flatnonzero((d >= 0) & ((np.abs(l) <= d * t) | (d < 1e-30)))


def _plan_cpu(frame: Frame, R: np.ndarray, res: Tuple[float, float], az_limit: Optional[float]) -> Optional[_BinPlan]:
    perm, _ = _rotation_parts(R)
    cand = _wedge_candidates(frame.xyz, perm, az_limit)
    p = (frame.xyz if cand is None else frame.xyz[cand]).astype(np.float64)
    r = np.sqrt((p ** 2).sum(axis=1))
    ok = np.isfinite(r) & (r > 0.5)
    sub = np.flatnonzero(ok)
    dlh = p[sub] @ R.T
    rr = r[sub]
    idx = sub if cand is None else cand[sub]
    az = np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0]))
    el = np.degrees(np.arcsin(np.clip(dlh[:, 2] / rr, -1, 1)))
    if az_limit is not None:
        m = np.abs(az) <= az_limit
        idx, rr, az, el = idx[m], rr[m], az[m], el[m]
    if len(idx) == 0:
        return None
    az_step, el_step = res
    az_hi, el_hi = float(az.max()), float(el.max())
    W = int((az_hi - az.min()) / az_step) + 1
    H = int((el_hi - el.min()) / el_step) + 1
    ci = ((az_hi - az) / az_step).astype(np.int64)
    ri = ((el_hi - el) / el_step).astype(np.int64)
    flat = ri * W + ci
    # per cell: the nearest return (first index among equal ranges) and the farthest (last index among
    # equal ranges) - what np.lexsort((rr, flat)) gives at the group ends, from one stable integer sort
    order = np.argsort(flat, kind="stable")
    f = flat[order]
    rs = rr[order]
    brk = f[1:] != f[:-1]
    starts = np.concatenate([[0], np.flatnonzero(brk) + 1])
    grp = np.concatenate([[0], np.cumsum(brk)])
    n_min = np.minimum.reduceat(rs, starts)
    n_max = np.maximum.reduceat(rs, starts)
    k = np.flatnonzero(rs == n_min[grp])
    first = k[np.concatenate([[True], grp[k][1:] != grp[k][:-1]])]
    k = np.flatnonzero(rs == n_max[grp])
    last = k[np.concatenate([grp[k][1:] != grp[k][:-1], [True]])]
    w0, w1 = order[first], order[last]
    return _BinPlan(H, W, az_hi, el_hi, f[starts], (idx[w0], idx[w1]),
                    (rr[w0].astype(np.float32), rr[w1].astype(np.float32)))


def _guard_q(step: float) -> float:
    """Closest (in bins) a quotient (hi - angle) / step may be to a bin edge: 10x its possible error."""
    return _GUARD_Q if _GUARD_Q is not None else 10.0 * (2.0 * _ERR_DEG / step) + 1e-12


def _near_integer(q, tol: float, torch):
    """Points whose bin quotient is within ``tol`` of an edge >= 1 (the edge at 0 is the grid origin itself)."""
    n = torch.round(q)
    return ((q - n).abs() < tol) & (n >= 1)


def _plan_torch(frame: Frame, R: np.ndarray, res: Tuple[float, float], az_limit: Optional[float], device):
    """GPU plan, or "guard" when a point is too close to a bin edge for CUDA's atan2/asin to be trusted."""
    import torch

    perm, _ = _rotation_parts(R)
    if perm is None:
        GUARD_WHY["axes"] += 1
        return "guard"
    pidx_, sign = perm
    t = torch.from_numpy(np.ascontiguousarray(frame.xyz)).to(device)
    p = t.double()
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = torch.sqrt(x * x + y * y + z * z)
    ok = torch.isfinite(r) & (r > 0.5)
    d = p[:, pidx_[0]] * float(sign[0]) + 0.0
    l = p[:, pidx_[1]] * float(sign[1]) + 0.0
    h = p[:, pidx_[2]] * float(sign[2]) + 0.0
    rs = torch.where(ok, r, torch.ones_like(r))
    az = torch.atan2(l, d) * _DEG
    el = torch.asin(torch.clamp(h / rs, -1.0, 1.0)) * _DEG
    # a return straight above/below the sensor (d = l = 0): atan2 depends on the sign of the zeros there
    near_lim = (ok & (d == 0) & (l == 0)).any()
    if az_limit is not None:
        keep = ok & (az.abs() <= az_limit)
        near_lim = near_lim | (ok & ((az.abs() - az_limit).abs() < _GUARD_DEG)).any()
    else:
        keep = ok
    k = torch.nonzero(keep).squeeze(1)
    nk = int(k.numel())
    if nk == 0:
        if bool(near_lim):
            GUARD_WHY["limit"] += 1
            return "guard"
        return None
    azk, elk, rrk = az[k], el[k], r[k]
    if bool(near_lim):
        GUARD_WHY["limit"] += 1
        return "guard"
    # the grid origin and size come from the extreme angles: recomputed with NumPy (the reference
    # functions) on every point within 4x the GPU error of an extreme, so they are exactly the reference's
    e = 4 * _ERR_DEG
    a_hi, a_lo, e_hi, e_lo = azk.max(), azk.min(), elk.max(), elk.min()
    near_ext = torch.nonzero((azk >= a_hi - e) | (azk <= a_lo + e) | (elk >= e_hi - e) | (elk <= e_lo + e)).squeeze(1)
    ke = k[near_ext]
    dd, ll, hh, re_ = (a.cpu().numpy() for a in (d[ke], l[ke], h[ke], r[ke]))
    az_x = np.degrees(np.arctan2(ll, dd))
    el_x = np.degrees(np.arcsin(np.clip(hh / re_, -1, 1)))
    az_step, el_step = res
    az_hi, az_lo, el_hi, el_lo = float(az_x.max()), float(az_x.min()), float(el_x.max()), float(el_x.min())
    W = int((az_hi - az_lo) / az_step) + 1
    H = int((el_hi - el_lo) / el_step) + 1
    qa = (az_hi - azk) / az_step
    qe = (el_hi - elk) / el_step
    bad = (_near_integer(qa, _guard_q(az_step), torch) | _near_integer(qe, _guard_q(el_step), torch)).any()
    HW = H * W
    # a guarded (bad) frame may put an extreme point one bin outside the grid: keep the scatter in range,
    # the frame is discarded below anyway (an out-of-range CUDA index would poison the context)
    flat = (qe.to(torch.int64) * W + qa.to(torch.int64)).clamp_(0, HW - 1)
    pos = torch.arange(nk, device=device)
    cmin = torch.full((HW,), float("inf"), dtype=torch.float64, device=device).scatter_reduce(0, flat, rrk, "amin")
    cmax = torch.full((HW,), float("-inf"), dtype=torch.float64, device=device).scatter_reduce(0, flat, rrk, "amax")
    e0 = rrk == cmin[flat]
    e1 = rrk == cmax[flat]
    first = torch.full((HW,), nk, dtype=torch.int64, device=device).scatter_reduce(0, flat[e0], pos[e0], "amin")
    last = torch.full((HW,), -1, dtype=torch.int64, device=device).scatter_reduce(0, flat[e1], pos[e1], "amax")
    cells = torch.nonzero(first < nk).squeeze(1)
    w0, w1 = first[cells], last[cells]
    if bool(bad):
        GUARD_WHY["edge"] += 1
        return "guard"
    out = [a.cpu().numpy() for a in (cells, k[w0], k[w1], rrk[w0].float(), rrk[w1].float())]
    return _BinPlan(H, W, az_hi, el_hi, out[0], (out[1], out[2]), (out[3], out[4]))


def _fill_binned(frame: Frame, R: np.ndarray, res: Tuple[float, float], plan: Optional[_BinPlan]) -> RangeImage:
    if plan is None:
        rng, inten, xyz, slots = _empty(1, 1)
        return RangeImage(rng, inten, xyz, xyz.copy(), rng > 0, np.zeros(1, np.float32), np.zeros(1, np.float32),
                          np.full(1, -1), slots, frame.profile.sensor, False)
    H, W = plan.H, plan.W
    az_step, el_step = res
    n_pix = H * W
    rng = np.zeros(2 * n_pix, np.float32)
    inten = np.zeros(2 * n_pix, np.float32)
    xyz = np.zeros((2 * n_pix, 3), np.float32)
    slots = np.full(2 * n_pix, -1, np.int64)
    # a signed permutation is element-wise: dlh of the winners only, scattered like xyz (empty pixels are
    # +0.0 in both); the general rotation stays the whole-image BLAS product (bit-identical to v1)
    perm, _ = _rotation_parts(R)
    dlh = np.zeros((2 * n_pix, 3), np.float32) if perm is not None else None
    for layer in (0, 1):
        tgt = plan.cells + layer * n_pix
        pi = plan.pidx[layer]
        rng[tgt] = plan.rng[layer]
        pts = frame.xyz[pi]
        xyz[tgt] = pts
        if dlh is not None:
            dlh[tgt] = _dlh_image(pts, R)
        if frame.intensity is not None:
            inten[tgt] = frame.intensity[pi]
        slots[tgt] = frame.slot[pi] if frame.slot is not None else pi
    rng, inten, slots = rng.reshape(2, H, W), inten.reshape(2, H, W), slots.reshape(2, H, W)
    xyz = xyz.reshape(2, H, W, 3)
    dlh = dlh.reshape(2, H, W, 3) if dlh is not None else _dlh_image(xyz, R)
    return RangeImage(rng=rng, intensity=inten, xyz=xyz, dlh=dlh, valid=rng > 0,
                      el_deg=(plan.el_hi - el_step * np.arange(H)).astype(np.float32),
                      az_deg=(plan.az_hi - az_step * np.arange(W)).astype(np.float32), ring_of_row=np.full(H, -1, np.int64),
                      slot_of_pixel=slots, sensor=frame.profile.sensor, organized=False)


def _binned(frame: Frame, R: np.ndarray, res: Tuple[float, float], az_limit: Optional[float],
            device=None, on_device_error=None) -> RangeImage:
    plan = "cpu"
    if device is not None:
        try:
            plan = _plan_torch(frame, R, res, az_limit, device)
        except Exception as exc:          # hot fallback: this frame on the CPU, the caller decides about the rest
            BIN_STATS["gpu_error"] += 1
            plan = "cpu"
            if on_device_error is not None:
                on_device_error(exc)
        if isinstance(plan, str):
            if plan == "guard":
                BIN_STATS["gpu_guard"] += 1
        else:
            BIN_STATS["gpu"] += 1
    if isinstance(plan, str):
        BIN_STATS["cpu"] += 1
        plan = _plan_cpu(frame, R, res, az_limit)
    return _fill_binned(frame, R, res, plan)
