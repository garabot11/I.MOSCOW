"""Sensor profile: point layout, dual returns, beam angles, forward crop and axis convention.

A profile is computed once per *signature* (frame_id, fields, number of points) from the
first cloud and reused for every following cloud of the same stream:

* organised layout — the development Hesai-like clouds are column-major with the ring
  index running fastest (slot ``i`` -> ring ``i % n_rings``, column ``i // n_rings``),
  and no-return slots keep their place as (0, 0, 0);
* dual returns — adjacent column pairs (even = far/last, odd = near/first return) are
  two echoes of one firing and are identical when there is a single echo;
* beam angles — per-ring elevation, per-ring azimuth offset and per-column azimuth in the
  analysis frame, so that every slot (also a no-return slot) has a direction;
* forward crop — the firing columns within ``forward_crop_deg`` of forward; columns
  outside cannot contribute to the analysed region and are dropped before any work;
* axes — the configured ``forward_axis``/``up_axis`` are kept while the track bed is found
  with them; otherwise (another lidar with another convention) the axes are detected from
  the cloud: forward = the axis with the largest extent, up = the remaining axis with the
  smaller angular spread (the vertical field of view is the narrow one), confirmed by the
  rail-level fit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import DetectorConfig

_AXES = ("x", "y", "z")


@dataclass
class SensorProfile:
    signature: Tuple
    organized: bool
    n_rings: int
    n_cols: int
    dual_return: bool
    sensor: str                       # "hesai128" | "generic"
    forward_axis: str
    up_axis: str
    axes_source: str                  # "config" | "auto"
    ring_el_deg: Optional[np.ndarray] = None      # (n_rings,)
    ring_az_off_deg: Optional[np.ndarray] = None  # (n_rings,) added to the column azimuth
    col_az_deg: Optional[np.ndarray] = None       # (n_cols,) analysis-frame azimuth of every column
    keep_cols: Optional[np.ndarray] = None        # sorted kept columns; None = all
    keep_slice: Optional[slice] = None            # the same as a slot slice when the kept columns are contiguous
    note: str = ""

    def summary(self) -> str:
        crop = "all" if self.keep_cols is None else f"{len(self.keep_cols)}/{self.n_cols}"
        return (f"sensor={self.sensor} organised={self.organized} rings={self.n_rings} cols={self.n_cols} "
                f"dual={self.dual_return} axes={self.forward_axis}/{self.up_axis} ({self.axes_source}) "
                f"columns kept={crop}{' ' + self.note if self.note else ''}")


def signature_of(msg, names) -> Tuple:
    return (str(msg.header.frame_id), tuple(names), int(msg.width) * int(msg.height), int(msg.point_step))


def axis_rotation(forward_axis: str, up_axis: str) -> np.ndarray:
    return dataclasses.replace(DetectorConfig(), forward_axis=forward_axis, up_axis=up_axis).axis_matrix()


# ---------------------------------------------------------------------------
def detect_layout(struct: np.ndarray) -> Tuple[bool, int, int, bool]:
    """(organised, n_rings, n_cols, dual_return) from the ring field and the slot order."""
    names = struct.dtype.names or ()
    n = len(struct)
    if "ring" not in names or n == 0:
        return False, 0, 0, False
    ring = np.asarray(struct["ring"])
    nr = int(ring.max()) + 1
    if nr < 2 or n % nr:
        return False, 0, 0, False
    probe = np.unique(np.concatenate([np.arange(min(n, 4096)), np.linspace(0, n - 1, 4096).astype(np.int64)]))
    if not np.array_equal(ring[probe].astype(np.int64), probe % nr):
        return False, 0, 0, False
    n_cols = n // nr
    dual = False
    if n_cols >= 2 and n_cols % 2 == 0:
        x = np.asarray(struct["x"]).reshape(n_cols, nr)
        y = np.asarray(struct["y"]).reshape(n_cols, nr)
        z = np.asarray(struct["z"]).reshape(n_cols, nr)
        va = (x[0::2] != 0) | (y[0::2] != 0) | (z[0::2] != 0)
        vb = (x[1::2] != 0) | (y[1::2] != 0) | (z[1::2] != 0)
        both = va & vb
        if both.sum() > 1000:
            same = (x[0::2] == x[1::2]) & (y[0::2] == y[1::2]) & (z[0::2] == z[1::2])
            dual = bool(same[both].mean() > 0.5)
    return True, nr, n_cols, dual


def beam_angles(struct: np.ndarray, R: np.ndarray, nr: int, n_cols: int):
    """Per-ring elevation, per-ring azimuth offset and per-column azimuth (degrees, analysis frame)."""
    xyz = np.stack([np.asarray(struct[k], dtype=np.float64) for k in _AXES], axis=1)
    dlh = xyz @ R.T
    r = np.linalg.norm(dlh, axis=1)
    valid = r > 0.5
    d, l, h = dlh[:, 0].reshape(n_cols, nr), dlh[:, 1].reshape(n_cols, nr), dlh[:, 2].reshape(n_cols, nr)
    rr = r.reshape(n_cols, nr)
    V = valid.reshape(n_cols, nr)
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.degrees(np.arcsin(np.clip(h / rr, -1, 1)))
        az = np.degrees(np.arctan2(l, d))
    el[~V] = np.nan
    az[~V] = np.nan
    ring_el = np.nanmedian(el, axis=0) if V.any() else np.full(nr, np.nan)
    counts = V.sum(axis=0)
    ref = int(np.argmax(np.where(np.abs(np.nan_to_num(ring_el, nan=90.0)) < 3.0, counts, -1)))
    off = np.zeros(nr)
    for k in range(nr):
        both = V[:, k] & V[:, ref]
        if both.sum() >= 10:
            off[k] = float(np.median(az[both, k] - az[both, ref]))
    col_az = np.nanmedian(az - off[None, :], axis=1) if V.any() else np.full(n_cols, np.nan)
    good = np.isfinite(col_az)
    if good.sum() >= 2:
        idx = np.arange(n_cols)
        col_az = np.interp(idx, idx[good], col_az[good])
    return ring_el, off, col_az


def forward_crop(col_az: Optional[np.ndarray], half_deg: float):
    """Kept columns and, if contiguous, the slot slice; (None, None) = keep everything."""
    if col_az is None or half_deg <= 0 or not np.isfinite(col_az).all():
        return None, None
    keep = np.flatnonzero(np.abs(col_az) <= half_deg)
    if len(keep) == 0 or len(keep) == len(col_az):
        return None, None
    return keep, None


def _rail_found(xyz_valid: np.ndarray, fwd: str, up: str, cfg: DetectorConfig) -> Tuple[bool, int, float, int, int]:
    """Rail-level fit with the given axes: (ok, n_bins, rms, n_beyond, n_surface).

    ``n_beyond`` counts returns 0.25-1.0 m *beyond* the fitted surface (farther from the sensor than
    the surface, within +-1 m of the centre, up to 40 m): a track bed has them (drainage trough,
    ballast between sleepers), a ceiling has none - this is what tells the floor from the ceiling."""
    from .track_model import fit_rail_level  # local import: track_model imports config only

    c = dataclasses.replace(cfg, forward_axis=fwd, up_axis=up)
    dlh = xyz_valid @ c.axis_matrix().T
    d, l, h = dlh[:, 0], dlh[:, 1], dlh[:, 2]
    m = (d >= c.d_min) & (d <= c.d_max) & (np.abs(l) <= c.l_abs_max) & (h >= c.h_abs_min) & (h <= c.h_abs_max)
    rail = fit_rail_level(d[m], l[m], h[m], c, 0.0)
    rms = float(rail.residual_rms) if np.isfinite(rail.residual_rms) else 9.9
    if not rail.ok:
        return False, int(rail.n_bins), rms, 0, 0
    near = m & (d <= 40.0) & (np.abs(l) <= 1.0)
    dh = h[near] - rail(d[near])
    n_surface = int((np.abs(dh) <= 0.25).sum())
    n_beyond = int(((dh < -0.25) & (dh > -1.0)).sum())
    return True, int(rail.n_bins), rms, n_beyond, n_surface


# Known lidar conventions, most likely first: ROS REP-103 (x forward, z up), the Falcon Prime native
# frame of the manual (x up, y right, z forward), the development Hesai bags (-y forward, z up) and
# the same three with the unit mounted upside down.
_KNOWN_CONVENTIONS = [("+x", "+z"), ("+z", "+x"), ("-y", "+z"), ("+y", "+z"), ("-x", "+z"),
                      ("+x", "-z"), ("+z", "-x"), ("-y", "-z"), ("+y", "-z"), ("-x", "-z")]


def detect_axes(xyz: np.ndarray, cfg: DetectorConfig) -> Tuple[str, str, str]:
    """Forward/up axes of an unknown sensor convention: (forward, up, note).

    Forward is the signed axis with the largest extent.  Up is chosen among the four remaining
    signed axes by the rail-level fit (`fit_rail_level`), but the floor and the ceiling of a tunnel
    are both planar, so a fit alone cannot tell them apart (on the Falcon Prime bag the ceiling even
    gave more bins than the dark, sparse track bed).  Candidates are therefore tried in the order of
    the known conventions, and candidates with at least 40 % of the best bin count stay; returns *beyond* the
    surface (trough, ballast) single out the floor; then the known conventions decide, and the bin
    count only between unknown conventions."""
    pts = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1)
    pts, r = pts[r > 1.0], r[r > 1.0]
    if len(pts) < 1000:
        return cfg.forward_axis, cfg.up_axis, "too few points to detect axes"
    ext = [np.percentile(np.abs(pts[:, i]), 98) for i in range(3)]
    fi = int(np.argmax(ext))
    far = np.abs(pts[:, fi]) > 15.0
    fsign = "+" if (pts[far, fi] > 0).sum() >= (pts[far, fi] < 0).sum() else "-"
    fwd = fsign + _AXES[fi]
    # up is the remaining axis with the narrower angular spread (the vertical field of view of these
    # lidars is much narrower than the horizontal one); the walls would otherwise fit a "rail level" too
    others = [i for i in range(3) if i != fi]
    spread = []
    for i in others:
        ang = np.degrees(np.arcsin(np.clip(pts[:, i] / r, -1, 1)))
        spread.append(np.percentile(ang, 97) - np.percentile(ang, 3))
    ui = others[int(np.argmin(spread))]
    fits = {}
    for usign in ("+", "-"):
        up = usign + _AXES[ui]
        ok, nb, rms, n_beyond, n_surf = _rail_found(pts, fwd, up, cfg)
        if ok:
            fits[up] = (nb, rms, n_beyond, n_surf)
    if not fits:
        return cfg.forward_axis, cfg.up_axis, "axes not detected (no track bed with any candidate)"
    best_bins = max(v[0] for v in fits.values())
    good = [up for up in fits if fits[up][0] >= max(4, 0.4 * best_bins)]
    # a floor has returns beyond its surface (trough, ballast); a ceiling has none
    pen = [u for u in good if fits[u][2] >= 10 and fits[u][2] >= 0.02 * max(fits[u][3], 1)]
    if pen and len(pen) < len(good):
        good, how = pen, "returns beyond the surface"
    else:
        how = ""
    known = [u for f, u in _KNOWN_CONVENTIONS if f == fwd and u in good]
    if known:
        up, how = known[0], how or "known convention"
    else:
        up, how = max(good, key=lambda u: (fits[u][0], -fits[u][1])), how or "most track-bed bins"
    detail = ", ".join(f"{u}: {v[0]} bins rms {v[1]:.3f} beyond {v[2]}/{v[3]}" for u, v in fits.items())
    return fwd, up, f"axes detected: forward {fwd}, up {up} ({how}; extent {ext[fi]:.0f} m, vertical spread {min(spread):.0f} deg; candidates {detail})"


def build_profile(msg, struct: np.ndarray, cfg: DetectorConfig) -> SensorProfile:
    names = struct.dtype.names or ()
    sig = signature_of(msg, names)
    organized, nr, n_cols, dual = detect_layout(struct)
    fwd, up, source, note = cfg.forward_axis, cfg.up_axis, "config", ""
    if cfg.axes_mode != "config":
        xyz = np.stack([np.asarray(struct[k], dtype=np.float32) for k in _AXES], axis=1)
        valid = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
        ok = _rail_found(xyz[valid].astype(np.float64), fwd, up, cfg)[0]
        if not ok:
            f2, u2, note = detect_axes(xyz[valid], cfg)
            if (f2, u2) != (fwd, up):
                fwd, up, source = f2, u2, "auto"
    prof = SensorProfile(signature=sig, organized=organized, n_rings=nr, n_cols=n_cols, dual_return=dual,
                         sensor="hesai128" if organized and nr == 128 else "generic",
                         forward_axis=fwd, up_axis=up, axes_source=source, note=note)
    if organized:
        R = axis_rotation(fwd, up)
        prof.ring_el_deg, prof.ring_az_off_deg, prof.col_az_deg = beam_angles(struct, R, nr, n_cols)
        keep, _ = forward_crop(prof.col_az_deg, cfg.forward_crop_deg)
        if keep is not None:
            if dual:  # keep whole firing pairs
                keep = np.unique(np.concatenate([keep - (keep % 2), keep - (keep % 2) + 1]))
                keep = keep[keep < n_cols]
            prof.keep_cols = keep
            if keep[-1] - keep[0] + 1 == len(keep):
                prof.keep_slice = slice(int(keep[0]) * nr, (int(keep[-1]) + 1) * nr)
    return prof
