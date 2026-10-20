"""Beam model of the forward LiDAR: the direction of every slot of a PointCloud2 message.

Interface A of ``work/specs/00_interfaces.md``.  Pure NumPy (no SciPy / torch / ROS).

Measured on the development bags (2026-10-17), the Hesai-like 128-ring sensor is exactly separable:

* slot ``i`` -> ring ``i % 128``, column ``i // 128`` (column-major, ring fastest);
* columns come in (far/last, near/first) pairs that share one firing (``dual_return``);
* elevation is constant per ring, ``el = elevation_deg[ring]``;
* azimuth ``az = col_az_deg[col] + ring_az_offset_deg[ring]`` with ``col_az`` linear in the firing
  index (step -0.1 deg per firing); the fit reproduces every valid return of all six bags to
  ~5e-6 deg (float32 precision of x, y, z);
* 64 rings (25..88) fire on every firing, 32 only on even and 32 only on odd firings
  (``ring_fire_parity``);
* the 307 200-point topic is a crop of 1200 firings whose *point* azimuth lies within
  ``az_window_deg`` (= [-50.000, +50.007] deg) - slots outside the window never return.

Conventions (as ``config.py``): analysis axes d (forward), l = h x d (lateral), h (up) from
``forward_axis`` / ``up_axis``; ``az = atan2(v.l, v.d)`` in degrees, 0 = forward, positive towards
+l; ``el = asin(v.h)``.  Ring azimuth offsets are normalised to zero median over the rings that fire
on every firing, so ``col_az_deg`` is the azimuth of the dense horizon band.

``fired_mask()`` tells which slots the sensor fires at all in this message layout.  An unfired slot is
always (0, 0, 0) and is **not** evidence of free space; a fired slot with (0, 0, 0) is "no return".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["BeamModel", "FalconBeamModel", "axis_matrix", "falcon_key", "hesai128"]


def _axis_vector(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    idx = {"x": 0, "y": 1, "z": 2}.get(spec.lstrip("+-"))
    if idx is None:
        raise ValueError(f"bad axis spec '{spec}', expected e.g. '-y' or '+z'")
    v = np.zeros(3)
    v[idx] = sign
    return v


def axis_matrix(forward_axis: str = "-y", up_axis: str = "+z") -> np.ndarray:
    """3x3 matrix R with rows (d, l, h) in sensor coordinates: (d, l, h) = R @ (x, y, z)."""
    fwd, up = _axis_vector(forward_axis), _axis_vector(up_axis)
    if abs(float(fwd @ up)) > 1e-9:
        raise ValueError("forward_axis and up_axis must be orthogonal")
    return np.vstack([fwd, np.cross(up, fwd), up])


def _wrap(a):
    """Wrap degrees to (-180, 180]."""
    w = np.mod(np.asarray(a, dtype=np.float64) + 180.0, 360.0) - 180.0
    return np.where(w == -180.0, 180.0, w)


@dataclass
class BeamModel:
    n_rings: int                      # 128
    n_cols: int                       # slot columns in the message (2400 or 7200)
    elevation_deg: np.ndarray         # (n_rings,) per-ring elevation (sensor frame)
    ring_az_offset_deg: np.ndarray    # (n_rings,) per-ring azimuth offset added to the column azimuth
    col_az_deg: np.ndarray            # (n_cols,) firing azimuth of every column (pairs share one value), wrapped to (-180, 180]
    dual_return: bool                 # columns come in (far, near) pairs
    forward_axis: str = "-y"
    up_axis: str = "+z"
    # ---- optional (defaults keep a minimal model valid) ----
    col_time_s: Optional[np.ndarray] = None        # (n_cols,) time of the column's firing after the header stamp
    ring_fire_parity: Optional[np.ndarray] = None  # (n_rings,) int8: -1 every firing, 0 even firings only, 1 odd only
    az_window_deg: Optional[Tuple[float, float]] = None  # crop window on the slot (point) azimuth; None = no crop
    sensor: str = "hesai128"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.elevation_deg = np.asarray(self.elevation_deg, dtype=np.float64)
        self.ring_az_offset_deg = np.asarray(self.ring_az_offset_deg, dtype=np.float64)
        self.col_az_deg = np.asarray(self.col_az_deg, dtype=np.float64)
        if self.elevation_deg.shape != (self.n_rings,) or self.ring_az_offset_deg.shape != (self.n_rings,):
            raise ValueError("elevation_deg / ring_az_offset_deg must have shape (n_rings,)")
        if self.col_az_deg.shape != (self.n_cols,):
            raise ValueError("col_az_deg must have shape (n_cols,)")
        if self.col_time_s is not None:
            self.col_time_s = np.asarray(self.col_time_s, dtype=np.float64)
        if self.ring_fire_parity is not None:
            self.ring_fire_parity = np.asarray(self.ring_fire_parity, dtype=np.int8)
        if self.az_window_deg is not None:
            self.az_window_deg = (float(self.az_window_deg[0]), float(self.az_window_deg[1]))

    # ------------------------------------------------------------------ indexing
    @property
    def n_slots(self) -> int:
        return self.n_cols * self.n_rings

    @property
    def n_firings(self) -> int:
        return self.n_cols // 2 if self.dual_return else self.n_cols

    def slot_index(self, ring, col) -> np.ndarray:
        """Message slot index of (ring, col)."""
        return np.asarray(col, dtype=np.int64) * self.n_rings + np.asarray(ring, dtype=np.int64)

    def ring_of_slot(self, slot) -> np.ndarray:
        return np.asarray(slot, dtype=np.int64) % self.n_rings

    def col_of_slot(self, slot) -> np.ndarray:
        return np.asarray(slot, dtype=np.int64) // self.n_rings

    def firing_of_col(self, col) -> np.ndarray:
        col = np.asarray(col, dtype=np.int64)
        return col // 2 if self.dual_return else col

    def echo_of_col(self, col) -> np.ndarray:
        """0 = far/last return (even column), 1 = near/first return (odd column); 0 for single return."""
        col = np.asarray(col, dtype=np.int64)
        return col % 2 if self.dual_return else np.zeros_like(col)

    # ------------------------------------------------------------------ geometry
    def axis_matrix(self) -> np.ndarray:
        return axis_matrix(self.forward_axis, self.up_axis)

    def slot_az_deg(self) -> np.ndarray:
        """(n_slots,) azimuth of every slot (message order), wrapped to (-180, 180]."""
        return _wrap(np.repeat(self.col_az_deg, self.n_rings) + np.tile(self.ring_az_offset_deg, self.n_cols))

    def slot_el_deg(self) -> np.ndarray:
        return np.tile(self.elevation_deg, self.n_cols)

    def directions_dlh(self) -> np.ndarray:
        """(n_slots, 3) float64 unit vectors in the analysis frame (d, l, h)."""
        az = np.radians(np.repeat(self.col_az_deg, self.n_rings) + np.tile(self.ring_az_offset_deg, self.n_cols))
        el = np.radians(self.slot_el_deg())
        ce = np.cos(el)
        return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1)

    def directions(self) -> np.ndarray:
        """(n_cols*n_rings, 3) float32 unit vectors in the sensor frame, message slot order, incl. no-return slots."""
        return (self.directions_dlh() @ self.axis_matrix()).astype(np.float32)

    def fired_mask(self) -> np.ndarray:
        """(n_slots,) bool: slots the sensor fires in this layout (firing parity of the ring and crop window)."""
        m = np.ones(self.n_slots, dtype=bool)
        if self.ring_fire_parity is not None:
            par = np.tile(self.ring_fire_parity, self.n_cols)
            fpar = np.repeat(self.firing_of_col(np.arange(self.n_cols)) % 2, self.n_rings)
            m &= (par < 0) | (par == fpar)
        if self.az_window_deg is not None:
            lo, hi = self.az_window_deg
            az = self.slot_az_deg()
            m &= (az >= lo) & (az <= hi)
        return m

    def slot_time_s(self) -> np.ndarray:
        """(n_slots,) time after the header stamp of each slot (zeros when unknown)."""
        if self.col_time_s is None:
            return np.zeros(self.n_slots)
        return np.repeat(self.col_time_s, self.n_rings)

    def angles_of_points(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(az_deg, el_deg) of sensor-frame points with this model's axis convention."""
        dlh = np.asarray(xyz, dtype=np.float64) @ self.axis_matrix().T
        r = np.linalg.norm(dlh, axis=1)
        az = np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0]))
        el = np.degrees(np.arcsin(np.clip(dlh[:, 2] / np.maximum(r, 1e-12), -1.0, 1.0)))
        return az, el

    def angular_error_deg(self, cloud) -> np.ndarray:
        """Angle (deg) between each valid return of ``cloud`` (structured array or (N,3)) and its slot direction."""
        xyz = _xyz(cloud)
        if len(xyz) != self.n_slots:
            raise ValueError(f"cloud has {len(xyz)} points, model has {self.n_slots} slots")
        valid = np.isfinite(xyz).all(1) & ~(xyz == 0).all(1)
        u = xyz[valid] / np.linalg.norm(xyz[valid], axis=1)[:, None]
        dirs = self.directions_dlh()[valid] @ self.axis_matrix()
        cosang = np.clip((u * dirs).sum(1), -1.0, 1.0)
        # arccos is ill-conditioned near 1: use the norm of the cross product as well
        sinang = np.linalg.norm(np.cross(u, dirs), axis=1)
        return np.degrees(np.arctan2(sinang, cosang))

    # ------------------------------------------------------------------ estimation
    @classmethod
    def estimate(cls, clouds: Sequence, forward_axis: str = "-y", up_axis: str = "+z", n_rings: int = 128,
                 stamps_ns: Optional[Sequence[int]] = None, az_window: str = "auto",
                 crop_pad_deg: float = 0.002) -> "BeamModel":
        """Fit the model to one or more organised clouds of the same layout.

        ``clouds``: structured arrays (cloud_io) with fields x, y, z (+ optional ring, timestamp) or (N, 3)
        arrays.  One cloud is enough; more clouds fill rings / firings without returns.
        ``stamps_ns``: header stamps (for ``col_time_s``); default: the timestamp of slot 0.
        ``az_window``: "auto" (a crop window is fitted only when the message is less than a full turn),
        "none", or ignored when a tuple is given.
        """
        if not len(clouds):
            raise ValueError("need at least one cloud")
        R = axis_matrix(forward_axis, up_axis)
        n = len(clouds[0])
        if n % n_rings:
            raise ValueError(f"{n} points is not a multiple of {n_rings} rings")
        n_cols = n // n_rings
        ring = np.arange(n) % n_rings
        col = np.arange(n) // n_rings
        az_l, el_l, v_l, t_l = [], [], [], []
        for k, c in enumerate(clouds):
            if len(c) != n:
                raise ValueError("all clouds must have the same number of points")
            if getattr(c, "dtype", None) is not None and c.dtype.names and "ring" in c.dtype.names:
                if not np.array_equal(np.asarray(c["ring"], dtype=np.int64), ring):
                    raise ValueError("ring field is not slot % n_rings: not a column-major organised cloud")
            xyz = _xyz(c)
            valid = np.isfinite(xyz).all(1) & ~(xyz == 0).all(1)
            dlh = xyz @ R.T
            r = np.linalg.norm(dlh, axis=1)
            az_l.append(np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0])))
            el_l.append(np.degrees(np.arcsin(np.clip(dlh[:, 2] / np.maximum(r, 1e-12), -1, 1))))
            v_l.append(valid)
            if getattr(c, "dtype", None) is not None and c.dtype.names and "timestamp" in c.dtype.names:
                ts = np.asarray(c["timestamp"], dtype=np.float64)
                ref = ts[0] if stamps_ns is None else stamps_ns[k] * 1e-9
                t_l.append(ts - ref)
        AZ, EL, V = np.concatenate(az_l), np.concatenate(el_l), np.concatenate(v_l)
        RING, COL = np.tile(ring, len(clouds)), np.tile(col, len(clouds))
        if not V.any():
            raise ValueError("no valid returns")
        counts = np.bincount(RING[V], minlength=n_rings)
        if (counts == 0).any():
            raise ValueError(f"rings without any valid return: {np.nonzero(counts == 0)[0].tolist()} - pass more clouds")
        # dual return: the two columns of a pair share one azimuth
        dual = False
        if n_cols % 2 == 0:
            a0 = np.stack(az_l)[:, :].reshape(len(clouds), n_cols // 2, 2, n_rings)
            v0 = np.stack(v_l).reshape(len(clouds), n_cols // 2, 2, n_rings)
            both = v0[:, :, 0] & v0[:, :, 1]
            if both.sum() > 100:
                dd = np.abs(_wrap(a0[:, :, 0] - a0[:, :, 1]))[both]
                dual = bool(np.mean(dd < 1e-3) > 0.99)
        FIR = COL // 2 if dual else COL
        n_fir = n_cols // 2 if dual else n_cols
        # elevation: per-ring median
        elev = np.array([np.median(EL[V & (RING == k)]) for k in range(n_rings)])
        # azimuth: per ring az = c_k + s * firing (unwrapped)
        c_k = np.zeros(n_rings)
        s_k = np.zeros(n_rings)
        for k in range(n_rings):
            m = V & (RING == k)
            f = FIR[m]
            a = AZ[m]
            order = np.argsort(f, kind="stable")
            f, a = f[order], a[order]
            uf, first = np.unique(f, return_index=True)
            a1 = np.unwrap(a[first], period=360.0)
            if len(uf) >= 2:
                s, c0 = np.polyfit(uf.astype(np.float64), a1, 1)
            else:
                s, c0 = np.nan, a1[0]
            s_k[k], c_k[k] = s, c0
        step = float(np.nanmedian(s_k))
        if not np.isfinite(step):
            raise ValueError("cannot determine the firing azimuth step")
        # with a common step, the per-ring intercept is the circular median of az - step * firing
        for k in range(n_rings):
            m = V & (RING == k)
            x = AZ[m] - step * FIR[m]
            ref = x[0]
            c_k[k] = ref + np.median(_wrap(x - ref))
        # parity of the firings of each ring
        par = np.full(n_rings, -1, dtype=np.int8)
        for k in range(n_rings):
            fk = FIR[V & (RING == k)]
            ne, no = int((fk % 2 == 0).sum()), int((fk % 2 == 1).sum())
            if no == 0 and ne >= 10:
                par[k] = 0
            elif ne == 0 and no >= 10:
                par[k] = 1
        dense = par < 0
        ref_ring = np.nonzero(dense)[0] if dense.any() else np.arange(n_rings)
        a0_ = c_k[ref_ring[0]]
        rel = _wrap(c_k - a0_)
        med = float(np.median(rel[ref_ring]))
        offsets = rel - med
        base = a0_ + med
        # column azimuth: linear model, refined per firing when the residual is not negligible
        fir_az = base + step * np.arange(n_fir)
        res = _wrap(AZ[V] - (fir_az[FIR[V]] + offsets[RING[V]]))
        meta = {"fit_max_residual_deg": float(np.abs(res).max()), "firing_step_deg": step,
                "n_clouds": len(clouds), "col_az_model": "linear"}
        if np.abs(res).max() > 1e-3:
            corr = np.full(n_fir, np.nan)
            fv = FIR[V]
            for fi in np.unique(fv):
                corr[fi] = np.median(res[fv == fi])
            ok = np.isfinite(corr)
            corr = np.interp(np.arange(n_fir), np.nonzero(ok)[0], corr[ok])
            fir_az = fir_az + corr
            meta["col_az_model"] = "linear+per-firing correction"
        col_az = _wrap(np.repeat(fir_az, 2) if dual else fir_az)
        # crop window on the point azimuth (only for messages that are less than a full turn)
        window = None
        if isinstance(az_window, (tuple, list)):
            window = (float(az_window[0]), float(az_window[1]))
        elif az_window == "auto" and n_fir * abs(step) < 359.0:
            # edges midway between the outermost observed return and the next parity-fired slot outward
            # (the real crop leaves only ~0.001 deg of margin, so a fixed pad would be wrong).  With few
            # clouds the window can come out narrower than the sensor's when edge slots had no return.
            centre = _wrap(base + step * (n_fir - 1) / 2.0)
            all_fir = np.arange(n_fir)
            slot_rel = _wrap(fir_az[all_fir][:, None] + offsets[None, :] - centre)
            ok_par = (par[None, :] < 0) | (par[None, :] == (all_fir % 2)[:, None])
            cand = slot_rel[ok_par]
            rel_pa = _wrap(fir_az[FIR[V]] + offsets[RING[V]] - centre)
            lo_o, hi_o = float(rel_pa.min()), float(rel_pa.max())
            below, above = cand[cand < lo_o - 1e-9], cand[cand > hi_o + 1e-9]
            lo = (lo_o + below.max()) / 2 if len(below) else lo_o - crop_pad_deg
            hi = (hi_o + above.min()) / 2 if len(above) else hi_o + crop_pad_deg
            window = (float(_wrap(centre + lo)), float(_wrap(centre + hi)))
        col_time = None
        if t_l:
            T = np.median(np.stack(t_l), axis=0).reshape(n_cols, n_rings)
            col_time = np.median(T, axis=1)
        meta["az_window_rule"] = "point azimuth within the window" if window else "none"
        return cls(n_rings=n_rings, n_cols=n_cols, elevation_deg=elev, ring_az_offset_deg=offsets, col_az_deg=col_az,
                   dual_return=dual, forward_axis=forward_axis, up_axis=up_axis, col_time_s=col_time,
                   ring_fire_parity=par, az_window_deg=window, meta=meta)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        arrays = {
            "n_rings": np.int64(self.n_rings), "n_cols": np.int64(self.n_cols),
            "elevation_deg": self.elevation_deg, "ring_az_offset_deg": self.ring_az_offset_deg,
            "col_az_deg": self.col_az_deg, "dual_return": np.bool_(self.dual_return),
            "forward_axis": np.str_(self.forward_axis), "up_axis": np.str_(self.up_axis),
            "sensor": np.str_(self.sensor), "meta_json": np.str_(json.dumps(self.meta)),
        }
        if self.col_time_s is not None:
            arrays["col_time_s"] = self.col_time_s
        if self.ring_fire_parity is not None:
            arrays["ring_fire_parity"] = self.ring_fire_parity
        if self.az_window_deg is not None:
            arrays["az_window_deg"] = np.asarray(self.az_window_deg, dtype=np.float64)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "BeamModel":
        with np.load(path, allow_pickle=False) as z:
            get = lambda k, d=None: z[k] if k in z.files else d  # noqa: E731
            win = get("az_window_deg")
            return cls(n_rings=int(z["n_rings"]), n_cols=int(z["n_cols"]), elevation_deg=z["elevation_deg"],
                       ring_az_offset_deg=z["ring_az_offset_deg"], col_az_deg=z["col_az_deg"],
                       dual_return=bool(z["dual_return"]), forward_axis=str(get("forward_axis", "-y")),
                       up_axis=str(get("up_axis", "+z")), col_time_s=get("col_time_s"),
                       ring_fire_parity=get("ring_fire_parity"),
                       az_window_deg=None if win is None else (float(win[0]), float(win[1])),
                       sensor=str(get("sensor", "hesai128")),
                       meta=json.loads(str(get("meta_json", "{}"))))


def _xyz(cloud) -> np.ndarray:
    if getattr(cloud, "dtype", None) is not None and cloud.dtype.names:
        return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float64)
    return np.asarray(cloud, dtype=np.float64)[:, :3]


# ====================================================================== Falcon Prime (Seyond) slot table
_KEY_IDX = 2048   # scan_idx < 2048
_KEY_FL = 64      # flags < 64


def falcon_key(scan_id, scan_idx, flags) -> np.ndarray:
    return (np.asarray(scan_id, dtype=np.int64) * _KEY_IDX + np.asarray(scan_idx, dtype=np.int64)) * _KEY_FL \
        + np.asarray(flags, dtype=np.int64)


@dataclass
class FalconBeamModel:
    """Slot table of the Falcon Prime as recorded (``/iv_points``, frame ``seyond``; x forward, y left,
    z up; the unit is pitched up ~2 deg, ~2.5 m above the track bed).  ``up_axis="-z"`` models an
    upside-down mount.

    The cloud is not a grid: every return carries ``scan_id`` (0..37, one slanted polygon-mirror line),
    ``scan_idx`` (position along the line) and ``flags`` (sub-beam = flags % 4, ~1.15 deg apart in
    elevation; flags // 4 = laser group).  A slot is one (scan_id, scan_idx, flags) triple; the table holds
    every slot that returned in the fit frames (their union is the fired-slot set, there are no zero points
    in the message) with its median direction and time.  Slots of the table absent from a cloud are its
    no-return positions.  The polygon phase jitters per frame and line: ``frame_line_shift`` fits
    az_shift = a + b * scan_idx per line from a cloud's own returns, ``directions_for_frame`` applies it.
    """

    scan_id: np.ndarray
    scan_idx: np.ndarray
    flags: np.ndarray
    az_deg: np.ndarray          # median azimuth (analysis frame of forward_axis / up_axis)
    el_deg: np.ndarray
    time_s: np.ndarray          # median point time minus header stamp (negative: before the stamp)
    seen_frac: np.ndarray       # fraction of fit frames in which the slot returned
    forward_axis: str = "+x"
    up_axis: str = "+z"
    sensor: str = "falcon_prime"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.scan_id = np.asarray(self.scan_id, dtype=np.uint16)
        self.scan_idx = np.asarray(self.scan_idx, dtype=np.uint16)
        self.flags = np.asarray(self.flags, dtype=np.uint8)
        for k in ("az_deg", "el_deg", "time_s"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64))
        self.seen_frac = np.asarray(self.seen_frac, dtype=np.float32)
        self._keys = falcon_key(self.scan_id, self.scan_idx, self.flags)
        if np.any(np.diff(self._keys) <= 0):
            order = np.argsort(self._keys)
            for k in ("scan_id", "scan_idx", "flags", "az_deg", "el_deg", "time_s", "seen_frac"):
                setattr(self, k, getattr(self, k)[order])
            self._keys = self._keys[order]

    @property
    def n_slots(self) -> int:
        return len(self._keys)

    @property
    def n_lines(self) -> int:
        return int(self.scan_id.max()) + 1

    def axis_matrix(self) -> np.ndarray:
        return axis_matrix(self.forward_axis, self.up_axis)

    def slot_index(self, scan_id, scan_idx, flags) -> np.ndarray:
        """Table index of each (scan_id, scan_idx, flags); -1 where the slot is not in the table."""
        k = falcon_key(scan_id, scan_idx, flags)
        j = np.searchsorted(self._keys, k)
        j = np.minimum(j, len(self._keys) - 1)
        return np.where(self._keys[j] == k, j, -1)

    def slot_of_cloud(self, cloud) -> np.ndarray:
        return self.slot_index(cloud["scan_id"], cloud["scan_idx"], cloud["flags"])

    def fired_mask(self) -> np.ndarray:
        return np.ones(self.n_slots, dtype=bool)

    @staticmethod
    def _dirs_dlh(az_deg, el_deg) -> np.ndarray:
        az, el = np.radians(az_deg), np.radians(el_deg)
        ce = np.cos(el)
        return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1)

    def directions_dlh(self, az_shift_deg: Optional[np.ndarray] = None) -> np.ndarray:
        az = self.az_deg if az_shift_deg is None else self.az_deg + az_shift_deg
        return self._dirs_dlh(az, self.el_deg)

    def directions(self, az_shift_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """(n_slots, 3) float32 unit vectors in the sensor frame (table order)."""
        return (self.directions_dlh(az_shift_deg) @ self.axis_matrix()).astype(np.float32)

    def angles_of_points(self, xyz) -> Tuple[np.ndarray, np.ndarray]:
        dlh = np.asarray(xyz, dtype=np.float64) @ self.axis_matrix().T
        r = np.linalg.norm(dlh, axis=1)
        return (np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0])),
                np.degrees(np.arcsin(np.clip(dlh[:, 2] / np.maximum(r, 1e-12), -1, 1))))

    def frame_line_shift(self, cloud, min_points: int = 20) -> np.ndarray:
        """(n_lines, 2) per-line (a, b): azimuth shift of this cloud = a + b * scan_idx (deg), from its first returns."""
        first = _first_returns(cloud)
        j = self.slot_of_cloud(cloud[first])
        ok = j >= 0
        az, _ = self.angles_of_points(_xyz(cloud[first][ok]))
        j = j[ok]
        d = _wrap(az - self.az_deg[j])
        sid = self.scan_id[j].astype(np.int64)
        idx = self.scan_idx[j].astype(np.float64)
        out = np.zeros((self.n_lines, 2))
        for s in range(self.n_lines):
            m = sid == s
            if m.sum() >= min_points:
                x, y = idx[m], d[m]
                med = np.median(y)
                keep = np.abs(y - med) < 0.5
                if keep.sum() >= min_points:
                    b, a = np.polyfit(x[keep], y[keep], 1)
                    out[s] = (a, b)
                else:
                    out[s] = (med, 0.0)
        return out

    def shift_of_slots(self, line_shift: np.ndarray) -> np.ndarray:
        s = self.scan_id.astype(np.int64)
        return line_shift[s, 0] + line_shift[s, 1] * self.scan_idx.astype(np.float64)

    def directions_for_frame(self, cloud) -> np.ndarray:
        """Sensor-frame unit vectors of all table slots with this cloud's per-line azimuth shift applied."""
        return self.directions(self.shift_of_slots(self.frame_line_shift(cloud)))

    def angular_error_deg(self, cloud, corrected: bool = True) -> np.ndarray:
        """Angle between each first return of ``cloud`` that is in the table and its slot direction."""
        first = _first_returns(cloud)
        c = cloud[first]
        j = self.slot_of_cloud(c)
        ok = j >= 0
        dirs = (self.directions_for_frame(cloud) if corrected else self.directions()).astype(np.float64)[j[ok]]
        xyz = _xyz(c[ok])
        u = xyz / np.linalg.norm(xyz, axis=1)[:, None]
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, dirs), axis=1), np.clip((u * dirs).sum(1), -1, 1)))

    def no_return_mask(self, cloud) -> np.ndarray:
        """(n_slots,) True for table slots without a first return in ``cloud``."""
        first = _first_returns(cloud)
        j = self.slot_of_cloud(cloud[first])
        m = np.ones(self.n_slots, dtype=bool)
        m[j[j >= 0]] = False
        return m

    @classmethod
    def estimate(cls, clouds: Sequence, stamps_ns: Sequence[int], forward_axis: str = "+x", up_axis: str = "+z",
                 iterations: int = 2) -> "FalconBeamModel":
        """Fit the table from clouds (structured arrays with x,y,z,timestamp,flags,scan_id,scan_idx[,is_2nd_return])."""
        R = axis_matrix(forward_axis, up_axis)
        per = []
        for c, st in zip(clouds, stamps_ns):
            c = c[_first_returns(c)]
            k = falcon_key(c["scan_id"], c["scan_idx"], c["flags"])
            k, first = np.unique(k, return_index=True)
            c = c[first]
            dlh = _xyz(c) @ R.T
            r = np.linalg.norm(dlh, axis=1)
            az = np.degrees(np.arctan2(dlh[:, 1], dlh[:, 0]))
            el = np.degrees(np.arcsin(np.clip(dlh[:, 2] / np.maximum(r, 1e-12), -1, 1)))
            t = np.asarray(c["timestamp"], dtype=np.float64) - st * 1e-9
            per.append((k, az, el, t, c))
        keys = np.unique(np.concatenate([p[0] for p in per]))
        nF = len(per)
        A = np.full((nF, len(keys)), np.nan)
        E, T = A.copy(), A.copy()
        for i, (k, az, el, t, _) in enumerate(per):
            j = np.searchsorted(keys, k)
            A[i, j], E[i, j], T[i, j] = az, el, t
        sid = (keys // _KEY_FL // _KEY_IDX).astype(np.int64)
        idx = ((keys // _KEY_FL) % _KEY_IDX).astype(np.float64)
        az_m = np.nanmedian(A, axis=0)
        shift_sd = []
        for _ in range(iterations):
            corr = np.zeros_like(A)
            for i in range(nF):
                d = _wrap(A[i] - az_m)
                for s in np.unique(sid):
                    m = (sid == s) & np.isfinite(d)
                    if m.sum() < 20:
                        continue
                    y, x = d[m], idx[m]
                    keep = np.abs(y - np.median(y)) < 0.5
                    b, a = np.polyfit(x[keep], y[keep], 1)
                    corr[i, sid == s] = a + b * idx[sid == s]
            az_m = np.nanmedian(A - corr, axis=0)
            shift_sd = corr
        el_m = np.nanmedian(E, axis=0)
        t_m = np.nanmedian(T, axis=0)
        seen = np.isfinite(A).mean(axis=0)
        flags = keys % _KEY_FL
        resid = _wrap(A - shift_sd - az_m)
        meta = {"n_frames": nF, "line_shift_sd_deg": float(np.nanstd(shift_sd[np.isfinite(A)])),
                "az_residual_after_shift_p95_deg": float(np.nanpercentile(np.abs(resid), 95)),
                "el_residual_p95_deg": float(np.nanpercentile(np.abs(E - el_m), 95))}
        return cls(scan_id=sid, scan_idx=idx.astype(np.int64), flags=flags, az_deg=az_m, el_deg=el_m, time_s=t_m,
                   seen_frac=seen, forward_axis=forward_axis, up_axis=up_axis, meta=meta)

    def save(self, path) -> None:
        np.savez_compressed(path, scan_id=self.scan_id, scan_idx=self.scan_idx, flags=self.flags,
                            az_deg=self.az_deg.astype(np.float32), el_deg=self.el_deg.astype(np.float32),
                            az_fine=(self.az_deg - self.az_deg.astype(np.float32)).astype(np.float32),
                            el_fine=(self.el_deg - self.el_deg.astype(np.float32)).astype(np.float32),
                            time_s=self.time_s, seen_frac=self.seen_frac,
                            forward_axis=np.str_(self.forward_axis), up_axis=np.str_(self.up_axis),
                            sensor=np.str_(self.sensor), meta_json=np.str_(json.dumps(self.meta)))

    @classmethod
    def load(cls, path) -> "FalconBeamModel":
        with np.load(path, allow_pickle=False) as z:
            az = z["az_deg"].astype(np.float64) + z["az_fine"].astype(np.float64)
            el = z["el_deg"].astype(np.float64) + z["el_fine"].astype(np.float64)
            return cls(scan_id=z["scan_id"], scan_idx=z["scan_idx"], flags=z["flags"], az_deg=az, el_deg=el,
                       time_s=z["time_s"], seen_frac=z["seen_frac"], forward_axis=str(z["forward_axis"]),
                       up_axis=str(z["up_axis"]), sensor=str(z["sensor"]), meta=json.loads(str(z["meta_json"])))


def _first_returns(cloud) -> np.ndarray:
    if cloud.dtype.names and "is_2nd_return" in cloud.dtype.names:
        return np.asarray(cloud["is_2nd_return"]) == 0
    return np.ones(len(cloud), dtype=bool)


# ---------------------------------------------------------------------- built-in Hesai model
# Fitted with BeamModel.estimate on the development bags (2026-10-17).  Reproduces every valid return of
# all six bags to < 5e-5 deg.  Generated by solution/synth/tools/fit_beam_model.py --print-const.
_HESAI_CONST = {'step_deg': -0.1, 'firing_period_s': 2.7780685949763026e-05, 'elevation_deg': [14.4019921, 13.5089839, 13.0250002, 12.5879688, 12.1200002, 11.6699998, 11.1879691, 10.748984, 10.2589842, 9.8079683, 9.317969, 8.8459762, 8.3389841, 7.8919924, 7.3759766, 6.9039842, 6.4149998, 5.9279688, 5.4409764, 4.9479685, 4.46, 3.9639844, 3.4659766, 2.9719922, 2.4500001, 1.9689845, 1.8409766, 1.715, 1.595, 1.4559765, 1.325, 1.2109766, 1.0569923, 0.9519923, 0.8209765, 0.7019922, 0.5909766, 0.4369922, 0.3229688, 0.2029688, 0.04, -0.0459766, -0.1909766, -0.3029688, -0.4249609, -0.5699609, -0.6899609, -0.8049609, -0.9709765, -1.0619922, -1.2069923, -1.3219922, -1.4309766, -1.5829688, -1.7099609, -1.8219921, -1.9819922, -2.0789843, -2.2249609, -2.3369923, -2.4499609, -2.5959767, -2.7389843, -2.8479688, -3.0159766, -3.1059764, -3.2519922, -3.3739844, -3.4789843, -3.6329689, -3.7579685, -3.8679689, -4.0189842, -4.1169921, -4.2739842, -4.3909765, -4.5009765, -4.6519919, -4.7689846, -4.8909766, -5.0459765, -5.1439843, -5.292969, -5.4059766, -5.5209768, -5.6729686, -5.7839844, -5.9089844, -6.0669919, -6.1579687, -6.6859767, -7.1819922, -7.7069925, -8.2059763, -8.7469922, -9.2159767, -9.7499611, -10.2309767, -10.7589837, -11.2429692, -11.7569928, -12.2449614, -12.7719927, -13.2429682, -13.7699614, -14.2399607, -14.7619921, -15.2469929, -15.7479682, -16.2209758, -16.7359777, -17.2069915, -17.70496, -18.1709769, -18.6619928, -19.1429687, -19.6129681, -20.0989842, -20.5619933, -21.0359762, -21.4929685, -21.9709766, -22.4239839, -22.8729687, -23.3279692, -23.7969935, -24.2369921, -25.1199613], 'ring_az_offset_deg': [-3.3139844, -3.3139844, -1.09, -3.3119922, -1.0989844, -3.3129688, -1.0969922, -3.3129688, -1.0929687, -3.3239843, -1.1039844, -3.3109766, -1.095, -3.3079688, -1.0909766, 1.0729687, -1.075, 1.0709766, 3.3489843, 1.0729688, 3.3309766, 1.0679688, 3.3389844, 1.0759766, 3.3539844, 1.0769922, -7.6909766, -5.4789844, -3.3119922, 3.3319922, -1.0829688, 5.5009766, 7.7069922, 1.0859765, -7.7059765, -5.4809766, -3.3129687, 3.3439844, -1.08, 5.5069922, 7.7119922, 1.0799609, -7.7089844, -5.5039844, -3.3209766, 3.3379688, -1.0879688, 5.5109766, 7.7209766, 1.0879688, -7.7169922, -5.4869922, -3.33, 3.3439844, -1.0769922, 5.5229687, 7.7309766, 1.0809766, -7.7309766, -5.5029687, -3.33, 3.3349609, -1.0769922, 5.5389844, 7.7469922, 1.1039844, -7.735, -5.5079687, -3.3269922, 3.3519922, -1.0819922, 5.5429688, 7.7589844, 1.1059765, -7.7439844, -5.5209766, -3.3229687, 3.3529688, -1.0819922, 5.5529688, 7.7709766, 1.1069922, -7.78, -5.5419922, -3.335, 3.3659765, -1.0859766, 5.5669922, 7.7759766, 1.1059766, 3.3559766, 1.1119922, 3.3809766, 1.1299609, 3.3979688, -3.3629688, 3.3959766, -3.3569922, -1.1, -3.3739844, -1.1, -3.3889844, -1.1039844, -3.3929687, -1.0889844, -3.3969922, -1.1019922, -3.3919922, -1.11, -3.4189843, -1.1119922, 1.1539844, -1.1269922, 1.1579687, 3.4659766, 1.1719922, 3.4759766, 1.1769922, 3.4709766, 1.1659765, 3.4869922, 1.1819922, 3.4939844, 1.1859766, 3.4899609, 1.1869922, 3.4959766, 3.4979688], 'ring_fire_parity': [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], 'layouts': {2400: {'az0_deg': 60.0, 't0_s': -1.1920928955078125e-07, 'az_window_deg': [-50.00102, 50.0075]}, 7200: {'az0_deg': 179.989998, 't0_s': 0.0, 'az_window_deg': None}}}


def hesai128(n_cols: int = 2400, forward_axis: str = "-y", up_axis: str = "+z") -> BeamModel:
    """Built-in beam model of the dataset's sensor for the 307 200-point (2400 cols, 120 deg, crop +-50 deg of
    point azimuth) or the 921 600-point (7200 cols, 360 deg) message layout.  No data file needed."""
    c = _HESAI_CONST
    if c is None:
        raise RuntimeError("built-in constants missing")
    if n_cols not in c["layouts"]:
        raise ValueError(f"unknown layout {n_cols}, known: {sorted(c['layouts'])}")
    lay = c["layouts"][n_cols]
    n_fir = n_cols // 2
    fir = np.arange(n_fir)
    fir_az = lay["az0_deg"] + c["step_deg"] * fir
    col_az = _wrap(np.repeat(fir_az, 2))
    col_time = np.repeat(fir * c["firing_period_s"] + lay["t0_s"], 2)
    win = lay.get("az_window_deg")
    return BeamModel(n_rings=128, n_cols=n_cols, elevation_deg=np.array(c["elevation_deg"]),
                     ring_az_offset_deg=np.array(c["ring_az_offset_deg"]), col_az_deg=col_az, dual_return=True,
                     forward_axis=forward_axis, up_axis=up_axis, col_time_s=col_time,
                     ring_fire_parity=np.array(c["ring_fire_parity"], dtype=np.int8),
                     az_window_deg=None if win is None else tuple(win), sensor="hesai128",
                     meta={"source": "built-in constants", "col_time_model": "linear (mean firing period)"})
