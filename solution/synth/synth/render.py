"""Per-beam echoes and their conversion to/from message layouts.

A *beam* is one firing of one ring (Hesai: ``n_firings * 128`` beams, two message columns per beam) or one
Falcon slot.  ``Echoes`` holds up to two returns per beam: index 0 = near/first, 1 = far/last; range 0 = none.
``src`` is -1 for background structure, otherwise the object / dust id.

Object overlay semantics (dual-return sensor, beam footprint sampled by ``n_sub`` sub-rays):
* the object return range is the nearest sub-ray hit, its detection uses the footprint fraction f;
* object nearer than the near echo -> it becomes the near echo; if the whole footprint is on the object
  (f = 1, solid) the far echo is the object too, otherwise the far echo keeps the background (mixed pixel);
* object between the near and the far echo -> it replaces the far echo when f = 1;
* object hit with f = 1 but not detected (dark / far) -> the background behind it is blocked (shadow).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .returns import ReturnModel


@dataclass
class Echoes:
    r: np.ndarray        # (n, 2) float64
    inten: np.ndarray    # (n, 2) float32
    src: np.ndarray      # (n, 2) int32
    strength: Optional[np.ndarray] = None   # (n, 2) rho * cos^a * footprint fraction (for attenuation by dust)

    def __post_init__(self):
        if self.strength is None:
            self.strength = np.zeros(self.r.shape)

    @classmethod
    def empty(cls, n):
        return cls(np.zeros((n, 2)), np.zeros((n, 2), dtype=np.float32), np.full((n, 2), -1, dtype=np.int32))

    def single(self, r, inten, src=-1, mask=None):
        """Set one echo (both slots) where mask."""
        m = np.ones(len(self.r), dtype=bool) if mask is None else mask
        self.r[m, 0] = self.r[m, 1] = np.asarray(r)[m] if np.ndim(r) else r
        self.inten[m, 0] = self.inten[m, 1] = np.asarray(inten)[m] if np.ndim(inten) else inten
        self.src[m, 0] = self.src[m, 1] = np.asarray(src)[m] if np.ndim(src) else src

    def strongest_only(self):
        """Single-return sensor: keep the echo with the larger intensity in both slots."""
        e = Echoes(self.r.copy(), self.inten.copy(), self.src.copy(), self.strength.copy())
        has0, has1 = e.r[:, 0] > 0, e.r[:, 1] > 0
        pick1 = has1 & (~has0 | (e.inten[:, 1] > e.inten[:, 0]))
        for k in (0, 1):
            e.r[pick1, k] = self.r[pick1, 1]
            e.inten[pick1, k] = self.inten[pick1, 1]
            e.src[pick1, k] = self.src[pick1, 1]
            e.strength[pick1, k] = self.strength[pick1, 1]
            e.strength[~pick1, k] = self.strength[~pick1, 0]
            e.r[~pick1, k] = self.r[~pick1, 0]
            e.inten[~pick1, k] = self.inten[~pick1, 0]
            e.src[~pick1, k] = self.src[~pick1, 0]
        return e


def edge_mixed_pixels(e: "Echoes", nb: np.ndarray, rng, p_mix: float, min_jump_m: float = 0.5, inten_factor: float = 0.4):
    """Second echoes at depth edges: a beam whose scan neighbour (``nb``: (n, 2) indices, -1 = none) returned
    at least ``min_jump_m`` farther gets, with probability ``p_mix``, a far echo at the neighbour's range (the
    beam footprint straddles the edge).  Calibrated to the real dual-echo fraction (Hesai 1.4 %, Falcon 2.1 %)."""
    if p_mix <= 0:
        return 0
    r1 = e.r[:, 0]
    best_r = np.zeros(len(r1))
    best_i = np.zeros(len(r1), dtype=np.float32)
    for k in range(nb.shape[1]):
        j = nb[:, k]
        ok = j >= 0
        rj = np.where(ok, e.r[np.maximum(j, 0), 0], 0.0)
        cand = ok & (r1 > 0) & (rj > r1 + min_jump_m) & (rj > best_r)
        best_r = np.where(cand, rj, best_r)
        best_i = np.where(cand, e.inten[np.maximum(j, 0), 0], best_i)
    single = (e.r[:, 1] <= 0) | (np.abs(e.r[:, 1] - r1) < 1e-6)
    m = (best_r > 0) & single & (rng.random(len(r1)) < p_mix)
    e.r[m, 1] = best_r[m]
    e.inten[m, 1] = np.maximum(1.0, np.round(best_i[m] * inten_factor))
    e.src[m, 1] = -1
    e.strength[m, 1] = e.strength[m, 0] * inten_factor
    return int(m.sum())


@dataclass
class SceneObject:
    id: int
    cls: str
    V: np.ndarray            # (n, 3) vertices in the frame of the rays (sensor frame or world)
    F: np.ndarray
    reflectivity: float
    is_obstacle: bool = True
    meta: Optional[dict] = None


def _cone_candidates(dirs, origins, V, margin_rad):
    c = 0.5 * (V.max(0) + V.min(0))
    R = 0.5 * np.linalg.norm(V.max(0) - V.min(0))
    o = origins if origins.ndim == 2 else origins[None]
    v = c[None] - o
    dist = np.linalg.norm(v, axis=1)
    u = v / np.maximum(dist, 1e-9)[:, None]
    half = np.arcsin(np.clip(R / np.maximum(dist, 1e-9), 0, 1)) + margin_rad
    half = np.where(dist <= R, np.pi, half)
    return (dirs * u).sum(1) >= np.cos(np.minimum(half, np.pi))


def sub_ray_offsets(n_sub: int, divergence_deg: float) -> np.ndarray:
    """(n_sub, 2) angular offsets (deg) of sub-rays inside a beam of full divergence ``divergence_deg``:
    the centre ray plus rings at 0.35 * divergence (a Gaussian beam's 1/e^2 footprint)."""
    if n_sub <= 1 or divergence_deg <= 0:
        return np.zeros((1, 2))
    k = n_sub - 1
    ang = 2 * np.pi * np.arange(k) / k
    return np.vstack([[0, 0], 0.35 * divergence_deg * np.stack([np.cos(ang), np.sin(ang)], 1)])


def overlay_objects(echoes: Echoes, objects: Sequence[SceneObject], origins, dirs, caster, ret: ReturnModel, rng,
                    divergence_deg: float = 0.0, n_sub: int = 7, up=(0.0, 0.0, 1.0), beams: Optional[np.ndarray] = None):
    """Render objects into ``echoes`` (modified in place).  ``beams``: indices of the beams to consider (default all).
    Returns per-object dict: geometric centre-ray hits, nearest geometric range, detected beams."""
    info = {o.id: {"geom_beams": 0, "geom_min_range": np.inf, "detected_beams": 0, "occluded_by_objects_frac": 0.0}
            for o in objects}
    if not objects:
        return info
    dirs = np.asarray(dirs, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    n = len(dirs)
    beams = np.arange(n) if beams is None else np.asarray(beams)
    offs = sub_ray_offsets(n_sub, divergence_deg)
    margin = np.radians(abs(offs).max() + 0.01)
    # one combined mesh of all objects with a face -> object map
    Vs, Fs, fobj, off = [], [], [], 0
    cand = np.zeros(n, dtype=bool)
    o_full = origins if origins.ndim == 2 else np.broadcast_to(origins, (n, 3))
    for k, o in enumerate(objects):
        Vs.append(o.V)
        Fs.append(o.F + off)
        fobj.append(np.full(len(o.F), k))
        off += len(o.V)
        cand[beams] |= _cone_candidates(dirs[beams], o_full[beams], o.V, margin)
    mh = caster.mesh(np.vstack(Vs), np.vstack(Fs))
    fobj = np.concatenate(fobj)
    idx = np.nonzero(cand)[0]
    if len(idx) == 0:
        return info
    d0 = dirs[idx]
    o0 = o_full[idx]
    upv = np.asarray(up, dtype=np.float64)
    e_az = np.cross(upv[None], d0)
    nrm = np.linalg.norm(e_az, axis=1, keepdims=True)
    e_az = np.where(nrm > 1e-9, e_az / np.maximum(nrm, 1e-12), np.array([[1.0, 0, 0]]))
    e_el = np.cross(d0, e_az)
    m = len(idx)
    S = len(offs)
    T = np.full((S, m), np.inf)
    OB = np.full((S, m), -1, dtype=np.int64)
    COS = np.zeros((S, m))
    for s, (u, v) in enumerate(offs):
        ds = d0 + np.radians(u) * e_az + np.radians(v) * e_el
        ds /= np.linalg.norm(ds, axis=1, keepdims=True)
        t, face, normal = caster.cast(mh, o0, ds, ret.max_range_m * 1.5 + 50)
        hit = face >= 0
        T[s, hit] = t[hit]
        OB[s, hit] = fobj[face[hit]]
        COS[s, hit] = np.abs((normal[hit] * ds[hit]).sum(1))
    # nearest object per beam and its footprint fraction
    s_near = np.argmin(T, axis=0)
    t_obj = T[s_near, np.arange(m)]
    ob = OB[s_near, np.arange(m)]
    anyhit = np.isfinite(t_obj)
    frac = np.where(anyhit, (OB == ob[None]).sum(0) / S, 0.0)
    cos_t = np.where(OB[0] == ob, COS[0], COS[s_near, np.arange(m)])
    rho = np.array([o.reflectivity for o in objects])[np.maximum(ob, 0)]
    for k, o in enumerate(objects):
        mk = (OB[0] == k)
        info[o.id]["geom_beams"] = int(mk.sum())
        if (OB == k).any():
            info[o.id]["geom_min_range"] = float(T[OB == k].min())
        if len(objects) > 1:
            # inter-object occlusion: centre rays that hit this object's own mesh but another object first
            _, f_own, _ = caster.cast(caster.mesh(o.V, o.F), o0, d0, ret.max_range_m * 1.5 + 50)
            own = f_own >= 0
            info[o.id]["occluded_by_objects_frac"] = (round(float((own & ~mk).sum() / own.sum()), 3)
                                                      if own.any() else None)
    det = anyhit & ret.detect(rng, rho, cos_t, t_obj, frac)
    r_meas = ret.measure_range(rng, t_obj)
    inten = ret.intensity(rng, rho, cos_t, t_obj)
    # background echoes of the candidate beams before any change
    rn, rf = echoes.r[idx, 0].copy(), echoes.r[idx, 1].copy()
    i_n, i_f = echoes.inten[idx, 0].copy(), echoes.inten[idx, 1].copy()
    s_n, s_f = echoes.src[idx, 0].copy(), echoes.src[idx, 1].copy()
    g_n, g_f = echoes.strength[idx, 0].copy(), echoes.strength[idx, 1].copy()
    has_n, has_f = rn > 0, rf > 0
    last_g = np.where(has_f, g_f, g_n)
    g_obj = rho * np.clip(cos_t, 0, 1) ** ret.detect_cos_exp * frac
    last_r = np.where(has_f, rf, rn)            # the background's last echo (single echo = both)
    last_i = np.where(has_f, i_f, i_n)
    last_s = np.where(has_f, s_f, s_n)
    solid = frac >= 0.999
    in_front = anyhit & (~has_n | (t_obj < rn))
    between = anyhit & has_n & ~in_front & (~has_f | (t_obj < rf))
    ids = np.array([o.id for o in objects], dtype=np.int32)[np.maximum(ob, 0)]
    new_r, new_i, new_s = echoes.r[idx].copy(), echoes.inten[idx].copy(), echoes.src[idx].copy()
    new_g = echoes.strength[idx].copy()
    # detected in front of everything: near = object; far = object (solid) or the background's last echo
    a = det & in_front
    new_r[a, 0], new_i[a, 0], new_s[a, 0], new_g[a, 0] = r_meas[a], inten[a], ids[a], g_obj[a]
    a_solid = a & solid
    new_r[a_solid, 1], new_i[a_solid, 1], new_s[a_solid, 1], new_g[a_solid, 1] = r_meas[a_solid], inten[a_solid], ids[a_solid], g_obj[a_solid]
    a_mix = a & ~solid
    keep = a_mix & (last_r > 0)
    new_r[keep, 1], new_i[keep, 1], new_s[keep, 1], new_g[keep, 1] = last_r[keep], last_i[keep], last_s[keep], last_g[keep]
    lone = a_mix & (last_r <= 0)
    new_r[lone, 1], new_i[lone, 1], new_s[lone, 1], new_g[lone, 1] = r_meas[lone], inten[lone], ids[lone], g_obj[lone]
    # detected between the near and the far echo, solid: replaces the far echo
    b = det & between & solid
    new_r[b, 1], new_i[b, 1], new_s[b, 1], new_g[b, 1] = r_meas[b], inten[b], ids[b], g_obj[b]
    # not detected but the footprint is fully on the object: the background behind it is blocked
    c = ~det & solid & in_front
    new_r[c], new_i[c], new_s[c], new_g[c] = 0.0, 0.0, -1, 0.0
    c2 = ~det & solid & between
    new_r[c2, 1], new_i[c2, 1], new_s[c2, 1], new_g[c2, 1] = rn[c2], i_n[c2], s_n[c2], g_n[c2]
    echoes.r[idx], echoes.inten[idx], echoes.src[idx], echoes.strength[idx] = new_r, new_i, new_s, new_g
    for k, o in enumerate(objects):
        info[o.id]["detected_beams"] = int((det & (ob == k)).sum())
    return info


# --------------------------------------------------------------------------- Hesai layout
def hesai_echoes_from_cloud(cloud, bm) -> Echoes:
    """Echoes per beam (firing-major, ring fastest) from a real organised cloud: odd column = near, even = far."""
    nf = bm.n_firings
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], 1).astype(np.float64).reshape(nf, 2, bm.n_rings, 3)
    r = np.linalg.norm(xyz, axis=3)                       # (nf, 2, rings)
    it = np.asarray(cloud["intensity"], dtype=np.float32).reshape(nf, 2, bm.n_rings)
    e = Echoes.empty(nf * bm.n_rings)
    e.r[:, 0] = r[:, 1].reshape(-1)
    e.r[:, 1] = r[:, 0].reshape(-1)
    e.inten[:, 0] = it[:, 1].reshape(-1)
    e.inten[:, 1] = it[:, 0].reshape(-1)
    return e


def hesai_beam_dirs(bm) -> np.ndarray:
    """(n_firings*n_rings, 3) sensor-frame direction per beam (the near-column slot's direction)."""
    d = bm.directions().astype(np.float64).reshape(bm.n_firings, 2, bm.n_rings, 3)
    return d[:, 1].reshape(-1, 3)


def hesai_beam_fired(bm) -> np.ndarray:
    return bm.fired_mask().reshape(bm.n_firings, 2, bm.n_rings)[:, 1].reshape(-1)


_BM_CACHE = {}


def _bm_arrays(bm):
    key = id(bm)
    if key not in _BM_CACHE:
        _BM_CACHE.clear()
        _BM_CACHE[key] = (bm, bm.directions().astype(np.float64).reshape(bm.n_firings, 2, bm.n_rings, 3), bm.slot_time_s(),
                          np.tile(np.arange(bm.n_rings, dtype=np.uint16), bm.n_firings * 2))
    return _BM_CACHE[key][1:]


def hesai_points(echoes: Echoes, bm, stamp_ns: int, timestamps=None):
    """Structured message array (column-major, ring fastest; even col = far, odd col = near) + per-slot source."""
    from .writer import HESAI_DTYPE

    nf, nr = bm.n_firings, bm.n_rings
    dirs, slot_t, rings = _bm_arrays(bm)
    r = np.zeros((nf, 2, nr))
    it = np.zeros((nf, 2, nr), dtype=np.float32)
    src = np.full((nf, 2, nr), -1, dtype=np.int32)
    near_r, far_r = echoes.r[:, 0].reshape(nf, nr), echoes.r[:, 1].reshape(nf, nr)
    far_use = np.where(far_r > 0, far_r, near_r)
    r[:, 1], r[:, 0] = near_r, far_use
    it[:, 1] = echoes.inten[:, 0].reshape(nf, nr)
    it[:, 0] = np.where(far_r > 0, echoes.inten[:, 1].reshape(nf, nr), echoes.inten[:, 0].reshape(nf, nr))
    src[:, 1] = echoes.src[:, 0].reshape(nf, nr)
    src[:, 0] = np.where(far_r > 0, echoes.src[:, 1].reshape(nf, nr), echoes.src[:, 0].reshape(nf, nr))
    pts = np.zeros(nf * 2 * nr, dtype=HESAI_DTYPE)
    xyz = (dirs * r[..., None]).reshape(-1, 3)
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    pts["intensity"] = np.where(r.reshape(-1) > 0, it.reshape(-1), 0.0)
    pts["ring"] = rings
    pts["timestamp"] = (stamp_ns * 1e-9 + slot_t) if timestamps is None else timestamps
    src = np.where(r > 0, src, -1).reshape(-1)
    return pts, src
