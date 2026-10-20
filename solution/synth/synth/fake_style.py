"""Object insertion in the organisers' style (``dataset/extra/cloud_with_fake_obj``).

Measured on that bag (``work/specs/11_fake_style.md``, probe: ``tools/fake_style_probe.py``):

* message: ``x, y, z, intensity`` float32 only (``point_step`` 16, no ring / timestamp), ``height`` 1;
* the base cloud (real organised Hesai frame) keeps its point order; every base *return* whose beam hits an inserted
  object nearer than the return is deleted (the cloud is compacted); (0, 0, 0) no-return points are kept;
* the object points come from an idealised sensor - 128 rows at the nominal OT128 elevation table, azimuth
  k * 0.15 deg, rays from the sensor origin, one point per ray, every hit returns, exact range (no noise, no
  quantisation, no 209 m cap; rendered up to 300 m), intensity exactly 1.0, any azimuth (no +-50 deg crop);
* they are appended at the end of the cloud: objects farthest first, each object row-major (top row first, inside a
  row column index ``k mod 2400`` ascending, i.e. azimuth 0, +0.15, ..., then the negative azimuths);
* no visibility test against the base scene: an object behind a real wall is still appended (``occlude_by_base``
  switches such a test on; off = the organisers' style);
* objects do not occlude each other (each object is rendered alone; observed: one row-major run per object).

Pure NumPy apart from the ray caster (``raycast.Caster``: Warp / Open3D / NumPy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# nominal elevation table of the organisers' renderer (degrees, ascending), measured residual <= 4e-6 deg
OT128_EL_DEG = np.array(
    [-25.0] + [round(-24.1 + 0.5 * k, 3) for k in range(37)] + [round(-6.0 + 0.125 * k, 3) for k in range(64)]
    + [1.9, 2.0] + [round(2.4 + 0.5 * k, 3) for k in range(23)] + [14.4])
AZ_STEP_DEG = 0.15
N_COLS = 2400
MAX_RANGE_M = 300.0
OBJ_INTENSITY = 1.0

FAKE_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity"], "formats": ["<f4", "<f4", "<f4", "<f4"],
                       "offsets": [0, 4, 8, 12], "itemsize": 16})

assert len(OT128_EL_DEG) == 128


def grid_directions(forward_axis: str = "-y", up_axis: str = "+z") -> np.ndarray:
    """(128 * 2400, 3) unit ray directions in the sensor frame, in the organisers' emission order:
    row-major, row 0 = top elevation (14.4 deg), column k = azimuth k * 0.15 deg (k = 0..2399, wrapped)."""
    from metro_obstacle_detector.beam_model import axis_matrix

    el = np.radians(OT128_EL_DEG[::-1])
    k = np.arange(N_COLS)
    az = np.radians(np.where(k < N_COLS // 2, k, k - N_COLS) * AZ_STEP_DEG)
    ce = np.cos(el)[:, None]
    dlh = np.stack([np.broadcast_to(ce * np.cos(az)[None, :], (128, N_COLS)),
                    np.broadcast_to(ce * np.sin(az)[None, :], (128, N_COLS)),
                    np.broadcast_to(np.sin(el)[:, None], (128, N_COLS))], axis=-1).reshape(-1, 3)
    R = axis_matrix(forward_axis, up_axis)          # rows d, l, h in sensor coordinates
    return (dlh @ R).astype(np.float64)


@dataclass
class FakeObject:
    """One inserted object in the sensor frame of the current frame."""
    id: int
    V: np.ndarray                 # (n, 3) vertices, sensor frame
    F: np.ndarray                 # (m, 3) triangles


@dataclass
class FakeResult:
    points: np.ndarray            # FAKE_DTYPE message points (base kept + appended)
    n_base: int                   # base points in the message
    deleted: np.ndarray           # bool over the input base points: deleted
    deleted_by: np.ndarray        # int32 over the input base points: object id that deleted it (0 = kept)
    obj_slices: dict = field(default_factory=dict)   # id -> (start, stop) message indices of its appended points
    obj_range: dict = field(default_factory=dict)    # id -> min range of its appended points (None if none)


def _cone(dirs: np.ndarray, V: np.ndarray, margin_deg: float = 0.3) -> np.ndarray:
    """Indices of rays (from the origin) that can hit the bounding sphere of V."""
    c = 0.5 * (V.max(0) + V.min(0))
    rad = 0.5 * float(np.linalg.norm(V.max(0) - V.min(0)))
    dist = float(np.linalg.norm(c))
    if dist <= rad + 0.05:
        return np.arange(len(dirs))
    ang = np.arcsin(min(1.0, rad / dist)) + np.radians(margin_deg)
    return np.nonzero(dirs @ (c / dist) >= np.cos(ang))[0]


class FakeStyleRenderer:
    def __init__(self, caster, forward_axis: str = "-y", up_axis: str = "+z", max_range: float = MAX_RANGE_M,
                 occlude_by_base: bool = False, delete_eps_m: float = 0.0):
        self.caster = caster
        self.grid = grid_directions(forward_axis, up_axis)
        self.max_range = float(max_range)
        self.occlude_by_base = bool(occlude_by_base)
        self.eps = float(delete_eps_m)

    def render(self, base: np.ndarray, objects: Sequence[FakeObject]) -> FakeResult:
        """``base``: structured cloud with x, y, z, intensity (any further fields are dropped)."""
        xyz = np.stack([base["x"], base["y"], base["z"]], axis=1).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        valid = r > 0.0
        vi = np.nonzero(valid)[0]
        u = np.zeros_like(xyz)
        u[vi] = xyz[vi] / r[vi, None]
        deleted_by = np.zeros(len(base), dtype=np.int32)
        blocks = []
        for ob in objects:
            mh = self.caster.mesh(ob.V, ob.F)
            # 1) deletion of base returns occluded by the object (real beam directions = the returns' own)
            cand = vi[_cone(u[vi], ob.V)]
            if len(cand):
                t, face, _ = self.caster.cast(mh, np.zeros(3), u[cand], float(r[cand].max()) + 1.0)
                hit = (face >= 0) & (t < r[cand] - self.eps)
                sel = cand[hit]
                deleted_by[sel] = np.where(deleted_by[sel] == 0, ob.id, deleted_by[sel])
            # 2) object points on the organisers' grid
            gc = _cone(self.grid, ob.V)
            pts = np.zeros((0, 3))
            if len(gc):
                t, face, _ = self.caster.cast(mh, np.zeros(3), self.grid[gc], self.max_range)
                hit = (face >= 0) & np.isfinite(t) & (t <= self.max_range)
                g = gc[hit]                      # already in emission order (ascending grid index)
                pts = self.grid[g] * t[hit, None]
            blocks.append((ob.id, pts))
        if self.occlude_by_base and blocks:
            blocks = [(oid, self._visible(p, xyz[vi], r[vi])) for oid, p in blocks]
        keep = deleted_by == 0
        n_base = int(keep.sum())
        # farthest object first (median range of its points)
        order = sorted(range(len(blocks)), key=lambda k: -(float(np.median(np.linalg.norm(blocks[k][1], axis=1)))
                                                           if len(blocks[k][1]) else -1.0))
        n_app = sum(len(b[1]) for b in blocks)
        out = np.zeros(n_base + n_app, dtype=FAKE_DTYPE)
        for f in ("x", "y", "z", "intensity"):
            out[f][:n_base] = base[f][keep]
        pos = n_base
        slices, ranges = {}, {}
        for k in order:
            oid, p = blocks[k]
            n = len(p)
            out["x"][pos:pos + n] = p[:, 0]
            out["y"][pos:pos + n] = p[:, 1]
            out["z"][pos:pos + n] = p[:, 2]
            out["intensity"][pos:pos + n] = OBJ_INTENSITY
            slices[oid] = (pos, pos + n)
            ranges[oid] = float(np.linalg.norm(p, axis=1).min()) if n else None
            pos += n
        return FakeResult(out, n_base, ~keep, deleted_by, slices, ranges)

    @staticmethod
    def _visible(p: np.ndarray, base_xyz: np.ndarray, base_r: np.ndarray, tol_deg: float = 0.12) -> np.ndarray:
        """Drop object points that have a nearer base return within ``tol_deg`` (optional realism switch)."""
        if not len(p):
            return p
        from scipy.spatial import cKDTree

        ub = base_xyz / base_r[:, None]
        tree = cKDTree(ub)
        rp = np.linalg.norm(p, axis=1)
        nb = tree.query_ball_point(p / rp[:, None], r=np.radians(tol_deg))
        keep = np.array([not any(base_r[j] < rp[k] - 0.5 for j in lst) for k, lst in enumerate(nb)], dtype=bool)
        return p[keep]


def base_beam_model(cloud: np.ndarray, n_cols: int = 2400):
    """Built-in Hesai beam model with the azimuth phase of this recording session (organised base cloud):
    median of (point azimuth - slot azimuth) over the valid returns (six bags 0.00, new_data +0.01,
    cloud_with_fake_obj / org_synth +0.05 deg)."""
    import dataclasses

    from metro_obstacle_detector.beam_model import _wrap, hesai128

    bm = hesai128(n_cols)
    if len(cloud) != bm.n_slots:
        raise ValueError(f"base cloud has {len(cloud)} points, expected the organised {bm.n_slots}")
    x, y, z = (np.asarray(cloud[k], dtype=np.float64) for k in ("x", "y", "z"))
    v = (x != 0) | (y != 0) | (z != 0)
    az = np.degrees(np.arctan2(x[v], -y[v]))
    res = _wrap(az - bm.slot_az_deg()[v])
    phase = float(np.median(res))
    if abs(phase) > 1e-4:
        bm = dataclasses.replace(bm, col_az_deg=_wrap(bm.col_az_deg + phase), meta=dict(bm.meta, phase_deg=phase))
    return bm, phase


def _restore_base(out: np.ndarray, src: np.ndarray, base: np.ndarray) -> None:
    """Background slots whose range is unchanged get the base's exact x, y, z, intensity back (the echo model
    rebuilds points from beam directions)."""
    rb = np.sqrt(base["x"].astype(np.float64) ** 2 + base["y"] ** 2 + base["z"] ** 2)
    ro = np.sqrt(out["x"].astype(np.float64) ** 2 + out["y"] ** 2 + out["z"] ** 2)
    m = (src < 0) & (np.abs(ro - rb) < 1e-3)
    for f in ("x", "y", "z", "intensity"):
        out[f][m] = base[f][m]


class InPlaceRenderer:
    """The organisers' ``org_synth`` style: the organised layout and all fields are kept; an object hit along the
    real beam (both columns of the pair) replaces a farther return or fills a no-return slot, never a nearer
    return; intensity 1.0; no range cap (inserted ranges up to 335 m seen); objects occlude each other."""

    def __init__(self, caster, max_range: float = 400.0):
        self.caster = caster
        self.max_range = max_range

    def render(self, base: np.ndarray, objects: Sequence[FakeObject], bm):
        from .render import hesai_beam_dirs, hesai_beam_fired

        out = np.array(base, copy=True)
        src = np.full(len(base), -1, dtype=np.int32)
        if not objects:
            return out, src
        dirs = hesai_beam_dirs(bm)
        fired = np.nonzero(hesai_beam_fired(bm))[0]
        nf, nr = bm.n_firings, bm.n_rings
        Vs, Fs, fobj, off = [], [], [], 0
        cand = np.zeros(len(dirs), bool)
        for k, ob in enumerate(objects):
            Vs.append(ob.V); Fs.append(ob.F + off); fobj.append(np.full(len(ob.F), ob.id)); off += len(ob.V)
            cand[fired[_cone(dirs[fired], ob.V)]] = True
        beams = np.nonzero(cand)[0]
        if not len(beams):
            return out, src
        mh = self.caster.mesh(np.vstack(Vs), np.vstack(Fs))
        fobj = np.concatenate(fobj)
        t, face, _ = self.caster.cast(mh, np.zeros(3), dirs[beams], self.max_range)
        hit = (face >= 0) & np.isfinite(t)
        b = beams[hit]; t = t[hit]; oid = fobj[face[hit]]
        fir, ring = b // nr, b % nr
        s_far = (2 * fir) * nr + ring          # even column = far / last
        s_near = s_far + nr
        rb = np.sqrt(base["x"].astype(np.float64) ** 2 + base["y"] ** 2 + base["z"] ** 2)
        near_r = np.where(rb[s_near] > 0, rb[s_near], np.inf)
        far_r = np.where(rb[s_far] > 0, rb[s_far], np.inf)
        rep = t < np.minimum(near_r, far_r)    # never replaces a nearer return
        p = dirs[b[rep]] * t[rep, None]
        for s in (s_far[rep], s_near[rep]):
            out["x"][s], out["y"][s], out["z"][s] = p[:, 0], p[:, 1], p[:, 2]
            out["intensity"][s] = OBJ_INTENSITY
            src[s] = oid[rep]
        return out, src


class PhysicalRenderer:
    """Physically plausible insertion (our ``insert_real`` model): real beam directions of the base session,
    dual returns, calibrated detection probability / range noise / intensity (``returns.ReturnModel.hesai``),
    objects occlude the background and each other.  The base layout (fields) is kept."""

    def __init__(self, caster, ret=None):
        from .returns import ReturnModel

        self.caster = caster
        self.ret = ret if ret is not None else ReturnModel.hesai()

    def render(self, base: np.ndarray, objects: Sequence[FakeObject], bm, rho: dict, rng, stamp_ns: int, up):
        from .render import SceneObject, hesai_beam_dirs, hesai_beam_fired, hesai_echoes_from_cloud, hesai_points, overlay_objects

        echoes = hesai_echoes_from_cloud(base, bm)
        objs = [SceneObject(id=o.id, cls="obj", V=o.V, F=o.F, reflectivity=float(rho[o.id])) for o in objects]
        if objs:
            overlay_objects(echoes, objs, np.zeros(3), hesai_beam_dirs(bm), self.caster, self.ret, rng,
                            divergence_deg=0.0, n_sub=1, up=up, beams=np.nonzero(hesai_beam_fired(bm))[0])
        ts = base["timestamp"] if "timestamp" in (base.dtype.names or ()) else None
        pts, src = hesai_points(echoes, bm, stamp_ns, timestamps=ts)
        if set(base.dtype.names) == set(FAKE_DTYPE.names):
            pts = to_fake_layout(pts)
        _restore_base(pts, src, base)
        return pts, src


def to_fake_layout(cloud: np.ndarray) -> np.ndarray:
    """A base cloud as the organisers publish it: only x, y, z, intensity (point_step 16), same order."""
    out = np.zeros(len(cloud), dtype=FAKE_DTYPE)
    for f in ("x", "y", "z", "intensity"):
        out[f] = cloud[f]
    return out
