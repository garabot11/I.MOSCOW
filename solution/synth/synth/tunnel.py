"""Parametric metro tunnels as triangle meshes (world frame: x forward at s = 0, y left, z up; rail level z = 0).

Centre-line: a straight lead-in of ``straight_m`` followed by an arc of radius ``radius_m`` (signed, + = curve to
the left) up to the total length; ``radius_m = None`` gives a straight tunnel.  Cross-sections (u lateral, + left;
v height above rail level), measured on the development bags (docs/data_notes.md):
* ``circular``: radius 2.7 m, ceiling 4.6 m above rail (3.3 m above a 1.3 m sensor), segment ribs every 1.2 m;
* ``rectangular``: 4.6 m wide, ceiling 4.3 m above rail;
* ``double``: wall 2.4 m from the track centre, column row at 2.1 m on the other side (0.4 m columns every 5 m),
  a second track 4.2 m away, far wall at 6.6 m, ceiling 4.6 m;
* optional platform (edge 1.45 m from the centre, 1.1 m above rail) over ``platform`` = [s0, s1].
Common equipment: rails (1520 mm gauge, head at v = 0), track bed at v = -0.05 (bed returns at ~rail-head level, as in the real bags) with a drainage trough
(0.9 m wide, v = -0.30), contact rail with its cover (1.2-1.6 m from the centre, 0.60-0.72 m above rail) on
``contact_side``, cable trays on the walls, lamps every ``lamp_spacing_m``.  A wall closes the tunnel at its end.
Every face carries a reflectivity and a surface class (index into ``SURFACES``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BED_V = -0.05     # track-bed surface relative to rail head (real detector rail level = median bed return ~ rail head)
SURFACES = ["wall", "floor", "rail", "ceiling", "equipment", "lamp", "platform", "column", "end_wall"]
RHO = {"wall": 0.12, "floor": 0.11, "rail": 0.25, "ceiling": 0.12, "equipment": 0.15, "lamp": 0.6, "platform": 0.15,
       "column": 0.15, "end_wall": 0.12}


@dataclass
class TunnelSpec:
    kind: str = "circular"                 # circular | rectangular | double
    length_m: float = 400.0
    straight_m: float = 400.0              # straight lead-in before the arc
    radius_m: Optional[float] = None       # signed curve radius (+ left); None = straight
    grade: float = 0.0                     # rail level slope dz/ds
    contact_side: int = -1                 # -1 right, +1 left
    lamp_spacing_m: float = 15.0
    rib_spacing_m: float = 1.2             # circular segment joints
    platform: Optional[Tuple[float, float]] = None
    platform_side: int = 1
    rho_scale: float = 1.0
    behind_m: float = 20.0                 # tunnel modelled behind the sensor start
    double_side: int = 1                   # double track: +1 second track / columns on the left, -1 on the right
    double_wall_m: float = 2.4             # double track: wall distance on the near side
    double_column_m: float = 2.1           # double track: column row distance
    seed: int = 0

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


class CentreLine:
    """Arc-length parametrised centre-line (straight + arc)."""

    def __init__(self, spec: TunnelSpec, ds: float = 0.5):
        self.spec = spec
        s = np.arange(-spec.behind_m, spec.length_m + ds, ds)
        self.s = s
        psi = np.zeros_like(s)
        if spec.radius_m:
            over = np.maximum(s - spec.straight_m, 0.0)
            psi = over / spec.radius_m
        # integrate the heading
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        c, sn = np.cos(psi), np.sin(psi)
        x[1:] = np.cumsum(0.5 * (c[1:] + c[:-1]) * np.diff(s))
        y[1:] = np.cumsum(0.5 * (sn[1:] + sn[:-1]) * np.diff(s))
        i0 = int(np.argmin(np.abs(s)))
        x -= x[i0]
        y -= y[i0]
        self.x, self.y, self.psi = x, y, psi
        self.z = spec.grade * s

    def at(self, s):
        s = np.asarray(s, dtype=np.float64)
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        z = np.interp(s, self.s, self.z)
        psi = np.interp(s, self.s, self.psi)
        return np.stack([x, y, z], -1), psi

    def frame(self, s):
        """(point, T, N, Z) on the centre-line: tangent, left normal, up."""
        p, psi = self.at(s)
        T = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], -1)
        N = np.stack([-np.sin(psi), np.cos(psi), np.zeros_like(psi)], -1)
        Z = np.zeros_like(T)
        Z[..., 2] = 1.0
        return p, T, N, Z

    def to_world(self, s, u, v):
        p, T, N, Z = self.frame(s)
        return p + np.asarray(u)[..., None] * N + np.asarray(v)[..., None] * Z


class MeshBuilder:
    def __init__(self):
        self.V: List[np.ndarray] = []
        self.F: List[np.ndarray] = []
        self.mat: List[np.ndarray] = []
        self.n = 0

    def add(self, V, F, surface: str, rho: float):
        self.V.append(V)
        self.F.append(F + self.n)
        self.mat.append(np.full((len(F), 2), [SURFACES.index(surface), 0], dtype=np.float64))
        self.mat[-1][:, 1] = rho
        self.n += len(V)

    def extrude(self, cl: CentreLine, s_samples, profile_uv, surface, rho, closed=False):
        """Sweep an open/closed 2D polyline (u, v) along the centre-line samples."""
        P = np.asarray(profile_uv, dtype=np.float64)
        m = len(P)
        S = np.asarray(s_samples)
        n = len(S)
        V = cl.to_world(np.repeat(S, m), np.tile(P[:, 0], n), np.tile(P[:, 1], n))
        segs = m if closed else m - 1
        i = np.arange(n - 1)[:, None]
        j = np.arange(segs)[None, :]
        a = i * m + j
        b = i * m + (j + 1) % m
        c = (i + 1) * m + j
        d = (i + 1) * m + (j + 1) % m
        F = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
        self.add(V, F.astype(np.int32), surface, rho)

    def box_at(self, cl: CentreLine, s, u0, u1, v0, v1, length, surface, rho):
        prof = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        S = np.array([s - length / 2, s + length / 2])
        self.extrude(cl, S, prof, surface, rho, closed=True)
        # caps
        for ss in S:
            V = cl.to_world(np.full(4, ss), np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]))
            self.add(V, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), surface, rho)

    def build(self):
        V = np.vstack(self.V)
        F = np.vstack(self.F).astype(np.int32)
        M = np.vstack(self.mat)
        return V, F, M[:, 0].astype(np.int32), M[:, 1]


def build_tunnel(spec: TunnelSpec):
    """Returns (centre_line, V, F, face_surface_index, face_rho, info)."""
    rng = np.random.default_rng(spec.seed)
    cl = CentreLine(spec)
    ds = 1.0 if not spec.radius_m else min(1.0, max(0.25, abs(spec.radius_m) / 600.0))
    S = np.arange(-spec.behind_m, spec.length_m + 1e-6, ds)
    mb = MeshBuilder()
    r = lambda k: RHO[k] * spec.rho_scale * float(rng.uniform(0.85, 1.15))  # noqa: E731
    # track bed with drainage trough
    bed = [(-6.0, BED_V), (-0.45, BED_V), (-0.45, -0.30), (0.45, -0.30), (0.45, BED_V), (6.0, BED_V)]
    ds_ = spec.double_side
    if spec.kind == "double":
        bed = [(-spec.double_wall_m, BED_V), (-0.45, BED_V), (-0.45, -0.30), (0.45, -0.30), (0.45, BED_V), (3.75, BED_V),
               (3.75, -0.30), (4.65, -0.30), (4.65, BED_V), (6.6, BED_V)]
        if ds_ < 0:
            bed = [(-u, v) for u, v in bed[::-1]]
    mb.extrude(cl, S, bed, "floor", r("floor"))
    # rails (ours, and the second track for double)
    for uc in ([0.0, 4.2 * ds_] if spec.kind == "double" else [0.0]):
        for side in (-1, 1):
            u = uc + side * 0.795
            mb.extrude(cl, S, [(u - 0.035, BED_V), (u - 0.035, 0.0), (u + 0.035, 0.0), (u + 0.035, BED_V)], "rail", r("rail"))
    # cross-section shell
    if spec.kind == "circular":
        R, vc = 2.7, 1.9
        # arc from the bed on the right (-u) over the crown to the bed on the left
        ang = np.linspace(np.pi + np.arcsin((vc - BED_V) / R), -np.arcsin((vc - BED_V) / R), 56)
        shell = [(R * np.cos(a), vc + R * np.sin(a)) for a in ang]
        mb.extrude(cl, S, shell, "wall", r("wall"))
        half_w = R * np.cos(np.arcsin((vc - BED_V) / R))
        wall_u = (-half_w, half_w)
        ceil_v = vc + R
        # segment ribs (along-track features)
        for s0 in np.arange(-spec.behind_m, spec.length_m, spec.rib_spacing_m):
            rib = [((R - 0.03) * np.cos(a), vc + (R - 0.03) * np.sin(a)) for a in ang[4:-4]]
            mb.extrude(cl, np.array([s0, s0 + 0.08]), rib, "wall", r("wall"))
    elif spec.kind == "rectangular":
        hw, ceil_v = 2.3, 4.3
        mb.extrude(cl, S, [(-hw, BED_V), (-hw, ceil_v), (hw, ceil_v), (hw, BED_V)], "wall", r("wall"))
        wall_u = (-hw, hw)
    else:  # double track: near wall at -double_wall_m, far wall at 6.6, columns at double_column_m (mirrored by double_side)
        ceil_v = 4.6
        w_near, w_far = -spec.double_wall_m * ds_, 6.6 * ds_
        lo_u, hi_u = sorted([w_near, w_far])
        mb.extrude(cl, S, [(lo_u, BED_V), (lo_u, ceil_v), (hi_u, ceil_v), (hi_u, BED_V)], "wall", r("wall"))
        wall_u = (lo_u, hi_u)
        cu = spec.double_column_m * ds_
        for s0 in np.arange(-spec.behind_m + 2.0, spec.length_m, 5.0):
            mb.box_at(cl, s0, cu - 0.2, cu + 0.2, BED_V, ceil_v, 0.4, "column", r("column"))
    # contact rail and its cover on the contact side
    cs = spec.contact_side
    u_lo, u_hi = (1.2, 1.6) if cs > 0 else (-1.6, -1.2)
    mb.extrude(cl, S, [(u_lo, 0.60), (u_lo, 0.72), (u_hi, 0.72), (u_hi, 0.60)], "equipment", r("equipment"))
    cu = 1.37 * cs
    mb.extrude(cl, S, [(cu - 0.05, 0.30), (cu - 0.05, 0.42), (cu + 0.05, 0.42), (cu + 0.05, 0.30)], "equipment", r("equipment"))
    for s0 in np.arange(-spec.behind_m, spec.length_m, 5.0):
        mb.box_at(cl, s0, cs * 1.47 - 0.04, cs * 1.47 + 0.04, BED_V, 0.60, 0.1, "equipment", r("equipment"))
    # cable trays on both walls
    for side, wu in ((-1, wall_u[0]), (1, wall_u[1])):
        for v in (1.2, 1.5, 1.8):
            if spec.kind == "circular":
                # the wall at height v
                uw = np.sign(wu) * np.sqrt(max(2.7 ** 2 - (v - 1.9) ** 2, 0.0))
            else:
                uw = wu
            inner = uw - np.sign(uw) * 0.22
            a, b = sorted([inner, uw - np.sign(uw) * 0.02])
            mb.extrude(cl, S, [(a, v), (a, v + 0.06), (b, v + 0.06), (b, v)], "equipment", r("equipment") * 0.5)
    # lamps
    for s0 in np.arange(-spec.behind_m + 3.0, spec.length_m, spec.lamp_spacing_m):
        uw = wall_u[1] if spec.kind != "circular" else np.sqrt(2.7 ** 2 - (2.6 - 1.9) ** 2)
        u0 = uw - 0.15 if uw > 0 else uw
        mb.box_at(cl, s0, u0 - 0.0, u0 + 0.12, 2.5, 2.65, 0.4, "lamp", r("lamp"))
    # platform
    if spec.platform:
        s0, s1 = spec.platform
        sp = np.arange(s0, s1 + 1e-6, ds)
        side = spec.platform_side
        e = 1.45 * side
        far = 6.0 * side
        prof = [(e, BED_V), (e, 1.1), (far, 1.1)] if side > 0 else [(far, 1.1), (e, 1.1), (e, BED_V)]
        mb.extrude(cl, sp, prof, "platform", r("platform"))
    # end wall
    se = spec.length_m
    p, T, N, Z = cl.frame(np.array([se]))
    u = np.array([-8.0, 8.0, 8.0, -8.0])
    v = np.array([-0.5, -0.5, 6.0, 6.0])
    V = cl.to_world(np.full(4, se), u, v)
    mb.add(V, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), "end_wall", r("end_wall"))
    V, F, surf, rho = mb.build()
    info = {"wall_u": list(wall_u), "ceiling_above_rail_m": ceil_v, "faces": int(len(F))}
    return cl, V, F, surf, rho, info
