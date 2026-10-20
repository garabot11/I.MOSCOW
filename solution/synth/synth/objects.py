"""Procedural obstacle objects as triangle meshes (object frame: x along the track, y lateral, z up;
origin on the ground at the footprint centre).

Classes (interface B): person_standing (1.5-1.9 m), person_crouching (~0.9-1.0 m), person_lying (~0.3 m),
box (0.3-1.0 m), debris (low irregular clusters).  Humans are built from capsules, spheres and boxes;
mesh detail does not matter beyond ~50 m (a 0.1 deg beam is 9 cm wide at 50 m).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

CLASSES = ("person_standing", "person_crouching", "person_lying", "box", "debris")


# --------------------------------------------------------------------------- primitive meshes
def mesh_box(centre, half, R=None):
    c = np.asarray(centre, dtype=np.float64)
    h = np.asarray(half, dtype=np.float64)
    R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64) * h
    V = corners @ R.T + c
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return V, F


def mesh_sphere(centre, r, n_lon=12, n_lat=8):
    c = np.asarray(centre, dtype=np.float64)
    V = [c + [0, 0, r]]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            V.append(c + r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    V.append(c + [0, 0, -r])
    V = np.array(V)
    F = []
    for j in range(n_lon):
        F.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for i in range(n_lat - 2):
        a0 = 1 + i * n_lon
        b0 = a0 + n_lon
        for j in range(n_lon):
            a, b = a0 + j, a0 + (j + 1) % n_lon
            c_, d = b0 + j, b0 + (j + 1) % n_lon
            F += [[a, c_, d], [a, d, b]]
    last = len(V) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        F.append([last, base + (j + 1) % n_lon, base + j])
    return V, np.array(F, dtype=np.int32)


def mesh_capsule(p0, p1, r, n_seg=10, n_cap=3):
    """Capsule = cylinder between p0 and p1 plus hemispherical caps (as full spheres; overlap is harmless)."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    meshes = [mesh_sphere(p0, r, n_seg, n_cap * 2), mesh_sphere(p1, r, n_seg, n_cap * 2)]
    if L > 1e-6:
        a = axis / L
        t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(a, t)
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        ang = 2 * np.pi * np.arange(n_seg) / n_seg
        ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
        V = np.vstack([p0 + r * ring, p1 + r * ring])
        F = []
        for j in range(n_seg):
            k = (j + 1) % n_seg
            F += [[j, k, n_seg + k], [j, n_seg + k, n_seg + j]]
        meshes.append((V, np.array(F, dtype=np.int32)))
    return merge(meshes)


def merge(meshes):
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs).astype(np.int32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


# --------------------------------------------------------------------------- object models
@dataclass
class ObjectModel:
    cls: str
    V: np.ndarray                 # (n, 3) object frame
    F: np.ndarray                 # (m, 3)
    reflectivity: float
    params: dict = field(default_factory=dict)

    @property
    def extent(self) -> np.ndarray:
        return self.V.max(0) - self.V.min(0)

    def placed(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Vertices after rotation R (object -> target frame) and translation t."""
        return self.V @ R.T + t


def person_standing(rng, height=None, walking=None, girth=None):
    H = float(height if height is not None else rng.uniform(1.5, 1.9))
    s = H / 1.75
    g = float(girth if girth is not None else rng.uniform(1.0, 1.3))  # clothing / build: scales widths and depths
    walk = float(rng.uniform(0, 0.45) if walking is None else walking)  # leg/arm swing amplitude (rad)
    phase = rng.uniform(-1, 1)
    parts = []
    r_head = 0.105 * s
    parts.append(mesh_sphere([0, 0, H - r_head], r_head))
    hip, sh = 0.52 * H, 0.81 * H
    parts.append(mesh_box([0, 0, (hip + sh) / 2], [0.11 * s * g, 0.18 * s * g, (sh - hip) / 2]))
    parts.append(mesh_capsule([0, 0, sh], [0, 0, H - 2 * r_head], 0.05 * s))  # neck
    for side in (-1, 1):
        sw = walk * phase * side
        hp = np.array([0, side * 0.09 * s * g, hip])
        knee = hp + rot_y(sw) @ np.array([0, 0, -(hip - 0.27 * H)])
        ankle = knee + rot_y(sw * 0.6) @ np.array([0, side * 0.01, -(0.27 * H - 0.06 * s)])
        parts.append(mesh_capsule(hp, knee, 0.075 * s * g))
        parts.append(mesh_capsule(knee, ankle, 0.055 * s * g))
        parts.append(mesh_box(ankle + [0.06 * s, 0, -0.03 * s], [0.12 * s, 0.05 * s, 0.035 * s]))
        shp = np.array([0, side * (0.18 * g + 0.03) * s, sh - 0.03 * s])
        elbow = shp + rot_y(-sw) @ np.array([0, side * 0.04 * s * g, -0.29 * s])
        hand = elbow + rot_y(-sw * 1.3) @ np.array([0, side * 0.02 * s, -0.27 * s])
        parts.append(mesh_capsule(shp, elbow, 0.045 * s * g))
        parts.append(mesh_capsule(elbow, hand, 0.04 * s * g))
    V, F = merge(parts)
    V[:, 2] -= V[:, 2].min()
    return V, F, {"height_m": H, "swing_rad": walk, "girth": g}


def person_crouching(rng, height=None):
    target = float(height if height is not None else rng.uniform(0.9, 1.0))
    s = target / 0.95 * rng.uniform(0.95, 1.05)
    parts = []
    hip = np.array([-0.15 * s, 0, 0.45 * s])
    for side in (-1, 1):
        hp = hip + [0, side * 0.1 * s, 0]
        knee = hp + [0.38 * s, side * 0.04 * s, 0.05 * s]
        ankle = np.array([knee[0] - 0.05 * s, knee[1], 0.07 * s])
        parts.append(mesh_capsule(hp, knee, 0.075 * s))
        parts.append(mesh_capsule(knee, ankle, 0.055 * s))
        parts.append(mesh_box(ankle + [0.05 * s, 0, -0.035 * s], [0.12 * s, 0.05 * s, 0.035 * s]))
    tilt = rng.uniform(0.35, 0.7)  # torso pitched forward
    torso_len = 0.42 * s
    top = hip + rot_y(tilt) @ np.array([0, 0, torso_len])
    centre = (hip + top) / 2
    parts.append(mesh_box(centre, [0.11 * s, 0.18 * s, torso_len / 2], rot_y(tilt)))
    r_head = 0.105 * s
    head = top + rot_y(tilt) @ np.array([0, 0, 0.17 * s])
    parts.append(mesh_sphere(head, r_head))
    for side in (-1, 1):
        shp = top + [0, side * 0.2 * s, -0.03 * s]
        hand = np.array([shp[0] + 0.3 * s, side * 0.18 * s, 0.35 * s])
        parts.append(mesh_capsule(shp, hand, 0.045 * s))
    V, F = merge(parts)
    V[:, 2] -= V[:, 2].min()
    V *= target / V[:, 2].max()
    return V, F, {"height_m": float(V[:, 2].max())}


def person_lying(rng, length=None):
    L = float(length if length is not None else rng.uniform(1.5, 1.9))
    s = L / 1.75
    parts = []
    r_head = 0.105 * s
    hip_x, sh_x = 0.48 * L, 0.80 * L
    parts.append(mesh_sphere([L - r_head, 0, r_head], r_head))
    parts.append(mesh_box([(hip_x + sh_x) / 2, 0, 0.11 * s], [(sh_x - hip_x) / 2, 0.18 * s, 0.11 * s]))
    for side in (-1, 1):
        parts.append(mesh_capsule([hip_x, side * 0.09 * s, 0.075 * s], [0.05 * L, side * 0.12 * s, 0.06 * s], 0.07 * s))
        parts.append(mesh_capsule([sh_x, side * 0.22 * s, 0.06 * s], [sh_x - 0.5 * s, side * 0.26 * s, 0.05 * s], 0.045 * s))
    V, F = merge(parts)
    V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
    V[:, 2] -= V[:, 2].min()
    return V, F, {"length_m": float(np.ptp(V[:, 0])), "height_m": float(V[:, 2].max())}


def box_object(rng, size=None):
    if size is None:
        a = rng.uniform(0.3, 1.0)
        size = np.array([a * rng.uniform(0.7, 1.3), a * rng.uniform(0.7, 1.3), a * rng.uniform(0.6, 1.1)])
        size = np.clip(size, 0.25, 1.1)
    size = np.asarray(size, dtype=np.float64)
    V, F = mesh_box([0, 0, size[2] / 2], size / 2)
    return V, F, {"size_m": size.round(3).tolist()}


def debris_object(rng):
    parts = []
    n = int(rng.integers(2, 6))
    for _ in range(n):
        h = np.array([rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25), rng.uniform(0.03, 0.12)])
        c = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), h[2]])
        parts.append(mesh_box(c, h, rot_z(rng.uniform(0, np.pi)) @ rot_x(rng.uniform(-0.3, 0.3))))
    V, F = merge(parts)
    V[:, 2] -= V[:, 2].min()
    return V, F, {"n_parts": n}


def make_object(cls: str, rng, reflectivity=None, **kw) -> ObjectModel:
    if cls == "person_standing":
        V, F, p = person_standing(rng, **kw)
    elif cls == "person_crouching":
        V, F, p = person_crouching(rng, **kw)
    elif cls == "person_lying":
        V, F, p = person_lying(rng, **kw)
    elif cls == "box":
        V, F, p = box_object(rng, **kw)
    elif cls == "debris":
        V, F, p = debris_object(rng)
    else:
        raise ValueError(f"unknown class {cls}")
    rho = float(reflectivity if reflectivity is not None else rng.uniform(0.05, 0.6))
    return ObjectModel(cls=cls, V=V, F=F, reflectivity=rho, params=p)
