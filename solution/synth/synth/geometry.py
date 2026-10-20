"""Small exact ray-casting helpers in NumPy (boxes, capsules/cylinders, spheres).

All rays start at the sensor origin.  ``dirs`` are unit vectors (N, 3); the functions return the
entry distance ``t`` (inf for a miss).  Oriented boxes are given by centre, 3x3 rotation (columns =
box axes in the sensor frame) and half extents.  These are used for the gate test and as the CPU
fallback of the Warp ray caster; scenes with many triangles use ``raycast.py``.
"""
from __future__ import annotations

import numpy as np


def ray_obb(dirs: np.ndarray, centre, axes, half, origin=None) -> np.ndarray:
    """Entry distance of rays (origin + t*dirs) into an oriented box; inf where missed or t <= 0."""
    dirs = np.asarray(dirs, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    A = np.asarray(axes, dtype=np.float64)          # columns = box axes
    h = np.asarray(half, dtype=np.float64)
    ol = (o - c) @ A                                 # origin in box coordinates
    dl = dirs @ A                                    # (N, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dl
        t1 = (-h - ol) * inv
        t2 = (h - ol) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    # parallel rays: inside slab -> (-inf, inf), outside -> empty
    par = dl == 0
    inside = np.abs(ol) <= h
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t_enter = tmin.max(axis=1)
    t_exit = tmax.min(axis=1)
    hit = (t_enter <= t_exit) & (t_exit > 0)
    t = np.where(t_enter > 0, t_enter, t_exit)
    return np.where(hit, t, np.inf)


def ray_sphere(dirs, centre, radius, origin=None):
    dirs = np.asarray(dirs, dtype=np.float64)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    oc = o - np.asarray(centre, dtype=np.float64)
    b = dirs @ oc
    cc = oc @ oc - radius * radius
    disc = b * b - cc
    with np.errstate(invalid="ignore"):
        s = np.sqrt(disc)
    t0, t1 = -b - s, -b + s
    t = np.where(t0 > 0, t0, t1)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def ray_capsule(dirs, p0, p1, radius, origin=None):
    """Entry distance into a capsule (segment p0-p1 swept by a sphere of ``radius``)."""
    dirs = np.asarray(dirs, dtype=np.float64)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    ba = b - a
    oa = o - a
    baba = ba @ ba
    bard = dirs @ ba
    baoa = ba @ oa
    rdoa = dirs @ oa
    oaoa = oa @ oa
    A = baba - bard * bard
    B = baba * rdoa - baoa * bard
    C = baba * oaoa - baoa * baoa - radius * radius * baba
    h = B * B - A * C
    t = np.full(len(dirs), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(h)
        tc = (-B - sq) / A
        y = baoa + tc * bard
        body = (h >= 0) & (A > 1e-12) & (y > 0) & (y < baba) & (tc > 0)
        t = np.where(body, tc, t)
    for cap in (a, b):
        tc = ray_sphere(dirs, cap, radius, origin)
        t = np.minimum(t, tc)
    return t


def box_axes_from_heading(heading_rad: float, R_dlh: np.ndarray) -> np.ndarray:
    """Columns: box axes (along-track, lateral, up) in the sensor frame for a box yawed by ``heading_rad``
    in the analysis frame (rotation about h)."""
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    a_dlh = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # columns in d,l,h
    return R_dlh.T @ a_dlh
