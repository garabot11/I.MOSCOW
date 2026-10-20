"""Gate of the organisers'-style renderer against their own data (``cloud_with_fake_obj``).

For frames where one planar object is inserted (the wall, the plate), fit its plane and outline from their appended
points, re-render that quad with ``fake_style.FakeStyleRenderer`` on its OT128 x 0.15 deg grid and compare: number
of points, set of grid cells, max position error, order.  Also checks the deletion rule on an unedited base frame
(their frames 804-1509) with the same quad: deleted = returns behind the quad, zeros kept, order preserved.

    env -u PYTHONPATH python tools/gate_fake_style.py [--frames 10,49,100,150,200,704,720,740] [--out results/gate_fake_style.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fake_style_probe as probe  # noqa: E402
from synth.fake_style import AZ_STEP_DEG, OT128_EL_DEG, FakeObject, FakeStyleRenderer, grid_directions  # noqa: E402
from synth.raycast import Caster  # noqa: E402
from synth.realbag import RealBag  # noqa: E402


def cells(xyz):
    r = np.linalg.norm(xyz, axis=1)
    el = np.degrees(np.arcsin(xyz[:, 2] / r)); az = np.degrees(np.arctan2(xyz[:, 0], -xyz[:, 1]))
    row = np.abs(el[:, None] - OT128_EL_DEG[::-1][None, :]).argmin(1)
    col = np.rint(az / AZ_STEP_DEG).astype(int) % 2400
    return row * 2400 + col


def fit_quad(P, vertical: bool):
    """Plane + outline of planar points P (N, 3).  vertical: plane y = a + b x + c z (wall); else z = a + b x + c y
    (plate top).  The outline is extended by half the grid spacing at that range."""
    r = float(np.median(np.linalg.norm(P, axis=1)))
    if vertical:
        A = np.c_[np.ones(len(P)), P[:, 0], P[:, 2]]
        coef = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
        u_lo, u_hi = P[:, 0].min(), P[:, 0].max()
        v_lo, v_hi = P[:, 2].min(), P[:, 2].max()
        du = 0.5 * r * np.radians(AZ_STEP_DEG)
        el = np.degrees(np.arcsin(P[:, 2] / np.linalg.norm(P, axis=1)))
        dv_lo = 0.5 * r * np.radians(_row_gap(el.min(), -1)); dv_hi = 0.5 * r * np.radians(_row_gap(el.max(), +1))
        U = [u_lo - du, u_hi + du]; Vv = [v_lo - dv_lo, v_hi + dv_hi]
        corners = [(u, v) for u in U for v in Vv]
        V = np.array([[u, coef[0] + coef[1] * u + coef[2] * v, v] for u, v in corners])
    else:
        A = np.c_[np.ones(len(P)), P[:, 0], P[:, 1]]
        coef = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        du = 0.5 * r * np.radians(AZ_STEP_DEG)
        U = [P[:, 0].min() - du, P[:, 0].max() + du]
        # along the track the outline is unknown beyond the hit rows: extend by half the row spacing projected
        el = np.degrees(np.arcsin(P[:, 2] / np.linalg.norm(P, axis=1)))
        h = abs(float(np.median(P[:, 2])))
        y_near = P[:, 1].max(); y_far = P[:, 1].min()
        dn = r * r / max(h, 0.1) * np.radians(0.5 * _row_gap(el.min(), -1))
        df = r * r / max(h, 0.1) * np.radians(0.5 * _row_gap(el.max(), +1))
        Y = [y_far - df, y_near + dn]
        corners = [(u, y) for u in U for y in Y]
        V = np.array([[u, y, coef[0] + coef[1] * u + coef[2] * y] for u, y in corners])
    F = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    return V, F


def _row_gap(el, side):
    t = np.sort(OT128_EL_DEG)
    k = int(np.abs(t - el).argmin())
    j = min(len(t) - 1, max(0, k + side))
    return abs(t[j] - t[k]) if j != k else 0.125


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="10,49,100,150,200,704,720,730,740")
    ap.add_argument("--base-frame", type=int, default=1300)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    bag = RealBag(probe.FAKE_BAG)
    ren = FakeStyleRenderer(Caster("auto"))
    rows = []
    base = bag.frame(a.base_frame).cloud
    for f in [int(x) for x in a.frames.split(",")]:
        c = bag.frame(f).cloud
        r, ring, fi, v, off, m = probe.split(c)
        P = np.stack([c["x"][m:], c["y"][m:], c["z"][m:]], 1).astype(np.float64)
        rr = np.linalg.norm(P, axis=1)
        main_obj = rr < 150 if f <= 231 else (P[:, 2] < -0.8) & (np.abs(P[:, 0]) < 1.5)
        P = P[main_obj]
        vertical = f <= 231
        V, F = fit_quad(P, vertical)
        res = ren.render(c[:0], [FakeObject(1, V, F)])
        Q = np.stack([res.points["x"], res.points["y"], res.points["z"]], 1).astype(np.float64)
        ct, co = cells(P), cells(Q)
        common = np.intersect1d(ct, co)
        err = None
        if len(common):
            it = {k: i for i, k in enumerate(ct)}; io = {k: i for i, k in enumerate(co)}
            err = float(max(np.linalg.norm(P[it[k]] - Q[io[k]]) for k in common))
        order_ok = bool(np.all(np.diff(co) > 0))
        # deletion on an unedited base frame with the same quad
        rb = ren.render(base, [FakeObject(1, V, F)])
        xyz = np.stack([base["x"], base["y"], base["z"]], 1).astype(np.float64)
        rbase = np.linalg.norm(xyz, axis=1)
        zeros_deleted = int((rb.deleted & (rbase == 0)).sum())
        order_kept = bool(np.array_equal(rb.points["x"][:rb.n_base], base["x"][~rb.deleted]))
        rows.append({"frame": f, "object": "wall" if vertical else "plate", "range_m": round(float(np.median(rr[main_obj])), 2),
                     "their_points": int(len(P)), "our_points": int(len(Q)), "common_cells": int(len(common)),
                     "only_theirs": int(len(np.setdiff1d(ct, co))), "only_ours": int(len(np.setdiff1d(co, ct))),
                     "max_pos_err_m": None if err is None else round(err, 4), "our_order_row_major": order_ok,
                     "base_frame": a.base_frame, "base_deleted": int(rb.deleted.sum()), "base_zeros_deleted": zeros_deleted,
                     "base_order_preserved": order_kept, "tail_appended": int(len(rb.points) - rb.n_base)})
        print(json.dumps(rows[-1]), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
