"""Probe of the organisers' insertion style (Y1, ``work/specs/11_fake_style.md``).

    env -u PYTHONPATH python tools/fake_style_probe.py scan  --out DIR      # per-frame stats + appended points
    env -u PYTHONPATH python tools/fake_style_probe.py align --frames 49,200 # deleted-point counts (order-preserving alignment)

Helpers used by ``tools/label_org.py``: ``to_slot`` (beam-grid slot of every point; the base of
``cloud_with_fake_obj`` / ``org_synth`` has an azimuth phase of +0.05 deg relative to ``data/hesai128_2400.npz``),
``split`` (base part / appended tail block), ``align`` (slot of every kept base point; counts of deleted points).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from synth.realbag import RealBag  # noqa: E402
from metro_obstacle_detector.beam_model import BeamModel  # noqa: E402

FAKE_BAG = "/home/gag/i.mos/dataset/extra/cloud_with_fake_obj"
BM = BeamModel.load(os.path.join(os.path.dirname(HERE), "data", "hesai128_2400.npz"))
EL = BM.elevation_deg
_ORDER = np.argsort(EL)
_ELS = EL[_ORDER]
PHASE_DEG = 0.05                       # session phase of cloud_with_fake_obj / org_synth (six bags: 0, new_data: 0.01)
SAZ = BM.slot_az_deg().reshape(BM.n_cols, BM.n_rings) + PHASE_DEG   # [col, ring]
FAZ = SAZ[::2]
NS = 307200
OFFGRID_DEG = 1e-3


def angles(c):
    x = c["x"].astype(np.float64); y = c["y"].astype(np.float64); z = c["z"].astype(np.float64)
    r = np.sqrt(x * x + y * y + z * z)
    az = np.degrees(np.arctan2(x, -y))
    el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-12), -1, 1)))
    return r, az, el


def to_slot(c):
    """r, ring, firing, elevation residual, azimuth residual (deg) of every point (ring/firing -1 for zeros)."""
    r, az, el = angles(c)
    valid = r > 1e-6
    k = np.clip(np.searchsorted(_ELS, el), 1, len(_ELS) - 1)
    kk = np.where(np.abs(el - _ELS[k - 1]) < np.abs(el - _ELS[k]), k - 1, k)
    ring = _ORDER[kk]
    f = np.full(len(r), -1, np.int64)
    az_res = np.full(len(r), np.nan)
    for rg in range(128):
        m = valid & (ring == rg)
        if not m.any():
            continue
        a = -FAZ[:, rg]
        j = np.clip(np.searchsorted(a, -az[m]), 1, len(a) - 1)
        jj = np.where(np.abs(-az[m] - a[j - 1]) < np.abs(-az[m] - a[j]), j - 1, j)
        f[m] = jj
        az_res[m] = az[m] - FAZ[jj, rg]
    ring = np.where(valid, ring, -1)
    return r, ring, f, np.where(valid, el - EL[np.maximum(ring, 0)], np.nan), az_res


def split(c):
    """(r, ring, firing, valid, offgrid, n_base): the appended (off-grid) points form one tail block."""
    r, ring, f, er, ar = to_slot(c)
    v = r > 1e-6
    off = v & ((np.abs(er) > OFFGRID_DEG) | (np.abs(ar) > OFFGRID_DEG))
    idx = np.nonzero(off)[0]
    n_app = 0
    if len(idx):
        if not (idx.max() == len(c) - 1 and idx.max() - idx.min() + 1 == len(idx)):
            raise ValueError("off-grid points do not form one tail block")
        n_app = len(idx)
    return r, ring, f, v, off, len(c) - n_app


def align(c):
    """Original slot of every kept base point (order preserved, deletions only, zeros kept); returns
    (slot, kept-mask over the 307 200 slots, n_appended, stats).  Counts are exact; which slot inside a gap of
    zeros was deleted is not identifiable (self-test: ~40 % of slots exact)."""
    r, ring, f, v, off, m = split(c)
    slot = np.full(len(c), -1, np.int64)
    kv = np.nonzero(v[:m])[0]
    c0 = (2 * f[kv]) * 128 + ring[kv]
    ds = np.empty(len(kv), np.int64)
    dprev, bad = 0, 0
    for t, k in enumerate(kv):
        d0 = c0[t] - k
        d = d0 if d0 >= dprev else (d0 + 128 if d0 + 128 >= dprev else dprev)
        bad += int(d0 < dprev and d0 + 128 < dprev)
        ds[t] = d
        dprev = d
    slot[kv] = kv + ds
    kz = np.nonzero(~v[:m])[0]
    pos = np.searchsorted(kv, kz) - 1
    slot[kz] = kz + (np.where(pos >= 0, ds[np.clip(pos, 0, None)], 0) if len(ds) else 0)
    kept = np.zeros(NS, bool)
    s = slot[:m]
    ok = (s >= 0) & (s < NS)
    kept[s[ok]] = True
    return slot, kept, len(c) - m, {"order_violations": bad, "deleted": int(NS - kept.sum()) if len(kv) else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "align"])
    ap.add_argument("--bag", default=FAKE_BAG)
    ap.add_argument("--out")
    ap.add_argument("--frames")
    a = ap.parse_args(argv)
    b = RealBag(a.bag)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else range(len(b))
    rows, pts = [], []
    for i in frames:
        c = b.frame(i).cloud
        if a.cmd == "scan":
            r, ring, f, v, off, m = split(c)
            rows.append({"frame": i, "n": len(c), "valid": int(v.sum()), "zero": int((~v).sum()), "appended": len(c) - m,
                         "appended_intensity_1": int((c["intensity"][m:] == 1.0).sum())})
            if len(c) > m:
                pts.append(np.stack([np.full(len(c) - m, i), np.arange(m, len(c)), c["x"][m:], c["y"][m:], c["z"][m:]], 1))
        else:
            slot, kept, n_app, st = align(c)
            rows.append(dict(st, frame=i, n=len(c), appended=n_app))
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(rows, open(os.path.join(a.out, f"{a.cmd}.json"), "w"))
        if pts:
            np.save(os.path.join(a.out, "appended_points.npy"), np.concatenate(pts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
