import sqlite3, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/src/metro_obstacle_detector')
from metro_obstacle_detector.cloud_io import cloud_to_xyz, parse_pointcloud2_cdr
from metro_obstacle_detector.config import DetectorConfig
from metro_obstacle_detector.detector import ObstacleDetector
cfg = DetectorConfig(); det = ObstacleDetector(cfg)
PI, PS = det._perm_idx, det._perm_sign
def current(xyz):
    x0, y0, z0 = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)
    zero = (x0 == 0) & (y0 == 0) & (z0 == 0)
    valid = finite & ~zero
    counts = (int((~finite).sum()), int(zero.sum()), int(valid.sum()))
    vxyz = xyz[valid]
    d = PS[0] * vxyz[:, PI[0]]; l = PS[1] * vxyz[:, PI[1]]; h = PS[2] * vxyz[:, PI[2]]
    roi = (d >= cfg.d_min) & (d <= cfg.d_max) & (np.abs(l) <= cfg.l_abs_max) & (h >= cfg.h_abs_min) & (h <= cfg.h_abs_max)
    return counts, d[roi], l[roi], h[roi], vxyz[roi]
def fused(xyz):
    cols = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    x0, y0, z0 = cols
    finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)
    zero = (x0 == 0) & (y0 == 0) & (z0 == 0)
    valid = finite & ~zero
    counts = (int((~finite).sum()), int(zero.sum()), int(valid.sum()))
    dA = PS[0] * cols[PI[0]]; lA = PS[1] * cols[PI[1]]; hA = PS[2] * cols[PI[2]]
    with np.errstate(invalid='ignore'):
        roi = valid & (dA >= cfg.d_min) & (dA <= cfg.d_max) & (np.abs(lA) <= cfg.l_abs_max) & (hA >= cfg.h_abs_min) & (hA <= cfg.h_abs_max)
    sel = np.flatnonzero(roi)
    return counts, dA[sel], lA[sel], hA[sel], xyz[sel]
for bag in ['doubleT_obstacle', 'roundT_doubleT']:
    db = next(Path('/data', bag).glob('*.db3'))
    raws = [r[0] for r in sqlite3.connect(db.as_uri() + '?mode=ro', uri=True).execute('SELECT data FROM messages ORDER BY timestamp LIMIT 20')]
    xyzs = [cloud_to_xyz(parse_pointcloud2_cdr(r), intensity=False)[0] for r in raws]
    same = all(a[0] == b[0] and all(np.array_equal(p, q) for p, q in zip(a[1:], b[1:])) for a, b in ((current(x), fused(x)) for x in xyzs))
    def T(f):
        ts = []
        for _ in range(3):
            for x in xyzs:
                t = time.perf_counter(); f(x); ts.append((time.perf_counter() - t) * 1e3)
        return round(float(np.median(ts)), 2)
    print(bag, 'identical', same, 'current ms', T(current), 'fused ms', T(fused))
