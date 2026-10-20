"""Object-free tunnel backgrounds from ``doubleT_obstacle`` in the 307 200-point ``/lidar_points`` layout.

``doubleT_obstacle`` is the only stationary *tunnel* recording (double-track tunnel; the other stationary
segments are stations), but it contains the real object.  In frames 95..200 the object stands at x ~ +0.95 m,
55-57 m ahead.  For such a frame k the object's angular window (its slots found by differencing against
frames 30..45 of the same parity, where the object stands 2.6 m away at x ~ -1.7 m, padded by 0.5 deg) is
replaced by the same slots of a reference frame of the same parity: object and shadow disappear, the rest of
the frame is untouched.  The 921 600-point frame (7200 columns, 360 deg) is then cropped to firings
1200..2399 with the +-50 deg point-azimuth window of the 307 200-point layout (directions within 0.01 deg of
``hesai128(2400)``); the header stamp becomes the time of firing 1200, per-point timestamps are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .realbag import RealBag, RealFrame, xyz_of

SRC_BAG = "doubleT_obstacle"
NAME = "doubleT_obstacle_clean307"
FRAMES = (95, 200)
REFS = {0: list(range(30, 46, 2)), 1: list(range(31, 46, 2))}
FIRING0 = 1200


class CleanTunnelSource:
    def __init__(self):
        from metro_obstacle_detector.beam_model import hesai128

        self.src = RealBag(SRC_BAG)
        self.name = NAME
        self.topic = "/lidar_points"
        self.bm7 = hesai128(7200)
        self.bm2 = hesai128(2400)
        self.fired2 = self.bm2.fired_mask()
        self.az7 = self.bm7.slot_az_deg()
        self.el7 = self.bm7.slot_el_deg()
        self._refs = {}
        self.stats = {}

    def __len__(self):
        return len(self.src)

    def _ref(self, i):
        if i not in self._refs:
            c = self.src.frame(i).cloud
            self._refs[i] = (c, np.linalg.norm(xyz_of(c).astype(np.float64), axis=1))
        return self._refs[i]

    def clean7(self, i):
        """Cleaned 921 600-point cloud of frame i (95 <= i <= 200) and the number of replaced slots."""
        if not FRAMES[0] <= i <= FRAMES[1]:
            raise IndexError(f"{NAME}: frame {i} outside {FRAMES}")
        f = self.src.frame(i)
        c = f.cloud.copy()
        xyz = xyz_of(c).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        refs = [self._ref(j) for j in REFS[i % 2]]
        d, x, z = -xyz[:, 1], xyz[:, 0], xyz[:, 2]
        m = (r > 0) & (d > 53.5) & (d < 59.5) & (x > -0.3) & (x < 2.2) & (z > -3.4) & (z < -1.0)
        for _, rr in refs:
            m &= (rr == 0) | (np.abs(r - rr) > 0.3)
        obj = np.nonzero(m)[0]
        n_rep = 0
        if len(obj) >= 5:
            az0, az1 = np.percentile(self.az7[obj], [1, 99])
            el0, el1 = np.percentile(self.el7[obj], [1, 99])
            win = (self.az7 > az0 - 0.5) & (self.az7 < az1 + 0.5) & (self.el7 > el0 - 0.5) & (self.el7 < el1 + 0.5)
            ref_c = refs[0][0]
            for name in ("x", "y", "z", "intensity"):
                c[name][win] = ref_c[name][win]
            n_rep = int(win.sum())
        self.stats[i] = {"object_slots": int(len(obj)), "replaced_slots": n_rep}
        return f, c

    def frame(self, i) -> RealFrame:
        f, c7 = self.clean7(i)
        nf2 = self.bm2.n_firings
        s0, s1 = FIRING0 * 256, (FIRING0 + nf2) * 256
        c2 = c7[s0:s1].copy()
        for name in ("x", "y", "z", "intensity"):
            c2[name][~self.fired2] = 0.0
        stamp = int(round(c7["timestamp"][s0] * 1e9))
        return RealFrame(i, f.bag_time_ns, stamp, "hesai_lidar", self.topic, c2, f.fields, f.point_step)


def open_source(name: str):
    if name == NAME:
        return CleanTunnelSource()
    return RealBag(name)
