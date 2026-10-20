"""RViz markers for the detection result (boxes, labels, corridor, status text)."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .config import DetectorConfig
from .detector import Candidate, FrameResult
from .track_model import curvature_shrink


def _color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _base(header, ns: str, mid: int, mtype: int, lifetime_s: float) -> Marker:
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.lifetime.sec = int(lifetime_s)
    m.lifetime.nanosec = int((lifetime_s - int(lifetime_s)) * 1e9)
    return m


def status_text_position(res: FrameResult, cfg: DetectorConfig, R: np.ndarray) -> np.ndarray:
    """Sensor-frame position of the status text: straight ahead and above the sensor, fixed relative to it
    (a text that follows the fitted track would wobble with the centre-line estimate)."""
    dlh = np.array([cfg.marker_status_ahead_m, 0.0, cfg.marker_status_height_m], dtype=np.float64)
    return np.asarray(R, dtype=np.float64).T @ dlh


class ObstacleGeometryMemory:
    """Keeps the last substantial geometry (extent, point count) of each confirmed track.

    A confirmed track that coasts, or is hit only by a 1-2 pixel D2 group in a frame, is written back by the
    fusion with the geometry of that hit: the status line then read "nearest object: ... 0.0x0.0 m, 1 pts"
    for a frame.  The object did not shrink, so the published size and support are those of the track's
    last substantial observation (within ``max_age`` frames); the range and the decision are untouched."""

    def __init__(self, max_age: int = 30, min_points: int = 3, min_extent_m: float = 0.05):
        self.max_age, self.min_points, self.min_extent_m = max_age, min_points, min_extent_m
        self._mem = {}      # track_id -> (frame, extent, n_points)
        self._frame = 0

    def reset(self) -> None:
        self._mem = {}

    def apply(self, candidates) -> None:
        self._frame += 1
        for c in candidates:
            if not getattr(c, "confirmed", False) or c.track_id is None or c.track_id < 0:
                continue
            ext = np.asarray(c.extent, dtype=np.float64)
            degenerate = c.n_points < self.min_points or float(np.max(ext[1:])) < self.min_extent_m
            if not degenerate:
                self._mem[c.track_id] = (self._frame, ext.copy(), int(c.n_points))
                continue
            m = self._mem.get(c.track_id)
            if m is not None and self._frame - m[0] <= self.max_age:
                c.extent, c.n_points = m[1].copy(), m[2]
        self._mem = {k: v for k, v in self._mem.items() if self._frame - v[0] <= self.max_age}


def build_markers(header, res: FrameResult, status: str, distance: Optional[float], cfg: DetectorConfig,
                  R: np.ndarray, lifetime_s: float = 0.6, mode: str = "") -> MarkerArray:
    """Markers are expressed in the cloud's own frame (header.frame_id); no TF is required."""
    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    mid = 0
    # ---- candidates: cubes + labels
    for cand in res.candidates:
        confirmed = cand.confirmed
        col = _color(1.0, 0.1, 0.1, 0.85) if confirmed else _color(1.0, 0.75, 0.1, 0.6)
        m = _base(header, "obstacles", mid, Marker.CUBE, lifetime_s)
        mid += 1
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in cand.center_sensor]
        size_dlh = np.maximum(cand.extent, 0.4)
        size_xyz = np.abs(R.T @ size_dlh)  # extents along sensor axes (axis-aligned mapping)
        m.scale.x, m.scale.y, m.scale.z = [float(max(v, 0.3)) for v in size_xyz]
        m.color = col
        arr.markers.append(m)
        t = _base(header, "labels", mid, Marker.TEXT_VIEW_FACING, lifetime_s)
        mid += 1
        t.pose.position.x, t.pose.position.y = float(cand.center_sensor[0]), float(cand.center_sensor[1])
        t.pose.position.z = float(cand.center_sensor[2]) + float(size_xyz[2]) / 2 + 1.0
        t.scale.z = max(1.0, 0.02 * cand.distance_m)
        t.color = _color(1, 1, 1, 1)
        t.text = f"{'OBSTACLE' if confirmed else 'candidate'} {cand.distance_m:.1f} m"
        arr.markers.append(t)
    # ---- corridor (track centre-line and box edges) in the sensor frame
    if cfg.marker_corridor and res.curve is not None and res.rail is not None and res.rail.ok:
        verified = getattr(res, "verified_to_m", float("nan"))
        d_end = verified if verified == verified else min(cfg.d_max, max(60.0, res.curve.max_d_supported + 40.0))
        d_end = float(min(cfg.d_max, max(cfg.d_min + 5.0, d_end)))
        dd = np.linspace(cfg.d_min, d_end, 60)
        lc = res.curve(dd)
        hr = res.rail(dd)
        hw = cfg.gauge_halfwidth_body - cfg.lateral_shrink_per_m * dd - curvature_shrink(cfg, res.curve, dd)
        open_ = hw >= cfg.lateral_min_halfwidth
        if open_.sum() >= 2:
            dd, lc, hr, hw = dd[open_], lc[open_], hr[open_], hw[open_]
        for name, lat, hh, col, width in (
            ("centre", lc, hr + cfg.box_bottom_above_rail, _color(0.2, 0.9, 0.3, 0.9), 0.12),
            ("left", lc - hw, hr + cfg.box_bottom_above_rail, _color(0.2, 0.6, 1.0, 0.8), 0.08),
            ("right", lc + hw, hr + cfg.box_bottom_above_rail, _color(0.2, 0.6, 1.0, 0.8), 0.08),
            ("left_top", lc - hw, hr + cfg.box_top_above_rail - cfg.top_margin_per_m * dd, _color(0.2, 0.6, 1.0, 0.5), 0.05),
            ("right_top", lc + hw, hr + cfg.box_top_above_rail - cfg.top_margin_per_m * dd, _color(0.2, 0.6, 1.0, 0.5), 0.05),
        ):
            m = _base(header, "corridor_" + name, mid, Marker.LINE_STRIP, lifetime_s)
            mid += 1
            m.scale.x = width
            m.color = col
            pts_dlh = np.column_stack([dd, lat, hh])
            pts_xyz = pts_dlh @ R  # (x,y,z) = R^T (d,l,h) row-wise
            for p in pts_xyz:
                m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            arr.markers.append(m)
        # the clearance volume itself: translucent floor and side walls of the corridor (one triangle list)
        if cfg.marker_corridor_volume and len(dd) >= 2:
            bot = hr + cfg.box_bottom_above_rail
            top = hr + cfg.box_top_above_rail - cfg.top_margin_per_m * dd
            vol = _base(header, "corridor_volume", mid, Marker.TRIANGLE_LIST, lifetime_s)
            mid += 1
            vol.scale.x = vol.scale.y = vol.scale.z = 1.0
            vol.color = _color(0.2, 0.6, 1.0, 0.18)
            quads = (
                (lc - hw, bot, lc + hw, bot),    # floor
                (lc - hw, bot, lc - hw, top),    # left wall
                (lc + hw, bot, lc + hw, top),    # right wall
            )
            for l_a, h_a, l_b, h_b in quads:
                A = np.column_stack([dd, l_a, h_a]) @ R
                B = np.column_stack([dd, l_b, h_b]) @ R
                for i in range(len(dd) - 1):
                    for p in (A[i], B[i], A[i + 1], B[i], B[i + 1], A[i + 1]):
                        vol.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            arr.markers.append(vol)
    # ---- status text: ahead of the sensor above the track (at the sensor it sat on the bottom edge of the
    # default view, which looks 32 m down the tunnel from behind and above the sensor)
    t = _base(header, "status", mid, Marker.TEXT_VIEW_FACING, lifetime_s)
    mid += 1
    pos = status_text_position(res, cfg, R)
    t.pose.position.x, t.pose.position.y, t.pose.position.z = [float(v) for v in pos]
    t.scale.z = 2.4
    verified = getattr(res, "verified_to_m", float("nan"))
    suffix = f"   track checked to {verified:.0f} m" if verified == verified else ""
    if status == "detected":
        t.color = _color(1, 0.2, 0.2, 1)
        t.text = (f"OBSTACLE  {distance:.1f} m" if distance is not None else "OBSTACLE") + suffix
    elif status == "clear":
        t.color = _color(0.3, 1, 0.3, 1)
        t.text = "CLEAR" + suffix
    else:
        t.color = _color(1, 0.8, 0.2, 1)
        t.text = f"UNKNOWN ({res.reason})" if res.reason else "UNKNOWN"
    arr.markers.append(t)
    # ---- compute mode warning (emergency CPU mode): red text under the status
    if mode and mode != "gpu":
        w = _base(header, "mode", mid, Marker.TEXT_VIEW_FACING, lifetime_s)
        mid += 1
        wpos = pos - 2.0 * R[2]           # 2 m below the status text (R[2] = the up axis in sensor coordinates)
        w.pose.position.x, w.pose.position.y, w.pose.position.z = [float(v) for v in wpos]
        w.scale.z = 1.3
        w.color = _color(1, 0.15, 0.15, 1)
        w.text = ("EMERGENCY CPU MODE - GPU unavailable; the GPU version detects farther and faster. Contact IRS-MIREA (Telegram: IRS-MIREA)"
                  if mode == "cpu_emergency" else f"compute mode: {mode}")
        arr.markers.append(w)
    return arr
