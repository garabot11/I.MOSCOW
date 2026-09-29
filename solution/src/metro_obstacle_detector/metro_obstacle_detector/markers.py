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


def build_markers(header, res: FrameResult, status: str, distance: Optional[float], cfg: DetectorConfig,
                  R: np.ndarray, lifetime_s: float = 0.6) -> MarkerArray:
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
        d_end = min(cfg.d_max, max(60.0, res.curve.max_d_supported + 40.0))
        dd = np.linspace(cfg.d_min, d_end, 60)
        lc = res.curve(dd)
        hr = res.rail(dd)
        hw = cfg.gauge_halfwidth_body - cfg.lateral_shrink_per_m * dd - cfg.lateral_curv_factor * abs(res.curve.b) * dd * dd
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
    # ---- status text above the sensor
    t = _base(header, "status", mid, Marker.TEXT_VIEW_FACING, lifetime_s)
    mid += 1
    t.pose.position.z = 4.0
    t.scale.z = 1.6
    if status == "detected":
        t.color = _color(1, 0.2, 0.2, 1)
        t.text = f"OBSTACLE  {distance:.1f} m" if distance is not None else "OBSTACLE"
    elif status == "clear":
        t.color = _color(0.3, 1, 0.3, 1)
        t.text = "CLEAR"
    else:
        t.color = _color(1, 0.8, 0.2, 1)
        t.text = f"UNKNOWN ({res.reason})" if res.reason else "UNKNOWN"
    arr.markers.append(t)
    return arr
