"""Insertion of synthetic objects into real Hesai frames (``source: insert_real``).

* Train motion: only *stationary* segments are used (detected from the data: the fraction of slots valid in two
  consecutive frames whose range changed by < 5 cm must exceed 0.9).  Moving segments are skipped.
* Placement: rail level and track centre-line are the medians of ``ObstacleDetector`` results on frames of the
  segment.  The object bottom is put on the local floor measured in the real frame around the footprint when
  enough returns exist there (``placement: local_floor``), else on the rail model (``rail_model`` /
  ``rail_model_extrapolated`` beyond the farthest observed track-bed bin).
* Sanity number per object: ``occluded_by_scene_frac`` = fraction of the fired beams inside the object's
  geometric footprint whose real near return is nearer than the object (a sunk / hidden object shows a high
  value even when nothing is in front of it).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .objects import make_object, rot_z
from .realbag import RealBag, xyz_of
from .render import Echoes, SceneObject, hesai_beam_dirs, hesai_beam_fired, hesai_echoes_from_cloud, hesai_points, overlay_objects

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# clearance gauge used for the labels (relative to the track centre-line and rail level)
GAUGE = {"halfwidth_low": 1.0, "low_zone_top": 0.8, "halfwidth_body": 1.4, "bottom": 0.10, "top": 3.4}


def stationary_segments(bag: RealBag, thr: float = 0.9, min_len: int = 30) -> List[List[int]]:
    path = os.path.join(CACHE, f"stationary_{bag.name}.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        frac = np.array(data["static_frac"])
    else:
        prev = None
        frac = [0.0]
        for i in range(len(bag)):
            r = np.linalg.norm(xyz_of(bag.frame(i).cloud).astype(np.float64), axis=1)
            if prev is not None:
                both = (r > 0) & (prev > 0)
                frac.append(float(np.mean(np.abs(r[both] - prev[both]) < 0.05)) if both.any() else 0.0)
            prev = r
        frac = np.array(frac)
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"bag": bag.name, "threshold": thr, "static_frac": frac.round(4).tolist()}, fh)
    st = frac > thr
    st[0] = st[1] if len(st) > 1 else False
    segs, i = [], 0
    while i < len(st):
        if st[i]:
            j = i
            while j + 1 < len(st) and st[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                segs.append([i, j])
            i = j + 1
        else:
            i += 1
    return segs


@dataclass
class Track:
    z0: float
    grade: float
    curv: float
    max_d_rail: float
    l0: float
    a: float
    b: float
    max_d_curve: float
    R: np.ndarray            # rows d, l, h (sensor coordinates)

    def rail(self, d):
        D = self.max_d_rail
        dd = np.minimum(d, D)
        base = self.z0 + self.grade * dd + self.curv * dd * dd
        return base + (self.grade + 2 * self.curv * D) * np.maximum(np.asarray(d) - D, 0.0)

    def lc(self, d):
        return self.l0 + self.a * d + self.b * d * d

    def heading(self, d):
        return np.arctan(self.a + 2 * self.b * d)


def track_from_detector(clouds, forward_axis="-y", up_axis="+z") -> Track:
    from metro_obstacle_detector.config import DetectorConfig
    from metro_obstacle_detector.detector import ObstacleDetector

    cfg = DetectorConfig(forward_axis=forward_axis, up_axis=up_axis)
    det = ObstacleDetector(cfg)
    rails, curves = [], []
    for c in clouds:
        res = det.process(xyz_of(c))
        if res.rail is not None and res.rail.ok:
            rails.append((res.rail.z0, res.rail.grade, res.rail.curv, res.rail.max_d_observed))
        if res.curve is not None and res.curve.ok:
            curves.append((res.curve.l0, res.curve.a, res.curve.b, res.curve.max_d_supported))
    if not rails or not curves:
        raise RuntimeError("detector found no rail level / centre-line in the segment")
    r = np.median(np.array(rails), axis=0)
    c = np.median(np.array(curves), axis=0)
    return Track(z0=r[0], grade=r[1], curv=r[2], max_d_rail=r[3], l0=c[0], a=c[1], b=c[2], max_d_curve=c[3],
                 R=cfg.axis_matrix())


def in_clearance(lat_lo, lat_hi, bot, top) -> bool:
    g = GAUGE
    if top < g["bottom"] or bot > g["top"]:
        return False
    low = min(top, g["low_zone_top"]) > max(bot, g["bottom"])
    hi = top > g["low_zone_top"]
    ok_low = low and (lat_lo < g["halfwidth_low"] and lat_hi > -g["halfwidth_low"])
    ok_hi = hi and (lat_lo < g["halfwidth_body"] and lat_hi > -g["halfwidth_body"])
    return bool(ok_low or ok_hi)


def local_floor(xyz_dlh, d, l, rail_h, half_d=1.5, half_l=0.5, min_pts=5):
    dd, ll, hh = xyz_dlh.T
    m = (np.abs(dd - d) < half_d) & (np.abs(ll - l) < half_l) & (hh < rail_h + 0.25) & (hh > rail_h - 0.6)
    if m.sum() >= min_pts:
        return float(np.median(hh[m])), int(m.sum())
    return None, int(m.sum())


def sight_limit(xyz_dlh, track: Track, max_range=209.0):
    d, l, h = xyz_dlh.T
    m = (d > 5) & (np.abs(l - track.lc(d)) < 3.0) & (h - track.rail(d) > 0.2) & (h - track.rail(d) < 5.5)
    if m.sum() < 20:
        return None
    return float(min(max_range, np.percentile(d[m], 99.5)))


def object_label(o: SceneObject, V_dlh: np.ndarray, track: Track, R: np.ndarray, n_hits: int, dist_hit: Optional[float],
                 geom_min: float, extra: dict) -> dict:
    lo, hi = V_dlh.min(0), V_dlh.max(0)
    c_dlh = 0.5 * (lo + hi)
    d_c = float(c_dlh[0])
    lat_c = float(c_dlh[1] - track.lc(d_c))
    rail_c = float(track.rail(d_c))
    bot, top = float(lo[2] - rail_c), float(hi[2] - rail_c)
    lat_lo, lat_hi = float(lo[1] - track.lc(d_c)), float(hi[1] - track.lc(d_c))
    inc = in_clearance(lat_lo, lat_hi, bot, top) if o.cls != "dust" else False
    dist = dist_hit if (n_hits > 0 and dist_hit is not None) else (geom_min if np.isfinite(geom_min)
                                                                  else float(np.linalg.norm(V_dlh, axis=1).min()))
    lab = {"id": int(o.id), "class": o.cls, "is_obstacle": bool(inc and o.is_obstacle), "center_sensor": (c_dlh @ R).tolist(),
           "extent_dlh": (hi - lo).tolist(), "d_along_m": d_c, "lateral_m": lat_c,
           "height_above_rail_m": float(c_dlh[2] - rail_c), "distance_m": float(dist), "n_hits": int(n_hits),
           "visible": bool(n_hits > 0), "in_clearance": bool(inc), "reflectivity": float(o.reflectivity),
           "bottom_above_rail_m": bot, "top_above_rail_m": top}
    lab.update(extra)
    return lab


def plan_objects(rng, spec: dict, track: Track, xyz_dlh0, d_max_vis: float) -> List[dict]:
    """Random object plan for one scene (static or crossing objects)."""
    plans = []
    n = int(rng.integers(spec.get("n_objects", [1, 2])[0], spec.get("n_objects", [1, 2])[1] + 1))
    classes = spec.get("classes", ["person_standing", "person_crouching", "person_lying", "box", "debris"])
    weights = np.asarray(spec.get("class_weights", [1.0] * len(classes)), dtype=np.float64)
    d_lo, d_hi = spec.get("d_range_m", [20.0, 210.0])
    d_hi = min(d_hi, d_max_vis)
    used = []
    for k in range(n):
        cls = str(rng.choice(classes, p=weights / weights.sum()))
        for _ in range(20):
            d = float(rng.uniform(d_lo, max(d_lo + 1, d_hi)))
            if all(abs(d - u) > 4.0 for u in used):
                break
        used.append(d)
        lat_mode = rng.random()
        if lat_mode < spec.get("p_out_of_clearance", 0.15):
            lat = float(rng.choice([-1, 1]) * rng.uniform(1.7, 2.3))
        else:
            lat = float(rng.uniform(-0.8, 0.8))
        crossing = rng.random() < spec.get("p_crossing", 0.2) and cls.startswith("person_standing")
        rho_lo, rho_hi = spec.get("reflectivity", [0.05, 0.6])
        speed = float(rng.choice([-1, 1]) * rng.uniform(0.6, 1.5)) if crossing else 0.0
        if crossing:  # start beside the track and walk across it
            lat = float(-np.sign(speed) * rng.uniform(1.5, 3.0))
        plans.append({"cls": cls, "d": d, "lat": lat, "yaw": float(np.pi / 2 * np.sign(speed)) if crossing else float(rng.uniform(-np.pi, np.pi)),
                      "rho": float(rng.uniform(rho_lo, rho_hi)), "seed": int(rng.integers(1 << 31)),
                      "crossing_speed": speed})
    return plans


class InsertScene:
    def __init__(self, bag_name: str, frames: List[int], plans: List[dict], seed: int, caster, ret, bm,
                 divergence_deg: float = 0.0, n_sub: int = 7, track: Optional[Track] = None):
        from .clean_source import open_source

        self.bag = open_source(bag_name)
        self.frames = frames
        self.plans = plans
        self.seed = seed
        self.caster = caster
        self.ret = ret
        self.bm = bm
        self.div = divergence_deg
        self.n_sub = n_sub
        sample = [self.bag.frame(i).cloud for i in frames[:: max(1, len(frames) // 6)][:6]]
        self.track = track if track is not None else track_from_detector(sample, bm.forward_axis, bm.up_axis)
        R = self.track.R
        xyz0 = xyz_of(sample[0]).astype(np.float64)
        v0 = ~(xyz0 == 0).all(1)
        self.xyz_dlh0 = xyz0[v0] @ R.T
        self.sight = sight_limit(self.xyz_dlh0, self.track)
        self.beam_dirs = hesai_beam_dirs(bm)
        self.beam_fired = hesai_beam_fired(bm)
        self.floor_offset_m, self.floor_offset_samples = self._floor_offset()
        self.models = []
        self.add_plans(plans)

    def _floor_offset(self):
        """Median (local floor - rail model) on the centre-line where the bed is observed; -0.05 m if unknown.
        Used for objects without enough returns around their own footprint."""
        offs = []
        for d in np.arange(8.0, min(60.0, self.track.max_d_rail) + 1e-6, 4.0):
            rail_h = float(self.track.rail(d))
            fl, _ = local_floor(self.xyz_dlh0, d, float(self.track.lc(d)), rail_h)
            if fl is not None and rail_h - 0.45 <= fl <= rail_h + 0.15:
                offs.append(fl - rail_h)
        return (float(np.median(offs)) if len(offs) >= 3 else -0.05), len(offs)

    def add_plans(self, plans):
        for k, p in enumerate(plans):
            ob = make_object(p["cls"], np.random.default_rng(p["seed"]), reflectivity=p["rho"])
            d = p["d"]
            lc = float(self.track.lc(d)) + p["lat"]
            rail_h = float(self.track.rail(d))
            fl, n_fl = local_floor(self.xyz_dlh0, d, lc, rail_h)
            if fl is not None and rail_h - 0.45 <= fl <= rail_h + 0.15:
                bottom, method = fl, "local_floor"
            else:
                bottom = rail_h + self.floor_offset_m
                method = "rail_model_extrapolated" if d > self.track.max_d_rail else "rail_model"
            self.models.append({"obj": ob, "plan": p, "bottom": bottom, "placement": method, "floor_pts": n_fl})

    def object_vertices(self, k: int, t_s: float):
        m = self.models[k]
        p = m["plan"]
        d = p["d"]
        lat = p["lat"] + p["crossing_speed"] * t_s
        yaw = p["yaw"] + float(self.track.heading(d))
        R_obj = rot_z(yaw)
        pos = np.array([d, float(self.track.lc(d)) + lat, m["bottom"]])
        V_dlh = m["obj"].V @ R_obj.T + pos
        return V_dlh, V_dlh @ self.track.R

    def render_frame(self, j: int, rng):
        fi = self.frames[j]
        f = self.bag.frame(fi)
        t_s = (f.stamp_ns - self.bag_t0) * 1e-9 if hasattr(self, "bag_t0") else 0.0
        if j == 0:
            self.bag_t0 = f.stamp_ns
            t_s = 0.0
        echoes = hesai_echoes_from_cloud(f.cloud, self.bm)
        objs, verts = [], []
        for k, m in enumerate(self.models):
            V_dlh, V_s = self.object_vertices(k, t_s)
            objs.append(SceneObject(id=k + 1, cls=m["obj"].cls, V=V_s, F=m["obj"].F, reflectivity=m["obj"].reflectivity))
            verts.append(V_dlh)
        bg_near = echoes.r[:, 0].copy()
        beams = np.nonzero(self.beam_fired)[0]
        info = overlay_objects(echoes, objs, np.zeros(3), self.beam_dirs, self.caster, self.ret, rng,
                               divergence_deg=self.div, n_sub=self.n_sub, up=self.track.R[2], beams=beams)
        pts, src = hesai_points(echoes, self.bm, f.stamp_ns, timestamps=f.cloud["timestamp"])
        slots = np.nonzero(src > 0)[0]
        labels = []
        for k, (o, V_dlh) in enumerate(zip(objs, verts)):
            mine = slots[src[slots] == o.id]
            n_hits = len(mine)
            dist = float(np.linalg.norm(np.stack([pts["x"][mine], pts["y"][mine], pts["z"][mine]], 1), axis=1).min()) if n_hits else None
            occ = self._occlusion(o, bg_near, info[o.id])
            m = self.models[k]
            extra = {"placement": m["placement"], "occluded_by_scene_frac": occ, "geom_beams": info[o.id]["geom_beams"],
                     "occluded_by_objects_frac": info[o.id]["occluded_by_objects_frac"],
                     "crossing_speed_mps": m["plan"]["crossing_speed"]}
            labels.append(object_label(o, V_dlh, self.track, self.track.R, n_hits, dist, info[o.id]["geom_min_range"], extra))
        return f, pts, slots, src[slots], labels

    def first_frame_occlusion(self, k: int) -> Optional[float]:
        f = self.bag.frame(self.frames[0])
        bg_near = hesai_echoes_from_cloud(f.cloud, self.bm).r[:, 0]
        _, V_s = self.object_vertices(k, 0.0)
        m = self.models[k]
        o = SceneObject(id=k + 1, cls=m["obj"].cls, V=V_s, F=m["obj"].F, reflectivity=m["obj"].reflectivity)
        return self._occlusion(o, bg_near, None)

    def _occlusion(self, o: SceneObject, bg_near, inf):
        # beams whose centre ray hits the object geometrically, and whose real near echo is in front of it
        mh = self.caster.mesh(o.V, o.F)
        cand = np.nonzero(self.beam_fired)[0]
        t, face, _ = self.caster.cast(mh, np.zeros(3), self.beam_dirs[cand], 400.0)
        hit = face >= 0
        if not hit.any():
            return None
        b = cand[hit]
        front = (bg_near[b] > 0) & (bg_near[b] < t[hit] - 0.3)
        return round(float(front.mean()), 3)
