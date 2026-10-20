"""Parametric scenes (``source: parametric``): a tunnel mesh, a moving train-mounted sensor, objects and dust.

Sensor rigs
* ``hesai128``: beam model of the dataset's sensor (307 200-point layout by default), mounted ``mount_height_m``
  (1.3 m) above rail level on the centre-line; every fired beam is cast from the sensor position at its own
  firing time (motion during the sweep); output identical to the real ``/lidar_points`` layout.
* ``falcon_prime``: the slot table measured on the real recording, with a fresh per-frame per-line azimuth
  shift (sigma measured 0.06 deg); mounted 2.0 m above rail, pitched up 2.3 deg (as recorded); only returned
  points are written (no zero points), second returns flagged ``is_2nd_return``; header stamp = end of scan.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

from .insert import GAUGE, in_clearance
from .objects import make_object, rot_z
from .render import Echoes, SceneObject, edge_mixed_pixels, hesai_points, overlay_objects
from .returns import ReturnModel
from .tunnel import BED_V, SURFACES, TunnelSpec, build_tunnel

FALCON_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "falcon_prime_seyond.npz")


def _rot_pitch(theta):
    """Rotation of (d, l, h) vectors by a nose-up pitch theta (about the l axis)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


class HesaiRig:
    sensor = "hesai128"

    def __init__(self, cfg):
        from metro_obstacle_detector.beam_model import hesai128

        self.bm = hesai128(int(cfg.get("layout", 2400)))
        self.height = float(cfg.get("mount_height_m", 1.3))
        self.lateral = float(cfg.get("mount_lateral_m", 0.0))
        self.pitch = np.radians(float(cfg.get("pitch_deg", 0.0)))
        self.frame_id = cfg.get("frame_id", "hesai_lidar")
        self.topic = "/lidar_points" if self.bm.n_cols == 2400 else "/sensing/lidar/hesai128/pointcloud"
        self.ret = ReturnModel.hesai(**cfg.get("return_model", {}))
        self.div = float(cfg.get("divergence_deg", 0.0))
        nf, nr = self.bm.n_firings, self.bm.n_rings
        fired = self.bm.fired_mask().reshape(nf, 2, nr)[:, 1].reshape(-1)
        self.beams = np.nonzero(fired)[0]
        d_s = self.bm.directions().astype(np.float64).reshape(nf, 2, nr, 3)[:, 1].reshape(-1, 3)
        self.dirs_s = d_s[self.beams]
        self.dirs_dlh = self.dirs_s @ self.bm.axis_matrix().T @ _rot_pitch(self.pitch).T
        t_col = self.bm.col_time_s.reshape(nf, 2)[:, 1]
        self.t_beam = np.repeat(t_col, nr)[self.beams]
        self.n_beams_all = nf * nr
        self.header_offset_s = 0.0  # header stamp = first firing
        inv = np.full(nf * nr, -1, dtype=np.int64)
        inv[self.beams] = np.arange(len(self.beams))
        f, k = self.beams // nr, self.beams % nr
        step = np.where(self.bm.ring_fire_parity[k] < 0, 1, 2)
        nb = []
        for sgn in (-1, 1):
            ff = f + sgn * step
            ok = (ff >= 0) & (ff < nf)
            nb.append(np.where(ok, inv[np.clip(ff, 0, nf - 1) * nr + k], -1))
        self.neighbours = np.stack(nb, 1)
        self.p_mix = float(cfg.get("edge_mix_p", 0.35))
        self.coordinate_mode = "hesai (forward -y, up +z)"

    def new_frame(self, rng):
        pass

    def message(self, echoes_fired: Echoes, stamp_ns):
        e = Echoes.empty(self.n_beams_all)
        e.r[self.beams], e.inten[self.beams], e.src[self.beams] = echoes_fired.r, echoes_fired.inten, echoes_fired.src
        pts, src = hesai_points(e, self.bm, stamp_ns)
        return pts, src

    def sensor_from_dlh(self, v_dlh):
        """Analysis-frame (track-aligned d,l,h incl. pitch) vectors -> sensor-frame vectors."""
        return v_dlh @ _rot_pitch(self.pitch) @ self.bm.axis_matrix()

    @staticmethod
    def _out_xyz(xyz):
        return xyz   # the Hesai message uses the sensor frame as is


class FalconRig:
    sensor = "falcon_prime"

    def __init__(self, cfg):
        from metro_obstacle_detector.beam_model import FalconBeamModel

        self.tab = FalconBeamModel.load(cfg.get("falcon_table", FALCON_TABLE))
        self.height = float(cfg.get("mount_height_m", 1.98))
        self.lateral = float(cfg.get("mount_lateral_m", 0.0))
        self.pitch = np.radians(float(cfg.get("pitch_deg", 2.33)))
        self.frame_id = cfg.get("frame_id", "seyond")
        self.topic = cfg.get("topic", "/iv_points")
        self.mode = cfg.get("coordinate_mode", "as_recorded")   # as_recorded (= ros) | upside_down | native
        self.ret = ReturnModel.falcon(**cfg.get("return_model", {}))
        self.div = float(cfg.get("divergence_deg", 0.1))
        self.shift_sd = float(cfg.get("line_shift_sd_deg", 0.06))
        self.beams = np.arange(self.tab.n_slots)
        self.t_beam = self.tab.time_s.copy()
        order = np.lexsort((self.tab.scan_idx, self.tab.flags, self.tab.scan_id))
        grp = self.tab.scan_id.astype(np.int64)[order] * 64 + self.tab.flags[order]
        prev = np.full(self.tab.n_slots, -1, dtype=np.int64)
        nxt = np.full(self.tab.n_slots, -1, dtype=np.int64)
        same = grp[1:] == grp[:-1]
        prev[order[1:][same]] = order[:-1][same]
        nxt[order[:-1][same]] = order[1:][same]
        self.neighbours = np.stack([prev, nxt], 1)
        self.p_mix = float(cfg.get("edge_mix_p", 0.30))
        self.keep_no_return = bool(cfg.get("keep_no_return", False))
        self.header_offset_s = 0.0   # header = end of scan; slot times are negative
        self.coordinate_mode = {"as_recorded": "as recorded (x forward, y left, z up)", "ros": "ros (x forward, y left, z up)",
                                "upside_down": "upside down (x forward, y right, z down)",
                                "native": "native (x up, y right, z forward)"}[self.mode]
        self._set_dirs(np.zeros(self.tab.n_slots))

    def _set_dirs(self, shift):
        d_dlh = self.tab.directions_dlh(shift)
        self.dirs_dlh = d_dlh @ _rot_pitch(self.pitch).T
        self.dirs_s = d_dlh @ self.tab.axis_matrix()

    def new_frame(self, rng):
        line = rng.normal(0.0, self.shift_sd, self.tab.n_lines)
        self._set_dirs(line[self.tab.scan_id.astype(np.int64)])

    def _out_xyz(self, xyz):
        if self.mode in ("as_recorded", "ros"):
            return xyz
        if self.mode == "upside_down":
            return xyz * np.array([1.0, -1.0, -1.0])
        return np.stack([xyz[:, 2], -xyz[:, 1], xyz[:, 0]], 1)  # native: x up, y right, z forward

    def message(self, echoes: Echoes, stamp_ns, rng=None):
        from .writer import FALCON_DTYPE

        rng = rng or np.random.default_rng(0)
        r1, r2 = echoes.r[:, 0], echoes.r[:, 1]
        has1 = r1 > 0
        has2 = (r2 > 0) & has1 & (np.abs(r2 - r1) > 0.3)
        # one or two points per slot, the second return placed just before its first (as in the recording);
        # with keep_no_return, slots without a return are written as (0, 0, 0) points (not in the recording)
        idx1 = np.nonzero(has1 | self.keep_no_return)[0]
        two = has2[idx1]
        counts = 1 + two.astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        n = int(counts.sum())
        slot = np.repeat(idx1, counts)
        is2 = np.zeros(n, dtype=bool)
        is2[starts[two]] = True
        r = np.where(is2, r2[slot], r1[slot])
        inten = np.where(is2, echoes.inten[slot, 1], echoes.inten[slot, 0])
        src = np.where(is2, echoes.src[slot, 1], echoes.src[slot, 0])
        xyz = self._out_xyz(self.dirs_s[slot] * r[:, None])
        pts = np.zeros(n, dtype=FALCON_DTYPE)
        pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        pts["timestamp"] = stamp_ns * 1e-9 + self.t_beam[slot]
        pts["intensity"] = inten
        pts["flags"] = self.tab.flags[slot]
        mu = np.interp(r, [0, 30, 100, 500], [13, 13, 7, 4])
        pts["elongation"] = np.clip(np.round(mu + rng.normal(0, 1.5, n)), 1, 15).astype(np.uint8)
        pts["scan_id"] = self.tab.scan_id[slot]
        pts["scan_idx"] = self.tab.scan_idx[slot]
        pts["is_2nd_return"] = is2.astype(np.uint8)
        return pts, src

    def sensor_from_dlh(self, v_dlh):
        return v_dlh @ _rot_pitch(self.pitch) @ self.tab.axis_matrix()


def grade_profile(spec: TunnelSpec, s0: float, mount_height_m: float = 1.3, pitch_rad: float = 0.0) -> dict:
    """Vertical (and horizontal) track profile used for the scene: rail level z(s) = grade * s along the arc length
    s of the centre-line (world frame, z up, s = 0 at the scene origin; the sensor starts at s0).  The sensor's
    analysis frame is NOT pitched with the grade (only by ``pitch_rad``), so in the sensor frame the rail level
    ahead is h(d) ~ -mount_height + (grade - tan(pitch)) * d: a constant grade looks like a sensor pitch error of
    -atan(grade) (real bags show relative slopes up to -2.4 %)."""
    g = float(spec.grade)
    rel = g - float(np.tan(pitch_rad))
    return {"model": "constant_grade", "grade": g, "grade_percent": round(100.0 * g, 4),
            "sensor_frame_rail_level_m": {"polynomial_in_d": [-float(mount_height_m), rel, 0.0],
                                          "slope_percent": round(100.0 * rel, 4),
                                          "note": "analysis frame of the sensor (d ahead, h up), straight track, small angles"},
            "rail_level_z_m": "grade * s", "polynomial_in_s": [0.0, g, 0.0],
            "segments": [{"s0_m": -float(spec.behind_m), "s1_m": float(spec.length_m), "grade_percent": round(100.0 * g, 4)}],
            "sensor_start_s_m": float(s0),
            "rail_level_relative_to_sensor_start": "z(d) - z(0) = grade * d for along-track distance d ahead",
            "horizontal": {"straight_m": float(spec.straight_m), "curve_radius_m": spec.radius_m,
                           "curve_starts_at_d_m": float(spec.straight_m - s0) if spec.radius_m else None}}


def make_rig(cfg):
    return HesaiRig(cfg) if cfg["sensor"] == "hesai128" else FalconRig(cfg)


def sample_tunnel(rng, tcfg: dict, length: float) -> TunnelSpec:
    kind = str(rng.choice(tcfg.get("kinds", ["circular", "rectangular", "double"])))
    radius = None
    straight = length
    if rng.random() < tcfg.get("p_curve", 0.4):
        lo, hi = tcfg.get("radius_m", [300, 2000])
        radius = float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(lo), np.log(hi))))
        straight = float(rng.uniform(*tcfg.get("straight_before_curve_m", [0, 150])))
    platform = None
    if rng.random() < tcfg.get("p_platform", 0.0):
        p0 = float(rng.uniform(40, 200))
        platform = (p0, p0 + float(rng.uniform(80, 160)))
    return TunnelSpec(kind=kind, length_m=length, straight_m=straight, radius_m=radius,
                      grade=float(rng.uniform(-0.02, 0.02)) if tcfg.get("grade", True) else 0.0,
                      contact_side=int(rng.choice([-1, 1])), lamp_spacing_m=float(rng.uniform(8, 25)),
                      platform=platform, platform_side=int(rng.choice([-1, 1])),
                      rho_scale=float(rng.uniform(*tcfg.get("rho_scale", [0.8, 1.25]))), seed=int(rng.integers(1 << 31)),
                      double_side=int(rng.choice([-1, 1])), double_wall_m=float(rng.uniform(*tcfg.get("double_wall_m", [2.4, 3.3]))),
                      double_column_m=float(rng.uniform(*tcfg.get("double_column_m", [1.8, 2.3]))))


class ParametricScene:
    def __init__(self, cfg, rng, caster, n_frames):
        self.cfg = cfg
        self.rig = make_rig(cfg)
        self.caster = caster
        sp = cfg.get("speed_mps", [0.0, 20.0])
        self.speed = 0.0 if rng.random() < cfg.get("p_stationary", 0.2) else float(rng.uniform(*sp))
        self.n = n_frames
        travel = self.speed * n_frames / 10.0
        self.s0 = float(rng.uniform(0.0, 20.0))
        length = max(float(cfg.get("tunnel", {}).get("length_m", 400.0)), self.s0 + travel + 260.0)
        self.spec = sample_tunnel(rng, cfg.get("tunnel", {}), length)
        self.cl, V, F, self.surf, self.rho, self.tinfo = build_tunnel(self.spec)
        self.mesh = caster.mesh(V, F)
        self.objects = []
        self.dust = None
        p_single = float(cfg.get("p_single_return", 0.0))
        self.single_return = bool(cfg.get("single_return", False)) or (p_single > 0 and rng.random() < p_single)

    # ---------------------------------------------------------------- poses
    def pose(self, t):
        """Sensor origin (world) and the d/l/h basis (world) at times t (array)."""
        s = self.s0 + self.speed * np.asarray(t, dtype=np.float64)
        p, T, N, Z = self.cl.frame(s)
        o = p + self.rig.height * Z + self.rig.lateral * N
        return s, o, T, N, Z

    def world_dirs(self, T, N, Z):
        d = self.rig.dirs_dlh
        return d[:, :1] * T + d[:, 1:2] * N + d[:, 2:3] * Z

    def world_to_sensor(self, X, t_header):
        _, o, T, N, Z = self.pose(np.array([t_header]))
        rel = X - o[0]
        dlh = np.stack([rel @ T[0], rel @ N[0], rel @ Z[0]], -1)
        # same output transform as the message points (Falcon coordinate_mode native / upside_down)
        return self.rig._out_xyz(self.rig.sensor_from_dlh(dlh))

    # ---------------------------------------------------------------- objects
    def add_object(self, obj_id, cls, rng, d0, lat, rho, yaw, crossing_speed=0.0, is_obstacle=True, **kw):
        ob = make_object(cls, rng, reflectivity=rho, **kw)
        s_obj = self.s0 + d0
        self.objects.append({"id": obj_id, "obj": ob, "s": s_obj, "lat": lat, "yaw": yaw, "v_lat": crossing_speed,
                             "bottom": BED_V, "is_obstacle": is_obstacle})

    def object_world(self, o, t):
        lat = o["lat"] + o["v_lat"] * t
        p, T, N, Z = self.cl.frame(np.array([o["s"]]))
        psi = np.arctan2(T[0, 1], T[0, 0])
        Vl = o["obj"].V @ rot_z(o["yaw"] + psi).T
        base = p[0] + lat * N[0] + o["bottom"] * Z[0]
        return Vl + base, lat

    # ---------------------------------------------------------------- one frame
    def render(self, j, stamp_ns, rng):
        rig = self.rig
        rig.new_frame(rng)
        t_header = j * 0.1
        t_beams = t_header + rig.t_beam      # Hesai: offsets after the header; Falcon: negative (scan ends at the header)
        s, o, T, N, Z = self.pose(t_beams)
        dirs = self.world_dirs(T, N, Z)
        t, face, normal = self.caster.cast(self.mesh, o, dirs, rig.ret.max_range_m + 10.0)
        hit = face >= 0
        cos_t = np.where(hit, np.abs((normal * dirs).sum(1)), 0.0)
        rho = np.where(hit, self.rho[np.maximum(face, 0)], 0.0)
        det = hit & rig.ret.detect(rng, rho, cos_t, t)
        e = Echoes.empty(len(dirs))
        r_meas = rig.ret.measure_range(rng, np.where(det, t, 0.0))
        inten = rig.ret.intensity(rng, rho, cos_t, t)
        e.single(np.where(det, r_meas, 0.0), np.where(det, inten, 0.0), -1)
        g = np.where(det, rho * cos_t ** rig.ret.detect_cos_exp, 0.0)
        e.strength[:, 0] = e.strength[:, 1] = g
        edge_mixed_pixels(e, rig.neighbours, rng, rig.p_mix)
        # objects (world frame, per-beam origins)
        objs, worlds = [], []
        for ob in self.objects:
            Vw, lat = self.object_world(ob, t_header)
            objs.append(SceneObject(ob["id"], ob["obj"].cls, Vw, ob["obj"].F, ob["obj"].reflectivity, ob["is_obstacle"]))
            worlds.append((Vw, lat))
        info = overlay_objects(e, objs, o, dirs, self.caster, rig.ret, rng, divergence_deg=rig.div,
                               n_sub=int(self.cfg.get("sub_rays", 7)), up=(0.0, 0.0, 1.0)) if objs else {}
        dust_labels = []
        if self.dust is not None:
            dust_labels = self.dust.apply(e, o, dirs, rig.ret, rng, t_header, self)
        if self.single_return:
            e = e.strongest_only()
        if rig.sensor == "falcon_prime":
            pts, src = rig.message(e, stamp_ns, rng)
        else:
            pts, src = rig.message(e, stamp_ns)
        for dl in dust_labels:
            mine = np.nonzero(src == dl["id"])[0]
            dl["n_hits"] = int(len(mine))
            dl["visible"] = bool(len(mine) > 0)
            if len(mine):
                xyz = np.stack([pts["x"][mine], pts["y"][mine], pts["z"][mine]], 1).astype(np.float64)
                dl["distance_m"] = float(np.linalg.norm(xyz, axis=1).min())
        labels = self.labels(objs, worlds, info, pts, src, t_header) + dust_labels
        sight = self.sight_limit(t_header)
        slots = np.nonzero(src > 0)[0]
        return pts, slots, src[slots], labels, sight, {"surface_hits": np.bincount(self.surf[face[det]], minlength=len(SURFACES)).tolist()}

    def labels(self, objs, worlds, info, pts, src, t_header):
        out = []
        s_h, o_h, T_h, N_h, Z_h = self.pose(np.array([t_header]))
        for so, (Vw, lat), ob in zip(objs, worlds, self.objects):
            mine = np.nonzero(src == so.id)[0]
            n_hits = len(mine)
            if n_hits:
                xyz = np.stack([pts["x"][mine], pts["y"][mine], pts["z"][mine]], 1).astype(np.float64)
                dist = float(np.linalg.norm(xyz, axis=1).min())
            else:
                g = info.get(so.id, {}).get("geom_min_range", np.inf)
                dist = float(g) if np.isfinite(g) else float(np.linalg.norm(Vw - o_h[0], axis=1).min())
            # track-aligned extents at the object
            p, T, N, Z = self.cl.frame(np.array([ob["s"]]))
            rel = Vw - p[0]
            loc = np.stack([rel @ T[0], rel @ N[0], rel @ Z[0]], 1)
            lo, hi = loc.min(0), loc.max(0)
            c_w = p[0] + 0.5 * ((lo[0] + hi[0]) * T[0] + (lo[1] + hi[1]) * N[0] + (lo[2] + hi[2]) * Z[0])
            c_s = self.world_to_sensor(c_w[None], t_header)[0]
            inc = in_clearance(lo[1], hi[1], lo[2], hi[2])
            out.append({"id": so.id, "class": so.cls, "is_obstacle": bool(inc and so.is_obstacle), "center_sensor": c_s.tolist(),
                        "extent_dlh": (hi - lo).tolist(), "d_along_m": float(ob["s"] - s_h[0]),
                        "lateral_m": float(0.5 * (lo[1] + hi[1])), "height_above_rail_m": float(0.5 * (lo[2] + hi[2])),
                        "distance_m": dist, "n_hits": int(n_hits), "visible": bool(n_hits > 0), "in_clearance": bool(inc),
                        "reflectivity": float(so.reflectivity), "bottom_above_rail_m": float(lo[2]),
                        "top_above_rail_m": float(hi[2]), "crossing_speed_mps": float(ob["v_lat"]),
                        "occluded_by_objects_frac": info.get(so.id, {}).get("occluded_by_objects_frac", 0.0)})
        return out

    def sight_limit(self, t_header, height=1.0, step=2.0):
        """Farthest along-track distance at which the track (centre-line, 1 m above rail) is in line of sight,
        within the sensor's range and horizontal field of view."""
        s_h, o_h, T_h, N_h, Z_h = self.pose(np.array([t_header]))
        d = np.arange(step, self.rig.ret.max_range_m + step, step)
        tgt = self.cl.to_world(s_h[0] + d, np.zeros_like(d), np.full_like(d, height))
        v = tgt - o_h[0]
        dist = np.linalg.norm(v, axis=1)
        u = v / dist[:, None]
        t, face, _ = self.caster.cast(self.mesh, np.broadcast_to(o_h[0], u.shape), u, float(dist.max()) + 1)
        az = np.degrees(np.arctan2(u @ N_h[0], u @ T_h[0]))
        half_fov = 50.0 if self.rig.sensor == "hesai128" else 60.0
        ok = ((face < 0) | (t >= dist - 0.05)) & (np.abs(az) <= half_fov) & (dist <= self.rig.ret.max_range_m)
        if not ok[0]:
            return 0.0
        bad = np.nonzero(~ok)[0]
        return float(d[bad[0] - 1]) if len(bad) else float(d[-1])


def plan_parametric_objects(scene: ParametricScene, rng, ocfg: dict, negative: bool):
    if negative:
        return []
    n_lo, n_hi = ocfg.get("n_objects", [1, 3])
    n = int(rng.integers(n_lo, n_hi + 1))
    classes = ocfg.get("classes", ["person_standing", "person_crouching", "person_lying", "box", "debris"])
    w = np.asarray(ocfg.get("class_weights", [1.0] * len(classes)), dtype=np.float64)
    d_lo, d_hi = ocfg.get("d_range_m", [20.0, 300.0])
    # only where the object becomes visible during the scene: line of sight and sensor range at frame 0,
    # plus most of the distance the train travels during the scene
    vis = min(scene.sight_limit(0.0), scene.rig.ret.max_range_m - 5.0)
    travel = scene.speed * scene.n / 10.0
    d_hi = float(max(d_lo + 5.0, min(d_hi, vis + 0.8 * travel)))
    scene.placement_limit_m = d_hi
    rho_ranges = ocfg.get("reflectivity_ranges", [ocfg.get("reflectivity", [0.05, 0.6])])
    used = []
    for k in range(n):
        cls = str(rng.choice(classes, p=w / w.sum()))
        for _ in range(20):
            d0 = float(rng.uniform(d_lo, d_hi))
            if all(abs(d0 - u) > 5 for u in used):
                break
        used.append(d0)
        rr = rho_ranges[int(rng.integers(len(rho_ranges)))]
        rho = float(rng.uniform(*rr))
        crossing = cls == "person_standing" and rng.random() < ocfg.get("p_crossing", 0.2)
        v = float(rng.choice([-1, 1]) * rng.uniform(0.6, 1.5)) if crossing else 0.0
        if crossing:
            lat = float(-np.sign(v) * rng.uniform(1.5, 3.0))
            yaw = float(np.pi / 2 * np.sign(v))
        elif rng.random() < ocfg.get("p_out_of_clearance", 0.15):
            lat = float(rng.choice([-1, 1]) * rng.uniform(1.6, 2.0))
            yaw = float(rng.uniform(-np.pi, np.pi))
        else:
            lat = float(rng.uniform(-0.8, 0.8))
            yaw = float(rng.uniform(-np.pi, np.pi))
        scene.add_object(k + 1, cls, np.random.default_rng(int(rng.integers(1 << 31))), d0, lat, rho, yaw, v)
    return [{"id": o["id"], "class": o["obj"].cls, "d0_m": o["s"] - scene.s0, "lat_m": o["lat"], "yaw": o["yaw"],
             "reflectivity": o["obj"].reflectivity, "crossing_speed_mps": o["v_lat"], "params": o["obj"].params}
            for o in scene.objects]


def run_parametric_scene(cfg, i, out, caster, max_frames=None):
    from .generate import GENERATOR_VERSION, _scene_dir
    from .writer import BagWriter, LabelWriter, serialize_pointcloud2, write_scene_json

    rng = np.random.default_rng([int(cfg["seed"]), i])
    sc = cfg["scenes"]
    n = min(int(sc.get("frames", 120)), 300)
    if max_frames:
        n = min(n, max_frames)
    scene = ParametricScene(cfg, rng, caster, n)
    negative = i < int(sc.get("negatives", 0))
    plan = plan_parametric_objects(scene, rng, cfg.get("objects", {}), negative)
    dust_info = None
    if cfg.get("dust"):
        from .dust import DustField

        scene.dust = DustField.sample(cfg["dust"], rng, scene, id_base=100)
        dust_info = scene.dust.describe()
    tag = ("neg" if negative else "pos") + ("_dust" if scene.dust is not None else "")
    sdir = _scene_dir(out, cfg["set"], i, tag)
    os.makedirs(sdir, exist_ok=True)
    rig = scene.rig
    bw = BagWriter(os.path.join(sdir, "bag"), rig.topic, cfg.get("storage", "mcap"))
    lw = LabelWriter(sdir)
    base_ns = int(cfg.get("base_stamp_s", 1_800_000_000)) * 1_000_000_000 + i * 1000 * 1_000_000_000
    rng_r = np.random.default_rng([int(cfg["seed"]), i, 7])
    t0 = time.perf_counter()
    surf = np.zeros(len(SURFACES), dtype=np.int64)
    for j in range(n):
        stamp = base_ns + j * 100_000_000
        pts, slots, ids, labels, sight, extra = scene.render(j, stamp, rng_r)
        surf += np.asarray(extra["surface_hits"])
        dense = rig.sensor == "falcon_prime" and not getattr(rig, "keep_no_return", False)
        bw.write(serialize_pointcloud2(pts, stamp, rig.frame_id, is_dense=dense), stamp)
        lw.frame(j, stamp, labels, sight, slots, ids)
    bw.close()
    lw.close()
    dt = time.perf_counter() - t0
    info = {"sensor": rig.sensor, "source": "parametric", "real_bag": None, "real_frames": None,
            "tunnel": {"type": scene.spec.kind, "radius_m": 2.7 if scene.spec.kind == "circular" else None,
                       "length_m": scene.spec.length_m, "curve_radius_m": scene.spec.radius_m,
                       "straight_before_curve_m": scene.spec.straight_m, "grade": scene.spec.grade,
                       "platform": scene.spec.platform, "width_m": 4.6 if scene.spec.kind == "rectangular" else None,
                       **scene.tinfo, "spec": scene.spec.to_dict()},
            "speed_mps": scene.speed, "seed": [int(cfg["seed"]), i], "generator_version": GENERATOR_VERSION,
            "coordinate_mode": rig.coordinate_mode, "frame_id": rig.frame_id, "topic": rig.topic,
            "mount": {"height_above_rail_m": rig.height, "lateral_m": rig.lateral, "pitch_deg": float(np.degrees(rig.pitch))},
            "sensor_start_s_m": scene.s0, "objects": plan, "dust": dust_info,
            "grade_profile": grade_profile(scene.spec, scene.s0, rig.height, rig.pitch),
            "object_placement_limit_m": getattr(scene, "placement_limit_m", None),
            "single_return": scene.single_return, "return_model": rig.ret.to_dict(),
            "divergence_deg": rig.div, "clearance_gauge": GAUGE, "frames": n, "render_s": round(dt, 2),
            "surface_returns": dict(zip(SURFACES, surf.tolist()))}
    if rig.sensor == "hesai128":
        info["layout"] = {"n_cols": rig.bm.n_cols, "points": rig.bm.n_slots, "dual_return": True}
    else:
        info["layout"] = {"slots": rig.tab.n_slots, "point_step": 48, "zero_points": bool(rig.keep_no_return)}
    write_scene_json(sdir, info)
    return sdir, n, dt
