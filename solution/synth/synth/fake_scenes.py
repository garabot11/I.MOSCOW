"""Scenes in the organisers' insertion style (``source: fake_style``) on real Hesai base frames.

* Base: consecutive frames of a real recording in the 307 200-point layout (the unedited frames of
  ``cloud_with_fake_obj`` - the organisers' own base -, ``new_data``, the five 307k development bags).
* Objects are static in the world (or walk with their own speed): along-track position ``s_obj``; the train's
  along-track displacement ``s(t)`` comes from ``motion.segment_motion_dp`` (1-DoF, structure alignment; 0 for
  segments declared standing).  At frame t the object stands at ``d = s_obj + v t - s(t)`` on the track model of
  that frame (v1 detector's per-frame rail level and centre-line from the chain JSONL, median-smoothed over 9
  frames) with its bottom ``bottom_m`` above the rail model + the scene's floor offset.
* Rendering and message layout: ``fake_style.FakeStyleRenderer`` (16-byte points, deleted occluded returns,
  appended OT128-grid points with intensity 1).  Labels: interface B; ``slot_labels.slot`` = message point index
  (the layout is not organised); per object also ``n_deleted`` (base returns it deleted).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .fake_style import FAKE_DTYPE, FakeObject, FakeStyleRenderer, to_fake_layout
from .insert import GAUGE, Track, in_clearance, local_floor
from .objects import make_object, merge, mesh_box, rot_z
from .realbag import RealBag

EVAL = "/home/gag/i.mos/work/eval/v1-gpu"
BASES = {
    "cloud_with_fake_obj": ("/home/gag/i.mos/dataset/extra/cloud_with_fake_obj", f"{EVAL}/extra/cloud_with_fake_obj.jsonl"),
    "new_data": ("/home/gag/i.mos/dataset/new_data_raw/new_data", f"{EVAL}/newdata/new_data.jsonl"),
}
for _b in ("doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT", "roundT_squareT_pressureGate_squareT",
           "squareT_platform_squareT_switch"):
    BASES[_b] = (f"/home/gag/i.mos/dataset/for_hackathon/{_b}", f"{EVAL}/real/{_b}.jsonl")
# frames of cloud_with_fake_obj that the organisers did not edit (307 200 points, nothing appended)
FAKE_BAG_CLEAN = [(804, 1246), (1248, 1509)]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fake_style")
GENERATOR_VERSION = "synth-fake-1.0"
STYLES = ("fake_a", "in_place", "physical")
STYLE_LAYOUT = {
    "fake_a": {"point_step": 16, "fields": ["x", "y", "z", "intensity"], "organized": False,
               "note": "base points in original order minus deleted (occluded) returns, object points appended; "
                       "slot_labels.slot = message point index"},
    "in_place": {"organized": True, "note": "base layout and fields kept (307 200 points); object hits replace "
                                             "farther returns / fill empty slots in place; slot = message index"},
    "physical": {"organized": True, "note": "base layout and fields kept (307 200 points); slot = message index"}}
STYLE_RENDERER = {
    "fake_a": {"grid": "OT128 nominal elevations x 0.15 deg azimuth, appended, every hit returns", "max_range_m": 300.0,
               "object_intensity": 1.0, "range_noise": None, "occlude_by_base": False},
    "in_place": {"rays": "real beam directions of the base session (phase-corrected built-in model)", "max_range_m": 400.0,
                 "object_intensity": 1.0, "range_noise": None, "rule": "nearer than both echoes -> both columns"},
    "physical": {"rays": "real beam directions of the base session", "return_model": "ReturnModel.hesai() (P(detect) "
                 "logistic in rho cos/r^2, 2 cm noise + 4 mm quantisation, intensity ~ rho, dual returns, 209 m cap)",
                 "object_reflectivity": "uniform 0.05-0.6 per object unless given"}}


# --------------------------------------------------------------------------- track per frame
def load_tracks(jsonl: str, frames: List[int], R: np.ndarray, win: int = 9) -> List[Track]:
    want = set(frames)
    rows = {}
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            fi = r["frame_index"]
            if fi in want:
                det = r.get("detector", {})
                rail, cur = det.get("rail") or {}, det.get("curve") or {}
                if rail.get("ok") and cur.get("ok"):
                    rows[fi] = [rail["z0"], rail["grade"], rail["curv"], rail.get("max_d_observed", 60.0),
                                cur["l0"], cur["a"], cur["b"], cur.get("max_d_supported", 100.0)]
    if not rows:
        raise RuntimeError(f"no rail/curve model in {jsonl} for frames {frames[0]}..{frames[-1]}")
    ok = sorted(rows)
    P = []
    for fi in frames:
        j = min(ok, key=lambda k: abs(k - fi))
        P.append(rows[j])
    P = np.array(P, dtype=np.float64)
    h = win // 2
    S = np.array([np.median(P[max(0, k - h):k + h + 1], axis=0) for k in range(len(P))])
    return [Track(z0=s[0], grade=s[1], curv=s[2], max_d_rail=s[3], l0=s[4], a=s[5], b=s[6], max_d_curve=s[7], R=R)
            for s in S]


def ego_motion(bag: RealBag, frames: List[int], name: str, standing: bool) -> np.ndarray:
    if standing:
        return np.zeros(len(frames))
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"motion_{name}_{frames[0]}_{frames[-1]}.npy")
    if os.path.exists(path):
        return np.load(path)
    from .motion import compact_frame, segment_motion_dp

    fr = []
    for i in frames:
        c = bag.frame(i).cloud
        x, y, z = c["x"].astype(np.float64), c["y"].astype(np.float64), c["z"].astype(np.float64)
        m = (x != 0) | (y != 0) | (z != 0)
        fr.append(compact_frame(np.stack([-y[m], x[m], z[m]], axis=1), seed=i))
    s = segment_motion_dp(fr)["s"]
    np.save(path, s)
    return s


# --------------------------------------------------------------------------- objects
def object_mesh(cls: str, rng, p: dict):
    """(V, F, params) in the object frame (x along track, y lateral, z up, origin on the bottom centre)."""
    if cls == "wall":
        w, h, t = p.get("size", [rng.uniform(1.5, 3.0), rng.uniform(1.5, 4.0), 0.02])
        V, F = mesh_box([0, 0, h / 2], [t / 2, w / 2, h / 2])
        return V, F, {"size_m": [t, w, h]}
    if cls == "plate":
        w, dep, t = p.get("size", [rng.uniform(0.8, 2.0), rng.uniform(0.3, 1.0), rng.uniform(0.02, 0.3)])
        V, F = mesh_box([0, 0, t / 2], [dep / 2, w / 2, t / 2])
        return V, F, {"size_m": [dep, w, t]}
    if cls in ("small_object", "cube"):
        a = p.get("size", [rng.uniform(0.2, 0.5)] * 3)
        V, F = mesh_box([0, 0, a[2] / 2], [a[0] / 2, a[1] / 2, a[2] / 2])
        return V, F, {"size_m": list(a)}
    if cls == "side_box":
        a = p.get("size", [rng.uniform(1.9, 2.4), rng.uniform(2.0, 2.5), rng.uniform(1.9, 2.1)])
        V, F = mesh_box([0, 0, a[2] / 2], [a[0] / 2, a[1] / 2, a[2] / 2])
        return V, F, {"size_m": list(a)}
    if cls == "overhead_box":
        a = p.get("size", [rng.uniform(1.8, 2.3), rng.uniform(2.0, 2.4), rng.uniform(1.0, 2.0)])
        V, F = mesh_box([0, 0, a[2] / 2], [a[0] / 2, a[1] / 2, a[2] / 2])
        return V, F, {"size_m": list(a)}
    if cls == "pole":
        a = p.get("size", [rng.uniform(0.05, 0.1), rng.uniform(0.05, 0.1), rng.uniform(1.0, 2.5)])
        V, F = mesh_box([0, 0, a[2] / 2], [a[0] / 2, a[1] / 2, a[2] / 2])
        return V, F, {"size_m": list(a)}
    ob = make_object(cls, rng, reflectivity=1.0, **({"size": p["size"]} if cls == "box" and "size" in p else {}))
    return ob.V, ob.F, ob.params


def default_bottom(cls: str, rng) -> float:
    """Bottom above the rail model (m) when the plan does not give one (organisers' placements in brackets)."""
    if cls == "plate":
        return float(rng.uniform(0.2, 0.8))                 # [their plate: top 0.4-0.6, ~0.25 thick]
    if cls == "small_object":
        return float(rng.choice([0.0, rng.uniform(0.3, 1.5)]))   # on the bed or floating [their 0.3 m cubes: 0.9-1.5]
    if cls == "pole":
        return float(rng.uniform(2.6, 3.3))                 # hanging from above [theirs: 3.0]
    if cls == "overhead_box":
        return float(rng.uniform(3.5, 4.0))                 # above the gauge (decoy) [theirs: 3.1-3.3]
    return 0.0


# --------------------------------------------------------------------------- scene
class FakeScene:
    def __init__(self, spec: dict, caster, seed, occlude_by_base=False):
        self.spec = spec
        self.name = spec["base"]
        path, jsonl = BASES[self.name]
        self.bag = RealBag(path)
        a, b = spec["frames"]
        self.frames = list(range(int(a), int(b) + 1))
        from metro_obstacle_detector.beam_model import axis_matrix

        self.R = axis_matrix("-y", "+z")
        self.tracks = load_tracks(jsonl, self.frames, self.R)
        self.s = ego_motion(self.bag, self.frames, self.name, bool(spec.get("standing", False)))
        self.style = spec.get("style", "fake_a")
        self.rng = np.random.default_rng(seed)
        c0 = self.bag.frame(self.frames[0]).cloud
        self.phase_deg = None
        if self.style == "fake_a":
            self.renderer = FakeStyleRenderer(caster, occlude_by_base=occlude_by_base)
        else:
            from .fake_style import InPlaceRenderer, PhysicalRenderer, base_beam_model

            self.bm, self.phase_deg = base_beam_model(c0)
            self.renderer = InPlaceRenderer(caster) if self.style == "in_place" else PhysicalRenderer(caster)
            self.rng_ret = np.random.default_rng([*np.atleast_1d(seed).tolist(), 7])
        x0 = np.stack([c0["x"], c0["y"], c0["z"]], 1).astype(np.float64)
        v0 = ~(x0 == 0).all(1)
        self.xyz_dlh0 = x0[v0] @ self.R.T
        self.floor_offset = self._floor_offset()
        self.objects = []
        for k, p in enumerate(spec.get("objects", [])):
            self.add_object(k + 1, p)

    def _floor_offset(self):
        tr = self.tracks[0]
        offs = []
        for d in np.arange(8.0, min(40.0, tr.max_d_rail) + 1e-6, 4.0):
            rail_h = float(tr.rail(d))
            fl, _ = local_floor(self.xyz_dlh0, d, float(tr.lc(d)), rail_h)
            if fl is not None and rail_h - 0.45 <= fl <= rail_h + 0.15:
                offs.append(fl - rail_h)
        return float(np.clip(np.median(offs), -0.3, 0.1)) if len(offs) >= 3 else -0.05

    def add_object(self, oid: int, p: dict):
        rng = np.random.default_rng([int(p.get("seed", self.rng.integers(1 << 31))), oid])
        cls = p["cls"]
        V, F, params = object_mesh(cls, rng, p)
        bottom = float(p["bottom_m"]) if "bottom_m" in p else default_bottom(cls, rng)
        s0 = float(p["d0"]) if "d0" in p else float(p["d_end"]) + float(self.s[-1]) - float(p.get("v_along", 0.0)) * 0.1 * (len(self.frames) - 1)
        self.objects.append({"id": oid, "cls": cls, "V": V, "F": F, "params": params, "s0": s0,
                             "lat": float(p.get("lat", 0.0)), "bottom": bottom, "yaw": float(p.get("yaw", 0.0)),
                             "v_along": float(p.get("v_along", 0.0)), "v_lat": float(p.get("v_lat", 0.0)),
                             "decoy": bool(p.get("decoy", False)),
                             "rho": float(p.get("rho", rng.uniform(0.05, 0.6)))})

    def placed(self, ob, j):
        """Vertices (dlh and sensor frame) of object ob in scene frame j, or None if it is behind the sensor."""
        t = 0.1 * j
        tr = self.tracks[j]
        d = ob["s0"] + ob["v_along"] * t - float(self.s[j])
        if d < 0.5:
            return None
        lat = ob["lat"] + ob["v_lat"] * t
        yaw = ob["yaw"] + float(tr.heading(d))
        pos = np.array([d, float(tr.lc(d)) + lat, float(tr.rail(d)) + self.floor_offset + ob["bottom"]])
        V_dlh = ob["V"] @ rot_z(yaw).T + pos
        return V_dlh, V_dlh @ self.R

    def render(self, j):
        f = self.bag.frame(self.frames[j])
        objs, placed = [], {}
        for ob in self.objects:
            pl = self.placed(ob, j)
            if pl is None:
                continue
            placed[ob["id"]] = pl
            objs.append(FakeObject(ob["id"], pl[1], ob["F"]))
        base = f.cloud
        if self.style == "fake_a":
            res = self.renderer.render(base, objs)
            points = res.points
            idx_of = {oid: np.arange(a, b) for oid, (a, b) in res.obj_slices.items()}
            n_del = {o.id: int((res.deleted_by == o.id).sum()) for o in objs}
            n_deleted_total = int(res.deleted.sum())
        else:
            if self.style == "in_place":
                points, src = self.renderer.render(base, objs, self.bm)
            else:
                points, src = self.renderer.render(base, objs, self.bm, {ob["id"]: ob["rho"] for ob in self.objects},
                                                   self.rng_ret, f.stamp_ns, self.R[2])
            rb = np.sqrt(base["x"].astype(np.float64) ** 2 + base["y"] ** 2 + base["z"] ** 2)
            rn = np.sqrt(points["x"].astype(np.float64) ** 2 + points["y"] ** 2 + points["z"] ** 2)
            idx_of = {o.id: np.nonzero(src == o.id)[0] for o in objs}
            n_del = {o.id: int(((src == o.id) & (rb > 0)).sum()) for o in objs}
            n_deleted_total = int(((rb > 0) & ((src >= 0) | (rn == 0))).sum())
        labels, slots, ids = [], [], []
        tr = self.tracks[j]
        for ob in self.objects:
            if ob["id"] not in placed:
                continue
            V_dlh, V_s = placed[ob["id"]]
            idx = idx_of.get(ob["id"], np.zeros(0, dtype=np.int64))
            rmin = float(np.sqrt(points["x"][idx].astype(np.float64) ** 2 + points["y"][idx] ** 2
                                 + points["z"][idx] ** 2).min()) if len(idx) else None
            slots.append(idx)
            ids.append(np.full(len(idx), ob["id"]))
            labels.append(self._label(ob, V_dlh, tr, len(idx), rmin, n_del.get(ob["id"], 0)))
        sl = np.concatenate(slots) if slots else np.zeros(0, dtype=np.int64)
        oi = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        return f, points, n_deleted_total, sl, oi, labels

    def _label(self, ob, V_dlh, tr: Track, n_hits, rmin, n_del):
        lo, hi = V_dlh.min(0), V_dlh.max(0)
        c = 0.5 * (lo + hi)
        d_c = float(c[0])
        lc = float(tr.lc(d_c))
        rail_c = float(tr.rail(d_c))
        bot, top = float(lo[2] - rail_c), float(hi[2] - rail_c)
        lat_lo, lat_hi = float(lo[1] - lc), float(hi[1] - lc)
        inc = in_clearance(lat_lo, lat_hi, bot, top)
        geom = float(np.linalg.norm(V_dlh, axis=1).min())
        dist = rmin if (n_hits > 0 and rmin is not None) else geom
        return {"id": ob["id"], "class": ob["cls"], "is_obstacle": bool(inc and not ob["decoy"]),
                "center_sensor": (c @ self.R).tolist(), "extent_dlh": (hi - lo).tolist(), "d_along_m": d_c,
                "lateral_m": float(c[1] - lc), "height_above_rail_m": float(c[2] - rail_c), "distance_m": float(dist),
                "n_hits": int(n_hits), "visible": bool(n_hits > 0), "in_clearance": bool(inc),
                "reflectivity": 1.0 if self.style != "physical" else ob["rho"], "style": self.style,
                "bottom_above_rail_m": bot, "top_above_rail_m": top, "lateral_min_m": lat_lo, "lateral_max_m": lat_hi,
                "n_deleted": int(n_del), "beyond_rail_model": bool(d_c > tr.max_d_rail),
                "beyond_curve_model": bool(d_c > tr.max_d_curve)}


# --------------------------------------------------------------------------- random plans (train sets)
def random_objects(rng, spec: dict, s_total: float) -> List[dict]:
    """Objects for a random scene: classes by weight, initial distance so that the object stays in view."""
    classes = spec.get("classes", ["wall", "plate", "box", "small_object", "person_standing", "person_crouching",
                                   "person_lying", "pole", "side_box", "overhead_box"])
    w = np.asarray(spec.get("class_weights", [1.0] * len(classes)), dtype=np.float64)
    n = int(rng.integers(spec.get("n_objects", [1, 4])[0], spec.get("n_objects", [1, 4])[1] + 1))
    d_lo, d_hi = spec.get("d_range_m", [10.0, 150.0])
    out = []
    for _ in range(n):
        cls = str(rng.choice(classes, p=w / w.sum()))
        d_end = float(rng.uniform(d_lo, d_hi))                   # distance at the end of the scene
        p = {"cls": cls, "d0": d_end + s_total, "seed": int(rng.integers(1 << 31))}
        if cls == "side_box":
            p["lat"] = float(rng.choice([-1, 1]) * rng.uniform(2.0, 3.4))
        elif cls in ("person_standing", "person_crouching") and rng.random() < 0.3:
            p["lat"] = float(rng.choice([-1, 1]) * rng.uniform(1.6, 2.4))     # beside the track (negative)
        else:
            p["lat"] = float(rng.uniform(-0.8, 0.8))
        p["yaw"] = float(rng.uniform(-np.pi, np.pi)) if cls.startswith("person") or cls in ("box", "small_object") else 0.0
        if cls == "person_standing" and rng.random() < 0.3:
            p["v_along"] = float(rng.uniform(-1.5, 1.5))
        out.append(p)
    return out


def run_fake_scene(cfg, i, out, caster, max_frames=None):
    from .writer import BagWriter, LabelWriter, serialize_pointcloud2, write_scene_json

    seed = [int(cfg["seed"]), i]
    rng = np.random.default_rng(seed)
    sc = cfg["scenes"]
    if "list" in sc:
        spec = dict(sc["list"][i])
    else:
        bases = sc["bases"]
        b = bases[int(rng.integers(len(bases)))]
        n = int(sc.get("frames", 150))
        a0, a1 = b["frames"]
        st = int(rng.integers(a0, max(a0 + 1, a1 - n + 2)))
        styles = sc.get("styles", {"fake_a": 1.0})
        names = sorted(styles)
        w = np.array([float(styles[k]) for k in names])
        spec = {"base": b["base"], "frames": [st, min(a1, st + n - 1)], "standing": bool(b.get("standing", False)),
                "negative": bool(rng.random() < float(sc.get("p_negative", 0.1))),
                "style": str(rng.choice(names, p=w / w.sum()))}
    if max_frames:
        spec["frames"] = [spec["frames"][0], min(spec["frames"][1], spec["frames"][0] + max_frames - 1)]
    t0 = time.perf_counter()
    scene = FakeScene(dict(spec, objects=[]), caster, seed, occlude_by_base=bool(cfg.get("occlude_by_base", False)))
    if spec.get("negative"):
        plans = list(spec.get("objects", []))     # negatives may carry decoys only
    elif "objects" in spec:
        plans = spec["objects"]
    else:
        plans = random_objects(rng, cfg.get("objects", {}), float(scene.s[-1]))
    for k, p in enumerate(plans):
        scene.add_object(k + 1, p)
    tag = "neg" if spec.get("negative") else "pos"
    sdir = os.path.join(out, f"{cfg['set']}_{i:03d}_{scene.style}_{tag}")
    os.makedirs(sdir, exist_ok=True)
    frame_id = cfg.get("frame_id", "hesai_lidar")
    topic = cfg.get("topic", "/lidar_points")
    bw = BagWriter(os.path.join(sdir, "bag"), topic, cfg.get("storage", "mcap"))
    lw = LabelWriter(sdir)
    n_pts = []
    for j in range(len(scene.frames)):
        f, points, n_deleted, sl, oi, labels = scene.render(j)
        bw.write(serialize_pointcloud2(points, f.stamp_ns, frame_id), f.bag_time_ns)
        lw.frame(j, f.stamp_ns, labels, None, sl, oi, extra={"real_frame": int(f.index), "n_points": int(len(points)),
                                                             "n_deleted": n_deleted, "style": scene.style,
                                                             "ego_s_m": round(float(scene.s[j]), 3)})
        n_pts.append(len(points))
    bw.close()
    lw.close()
    dt = time.perf_counter() - t0
    speed = float(scene.s[-1] - scene.s[0]) / max(0.1 * (len(scene.frames) - 1), 1e-9)
    info = {"sensor": "hesai128", "source": "insert_real", "style": scene.style, "real_bag": scene.name,
            "real_frames": [scene.frames[0], scene.frames[-1]],
            "tunnel": {"type": "real", "radius_m": None, "length_m": None, "description": "real recording"},
            "speed_mps": round(speed, 2), "standing": bool(spec.get("standing", False)), "seed": seed,
            "generator_version": GENERATOR_VERSION, "coordinate_mode": "hesai (forward -y, up +z)",
            "frame_id": frame_id, "topic": topic,
            "layout": STYLE_LAYOUT[scene.style], "base_phase_deg": scene.phase_deg,
            "renderer": STYLE_RENDERER[scene.style],
            "motion_model": "standing (declared)" if spec.get("standing") else
                            "1-DoF along-track displacement from motion.segment_motion_dp",
            "ego_s_total_m": round(float(scene.s[-1]), 2),
            "placement_method": "v1 per-frame rail level / centre-line (chain JSONL v1-gpu), 9-frame median; bottom = "
                                "rail model + floor_offset_m + bottom_m",
            "floor_offset_m": scene.floor_offset, "clearance_gauge": GAUGE,
            "objects": [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in ob.items() if k not in ("V", "F")}
                        for ob in scene.objects],
            "frames": len(scene.frames), "points_per_frame": [int(min(n_pts)), int(np.median(n_pts)), int(max(n_pts))],
            "render_s": round(dt, 2)}
    write_scene_json(sdir, info)
    return sdir, len(scene.frames), dt
