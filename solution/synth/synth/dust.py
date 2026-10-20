"""Dust clouds: Gaussian puffs of extinction, per-ray analytic optical depth (after Hahner et al., ICCV 2021,
"Fog simulation on real LiDAR point clouds", applied to a puff density field instead of homogeneous fog).

Extinction field  alpha(x) = sum_k a_k exp(-|x - c_k|^2 / (2 s_k^2))   [1/m]
Along a ray o + t d:  alpha(t) = sum_k a_k exp(-(b_k^2 + (t - t0_k)^2) / (2 s_k^2)),
optical depth tau(t) = sum_k a_k exp(-b_k^2 / 2 s_k^2) s_k sqrt(pi/2) [erf((t - t0_k)/(sqrt2 s_k)) + erf(t0_k/(sqrt2 s_k))].
* hard returns (background or objects) at range R are attenuated by T^2 = exp(-2 tau(R)): a detected echo of
  strength g survives with probability P(g T^2) / P(g), its intensity is multiplied by T^2;
* the medium returns a soft echo where alpha(t) exp(-2 tau(t)) / t^2 peaks, with the equivalent Lambertian
  reflectivity rho_eq = pi * (alpha / LR) * dR * exp(-2 tau) (lidar ratio LR = 40 sr, range bin dR = 0.75 m);
* dual-return sensors report the soft echo as the near and the surviving hard echo as the far return; single-
  return sensors keep the stronger one (``Echoes.strongest_only``).
Puffs drift with the cloud's wind velocity and are re-drawn every frame (amplitude jitter 15 %, position jitter
5 cm).  Clouds are labelled class ``dust``, ``is_obstacle: false``.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

try:
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover
    _erf = np.vectorize(math.erf)

LIDAR_RATIO_SR = 40.0
RANGE_BIN_M = 0.75


class DustField:
    def __init__(self, clouds: List[dict], rng_seed: int):
        self.clouds = clouds
        self.seed = rng_seed

    @classmethod
    def sample(cls, dcfg: dict, rng, scene, id_base: int = 100):
        clouds = []
        n_lo, n_hi = dcfg.get("n_clouds", [1, 2])
        for k in range(int(rng.integers(n_lo, n_hi + 1))):
            d0 = float(rng.uniform(*dcfg.get("d_range_m", [8.0, 80.0])))
            lat = float(rng.uniform(*dcfg.get("lateral_m", [-1.5, 1.5])))
            h = float(rng.uniform(*dcfg.get("height_m", [0.3, 1.8])))
            radius = float(rng.uniform(*dcfg.get("radius_m", [0.8, 3.0])))
            n_p = int(rng.integers(*dcfg.get("n_puffs", [6, 20])))
            a_peak = float(np.exp(rng.uniform(*np.log(dcfg.get("extinction_per_m", [0.05, 0.8])))))
            s = scene.s0 + d0
            centre = scene.cl.to_world(np.array([s]), np.array([lat]), np.array([h]))[0]
            off = rng.normal(0, radius / 2, (n_p, 3)) * np.array([1.0, 1.0, 0.5])
            sig = rng.uniform(*dcfg.get("puff_sigma_m", [0.3, 1.0]), n_p)
            amp = a_peak * rng.uniform(0.4, 1.0, n_p)
            ang = rng.uniform(0, 2 * np.pi)
            speed = float(rng.uniform(*dcfg.get("drift_mps", [0.1, 1.5])))
            vel = speed * np.array([np.cos(ang), np.sin(ang), rng.uniform(-0.1, 0.1)])
            clouds.append({"id": id_base + k, "centre0": centre, "offsets": off, "sigma": sig, "amp": amp,
                           "velocity": vel, "d0_m": d0, "lateral_m": lat, "height_m": h, "radius_m": radius,
                           "extinction_peak_per_m": a_peak})
            # an object inside or behind the cloud
            p_obj = float(dcfg.get("p_object_behind", 0.0))
            if rng.random() < p_obj:
                inside = rng.random() < 0.4
                dd = d0 + (0.0 if inside else float(rng.uniform(3.0, 25.0)))
                cls_ = str(rng.choice(dcfg.get("object_classes", ["person_standing", "person_crouching", "box"])))
                oid = len(scene.objects) + 1
                scene.add_object(oid, cls_, np.random.default_rng(int(rng.integers(1 << 31))), dd,
                                 float(rng.uniform(-0.6, 0.6)), float(rng.uniform(0.08, 0.5)), float(rng.uniform(-np.pi, np.pi)))
                clouds[-1]["object_id"] = oid
                clouds[-1]["object_inside"] = bool(inside)
        return cls(clouds, int(rng.integers(1 << 31)))

    def describe(self):
        return [{k: (v.tolist() if isinstance(v, np.ndarray) and v.size <= 3 else v) for k, v in c.items()
                 if k not in ("offsets", "sigma", "amp")} | {"n_puffs": int(len(c["sigma"]))} for c in self.clouds]

    def puffs_at(self, t: float, frame_rng):
        C, S, A, ID = [], [], [], []
        for c in self.clouds:
            n = len(c["sigma"])
            cen = c["centre0"] + c["velocity"] * t + c["offsets"] + frame_rng.normal(0, 0.05, (n, 3))
            C.append(cen)
            S.append(c["sigma"])
            A.append(c["amp"] * np.exp(frame_rng.normal(0, 0.15, n)))
            ID.append(np.full(n, c["id"]))
        return np.vstack(C), np.concatenate(S), np.concatenate(A), np.concatenate(ID)

    def apply(self, e, origins, dirs, ret, rng, t_header, scene):
        C, S, A, ID = self.puffs_at(t_header, rng)
        o = np.broadcast_to(np.asarray(origins, dtype=np.float64), dirs.shape)
        # candidate rays: within 3 sigma of some puff
        cand = np.zeros(len(dirs), dtype=bool)
        for c, s in zip(C, S):
            v = c - o
            t0 = (v * dirs).sum(1)
            b2 = (v * v).sum(1) - t0 * t0
            cand |= (b2 < (3 * s) ** 2) & (t0 > -3 * s)
        all_idx = np.nonzero(cand)[0]
        if len(all_idx) == 0:
            return self._labels(C, S, A, ID, e, None, None, scene, t_header, rng)
        rho_soft_all = []
        for c0 in range(0, len(all_idx), 4096):
            rho_soft_all.append(self._apply_chunk(all_idx[c0:c0 + 4096], e, o, dirs, ret, rng, C, S, A, ID))
        return self._labels(C, S, A, ID, e, all_idx, np.concatenate(rho_soft_all), scene, t_header, rng)

    def _apply_chunk(self, idx, e, o, dirs, ret, rng, C, S, A, ID):
        d = dirs[idx]
        oo = o[idx]
        V = C[None, :, :] - oo[:, None, :]                       # (m, P, 3)
        t0 = (V * d[:, None, :]).sum(2)                          # (m, P)
        b2 = (V * V).sum(2) - t0 * t0
        w = A[None] * np.exp(-b2 / (2 * S[None] ** 2))           # peak extinction on the ray per puff
        # keep only the puffs that touch this chunk of rays (consecutive beams look in one direction)
        keep = (w.max(0) > 1e-4 * A) & (t0.max(0) > -3 * S)
        if not keep.any():
            return np.zeros(0)
        C, S, A, ID = C[keep], S[keep], A[keep], ID[keep]
        t0, b2, w = t0[:, keep], b2[:, keep], w[:, keep]
        sq = np.sqrt(2.0) * S[None]

        def tau(t):  # t (m,) or (m, k)
            tt = t[..., None] if t.ndim == 2 else t[:, None]
            if t.ndim == 2:
                val = w[:, None, :] * S[None, None, :] * np.sqrt(np.pi / 2) * (
                    _erf((tt - t0[:, None, :]) / sq[:, None, :]) + _erf(t0[:, None, :] / sq[:, None, :]))
                return val.sum(-1)
            val = w * S[None] * np.sqrt(np.pi / 2) * (_erf((tt - t0) / sq) + _erf(t0 / sq))
            return val.sum(-1)

        r_near = e.r[idx, 0]
        r_far = e.r[idx, 1]
        hard_end = np.where(r_far > 0, r_far, np.where(r_near > 0, r_near, ret.max_range_m))
        # soft return: sample the ray where puffs are
        lo = np.clip(np.min(np.where(w > 1e-6, t0 - 3 * S[None], np.inf), axis=1), 0.3, None)
        hi = np.max(np.where(w > 1e-6, t0 + 3 * S[None], -np.inf), axis=1)
        hi = np.minimum(hi, np.where(r_near > 0, r_near, hard_end))
        K = 24
        u = (np.arange(K) + 0.5) / K
        T = lo[:, None] + (np.maximum(hi - lo, 0.0))[:, None] * u[None]           # (m, K)
        alpha = (w[:, None, :] * np.exp(-((T[..., None] - t0[:, None, :]) ** 2) / (2 * S[None, None, :] ** 2))).sum(-1)
        tauT = tau(T)
        rho_eq = np.pi * (alpha / LIDAR_RATIO_SR) * RANGE_BIN_M * np.exp(-2 * tauT)
        score = rho_eq / np.maximum(T, 0.3) ** 2
        k_best = np.argmax(score, axis=1)
        ts = T[np.arange(len(idx)), k_best]
        rho_s = rho_eq[np.arange(len(idx)), k_best]
        valid_seg = hi > lo
        soft = valid_seg & ret.detect(rng, rho_s, 1.0, ts)
        # which cloud dominates the soft return
        contrib = w * np.exp(-((ts[:, None] - t0) ** 2) / (2 * S[None] ** 2))
        soft_id = ID[np.argmax(contrib, axis=1)]
        # hard echoes: conditional survival under the two-way transmission
        out_r, out_i, out_s, out_g = e.r[idx].copy(), e.inten[idx].copy(), e.src[idx].copy(), e.strength[idx].copy()
        for k in (0, 1):
            rk = out_r[:, k]
            has = rk > 0
            T2 = np.exp(-2 * tau(np.where(has, rk, 0.0)))
            g = out_g[:, k]
            p_old = ret.p_detect(np.maximum(g, 1e-9), 1.0, rk)
            p_new = ret.p_detect(np.maximum(g * T2, 1e-12), 1.0, rk)
            keep = has & (rng.random(len(rk)) < np.where(p_old > 0, p_new / np.maximum(p_old, 1e-12), 0.0))
            out_i[:, k] = np.where(keep, np.maximum(1.0, np.round(out_i[:, k] * T2)), 0.0)
            out_r[:, k] = np.where(keep, rk, 0.0)
            out_s[:, k] = np.where(keep, out_s[:, k], -1)
            out_g[:, k] = np.where(keep, g * T2, 0.0)
        # compact: near = first surviving hard echo, far = last surviving one
        n_ok, f_ok = out_r[:, 0] > 0, out_r[:, 1] > 0
        hr = np.where(n_ok, out_r[:, 0], out_r[:, 1])
        hi_ = np.where(n_ok, out_i[:, 0], out_i[:, 1])
        hs = np.where(n_ok, out_s[:, 0], out_s[:, 1])
        hg = np.where(n_ok, out_g[:, 0], out_g[:, 1])
        fr = np.where(f_ok, out_r[:, 1], hr)
        fi = np.where(f_ok, out_i[:, 1], hi_)
        fs = np.where(f_ok, out_s[:, 1], hs)
        fg = np.where(f_ok, out_g[:, 1], hg)
        s_r = ret.measure_range(rng, ts)
        # range noise must not push the soft (near) echo behind the surviving hard (far) echo: near <= far
        s_r = np.where(fr > 0, np.minimum(s_r, fr - ret.quant_m), s_r)
        s_i = ret.intensity(rng, rho_s, 1.0, ts)
        new_r = np.stack([hr, fr], 1)
        new_i = np.stack([hi_, fi], 1)
        new_s = np.stack([hs, fs], 1)
        new_g = np.stack([hg, fg], 1)
        new_r[soft, 0], new_i[soft, 0], new_s[soft, 0], new_g[soft, 0] = s_r[soft], s_i[soft], soft_id[soft], rho_s[soft]
        no_hard = soft & (fr <= 0)
        new_r[no_hard, 1], new_i[no_hard, 1], new_s[no_hard, 1], new_g[no_hard, 1] = s_r[no_hard], s_i[no_hard], soft_id[no_hard], rho_s[no_hard]
        e.r[idx], e.inten[idx], e.src[idx], e.strength[idx] = new_r, new_i, new_s, new_g
        return rho_s[soft]

    def _labels(self, C, S, A, ID, e, idx, rho_soft, scene, t_header, rng):
        """Per-cloud labels (n_hits filled later from the message by the scene)."""
        from .insert import in_clearance

        out = []
        s_h, o_h, T_h, N_h, Z_h = scene.pose(np.array([t_header]))
        for c in self.clouds:
            m = ID == c["id"]
            Cc, Sc = C[m], S[m]
            lo_w = (Cc - 2 * Sc[:, None]).min(0)
            hi_w = (Cc + 2 * Sc[:, None]).max(0)
            cen_w = Cc.mean(0)
            s_c = float(scene.cl.s[np.argmin(np.linalg.norm(np.stack([scene.cl.x, scene.cl.y], 1) - cen_w[:2], axis=1))])
            p, T, N, Z = scene.cl.frame(np.array([s_c]))
            rel = np.vstack([Cc - 2 * Sc[:, None], Cc + 2 * Sc[:, None]]) - p[0]
            loc = np.stack([rel @ T[0], rel @ N[0], rel @ Z[0]], 1)
            lo, hi = loc.min(0), loc.max(0)
            out.append({"id": int(c["id"]), "class": "dust", "is_obstacle": False,
                        "center_sensor": scene.world_to_sensor(cen_w[None], t_header)[0].tolist(),
                        "extent_dlh": (hi - lo).tolist(), "d_along_m": float(s_c - s_h[0]),
                        "lateral_m": float(0.5 * (lo[1] + hi[1])), "height_above_rail_m": float(0.5 * (lo[2] + hi[2])),
                        "distance_m": float(max(0.0, np.linalg.norm(cen_w - o_h[0]) - 2 * Sc.max())), "n_hits": 0,
                        "visible": False, "in_clearance": bool(in_clearance(lo[1], hi[1], lo[2], hi[2])),
                        "reflectivity": float(np.median(rho_soft)) if rho_soft is not None and len(rho_soft) else 0.0,
                        "extinction_peak_per_m": float(c["extinction_peak_per_m"]), "visibility_m": float(3.0 / c["extinction_peak_per_m"]),
                        "n_puffs": int(m.sum()), "object_id_behind": c.get("object_id"), "object_inside": c.get("object_inside")})
        return out
