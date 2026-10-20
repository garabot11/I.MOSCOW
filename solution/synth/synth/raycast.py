"""Ray casting of triangle meshes: NVIDIA Warp on the GPU, Open3D (CPU) or brute-force NumPy as fallbacks.

``Caster.cast(mesh, origins, dirs, tmax)`` returns ``(t, face, normal)``: entry distance (inf = miss),
triangle index (-1) and the unit face normal of the hit.  ``mesh`` is a ``MeshHandle`` built once per static
scene (``Caster.mesh(V, F)``); dynamic objects are cheap to rebuild per frame.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_WARP = None


def _warp():
    global _WARP
    if _WARP is None:
        try:
            import warp as wp

            wp.config.quiet = True
            wp.init()
            if not wp.get_cuda_device_count():
                raise RuntimeError("no CUDA device")

            @wp.kernel
            def _cast(mesh: wp.uint64, orig: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3), tmax: float,
                      out_t: wp.array(dtype=float), out_f: wp.array(dtype=int), out_n: wp.array(dtype=wp.vec3)):
                i = wp.tid()
                q = wp.mesh_query_ray(mesh, orig[i], dirs[i], tmax)
                if q.result:
                    out_t[i] = q.t
                    out_f[i] = q.face
                    out_n[i] = q.normal
                else:
                    out_t[i] = -1.0
                    out_f[i] = -1

            _WARP = (wp, _cast)
        except Exception as exc:  # pragma: no cover - depends on the host
            _WARP = exc
    return _WARP


class MeshHandle:
    def __init__(self, V, F, backend, obj=None):
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int32)
        self.backend = backend
        self.obj = obj
        e1 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]
        e2 = self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        n = np.cross(e1, e2)
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


class Caster:
    def __init__(self, backend: str = "auto"):
        backend = os.environ.get("IMOS_RAYCAST", backend)
        if backend in ("auto", "warp"):
            w = _warp()
            if isinstance(w, tuple):
                backend = "warp"
            elif backend == "warp":
                raise RuntimeError(f"Warp unavailable: {w}")
            else:
                backend = "open3d"
        if backend == "open3d":
            try:
                import open3d  # noqa: F401
            except Exception:
                backend = "numpy"
        self.backend = backend

    def mesh(self, V, F) -> MeshHandle:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int32)
        if len(F) == 0:
            return MeshHandle(V, F, "empty")
        if self.backend == "warp":
            wp, _ = _warp()
            m = wp.Mesh(points=wp.array(V.astype(np.float32), dtype=wp.vec3, device="cuda:0"),
                        indices=wp.array(F.reshape(-1), dtype=int, device="cuda:0"))
            return MeshHandle(V, F, "warp", m)
        if self.backend == "open3d":
            import open3d as o3d

            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
            return MeshHandle(V, F, "open3d", sc)
        return MeshHandle(V, F, "numpy")

    def cast(self, mh: MeshHandle, origins, dirs, tmax: float = 1000.0):
        dirs = np.asarray(dirs, dtype=np.float64)
        n = len(dirs)
        origins = np.broadcast_to(np.asarray(origins, dtype=np.float64), (n, 3))
        t = np.full(n, np.inf)
        face = np.full(n, -1, dtype=np.int64)
        if mh.backend == "empty" or n == 0:
            return t, face, np.zeros((n, 3))
        if mh.backend == "warp":
            wp, kern = _warp()
            o = wp.array(np.ascontiguousarray(origins, dtype=np.float32), dtype=wp.vec3, device="cuda:0")
            d = wp.array(np.ascontiguousarray(dirs, dtype=np.float32), dtype=wp.vec3, device="cuda:0")
            ot = wp.empty(n, dtype=float, device="cuda:0")
            of = wp.empty(n, dtype=int, device="cuda:0")
            on = wp.empty(n, dtype=wp.vec3, device="cuda:0")
            wp.launch(kern, dim=n, inputs=[mh.obj.id, o, d, float(tmax), ot, of, on], device="cuda:0")
            tt = ot.numpy()
            ff = of.numpy()
            hit = tt >= 0
            t[hit] = tt[hit]
            face[hit] = ff[hit]
        elif mh.backend == "open3d":
            import open3d as o3d

            rays = np.hstack([origins, dirs]).astype(np.float32)
            ans = mh.obj.cast_rays(o3d.core.Tensor(rays))
            tt = ans["t_hit"].numpy().astype(np.float64)
            ff = ans["primitive_ids"].numpy().astype(np.int64)
            hit = np.isfinite(tt) & (tt <= tmax)
            t[hit] = tt[hit]
            face[hit] = ff[hit]
        else:
            t, face = _brute(mh.V, mh.F, origins, dirs, tmax)
        normal = np.zeros((n, 3))
        hit = face >= 0
        normal[hit] = mh.normals[face[hit]]
        return t, face, normal


def _brute(V, F, origins, dirs, tmax, chunk=4096):
    """Moller-Trumbore over all rays x triangles (for tests and tiny meshes)."""
    n = len(dirs)
    t_best = np.full(n, np.inf)
    f_best = np.full(n, -1, dtype=np.int64)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    for s in range(0, n, chunk):
        d = dirs[s:s + chunk, None, :]
        o = origins[s:s + chunk, None, :]
        p = np.cross(d, e2[None])
        det = (e1[None] * p).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tv = o - v0[None]
            u = (tv * p).sum(-1) * inv
            q = np.cross(tv, e1[None])
            v = (d * q).sum(-1) * inv
            tt = (e2[None] * q).sum(-1) * inv
        ok = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 1e-6) & (tt <= tmax)
        tt = np.where(ok, tt, np.inf)
        j = np.argmin(tt, axis=1)
        tb = tt[np.arange(len(j)), j]
        t_best[s:s + chunk] = tb
        f_best[s:s + chunk] = np.where(np.isfinite(tb), j, -1)
    return t_best, f_best
