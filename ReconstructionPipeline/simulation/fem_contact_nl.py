"""
Large-strain FEM with probe contact re-solved on the deformed skin (DEC-071). Offline only: not used by the gym or the
1 kHz loop (AGENTS invariant 2).

    nh = NeoHookeanContact(model)                       # model: a built FEMTissueModel (mesh, materials, region)
    out = nh.sweep(pose_P, d_max_mm=25.0, step_mm=0.5, on_step=callback)   # displacement-controlled probe sweep

Material: compressible Neo-Hookean per tet, psi = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2, with mu, lam from the SAME
per-cell rule as `FEMTissueModel._precompute`: fat / muscle by HU (x e_scale), the optional `region_modulus` region
(e.g. a near-empty vein lumen) with its own Poisson ratio. Its linearisation at u = 0 is the linear model's stiffness
(as `fem_nonlinear.NeoHookeanReference`, DEC-047).

Contact: the probe is a rigid rectangular face (`model.face_dims`, lateral x elevation) at the pose's origin moved by the
indentation d along the beam b (into the tissue). Each skin column's contact point is its reference skin-surface point
(`mesh.skin_vertices_ref`, the linear model's contact geometry) moved by its skin node's displacement; in the DEFORMED
configuration such a point x penetrates the probe
when s = (x - c) . b < 0 (c = o + d b) and it lies under the face footprint; penetration p = max(0, -s). Frictionless
penalty energy E_c = k/2 sum_i w_i p_i^2 with k = PENALTY x E_muscle x h (N/mm per node, as the DEC-060 sliding
penalty) and w_i in [0, 1] a footprint weight with a linear edge taper of one lattice spacing, computed from the
reference skin positions (fixed per pose). Probe force F = sum_i k w_i p_i.

Solver: indentation increments from first contact (warm start), Newton on elastic + contact energy with a sparse direct
solve on the free dofs and backtracking line search keeping every J > 0; an increment that fails is halved (down to
min_step_mm), as the adaptive load stepping of DEC-047.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import torch
from torch.func import grad, hessian, vmap

from ReconstructionPipeline.simulation.fem_nonlinear import _psi
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel

PENALTY = 100.0


def _frame(T: np.ndarray):
    o = T[:3, 3]
    ex = T[:3, 0] / np.linalg.norm(T[:3, 0])
    b = T[:3, 1] / np.linalg.norm(T[:3, 1])
    ez = T[:3, 2] / np.linalg.norm(T[:3, 2])
    return o, ex, b, ez


class NeoHookeanContact:
    def __init__(self, model: FEMTissueModel, e_scale: Optional[float] = None, penalty: float = PENALTY):
        m = model.mesh
        dt = torch.float64
        self.model, self.mesh = model, m
        self.X = torch.as_tensor(m.nodes, dtype=dt)
        self.tets = torch.as_tensor(m.tets, dtype=torch.long)
        es = model.e_scale if e_scale is None else float(e_scale)
        E_cell = np.where(m.cell_hu < model.fat_hu, model.e_fat, model.e_muscle) * 1e-3
        nu_cell = np.full(len(E_cell), float(model.nu))
        reg = model.cells_in_region()
        if reg is not None:
            E_cell = np.where(reg, model.e_region * 1e-3, E_cell)
            if model.nu_region is not None:
                nu_cell = np.where(reg, float(model.nu_region), nu_cell)
        E = E_cell[m.tet_cell] * es
        nu = nu_cell[m.tet_cell]
        self.mu = torch.as_tensor(E / (2 * (1 + nu)), dtype=dt)
        self.lam = torch.as_tensor(E * nu / ((1 + nu) * (1 - 2 * nu)), dtype=dt)
        Xe = self.X[self.tets]
        Dm = torch.stack([Xe[:, 1] - Xe[:, 0], Xe[:, 2] - Xe[:, 0], Xe[:, 3] - Xe[:, 0]], 2)
        self.V = torch.linalg.det(Dm).abs() / 6.0
        self.Dm_inv = torch.linalg.inv(Dm)
        self.n = 3 * len(self.X)
        self.free = np.repeat(~m.fixed, 3)
        self.fidx = np.nonzero(self.free)[0]
        dofs = (m.tets[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        self.dofs = torch.as_tensor(dofs, dtype=torch.long)
        self.rows = np.repeat(dofs, 12, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 12)).ravel()
        self._g, self._h, self._e = vmap(grad(_psi)), vmap(hessian(_psi)), vmap(_psi)
        self.skin = np.asarray(m.skin_nodes, dtype=np.int64)
        # contact points: the skin reference surface (x, y_skin(x, z), z) of each column, as the linear model's contact
        # geometry, moved by its skin node's displacement (the top lattice node can sit up to h above the true skin)
        self.skin_ref = np.asarray(m.skin_vertices_ref, dtype=np.float64)
        self.k = float(penalty) * model.e_muscle * 1e-3 * es * m.h
        self.L, self.W = model.face_dims
        self.taper = m.h

    # ------------------------------------------------------------------ contact
    def footprint(self, T: np.ndarray) -> np.ndarray:
        """Footprint weight of every skin column from its REFERENCE position (fixed for a pose, so the contact energy
        is a smooth function of u and Newton converges; lateral sliding of a few mm against a 45 x 10 mm face changes
        the footprint only at the taper)."""
        o, ex, _, ez = _frame(T)
        r = self.skin_ref - o
        wl = np.clip((0.5 * self.L - np.abs(r @ ex)) / self.taper + 0.5, 0.0, 1.0)
        we = np.clip((0.5 * self.W - np.abs(r @ ez)) / self.taper + 0.5, 0.0, 1.0)
        return wl * we

    def _contact(self, u: np.ndarray, T: np.ndarray, d: float, w: Optional[np.ndarray] = None):
        """Penetration p and footprint weight w (from the reference positions if None) of every skin column, beam b."""
        o, ex, b, ez = _frame(T)
        c = o + d * b
        x = self.skin_ref + u.reshape(-1, 3)[self.skin]
        if w is None:
            w = self.footprint(T)
        p = np.maximum(0.0, -((x - c) @ b))
        return p, w, b

    def _elastic(self, u: torch.Tensor):
        xe = (self.X + u.view(-1, 3))[self.tets]
        return xe

    def energy(self, u: np.ndarray, T, d, w) -> float:
        ut = torch.as_tensor(u)
        xe = self._elastic(ut)
        Ds = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], 2)
        if bool((torch.linalg.det(Ds @ self.Dm_inv) <= 0).any()):
            return float("inf")
        p, _, _ = self._contact(u, T, d, w)
        return float(self._e(xe, self.Dm_inv, self.V, self.mu, self.lam).sum()) + 0.5 * self.k * float((w * p * p).sum())

    def grad_hess(self, u: np.ndarray, T, d, w):
        ut = torch.as_tensor(u)
        xe = self._elastic(ut)
        ge = self._g(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1, 12)
        g = torch.zeros(self.n, dtype=torch.float64).index_add_(0, self.dofs.reshape(-1), ge.reshape(-1)).numpy()
        He = self._h(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1)
        K = sps.coo_matrix((He.numpy(), (self.rows, self.cols)), shape=(self.n, self.n)).tocsr()
        p, w, b = self._contact(u, T, d, w)
        act = np.nonzero(p > 0)[0]
        if len(act):
            nd = self.skin[act]
            kw = self.k * w[act]
            idx = nd[:, None] * 3 + np.arange(3)                                     # (A,3)
            np.add.at(g, idx.ravel(), (-(kw * p[act])[:, None] * b[None, :]).ravel())
            bb = np.outer(b, b)
            r_ = np.repeat(idx, 3, axis=1).ravel()
            c_ = np.tile(idx, (1, 3)).ravel()
            K = K + sps.coo_matrix(((kw[:, None, None] * bb[None]).ravel(), (r_, c_)), shape=(self.n, self.n)).tocsr()
        return g, K, p, w

    def force(self, u: np.ndarray, T, d) -> float:
        p, w, _ = self._contact(u, T, d)
        return float(self.k * (w * p).sum())

    # ------------------------------------------------------------------ solve
    def _newton(self, u: np.ndarray, T, d, iters: int, tol: float):
        fi = self.fidx
        res, scale = float("inf"), None
        for it in range(iters):
            g, K, p, w = self.grad_hess(u, T, d, None)
            gf = g[fi]
            if scale is None:
                scale = max(float(np.linalg.norm(gf)), self.k * 1e-3)
            res = float(np.linalg.norm(gf)) / scale
            if not np.isfinite(res):
                return u, it + 1, res
            if res < tol:
                return u, it + 1, res
            du = np.zeros(self.n)
            du[fi] = spla.spsolve(K[fi][:, fi].tocsc(), -gf)
            e0, a = self.energy(u, T, d, w), 1.0
            while a > 1e-6 and not (self.energy(u + a * du, T, d, w) <= e0 + 1e-12 * abs(e0)):
                a *= 0.5
            if a <= 1e-6:
                return u, it + 1, res
            u = u + a * du
        return u, iters, res

    def sweep(self, pose_P: np.ndarray, d_max_mm: float, step_mm: float = 0.5, min_step_mm: float = 1.0 / 32,
              newton_iters: int = 30, tol: float = 1e-6, f_stop_n: Optional[float] = None,
              on_step: Optional[Callable[[float, float, np.ndarray], Optional[bool]]] = None) -> Dict[str, object]:
        """Indent from first contact to first contact + d_max_mm. After each converged increment calls
        on_step(indentation_beyond_first_contact, force, u (Nn,3)); a True return stops the sweep. Stops also when the
        force exceeds f_stop_n. Returns the force curve and convergence history."""
        T = np.asarray(pose_P, dtype=np.float64)
        d0 = float(self.model.first_contact_indentation(T))
        d0 = 0.0 if not np.isfinite(d0) else d0
        u = np.zeros(self.n)
        dd, step, curve, hist, converged = 0.0, step_mm, [], [], True
        while dd < d_max_mm - 1e-9:
            s = min(step, d_max_mm - dd)
            u_try, it, res = self._newton(u, T, d0 + dd + s, newton_iters, tol)
            if res < tol:
                u, dd = u_try, dd + s
                F = self.force(u, T, d0 + dd)
                curve.append({"indent_mm": dd, "force_n": F, "newton_iters": it})
                stop = on_step(dd, F, u.reshape(-1, 3)) if on_step else None
                step = min(step_mm, s * 1.5)
                if stop or (f_stop_n is not None and F >= f_stop_n):
                    break
            else:
                step = s / 2.0
                hist.append({"indent_mm": dd, "failed_step_mm": s, "rel_residual": res})
                if step < min_step_mm:
                    converged = False
                    break
        return {"first_contact_mm": d0, "curve": curve, "history": hist, "converged": converged,
                "indent_reached_mm": dd}

    # ------------------------------------------------------------------ outputs
    def displacement_at(self, u: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Trilinear displacement (mm) at reference points (n,3) from nodal u (Nn,3), on the mesh lattice. Lattice
        nodes without a mesh node take the value of the nearest present corner of their cell."""
        m = self.mesh
        NX, NY, NZ = m.lattice_shape
        grid = np.full((NX, NY, NZ, 3), np.nan)
        ijk = m.node_ijk
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = u.reshape(-1, 3)
        f = (np.asarray(points, dtype=np.float64) - m.lattice_origin) / m.h
        i0 = np.clip(np.floor(f).astype(int), 0, np.array([NX, NY, NZ]) - 2)
        t = np.clip(f - i0, 0.0, 1.0)
        out = np.zeros((len(f), 3))
        wsum = np.zeros(len(f))
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    wgt = ((t[:, 0] if cx else 1 - t[:, 0]) * (t[:, 1] if cy else 1 - t[:, 1])
                           * (t[:, 2] if cz else 1 - t[:, 2]))
                    v = grid[i0[:, 0] + cx, i0[:, 1] + cy, i0[:, 2] + cz]
                    ok = np.isfinite(v[:, 0])
                    out[ok] += wgt[ok, None] * v[ok]
                    wsum[ok] += wgt[ok]
        return out / np.maximum(wsum, 1e-12)[:, None]
