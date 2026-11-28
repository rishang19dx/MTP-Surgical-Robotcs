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

Cavity lumen (DEC-075, `cavity=True`): the `region_modulus` cells (a vein lumen) are REMOVED from the mesh instead of
being a 0.1 kPa solid that inverts under collapse (DEC-073: stall at AP ratio 0.36). Nodes left without a tet are held
fixed. The cavity surface (faces shared by a kept and a removed tet) gets frictionless wall-to-wall penalty contact:
for a probe pose, every cavity-surface node casts a ray along +b and -b in the REFERENCE configuration; the first
cavity face it hits through the cavity (segment midpoint in a removed cell) is its partner, with fixed barycentric
weights (like the probe footprint, fixed per pose so Newton sees a smooth energy). Gap g = s b . (sum_j w_j x_j - x_p)
in the deformed configuration (reference gap g0 > 0); energy k/2 max(0, -g)^2 with the probe's k. Valid while the walls
slide little relative to each other across the cavity (transverse compression); `cavity_ratio` gives the AP ratio
g / g0 of the pairs near a line along b that span the full AP diameter there.

Rounded probe edge (DEC-077, `edge_radius_mm` = r > 0): instead of the sharp face with a one-spacing linear taper, the
probe is a rounded rectangle: flat over (L - 2r) x (W - 2r), then a quarter circle of radius r out to the L x W
footprint. For a skin column at reference lateral excess q (distance beyond the flat part, in the face plane) the
face lies h(q) = r - sqrt(r^2 - q^2) further back, so p = max(0, -((x - c) . b + h)), with weight 1 for q < r and 0
beyond; h is fixed per pose from the reference positions, like the footprint. The force direction stays b (no
lateral contact force on the curved edge, as the flat-face model). DEC-075 found the patient solves failing at the
sharp edge.

Projected-Newton fallback (DEC-077, `psd_fallback=True`): when the line search finds no decrease along the exact
Newton step, the step is recomputed once with every tet's 12 x 12 Hessian projected onto its positive-semidefinite
part (eigenvalues clamped at 0) and the line search retried; the converged state is the same (only the step changes).

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
FULL_GAP = 0.8          # cavity_ratio: pairs spanning >= 0.8 x the site's largest reference gap (DEC-075)


def _frame(T: np.ndarray):
    o = T[:3, 3]
    ex = T[:3, 0] / np.linalg.norm(T[:3, 0])
    b = T[:3, 1] / np.linalg.norm(T[:3, 1])
    ez = T[:3, 2] / np.linalg.norm(T[:3, 2])
    return o, ex, b, ez


class NeoHookeanContact:
    def __init__(self, model: FEMTissueModel, e_scale: Optional[float] = None, penalty: float = PENALTY,
                 cavity: bool = False, edge_radius_mm: float = 0.0, psd_fallback: bool = False):
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
        self.cavity = bool(cavity)
        tets_np = np.asarray(m.tets)
        if self.cavity:
            if reg is None:
                raise ValueError("cavity=True needs the model's region_modulus mask (the lumen)")
            gone = reg[m.tet_cell]
            self._cavity_faces = self._interface_faces(tets_np[~gone], tets_np[gone])
            ijk = np.asarray(m.cell_ijk)[reg]
            self._cav_grid = np.zeros(tuple(int(v) for v in ijk.max(0) + 2), dtype=bool)
            self._cav_grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
            tets_np, E, nu = tets_np[~gone], E[~gone], nu[~gone]
            self.tets = torch.as_tensor(tets_np, dtype=torch.long)
        self._pairs = None
        self.mu = torch.as_tensor(E / (2 * (1 + nu)), dtype=dt)
        self.lam = torch.as_tensor(E * nu / ((1 + nu) * (1 - 2 * nu)), dtype=dt)
        Xe = self.X[self.tets]
        Dm = torch.stack([Xe[:, 1] - Xe[:, 0], Xe[:, 2] - Xe[:, 0], Xe[:, 3] - Xe[:, 0]], 2)
        self.V = torch.linalg.det(Dm).abs() / 6.0
        self.Dm_inv = torch.linalg.inv(Dm)
        self.n = 3 * len(self.X)
        used = np.zeros(len(m.nodes), dtype=bool)
        used[tets_np.ravel()] = True
        self.free = np.repeat(~m.fixed & used, 3)
        self.fidx = np.nonzero(self.free)[0]
        dofs = (tets_np[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
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
        self.edge_r = float(edge_radius_mm)
        self.psd_fallback = bool(psd_fallback)
        if self.edge_r * 2 >= min(self.L, self.W):
            raise ValueError("edge_radius_mm must be below half the face width")
        self.stats = {"psd_steps": 0}

    # ------------------------------------------------------------------ cavity
    @staticmethod
    def _interface_faces(kept: np.ndarray, gone: np.ndarray) -> np.ndarray:
        """Triangles shared by a kept tet and a removed tet (node ids, (F,3))."""
        fl = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        fk = np.sort(kept[:, fl].reshape(-1, 3), 1)
        fg = np.sort(gone[:, fl].reshape(-1, 3), 1)
        sk = {tuple(f) for f in fk}
        out = {tuple(f) for f in fg if tuple(f) in sk}
        return np.array(sorted(out), dtype=np.int64).reshape(-1, 3)

    def _in_cavity(self, pts: np.ndarray) -> np.ndarray:
        m = self.mesh
        c = np.floor((pts - m.lattice_origin) / m.h).astype(np.int64)
        sh = np.array(self._cav_grid.shape)
        ok = np.all((c >= 0) & (c < sh), 1)
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = self._cav_grid[c[ok, 0], c[ok, 1], c[ok, 2]]
        return out

    def cavity_pairs(self, T: np.ndarray) -> Dict[str, np.ndarray]:
        """Reference wall-to-wall pairs along the beam of pose T (see module docstring)."""
        _, _, b, _ = _frame(np.asarray(T, dtype=np.float64))
        F = self._cavity_faces
        X = self.X.numpy()
        A, B, C = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        e1, e2 = B - A, C - A
        nodes = np.unique(F)
        P, S, Fi, W, G0 = [], [], [], [], []
        eps = 1e-9
        for sgn in (1.0, -1.0):
            dvec = sgn * b
            h = np.cross(dvec, e2)                              # (F,3)
            det = (e1 * h).sum(1)
            okf = np.abs(det) > 1e-12
            inv = np.where(okf, 1.0 / np.where(okf, det, 1.0), 0.0)
            for ch in np.array_split(nodes, max(1, len(nodes) // 256)):
                o = X[ch]                                       # (n,3)
                sv = o[:, None, :] - A[None]                    # (n,F,3)
                uu = (sv * h[None]).sum(2) * inv[None]
                q = np.cross(sv, e1[None])
                vv = (q * dvec).sum(2) * inv[None]
                tt = (q * e2[None]).sum(2) * inv[None]
                hit = okf[None] & (uu >= -eps) & (vv >= -eps) & (uu + vv <= 1 + eps) & (tt > 1e-6)
                hit &= ~np.any(F[None, :, :] == ch[:, None, None], 2)
                tt = np.where(hit, tt, np.inf)
                j = np.argmin(tt, 1)
                tmin = tt[np.arange(len(ch)), j]
                good = np.isfinite(tmin)
                if not good.any():
                    continue
                mid = o[good] + 0.5 * tmin[good, None] * dvec
                good_idx = np.nonzero(good)[0][self._in_cavity(mid)]
                for k in good_idx:
                    f = j[k]
                    w = np.array([1 - uu[k, f] - vv[k, f], uu[k, f], vv[k, f]])
                    P.append(ch[k]); S.append(sgn); Fi.append(f); W.append(np.clip(w, 0, 1) / np.clip(w, 0, 1).sum())
                    G0.append(tmin[k])
        if not P:
            return {"node": np.zeros(0, np.int64), "sign": np.zeros(0), "tri": np.zeros((0, 3), np.int64),
                    "w": np.zeros((0, 3)), "g0": np.zeros(0), "b": b}
        return {"node": np.array(P, np.int64), "sign": np.array(S), "tri": F[np.array(Fi)], "w": np.array(W),
                "g0": np.array(G0), "b": b}

    def _cavity_gap(self, u: np.ndarray) -> np.ndarray:
        pr = self._pairs
        x = self.X.numpy() + u.reshape(-1, 3)
        y = (pr["w"][:, :, None] * x[pr["tri"]]).sum(1)
        return pr["sign"] * ((y - x[pr["node"]]) @ pr["b"])

    def cavity_ratio(self, u: np.ndarray, point: np.ndarray, radius: Optional[float] = None) -> float:
        """AP ratio of the cavity at the line through `point` along b: mean g / g0 (g clipped at 0) over the +b pairs
        whose node lies within `radius` (default h) of that line in the reference configuration and whose reference gap
        g0 is at least `FULL_GAP` x the largest such g0 (the vessel's AP diameter there; pairs across a one-cell step of
        the voxel cavity edge are left out, DEC-075)."""
        pr = self._pairs
        if pr is None or not len(pr["node"]):
            return float("nan")
        b = pr["b"]
        r = self.X.numpy()[pr["node"]] - np.asarray(point, dtype=np.float64)
        lat = np.linalg.norm(r - (r @ b)[:, None] * b[None], axis=1)
        sel = (pr["sign"] > 0) & (lat <= (self.mesh.h if radius is None else radius) + 1e-9)
        if not sel.any():
            return float("nan")
        sel &= pr["g0"] >= FULL_GAP * pr["g0"][sel].max()
        g = self._cavity_gap(u)[sel]
        return float(np.mean(np.maximum(g, 0.0) / pr["g0"][sel]))

    def _cavity_terms(self, u: np.ndarray):
        """Energy, and the active pairs' dof indices (A,4,3) and coefficients (A,4) with penetration q (A,)."""
        g = self._cavity_gap(u)
        q = np.maximum(0.0, -g)
        act = np.nonzero(q > 0)[0]
        pr = self._pairs
        nd = np.concatenate([pr["node"][act, None], pr["tri"][act]], 1)            # (A,4)
        cf = np.concatenate([-np.ones((len(act), 1)), pr["w"][act]], 1) * pr["sign"][act, None]
        return 0.5 * self.k * float((q * q).sum()), nd, cf, q[act]

    # ------------------------------------------------------------------ contact
    def _edge(self, T: np.ndarray):
        """Rounded-edge weight (1 inside the rounded footprint, else 0) and face set-back h per skin column."""
        o, ex, _, ez = _frame(T)
        r = self.skin_ref - o
        R = self.edge_r
        qa = np.maximum(0.0, np.abs(r @ ex) - (0.5 * self.L - R))
        qe = np.maximum(0.0, np.abs(r @ ez) - (0.5 * self.W - R))
        q = np.sqrt(qa * qa + qe * qe)
        inside = q < R
        h = np.where(inside, R - np.sqrt(np.maximum(R * R - q * q, 0.0)), 0.0)
        return inside.astype(np.float64), h

    def footprint(self, T: np.ndarray) -> np.ndarray:
        """Footprint weight of every skin column from its REFERENCE position (fixed for a pose, so the contact energy
        is a smooth function of u and Newton converges; lateral sliding of a few mm against a 45 x 10 mm face changes
        the footprint only at the taper)."""
        if self.edge_r > 0:
            return self._edge(T)[0]
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
        h = self._edge(T)[1] if self.edge_r > 0 else 0.0
        p = np.maximum(0.0, -((x - c) @ b + h))
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
        e = float(self._e(xe, self.Dm_inv, self.V, self.mu, self.lam).sum()) + 0.5 * self.k * float((w * p * p).sum())
        if self.cavity and self._pairs is not None:
            e += self._cavity_terms(u)[0]
        return e

    def grad_hess(self, u: np.ndarray, T, d, w, psd: bool = False):
        ut = torch.as_tensor(u)
        xe = self._elastic(ut)
        ge = self._g(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1, 12)
        g = torch.zeros(self.n, dtype=torch.float64).index_add_(0, self.dofs.reshape(-1), ge.reshape(-1)).numpy()
        He = self._h(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1, 12, 12)
        if psd:
            ev, Q = torch.linalg.eigh(0.5 * (He + He.transpose(1, 2)))
            He = (Q * ev.clamp_min(0.0)[:, None, :]) @ Q.transpose(1, 2)
        He = He.reshape(-1)
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
        if self.cavity and self._pairs is not None:
            _, nd, cf, q = self._cavity_terms(u)
            if len(q):
                bv = self._pairs["b"]
                idx = (nd[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)                # (A,12)
                gv = (-(self.k * q)[:, None, None] * cf[:, :, None] * bv[None, None, :]).reshape(-1, 12)
                np.add.at(g, idx.ravel(), gv.ravel())
                Hc = self.k * np.einsum("ai,aj,kl->aikjl", cf, cf, np.outer(bv, bv)).reshape(-1, 12, 12)
                K = K + sps.coo_matrix((Hc.ravel(), (np.repeat(idx, 12, 1).ravel(), np.tile(idx, (1, 12)).ravel())),
                                       shape=(self.n, self.n)).tocsr()
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
            e0 = self.energy(u, T, d, w)
            for psd in ((False, True) if self.psd_fallback else (False,)):
                if psd:
                    _, K, _, _ = self.grad_hess(u, T, d, None, psd=True)
                    K = K + sps.identity(self.n, format="csr") * (1e-12 * float(abs(K.diagonal()[fi]).mean()))
                du = np.zeros(self.n)
                du[fi] = spla.spsolve(K[fi][:, fi].tocsc(), -gf)
                a = 1.0
                while a > 1e-6 and not (self.energy(u + a * du, T, d, w) <= e0 + 1e-12 * abs(e0)):
                    a *= 0.5
                if a > 1e-6:
                    self.stats["psd_steps"] += int(psd)
                    break
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
        if self.cavity:
            self._pairs = self.cavity_pairs(T)
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
