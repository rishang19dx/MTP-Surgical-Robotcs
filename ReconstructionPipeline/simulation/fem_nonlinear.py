"""
Nonlinear (compressible Neo-Hookean, finite strain) reference FEM on the SAME tetrahedral mesh, Dirichlet set and
moduli as the precomputed linear model (DEC-041 task T14, hypothesis H4).  Offline reference only; not used by the
gym.

Energy per tet (reference volume V, deformation gradient F = Ds Dm^-1, J = det F, I1 = tr F^T F):
    psi = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
with mu, lam from the same per-cell Young's modulus (fat / muscle by HU, x e_scale) and Poisson ratio as
`FEMTissueModel._precompute`.  Its linearisation at u = 0 is exactly the linear-elastic stiffness of that model
(checked in tests/test_fem_fidelity.py against the precomputed skin compliance), so the two models differ only by
finite-strain geometry and material nonlinearity.

Solver: external nodal forces applied in load steps; per step Newton iterations with the tangent assembled from
per-element 12x12 Hessians (torch.func, vectorised over tets) and solved by a sparse direct solver on the free dofs;
backtracking line search on the total potential energy keeping every J > 0.

What is compared (and what is not): both models get IDENTICAL nodal contact forces (from the linear contact solve
at a probe pose), so the comparison isolates the elasticity model.  The contact itself is not re-solved on the
deformed nonlinear geometry, and the linear model's strain-stiffening force factor (phi) is a separate empirical
correction that does not enter here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import torch
from torch.func import grad, hessian, vmap

from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel


def _det3(F: torch.Tensor) -> torch.Tensor:
    """3x3 determinant as an explicit polynomial (torch.linalg.det's double backward is NaN under vmap/hessian)."""
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def _psi(xe: torch.Tensor, Dm_inv: torch.Tensor, V: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor):
    """Neo-Hookean energy of one tet; xe (4,3) deformed node positions."""
    Ds = torch.stack([xe[1] - xe[0], xe[2] - xe[0], xe[3] - xe[0]], 1)
    F = Ds @ Dm_inv
    J = _det3(F)
    lnJ = torch.log(J)
    I1 = (F * F).sum()
    return V * (0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2)


class NeoHookeanReference:
    def __init__(self, model: FEMTissueModel, e_scale: Optional[float] = None):
        m = model.mesh
        dt = torch.float64
        self.model = model
        self.X = torch.as_tensor(m.nodes, dtype=dt)                                   # (Nn,3)
        self.tets = torch.as_tensor(m.tets, dtype=torch.long)                         # (Nt,4)
        es = model.e_scale if e_scale is None else float(e_scale)
        E = np.where(m.cell_hu < model.fat_hu, model.e_fat, model.e_muscle)[m.tet_cell] * 1e-3 * es   # N/mm^2
        nu = model.nu
        self.mu = torch.as_tensor(E / (2 * (1 + nu)), dtype=dt)
        self.lam = torch.as_tensor(E * nu / ((1 + nu) * (1 - 2 * nu)), dtype=dt)
        Xe = self.X[self.tets]
        Dm = torch.stack([Xe[:, 1] - Xe[:, 0], Xe[:, 2] - Xe[:, 0], Xe[:, 3] - Xe[:, 0]], 2)
        self.V = torch.linalg.det(Dm).abs() / 6.0
        self.Dm_inv = torch.linalg.inv(Dm)
        n = 3 * len(self.X)
        self.free = np.repeat(~m.fixed, 3)
        self.fidx = np.nonzero(self.free)[0]
        dofs = (m.tets[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        self.dofs = torch.as_tensor(dofs, dtype=torch.long)
        self.rows = np.repeat(dofs, 12, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 12)).ravel()
        self.n = n
        self._g = vmap(grad(_psi))
        self._h = vmap(hessian(_psi))
        self._e = vmap(_psi)

    def _state(self, u: torch.Tensor):
        return (self.X + u.view(-1, 3))[self.tets]

    def energy(self, u: torch.Tensor, f: torch.Tensor) -> float:
        xe = self._state(u)
        Ds = torch.stack([xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], 2)
        J = torch.linalg.det(Ds @ self.Dm_inv)       # value only (no derivative needed here)
        if bool((J <= 0).any()):
            return float("inf")
        return float(self._e(xe, self.Dm_inv, self.V, self.mu, self.lam).sum() - (f * u).sum())

    def gradient_and_tangent(self, u: torch.Tensor, f: torch.Tensor):
        xe = self._state(u)
        ge = self._g(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1, 12)
        g = torch.zeros(self.n, dtype=torch.float64).index_add_(0, self.dofs.reshape(-1), ge.reshape(-1)) - f
        He = self._h(xe, self.Dm_inv, self.V, self.mu, self.lam).reshape(-1, 12, 12)
        K = sps.coo_matrix((He.reshape(-1).numpy(), (self.rows, self.cols)), shape=(self.n, self.n)).tocsr()
        return g, K

    def solve(self, f_nodes: np.ndarray, load_steps: int = 5, newton_iters: int = 25, tol: float = 1e-8
              ) -> Dict[str, object]:
        """f_nodes (Nn,3) N -> displacement (Nn,3) mm and convergence info."""
        f_full = torch.as_tensor(np.asarray(f_nodes, dtype=np.float64).reshape(-1))
        f_full[torch.as_tensor(~self.free)] = 0.0
        u = torch.zeros(self.n, dtype=torch.float64)
        fi = self.fidx
        hist = []
        for s in range(1, load_steps + 1):
            f = f_full * (s / load_steps)
            fnorm = float(f.norm()) or 1.0
            for it in range(newton_iters):
                g, K = self.gradient_and_tangent(u, f)
                gf = g.numpy()[fi]
                res = float(np.linalg.norm(gf)) / fnorm
                if res < tol:
                    break
                du = np.zeros(self.n)
                du[fi] = spla.spsolve(K[fi][:, fi].tocsc(), -gf)
                du = torch.as_tensor(du)
                e0, a = self.energy(u, f), 1.0
                while a > 1e-6 and not (self.energy(u + a * du, f) <= e0 + 1e-12 * abs(e0)):
                    a *= 0.5
                u = u + a * du
            hist.append({"step": s, "newton_iters": it + 1, "rel_residual": res})
        return {"u": u.view(-1, 3).numpy(), "history": hist}

    def linear_solve(self, f_nodes: np.ndarray) -> np.ndarray:
        """K(u=0)^-1 f: the linearisation of this energy (must equal the precomputed linear model)."""
        f = torch.as_tensor(np.asarray(f_nodes, dtype=np.float64).reshape(-1))
        _, K = self.gradient_and_tangent(torch.zeros(self.n, dtype=torch.float64), torch.zeros(self.n, dtype=torch.float64))
        u = np.zeros(self.n)
        u[self.fidx] = spla.spsolve(K[self.fidx][:, self.fidx].tocsc(), f.numpy()[self.fidx])
        return u.reshape(-1, 3)


def contact_nodal_forces(model: FEMTissueModel, pose: np.ndarray, indentation_mm: float):
    """Linear model's contact solution at (pose, indentation) as mesh nodal forces (Nn,3) N (before the strain-
    stiffening factor), plus the loaded skin indices and the beam b."""
    ct = model._contact(pose, indentation_mm)
    f = np.zeros((model.mesh.num_nodes, 3))
    if len(ct["f"]) == 0:
        return f, np.zeros(0, dtype=np.int64), ct["b"]
    nodes = model.mesh.skin_nodes[ct["cand"]]
    f[nodes] = ct["f"][:, None] * ct["b"][None, :]
    return f, ct["cand"][ct["f"] > 0], ct["b"]
