"""
GPU-batched contact + field evaluation for the precomputed linear FEM (DEC-041, task T1).

`BatchedFEMSolver(model)` wraps one patient's `FEMTissueModel` and solves B probe queries at once:

    st = solver.solve(poses_P (B,4,4), indentations_mm (B,), e_scale=None | (B,))

It is the same model as `FEMTissueModel._solve` (module docstring of fem_tissue.py, steps 4-6), written
as batched tensor algebra so that N gym environments share one patient's compliance C and lattice field
G_lat and need no Python loop over envs:

  * face sampling, the fixed-point normal gap along the beam b, and bilinear distribution of face samples
    onto skin columns are elementwise over (B, n_samples);
  * each env's distinct candidate skin nodes are compacted with a sort + cumsum into a padded
    (B, K) index (K = max over the batch); padded slots carry zero compliance and zero interpolation
    weight, so their equations read f = 0;
  * the active-set iteration over face samples runs for the whole batch with batched
    `torch.linalg.solve`, until every env's active set is stable (or `max_active_set_iters`);
  * strain stiffening, force vector, torque and the frozen-active-set contact stiffness dF/dd follow
    `_forces` / `_stiffness` exactly;
  * the displacement field is one dense GEMM (B, 3 Sf) @ G_lat (3 Sf, 3 N_lat) -> (B, 3, NZ, NY, NX).

Per-env `e_scale` (a (B,) tensor) gives per-env stiffness randomisation for free: linear elasticity
scales C and G by 1 / e_scale.

Parity with the single-env CPU path is a test (`tests/test_fem_batched.py`).  The contact solve runs in
`contact_dtype`.  The default is float32: consumer and workstation GPUs run fp64 at a small fraction of fp32,
and the test bounds the fp32 force error against fp64.  Pass float64 for bit-level parity with the
CPU path.  The field GEMM runs in the model's field dtype (fp16 on the GPU).

Not batched here (per-env CPU geometry, cheap, only needed for diagnostics): bone depth, tilt cosine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel


@dataclass
class BatchedFEMState:
    """Batched tissue response, frame P, mm / N / N mm.  All tensors have leading dim B."""
    indentation_mm: torch.Tensor          # (B,)
    reaction_force_n: torch.Tensor        # (B,)   along the beam, after strain stiffening
    linear_force_n: torch.Tensor          # (B,)
    stiffening_factor: torch.Tensor       # (B,)
    compression_mm: torch.Tensor          # (B,)
    contact_stiffness_n_per_mm: torch.Tensor  # (B,) dF/dd, frozen active set
    force_vec_n: torch.Tensor             # (B,3)  force ON the probe = -b F
    torque_nmm: torch.Tensor              # (B,3)  about the face centre
    in_contact: torch.Tensor              # (B,) bool
    num_active_nodes: torch.Tensor        # (B,) int
    active_set_iters: int                 # iterations the batch needed
    active_set_converged: torch.Tensor    # (B,) bool
    u_grid: Optional[torch.Tensor]        # (B,3,NZ,NY,NX) float32, or None if field not requested
    grid_origin_mm: torch.Tensor          # (3,)
    grid_extent_mm: torch.Tensor          # (3,)

    def probe_face_poses(self, poses_P: torch.Tensor) -> torch.Tensor:
        """(B,4,4) skin probe poses -> poses of the indented face (image plane), as
        `TissueState.probe_face_pose`: origin moved by indentation along the beam (+Y_probe)."""
        T = torch.as_tensor(poses_P, device=self.indentation_mm.device).to(torch.float64).clone()
        b = T[:, :3, 1] / T[:, :3, 1].norm(dim=1, keepdim=True)
        T[:, :3, 3] = T[:, :3, 3] + self.indentation_mm.to(torch.float64)[:, None] * b
        return T

    def warp_to_reference(self, x: torch.Tensor, iterations: int = 5) -> torch.Tensor:
        """(B,N,3) deformed points -> reference points, inverting x = X + u(X) by fixed point, per env
        (`TissueState.warp_to_reference`).  Out-of-contact envs have a zero field, so X = x exactly."""
        if self.u_grid is None:
            raise ValueError("solve(..., field=True) is required for warp_to_reference()")
        x = x.to(self.u_grid)
        X = x
        for _ in range(iterations):
            X = x - self.displacement(X)
        return X

    def displacement(self, X: torch.Tensor) -> torch.Tensor:
        """Displacement (mm) at reference points X (B, N, 3) in frame P -> (B, N, 3).  Zero outside the ROI
        is NOT applied here; `padding_mode='border'` matches the single-env `_make_displacement_fn`."""
        if self.u_grid is None:
            raise ValueError("solve(..., field=True) is required for displacement()")
        B, N, _ = X.shape
        g = (2.0 * (X.to(self.u_grid) - self.grid_origin_mm) / self.grid_extent_mm - 1.0).view(B, N, 1, 1, 3)
        u = F.grid_sample(self.u_grid, g, mode="bilinear", padding_mode="border", align_corners=True)
        return u.view(B, 3, N).transpose(1, 2)


class BatchedFEMSolver:
    """Batched evaluation of one patient's `FEMTissueModel` on its device (see module docstring)."""

    def __init__(self, model: FEMTissueModel, contact_dtype: torch.dtype = torch.float32,
                 max_active_set_iters: int = 30):
        self.model = model
        self.dev = model.device
        self.dt = contact_dtype
        self.max_iters = int(max_active_set_iters)
        m = model.mesh
        dev, dt = self.dev, self.dt
        self.hmap = torch.as_tensor(model.hmap, dtype=dt, device=dev)
        self.hmap_o = [float(v) for v in model.hmap_o]
        self.hmap_s = [float(v) for v in model.hmap_s]
        self.C = torch.as_tensor(model.C_base, dtype=dt, device=dev)                  # (3Sf, 3Sf)
        self.column_skin = torch.as_tensor(m.column_skin, dtype=torch.long, device=dev)  # (NX, NZ)
        self.skin_to_free = torch.as_tensor(model.skin_to_free, dtype=torch.long, device=dev)
        self.skin_ref = torch.as_tensor(model.skin_ref, dtype=dt, device=dev)            # (S, 3)
        self.lat_origin = torch.as_tensor(m.lattice_origin, dtype=dt, device=dev)
        ls, es, da = model._face_samples()
        self.ls = torch.as_tensor(ls, dtype=dt, device=dev)
        self.es = torch.as_tensor(es, dtype=dt, device=dev)
        self.kpa = model.k_p * da
        self.Sf = len(model.free_skin)

    # ------------------------------------------------------------ geometry
    def _skin_height(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Torch port of `tet_mesh.bilinear_height` (same clamping)."""
        H = self.hmap
        nz, nx = H.shape
        fx = ((x - self.hmap_o[0]) / self.hmap_s[0]).clamp(0, nx - 1)
        fz = ((z - self.hmap_o[1]) / self.hmap_s[1]).clamp(0, nz - 1)
        x0 = fx.floor().long().clamp(0, max(nx - 2, 0))
        z0 = fz.floor().long().clamp(0, max(nz - 2, 0))
        x1 = (x0 + 1).clamp_max(nx - 1)
        z1 = (z0 + 1).clamp_max(nz - 1)
        tx, tz = fx - x0, fz - z0
        return (H[z0, x0] * (1 - tx) * (1 - tz) + H[z0, x1] * tx * (1 - tz)
                + H[z1, x0] * (1 - tx) * tz + H[z1, x1] * tx * tz)

    def _normal_gap(self, q: torch.Tensor, b: torch.Tensor, iters: int = 3):
        """Batched `FEMTissueModel._normal_gap`: q (B,n,3), b (B,3) -> g (B,n), p (B,n,3)."""
        cb = -b[:, 1:2]
        bb = b[:, None, :]
        t = (self._skin_height(q[..., 0], q[..., 2]) - q[..., 1]) / cb
        for _ in range(iters):
            p = q - t[..., None] * bb
            t = (self._skin_height(p[..., 0], p[..., 2]) - q[..., 1]) / cb
        return t, q - t[..., None] * bb

    # ------------------------------------------------- controller queries
    @staticmethod
    def _frames(T: torch.Tensor):
        return T[:, :3, 3], F.normalize(T[:, :3, 0], dim=1), F.normalize(T[:, :3, 1], dim=1), \
            F.normalize(T[:, :3, 2], dim=1)

    @torch.no_grad()
    def first_contact(self, poses_P, iters: int = 4) -> torch.Tensor:
        """Batched `FEMTissueModel.first_contact_indentation` -> (B,) mm; NaN where the beam does not
        point into the dorsal skin."""
        T = torch.as_tensor(poses_P, dtype=self.dt, device=self.dev)
        o, ex, b, ez = self._frames(T)
        d = torch.zeros(T.shape[0], dtype=self.dt, device=self.dev)
        for _ in range(iters):
            q = ((o + d[:, None] * b)[:, None, :] + self.ls[None, :, None] * ex[:, None, :]
                 + self.es[None, :, None] * ez[:, None, :])
            g, _ = self._normal_gap(q, b)
            d = d - g.max(1).values
        return torch.where(b[:, 1] > -0.2, torch.full_like(d, float("nan")), d)

    @torch.no_grad()
    def tissue_thickness(self, poses_P) -> torch.Tensor:
        """Batched `FEMTissueModel.tissue_thickness` (nearest bone-top column) -> (B,) mm, NaN if no bone."""
        m = self.model
        if not hasattr(self, "_bone_top"):
            self._bone_top = torch.as_tensor(m._bone_top, dtype=self.dt, device=self.dev)
        T = torch.as_tensor(poses_P, dtype=self.dt, device=self.dev)
        o, _, b, _ = self._frames(T)
        nz, nx = self._bone_top.shape
        ix = torch.round((o[:, 0] - self.hmap_o[0]) / self.hmap_s[0]).long().clamp(0, nx - 1)
        iz = torch.round((o[:, 2] - self.hmap_o[1]) / self.hmap_s[1]).long().clamp(0, nz - 1)
        return (o[:, 1] - self._bone_top[iz, ix]) / (-b[:, 1]).clamp_min(1e-3)

    @torch.no_grad()
    def max_indentation(self, poses_P, first_contact: Optional[torch.Tensor] = None,
                        default_mm: float = 30.0) -> torch.Tensor:
        """Batched `FEMTissueModel.max_indentation` -> (B,) mm."""
        t = self.tissue_thickness(poses_P)
        lim = torch.where(torch.isfinite(t), self.model.max_indent_fraction * t, torch.full_like(t, default_mm))
        d0 = self.first_contact(poses_P) if first_contact is None else first_contact
        return torch.where(torch.isfinite(d0), torch.maximum(lim, d0 + 1.0), lim)

    @torch.no_grad()
    def force(self, poses_P, indentations_mm, e_scale=None) -> torch.Tensor:
        """Batched `FEMTissueModel.force_at` (no field) -> (B,) N."""
        return self.solve(poses_P, indentations_mm, e_scale=e_scale, field=False, stiffness=False).reaction_force_n

    # --------------------------------------------------------------- solve
    @torch.no_grad()
    def solve(self, poses_P, indentations_mm, e_scale=None, field: bool = True,
              stiffness: bool = True) -> BatchedFEMState:
        m = self.model
        dev, dt = self.dev, self.dt
        T = torch.as_tensor(poses_P, dtype=dt, device=dev)
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        d = torch.as_tensor(indentations_mm, dtype=dt, device=dev).reshape(-1).expand(B).clone()
        if e_scale is None:
            es_env = torch.full((B,), m.e_scale, dtype=dt, device=dev)
        else:
            es_env = torch.as_tensor(e_scale, dtype=dt, device=dev).reshape(-1).expand(B).clone()

        o = T[:, :3, 3]
        ex = F.normalize(T[:, :3, 0], dim=1)
        b = F.normalize(T[:, :3, 1], dim=1)
        ez = F.normalize(T[:, :3, 2], dim=1)
        env_ok = b[:, 1] <= -0.2                                   # beam points into the dorsal skin
        c = o + d[:, None] * b
        q = (c[:, None, :] + self.ls[None, :, None] * ex[:, None, :] + self.es[None, :, None] * ez[:, None, :])
        g, q = self._normal_gap(q, b)                              # (B,n), (B,n,3)
        pen = (g > 0) & env_ok[:, None]

        # bilinear distribution of each sample onto the 4 surrounding skin columns
        h = m.h
        NX, NZ = m.NX, m.NZ
        fi = (q[..., 0] - self.lat_origin[0]) / h
        fk = (q[..., 2] - self.lat_origin[2]) / h
        i0 = fi.floor().long().clamp(0, NX - 2)
        k0 = fk.floor().long().clamp(0, NZ - 2)
        ti = (fi - i0).clamp(0, 1)
        tk = (fk - k0).clamp(0, 1)
        nodes = torch.stack([self.column_skin[i0, k0], self.column_skin[i0 + 1, k0],
                             self.column_skin[i0, k0 + 1], self.column_skin[i0 + 1, k0 + 1]], -1)  # (B,n,4)
        wts = torch.stack([(1 - ti) * (1 - tk), ti * (1 - tk), (1 - ti) * tk, ti * tk], -1)
        wts = wts * (nodes >= 0)
        wsum = wts.sum(-1, keepdim=True)
        wts = torch.where(wsum > 1e-9, wts / wsum.clamp_min(1e-12), torch.zeros_like(wts))
        keep = pen[..., None] & (nodes >= 0) & (wts > 0)           # (B,n,4)

        # compact the distinct candidate nodes of each env into padded slots 0..K-1
        n_s = g.shape[1]
        BIG = int(self.skin_ref.shape[0]) + 1
        key = torch.where(keep, nodes, torch.full_like(nodes, BIG)).view(B, -1)   # (B, 4n)
        skey, order = key.sort(dim=1)
        first = torch.ones_like(skey, dtype=torch.bool)
        first[:, 1:] = skey[:, 1:] != skey[:, :-1]
        first &= skey < BIG
        slot_sorted = first.long().cumsum(1) - 1                   # compact slot of each sorted entry
        counts = first.sum(1)
        K = max(int(counts.max()), 1)
        slot = torch.empty_like(slot_sorted).scatter_(1, order, slot_sorted)        # back to original order
        slot = torch.where(key < BIG, slot, torch.full_like(slot, K))               # padding column K
        cand = torch.full((B, K + 1), -1, dtype=torch.long, device=dev)
        cand.scatter_(1, torch.where(first, slot_sorted, torch.full_like(slot_sorted, K)),
                      torch.where(first, skey, torch.full_like(skey, -1)))
        cand = cand[:, :K]                                         # (B,K), -1 = padding
        cand_ok = cand >= 0

        # interpolation matrix N (B, n, K)
        Nm = torch.zeros(B, n_s, K + 1, dtype=dt, device=dev)
        Nm.scatter_add_(2, slot.view(B, n_s, 4), (wts * keep).to(dt))
        Nm = Nm[..., :K]
        has = Nm.sum(-1) > 1e-12
        base = pen & has                                           # samples taking part (CPU: pen, then has)
        g = torch.where(base, g, torch.zeros_like(g))

        # normal compliance among candidates: Cb[a,c] = b^T C_ac b / e_scale
        fr = torch.where(cand_ok, self.skin_to_free[cand.clamp_min(0)], torch.full_like(cand, -1))
        okf = fr >= 0
        r3 = (3 * fr.clamp_min(0)[..., None] + torch.arange(3, device=dev)).view(B, 3 * K)
        sub = self.C[r3[:, :, None], r3[:, None, :]].view(B, K, 3, K, 3)
        Cc = torch.einsum("bi,baicj,bj->bac", b, sub, b) / es_env[:, None, None]
        Cc = Cc * (okf[:, :, None] & okf[:, None, :])

        kpa = self.kpa
        I = torch.eye(K, dtype=dt, device=dev).expand(B, K, K)
        act = base.clone()
        conv = torch.zeros(B, dtype=torch.bool, device=dev)
        f = torch.zeros(B, K, dtype=dt, device=dev)
        it = 0
        for it in range(1, self.max_iters + 1):
            NA = Nm * act[..., None]
            Asys = I + kpa * (NA.transpose(1, 2) @ NA) @ Cc
            rhs = kpa * (NA.transpose(1, 2) @ (g * act)[..., None])
            f = torch.linalg.solve(Asys, rhs)[..., 0]
            new_act = ((g - (Nm @ (Cc @ f[..., None]))[..., 0]) > 0) & base
            conv = (new_act == act).all(1)
            if bool(conv.all()):
                break
            act = torch.where(conv[:, None], act, new_act)        # freeze envs that already converged
        f = f.clamp_min(0.0)

        fo = self._forces(f, Cc, b, cand, c)
        Fs = f.sum(1)
        if stiffness:
            # contact stiffness dF/dd with the active set frozen (dg_j/dd = 1)
            NA = Nm * act[..., None]
            Asys = I + kpa * (NA.transpose(1, 2) @ NA) @ Cc
            df = torch.linalg.solve(Asys, kpa * (NA.transpose(1, 2) @ act.to(dt)[..., None]))[..., 0]
            eps = 1e-3
            K_c = (self._forces(f + eps * df, Cc, b, cand, c)["F"] - fo["F"]) / eps
        else:
            K_c = torch.full_like(Fs, float("nan"))
        contact = Fs > 0
        K_c = torch.where(contact, K_c, torch.zeros_like(K_c))

        u_grid = None
        if field:
            u_grid = self._field(f, fr, b, es_env)

        z = torch.zeros_like(Fs)
        return BatchedFEMState(
            indentation_mm=d, reaction_force_n=torch.where(contact, fo["F"], z),
            linear_force_n=torch.where(contact, fo["Flin"], z),
            stiffening_factor=torch.where(contact, fo["phi"], torch.ones_like(z)),
            compression_mm=torch.where(contact, fo["comp"], z), contact_stiffness_n_per_mm=K_c,
            force_vec_n=fo["Fvec"] * contact[:, None], torque_nmm=fo["tau"] * contact[:, None],
            in_contact=contact, num_active_nodes=(f > 0).sum(1), active_set_iters=it,
            active_set_converged=conv, u_grid=u_grid,
            grid_origin_mm=self.lat_origin.float(),
            grid_extent_mm=torch.tensor([(m.NX - 1) * h, (m.NY - 1) * h, (m.NZ - 1) * h],
                                        dtype=torch.float32, device=dev),
        )

    def _forces(self, f, Cc, b, cand, c):
        """Batched `FEMTissueModel._forces`."""
        Fs = f.sum(1)
        u = (Cc @ f[..., None])[..., 0]
        comp = (f * u).sum(1) / Fs.clamp_min(1e-30)
        bu = self.model.beta * u.clamp_min(0.0)
        w = torch.where(bu > 1e-9, torch.expm1(bu) / bu.clamp_min(1e-12), torch.ones_like(bu))
        fw = f * w
        Fw = fw.sum(1)
        phi = Fw / Fs.clamp_min(1e-30)
        r = self.skin_ref[cand.clamp_min(0)] - c[:, None, :]
        tau = torch.cross(r, -fw[..., None] * b[:, None, :], dim=-1).sum(1)
        return {"F": Fs * phi, "Flin": Fs, "phi": phi, "comp": comp, "Fvec": -b * Fw[:, None], "tau": tau}

    def _field(self, f, fr, b, es_env):
        """u_lat = sum_c f_c (b . G_lat[c]) / e_scale for every env, as one dense GEMM."""
        m = self.model
        B = f.shape[0]
        use = (fr >= 0) & (f > 0)
        W = torch.zeros(B, 3 * self.Sf + 1, dtype=self.dt, device=self.dev)
        w3 = (f * use)[..., None] * b[:, None, :] / es_env[:, None, None]                 # (B,K,3)
        idx = torch.where(use[..., None], 3 * fr.clamp_min(0)[..., None] + torch.arange(3, device=self.dev),
                          torch.full_like(fr, 3 * self.Sf)[..., None].expand(-1, -1, 3))
        W.scatter_add_(1, idx.reshape(B, -1), w3.reshape(B, -1))
        u = W[:, :-1].to(m.G_lat.dtype) @ m.G_lat                                          # (B, 3 N_lat)
        return u.float().view(B, 3, m.NZ, m.NY, m.NX)
