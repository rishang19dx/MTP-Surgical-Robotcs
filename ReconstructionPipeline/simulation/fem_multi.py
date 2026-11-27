"""
Cross-patient batched FEM contact queries (DEC-066).

`BatchedFEMSolver` (fem_batched.py) batches the envs of ONE patient.  With many patients in one vector env, the
env looped over patients in Python, and every patient's force-control loop ran its own short GPU kernels with
host syncs (active-set convergence checks, the candidate count K).  With 32 patients the GPU sat mostly idle
(DEC-066 profile: 90 env-steps/s at 512 envs).

`MultiPatientFEMSolver([BatchedFEMSolver, ...])` stacks the per-patient contact data, zero-padded to the largest
patient, and answers the controller queries for envs of ALL patients in one batch; each env carries its patient
index `pid`:

    first_contact(pid, T), tissue_thickness(pid, T), max_indentation(pid, T, d0), contact(pid, T, d, ...)

The algebra is `BatchedFEMSolver.solve` line for line, with every per-patient tensor gathered by `pid`:
skin height map, skin columns, skin->free map, skin reference points, lattice origin, compliance C, bone top.
Padding is never read: indices are clamped to each env's own map size, and free-node indices come from each
patient's own `skin_to_free`, so padded C rows are never gathered.

Differences from `BatchedFEMSolver.solve`, neither of which changes a returned force:
  * the active-set loop checks convergence only at iterations in CHECK_AT (not every iteration).  An env whose
    active set has converged is frozen (`act` unchanged), so extra iterations recompute the same f;
  * K (padded candidate count) is the maximum over all patients in the batch, so a patient's envs may carry more
    zero padding than in a per-patient call; padded slots solve f = 0 exactly (identity rows, zero rhs).
Parity with per-patient `BatchedFEMSolver` is a test (`tests/test_fem_multi.py`).

The displacement field (one GEMM per patient: lattices differ in NY) and the warped slice stay per patient; the
env calls `BatchedFEMSolver._field` with the contact result of this class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver

CHECK_AT = (2, 4, 8, 16)   # plus the last iteration


def _pad_stack(maps: Sequence[torch.Tensor], fill=0.0) -> torch.Tensor:
    shape = [max(m.shape[i] for m in maps) for i in range(maps[0].dim())]
    out = torch.full((len(maps), *shape), fill, dtype=maps[0].dtype, device=maps[0].device)
    for p, m in enumerate(maps):
        out[(p, *[slice(0, s) for s in m.shape])] = m
    return out


def bilinear_multi(maps, nx, nz, o, s, pid, x, z):
    """Per-env bilinear lookup with the clamping of `BatchedFEMSolver._skin_height` / `core._bilinear`.
    maps (P, nz_max, nx_max); nx, nz (P,) long; o, s (P, 2); pid broadcastable to x; x, z same shape."""
    pid = pid.expand_as(x) if pid.shape != x.shape else pid
    nxe, nze = nx[pid], nz[pid]
    fx = torch.minimum(((x - o[pid, 0]) / s[pid, 0]).clamp_min(0), (nxe - 1).to(x.dtype))
    fz = torch.minimum(((z - o[pid, 1]) / s[pid, 1]).clamp_min(0), (nze - 1).to(x.dtype))
    x0 = torch.minimum(fx.floor().long().clamp_min(0), (nxe - 2).clamp_min(0))
    z0 = torch.minimum(fz.floor().long().clamp_min(0), (nze - 2).clamp_min(0))
    x1 = torch.minimum(x0 + 1, nxe - 1)
    z1 = torch.minimum(z0 + 1, nze - 1)
    tx, tz = fx - x0, fz - z0
    return (maps[pid, z0, x0] * (1 - tx) * (1 - tz) + maps[pid, z0, x1] * tx * (1 - tz)
            + maps[pid, z1, x0] * (1 - tx) * tz + maps[pid, z1, x1] * tx * tz)


class MultiPatientFEMSolver:
    def __init__(self, solvers: List[BatchedFEMSolver]):
        s0 = solvers[0]
        self.solvers = solvers
        self.dev, self.dt, self.max_iters = s0.dev, s0.dt, s0.max_iters
        m0 = s0.model
        for s in solvers[1:]:
            m = s.model
            same = (s.dt == s0.dt and s.dev == s0.dev and s.max_iters == s0.max_iters and s.kpa == s0.kpa
                    and m.h == m0.h and m.beta == m0.beta and m.max_indent_fraction == m0.max_indent_fraction
                    and m.NX == m0.NX and m.NZ == m0.NZ
                    and torch.equal(s.ls, s0.ls) and torch.equal(s.es, s0.es))
            if not same:
                raise ValueError("MultiPatientFEMSolver: patients differ in probe/mesh constants; batch them apart")
        self.ls, self.es, self.kpa, self.h = s0.ls, s0.es, s0.kpa, m0.h
        self.beta, self.max_frac = m0.beta, m0.max_indent_fraction
        self.NX, self.NZ = m0.NX, m0.NZ
        dt, dev = self.dt, self.dev
        L = torch.long
        self.hmap = _pad_stack([s.hmap for s in solvers])
        self.h_nz = torch.tensor([s.hmap.shape[0] for s in solvers], dtype=L, device=dev)
        self.h_nx = torch.tensor([s.hmap.shape[1] for s in solvers], dtype=L, device=dev)
        self.h_o = torch.tensor([s.hmap_o for s in solvers], dtype=dt, device=dev)
        self.h_s = torch.tensor([s.hmap_s for s in solvers], dtype=dt, device=dev)
        bt = [torch.as_tensor(s.model._bone_top, dtype=dt, device=dev) for s in solvers]
        self.bone_top = _pad_stack(bt)
        self.b_nz = torch.tensor([b.shape[0] for b in bt], dtype=L, device=dev)
        self.b_nx = torch.tensor([b.shape[1] for b in bt], dtype=L, device=dev)
        self.C = _pad_stack([s.C for s in solvers])                                   # (P, 3Sf_max, 3Sf_max)
        self.column_skin = torch.stack([s.column_skin for s in solvers])               # (P, NX, NZ)
        # skin-node count may differ (partial skin coverage, e.g. ts2_s0353): pad; candidates always come from a
        # patient's own skin columns, so padded skin nodes are never referenced
        self.skin_to_free = _pad_stack([s.skin_to_free for s in solvers], fill=-1)     # (P, S_max)
        self.skin_ref = _pad_stack([s.skin_ref for s in solvers])                      # (P, S_max, 3)
        self.lat_origin = torch.stack([s.lat_origin for s in solvers])                 # (P, 3)
        self.S = int(self.skin_ref.shape[1])

    # ------------------------------------------------------------ geometry
    def _skin_height(self, pid, x, z):
        return bilinear_multi(self.hmap, self.h_nx, self.h_nz, self.h_o, self.h_s, pid, x, z)

    def _normal_gap(self, pid, q, b, iters: int = 3):
        pq = pid[:, None].expand(q.shape[:2])
        cb = -b[:, 1:2]
        bb = b[:, None, :]
        t = (self._skin_height(pq, q[..., 0], q[..., 2]) - q[..., 1]) / cb
        for _ in range(iters):
            p = q - t[..., None] * bb
            t = (self._skin_height(pq, p[..., 0], p[..., 2]) - q[..., 1]) / cb
        return t, q - t[..., None] * bb

    @torch.no_grad()
    def first_contact(self, pid, poses_P, iters: int = 4):
        T = torch.as_tensor(poses_P, dtype=self.dt, device=self.dev)
        o, ex, b, ez = BatchedFEMSolver._frames(T)
        d = torch.zeros(T.shape[0], dtype=self.dt, device=self.dev)
        for _ in range(iters):
            q = ((o + d[:, None] * b)[:, None, :] + self.ls[None, :, None] * ex[:, None, :]
                 + self.es[None, :, None] * ez[:, None, :])
            g, _ = self._normal_gap(pid, q, b)
            d = d - g.max(1).values
        return torch.where(b[:, 1] > -0.2, torch.full_like(d, float("nan")), d)

    @torch.no_grad()
    def tissue_thickness(self, pid, poses_P):
        T = torch.as_tensor(poses_P, dtype=self.dt, device=self.dev)
        o, _, b, _ = BatchedFEMSolver._frames(T)
        ix = torch.minimum(torch.round((o[:, 0] - self.h_o[pid, 0]) / self.h_s[pid, 0]).long().clamp_min(0),
                           self.b_nx[pid] - 1)
        iz = torch.minimum(torch.round((o[:, 2] - self.h_o[pid, 1]) / self.h_s[pid, 1]).long().clamp_min(0),
                           self.b_nz[pid] - 1)
        return (o[:, 1] - self.bone_top[pid, iz, ix]) / (-b[:, 1]).clamp_min(1e-3)

    @torch.no_grad()
    def max_indentation(self, pid, poses_P, first_contact=None, default_mm: float = 30.0):
        t = self.tissue_thickness(pid, poses_P)
        lim = torch.where(torch.isfinite(t), self.max_frac * t, torch.full_like(t, default_mm))
        d0 = self.first_contact(pid, poses_P) if first_contact is None else first_contact
        return torch.where(torch.isfinite(d0), torch.maximum(lim, d0 + 1.0), lim)

    # ------------------------------------------------------------ contact
    @torch.no_grad()
    def contact(self, pid, poses_P, indentations_mm, e_scale, preload: Optional[torch.Tensor] = None,
                preload_margin_mm: float = 0.0, groups=None) -> Dict[str, torch.Tensor]:
        """Contact solve of `BatchedFEMSolver.solve` (no field, no stiffness) for envs of any patients.
        Returns F (B,) reaction force after stiffening, and f, fr, b, Wbg for the per-patient field.
        `groups`: list of (patient index, row indices into this batch); needed only with `preload` (one skin
        GEMM per patient); computed from `pid` (one host sync) when omitted."""
        dev, dt = self.dev, self.dt
        T = torch.as_tensor(poses_P, dtype=dt, device=dev)
        B = T.shape[0]
        d = torch.as_tensor(indentations_mm, dtype=dt, device=dev).reshape(-1).expand(B).clone()
        es_env = torch.as_tensor(e_scale, dtype=dt, device=dev).reshape(-1).expand(B).clone()
        o = T[:, :3, 3]
        ex = F.normalize(T[:, :3, 0], dim=1)
        b = F.normalize(T[:, :3, 1], dim=1)
        ez = F.normalize(T[:, :3, 2], dim=1)
        env_ok = b[:, 1] <= -0.2
        c = o + d[:, None] * b
        q = (c[:, None, :] + self.ls[None, :, None] * ex[:, None, :] + self.es[None, :, None] * ez[:, None, :])
        g, q = self._normal_gap(pid, q, b)
        pc = pid[:, None, None]
        Wbg = None
        if preload is not None:
            Wbg = preload.to(dt) / es_env[:, None]                                   # (B, 3Sf_max)
            if groups is None:
                groups = [(p, (pid == p).nonzero()[:, 0]) for p in pid.unique().tolist()]
            u_bg_skin = torch.zeros_like(Wbg)
            for p, rows in groups:
                n3 = self.solvers[p].C.shape[0]
                u_bg_skin[rows, :n3] = Wbg[rows, :n3] @ self.solvers[p].C
        margin = float(preload_margin_mm) if preload is not None else 0.0
        pen = (g > -margin) & env_ok[:, None]

        h, NX, NZ = self.h, self.NX, self.NZ
        lo = self.lat_origin[pid]
        fi = (q[..., 0] - lo[:, None, 0]) / h
        fk = (q[..., 2] - lo[:, None, 2]) / h
        i0 = fi.floor().long().clamp(0, NX - 2)
        k0 = fk.floor().long().clamp(0, NZ - 2)
        ti = (fi - i0).clamp(0, 1)
        tk = (fk - k0).clamp(0, 1)
        cs = self.column_skin
        pq = pid[:, None].expand_as(i0)
        nodes = torch.stack([cs[pq, i0, k0], cs[pq, i0 + 1, k0], cs[pq, i0, k0 + 1], cs[pq, i0 + 1, k0 + 1]], -1)
        wts = torch.stack([(1 - ti) * (1 - tk), ti * (1 - tk), (1 - ti) * tk, ti * tk], -1)
        wts = wts * (nodes >= 0)
        wsum = wts.sum(-1, keepdim=True)
        wts = torch.where(wsum > 1e-9, wts / wsum.clamp_min(1e-12), torch.zeros_like(wts))
        keep = pen[..., None] & (nodes >= 0) & (wts > 0)

        n_s = g.shape[1]
        BIG = self.S + 1
        key = torch.where(keep, nodes, torch.full_like(nodes, BIG)).view(B, -1)
        skey, order = key.sort(dim=1)
        first = torch.ones_like(skey, dtype=torch.bool)
        first[:, 1:] = skey[:, 1:] != skey[:, :-1]
        first &= skey < BIG
        slot_sorted = first.long().cumsum(1) - 1
        counts = first.sum(1)
        K = max(int(counts.max()), 1)                                                # one host sync per call
        slot = torch.empty_like(slot_sorted).scatter_(1, order, slot_sorted)
        slot = torch.where(key < BIG, slot, torch.full_like(slot, K))
        cand = torch.full((B, K + 1), -1, dtype=torch.long, device=dev)
        cand.scatter_(1, torch.where(first, slot_sorted, torch.full_like(slot_sorted, K)),
                      torch.where(first, skey, torch.full_like(skey, -1)))
        cand = cand[:, :K]
        cand_ok = cand >= 0
        pk = pid[:, None].expand_as(cand)

        Nm = torch.zeros(B, n_s, K + 1, dtype=dt, device=dev)
        Nm.scatter_add_(2, slot.view(B, n_s, 4), (wts * keep).to(dt))
        Nm = Nm[..., :K]
        has = Nm.sum(-1) > 1e-12
        fr = torch.where(cand_ok, self.skin_to_free[pk, cand.clamp_min(0)], torch.full_like(cand, -1))
        if preload is not None:
            r3b = 3 * fr.clamp_min(0)[..., None] + torch.arange(3, device=dev)
            ub = (u_bg_skin.gather(1, r3b.reshape(B, -1)).view(B, K, 3) * b[:, None, :]).sum(-1)
            ub = torch.where(fr >= 0, ub, torch.zeros_like(ub))
            g = g - (Nm @ ub[..., None])[..., 0]
            pen = pen & (g > 0)
        base = pen & has
        g = torch.where(base, g, torch.zeros_like(g))

        okf = fr >= 0
        r3 = (3 * fr.clamp_min(0)[..., None] + torch.arange(3, device=dev)).view(B, 3 * K)
        sub = self.C[pc, r3[:, :, None], r3[:, None, :]].view(B, K, 3, K, 3)
        Cc = torch.einsum("bi,baicj,bj->bac", b, sub, b) / es_env[:, None, None]
        Cc = Cc * (okf[:, :, None] & okf[:, None, :])

        kpa = self.kpa
        I = torch.eye(K, dtype=dt, device=dev).expand(B, K, K)
        act = base.clone()
        conv = torch.zeros(B, dtype=torch.bool, device=dev)
        f = torch.zeros(B, K, dtype=dt, device=dev)
        for it in range(1, self.max_iters + 1):
            NA = Nm * act[..., None]
            Asys = I + kpa * (NA.transpose(1, 2) @ NA) @ Cc
            rhs = kpa * (NA.transpose(1, 2) @ (g * act)[..., None])
            f = torch.linalg.solve(Asys, rhs)[..., 0]
            new_act = ((g - (Nm @ (Cc @ f[..., None]))[..., 0]) > 0) & base
            conv = (new_act == act).all(1)
            if (it in CHECK_AT or it == self.max_iters) and bool(conv.all()):
                break
            act = torch.where(conv[:, None], act, new_act)
        f = f.clamp_min(0.0)

        # strain stiffening as BatchedFEMSolver._forces (force magnitude only)
        Fs = f.sum(1)
        u = (Cc @ f[..., None])[..., 0]
        bu = self.beta * u.clamp_min(0.0)
        w = torch.where(bu > 1e-9, torch.expm1(bu) / bu.clamp_min(1e-12), torch.ones_like(bu))
        Fw = (f * w).sum(1)
        phi = Fw / Fs.clamp_min(1e-30)
        contact = Fs > 0
        Fr = torch.where(contact, Fs * phi, torch.zeros_like(Fs))
        return {"F": Fr, "f": f, "fr": fr, "b": b, "es": es_env, "Wbg": Wbg, "d": d, "in_contact": contact}
