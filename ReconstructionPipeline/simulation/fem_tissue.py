"""
Patient-specific, precomputed, quasi-static linear-elastic tetrahedral FEM tissue model (DEC-037).
Second backend of contract C1 (`TissueModel`, see tissue_model.py).

Method (Bro-Nielsen & Cotin 1996; Cotin, Delingette & Ayache 1999 "condensation"):
  1. `tet_mesh.build_tet_mesh`: CT soft tissue (HU > -200, not bone) on an h-mm lattice, Kuhn tets,
     bone cells + ROI deep/side faces Dirichlet.
  2. Linear elasticity, per-cell Young's modulus from HU (fat: HU < -20 -> E_fat, else E_muscle),
     Poisson ratio nu (default 0.45; nu -> 0.5 locks linear tets, so 0.45 is the documented upper limit).
  3. One-off precompute (GPU, batched Jacobi-PCG, multi-RHS): the full displacement response
     G[:, 3s+j] = K^-1 e_{s,j} to a unit force along each axis j = x, y, z of frame P at every free dorsal
     skin node s (DEC-040; before, -Y only).  From it:
       * C = 3-direction skin-node compliance (3S x 3S, float64, CPU), C[3t+j, 3s+i] = displacement i of s per
         N along j at t.  Its relative asymmetry before symmetrising (Maxwell-Betti) is stored in
         `timings["compliance_asymmetry"]` as a solver check (2.4e-6 on the sample patient, test_fem_tissue.py),
       * G_lat = the response sampled on the regular lattice (air above the skin gets the value of the
         skin node of its column, bone / empty = 0) so a field is one GEMV and directly grid_sample-able.
         Stored fp16 (on the GPU by default), 233 MB for the sample patient's L4 ROI.
     Cached on disk (`ReconstructionPipeline/.fem_cache`, gitignored) keyed by a hash of the volume,
     the ROI and all mesh/material parameters.  A uniform modulus scale `e_scale` is applied at run
     time as G / e_scale (exact for linear elasticity), so stiffness randomisation needs no recompute.
  4. Per query (CPU, small dense algebra): frictionless contact along the face normal b = beam (+Y_probe):
       * the probe face (lateral x elevation rectangle, `face_dims_mm`, default 45 x 10) is placed at
         origin + indentation * b and sampled every ~1 mm;
       * each sample's overlap g_j with the undeformed skin is measured ALONG b (fixed point on the
         sub-voxel skin height map, the env's `skin_height_map`), and its skin point p_j = q_j - g_j b;
       * samples are distributed to skin nodes with bilinear column weights at p_j; contact is a stiff
         compliant interface (skin + gel layer, `interface_stiffness_n_per_mm3`, default 0.1 N/mm^3,
         ~15x the tissue's areal stiffness) so partial footprint coverage changes the force smoothly
         instead of in 4 mm steps.  Per face sample j: lambda_j = k_p a_j (g_j - N_j u)^+ with u the
         normal (along b) skin-node displacement, u = Cb f, Cb[a,c] = b^T C_ac b; nodal forces f >= 0 act
         along b (no node is pulled, no tangential force); solved with an active set over samples;
       * displacement field u_lat = sum_c f_c (b . G_lat[c]) (GPU, one GEMV) -> `displace` /
         `warp_to_reference`.  A tilted probe therefore shears the tissue along its beam.
       For an untilted probe b = -Y_P and this is identical to the vertical-only model it replaced.
  5. Strain stiffening (APPROXIMATION, documented): linear FEM cannot reproduce the exponential
     force-indentation curve of fat/muscle.  Each nodal contact force of the linear solve is multiplied by
         phi(u_s) = (exp(beta u_s) - 1) / (beta u_s),   u_s = normal compression of skin node s (Cb f)_s
     (a local, Winkler-like correction), which turns a linear F = k d into the literature law
     F = (k/beta)(exp(beta d) - 1) when the compression is uniform.  A global factor based on the mean
     compression was rejected: it made F non-monotonic when a lightly loaded node enters contact.  The
     displacement field is NOT scaled (it is kinematically prescribed by the probe).  beta = 0 disables it.
  6. `contact_stiffness_n_per_mm` = dF/d(indentation) at the solution (frozen active set, dg_j/dd = 1), so a
     1 kHz loop can extrapolate F = F0 + K (d - d0) without calling the solver.

Limitations: small strain, reference-configuration geometry; frictionless (no tangential force, so
`force_vec_n` = -b F exactly); the contact direction is the face normal, skin points are found along
it on the undeformed surface; surface bulging into the face outside the initial penetration region is
ignored; outside the ROI the displacement is zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ReconstructionPipeline.simulation.tet_mesh import (
    CUBE_CORNERS,
    KUHN_TETS,
    TetMesh,
    bilinear_height,
    build_tet_mesh,
    default_roi,
    skin_height_map_from_patient,
    volume_hash,
)
from ReconstructionPipeline.simulation.tissue_model import SkinSurface, TissueState, tilt_cosine

_CACHE_VERSION = 4   # 4: three force directions per skin node (DEC-040)
DEFAULT_CACHE_DIR = os.environ.get(
    "MTP_FEM_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".fem_cache")
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class FEMTissueState(TissueState):
    """TissueState plus FEM-specific outputs (all in frame P, mm / N / N mm)."""
    contact_stiffness_n_per_mm: float = 0.0     # dF/d(indentation), frozen active set
    force_vec_n: np.ndarray = field(default_factory=lambda: np.zeros(3))   # force ON the probe
    torque_nmm: np.ndarray = field(default_factory=lambda: np.zeros(3))    # about the face centre
    linear_force_n: float = 0.0                 # before strain stiffening
    stiffening_factor: float = 1.0
    compression_mm: float = 0.0                 # force-weighted mean compression of active nodes
    num_active_nodes: int = 0
    active_skin_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    u_grid: Optional[torch.Tensor] = field(default=None, repr=False)        # (1,3,NZ,NY,NX)
    grid_origin_mm: Optional[np.ndarray] = None
    grid_spacing_mm: float = 0.0


# ---------------------------------------------------------------------------
# Element stiffness
# ---------------------------------------------------------------------------
def _tet_stiffness(P: np.ndarray, E: float, nu: float) -> np.ndarray:
    Dm = np.stack([P[1] - P[0], P[2] - P[0], P[3] - P[0]], 1)
    V = abs(np.linalg.det(Dm)) / 6.0
    Minv = np.linalg.inv(Dm)
    Gr = np.vstack([-Minv.sum(0), Minv])  # rows: grad N_a
    B = np.zeros((6, 12))
    for a in range(4):
        gx, gy, gz = Gr[a]
        B[:, 3 * a:3 * a + 3] = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[3:, 3:] = np.eye(3) * mu
    return V * B.T @ C @ B


class _BlasScope:
    """Re-entrant context limiting numpy/scipy OpenBLAS to one thread while a contact solve runs.

    numpy's bundled OpenBLAS busy-waits its thread pool between the many tiny (50x50) BLAS calls of the
    contact solve, which doubles the latency (measured: force_at median 1.8 ms unlimited vs 0.9 ms with
    one thread).  The limit is applied only inside this scope (torch's own BLAS is untouched), so the
    rest of the process keeps multi-threaded BLAS.  Entering costs ~0.1 ms, so callers doing many
    queries (the env's admittance loop) open one scope around all of them; nested scopes are free.
    No-op without threadpoolctl."""

    _depth = 0
    _ctl = None
    _limiter = None

    def __enter__(self):
        cls = _BlasScope
        if cls._depth == 0:
            try:
                if cls._ctl is None:
                    import threadpoolctl
                    cls._ctl = threadpoolctl.ThreadpoolController()
                cls._limiter = cls._ctl.limit(limits=1, user_api="blas")
            except Exception:
                cls._limiter = None
        cls._depth += 1
        return self

    def __exit__(self, *exc):
        cls = _BlasScope
        cls._depth -= 1
        if cls._depth == 0 and cls._limiter is not None:
            cls._limiter.restore_original_limits()
            cls._limiter = None
        return False


def _gen_solve(M: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Small dense solve through torch (CPU, float64).  numpy's bundled OpenBLAS busy-waits on its
    thread pool when torch is also loaded (measured 43 ms vs 0.3 ms for a 70x70 system)."""
    if M.shape[0] == 0:
        return np.zeros(0)
    return torch.linalg.solve(torch.from_numpy(M), torch.from_numpy(rhs)).numpy()


def _batched_pcg(A: torch.Tensor, diag: torch.Tensor, B: torch.Tensor, rtol: float, maxit: int):
    Minv = (1.0 / diag).unsqueeze(1)
    X = torch.zeros_like(B)
    R = B.clone()
    Z = Minv * R
    P = Z.clone()
    rz = (R * Z).sum(0)
    bn = B.norm(dim=0).clamp_min(1e-30)
    it = 0
    for it in range(1, maxit + 1):
        AP = A @ P
        al = rz / (P * AP).sum(0)
        X += al * P
        R -= al * AP
        if it % 10 == 0 and bool((R.norm(dim=0) < rtol * bn).all()):
            break
        Z = Minv * R
        rzn = (R * Z).sum(0)
        P = Z + (rzn / rz) * P
        rz = rzn
    rel = float((R.norm(dim=0) / bn).max())
    return X, it, rel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FEMTissueModel:
    """Precomputed condensed linear FEM implementing `TissueModel` (see module docstring)."""

    def __init__(
        self,
        patient_data: Dict[str, Any],
        roi_min_mm: Optional[np.ndarray] = None,
        roi_max_mm: Optional[np.ndarray] = None,
        element_size_mm: float = 4.0,
        e_fat_kpa: float = 10.0,
        e_muscle_kpa: float = 20.0,
        fat_hu_threshold: float = -20.0,
        poissons_ratio: float = 0.45,
        e_scale: float = 1.0,
        beta_per_mm: float = 0.15,
        face_dims_mm: Tuple[float, float] = (45.0, 10.0),
        face_sample_mm: float = 1.0,
        interface_stiffness_n_per_mm3: float = 0.1,
        warp_iterations: int = 5,
        max_indent_fraction: float = 0.5,
        body_hu: float = -200.0,
        bone_hu: float = 250.0,
        pcg_rtol: float = 1e-5,
        pcg_batch: int = 128,
        field_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.nu = float(poissons_ratio)
        if self.nu > 0.45 + 1e-9:
            print(f"[FEMTissueModel] warning: nu={self.nu} > 0.45 -> volumetric locking of linear tets (over-stiff).")
        self.e_fat = float(e_fat_kpa)
        self.e_muscle = float(e_muscle_kpa)
        self.fat_hu = float(fat_hu_threshold)
        self.e_scale = float(e_scale)
        self.beta = float(beta_per_mm)
        self.face_dims = (float(face_dims_mm[0]), float(face_dims_mm[1]))
        self.face_sample = float(face_sample_mm)
        self.k_p = float(interface_stiffness_n_per_mm3)
        self.warp_iterations = int(warp_iterations)
        self.max_indent_fraction = float(max_indent_fraction)
        if field_dtype is None:   # fp16 halves the 3-direction field on the GPU (DEC-040); CPU matmul wants fp32
            field_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.field_dtype = field_dtype
        self.timings: Dict[str, float] = {}
        self._pcg_rtol = float(pcg_rtol)
        self._pcg_batch = int(pcg_batch)

        if roi_min_mm is None or roi_max_mm is None:
            lo, hi = default_roi(patient_data)
            roi_min_mm = lo if roi_min_mm is None else roi_min_mm
            roi_max_mm = hi if roi_max_mm is None else roi_max_mm
        self.roi_min = np.asarray(roi_min_mm, dtype=np.float64)
        self.roi_max = np.asarray(roi_max_mm, dtype=np.float64)

        t0 = time.perf_counter()
        self.mesh: TetMesh = build_tet_mesh(patient_data, self.roi_min, self.roi_max, h=element_size_mm,
                                            body_hu=body_hu, bone_hu=bone_hu)
        self.timings["mesh_s"] = time.perf_counter() - t0
        m = self.mesh
        self.h = m.h
        self.NX, self.NY, self.NZ = m.lattice_shape

        # skin geometry
        self.hmap, self.hmap_o, self.hmap_s = skin_height_map_from_patient(patient_data)
        self.skin_surface = SkinSurface(self.hmap, self.hmap_o, self.hmap_s)
        self.skin_ref = m.skin_vertices_ref                     # (S,3)
        self.skin_fixed = m.fixed[m.skin_nodes]
        self._bone_top = self._bone_top_map(patient_data)

        key_params = dict(v=_CACHE_VERSION, vol=volume_hash(patient_data), roi=[*self.roi_min, *self.roi_max],
                          h=self.h, nu=self.nu, ef=self.e_fat, em=self.e_muscle, fhu=self.fat_hu,
                          bhu=body_hu, bone=bone_hu, rtol=pcg_rtol)
        self.cache_key = hashlib.sha1(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:20]
        loaded = False
        path = None
        if cache_dir:
            path = os.path.join(cache_dir, f"fem_{self.cache_key}.pt")
            if os.path.exists(path):
                t0 = time.perf_counter()
                blob = torch.load(path, map_location="cpu", weights_only=True)
                self.C_base = blob["C"].numpy()
                G_lat = blob["G_lat"]
                self.free_skin = blob["free_skin"].numpy()
                self.timings["cache_load_s"] = time.perf_counter() - t0
                self.timings.update({k: float(v) for k, v in blob.get("timings", {}).items()})
                loaded = True
        if not loaded:
            G_lat = self._precompute()
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"C": torch.from_numpy(self.C_base), "G_lat": G_lat.half(),
                            "free_skin": torch.from_numpy(self.free_skin),
                            "timings": {k: v for k, v in self.timings.items() if k != "mesh_s"},
                            "params": json.dumps(key_params)}, path)
        self.cache_path = path
        self.loaded_from_cache = loaded
        # free-skin column lookup: skin index -> row in G_lat / C, -1 for fixed skin nodes
        self.skin_to_free = -np.ones(len(m.skin_nodes), dtype=np.int64)
        self.skin_to_free[self.free_skin] = np.arange(len(self.free_skin))
        self.G_lat = G_lat.to(self.device, self.field_dtype).contiguous()   # (Sf, 3*NZ*NY*NX)
        self._lat_origin_t = torch.tensor(m.lattice_origin, dtype=torch.float32, device=self.device)
        self._lat_ext_t = torch.tensor([(self.NX - 1) * self.h, (self.NY - 1) * self.h, (self.NZ - 1) * self.h],
                                       dtype=torch.float32, device=self.device)
        self._cache_last: Optional[Tuple[bytes, float, FEMTissueState]] = None
        if self.verbose:
            src = "cache" if loaded else "precompute"
            print(f"[FEMTissueModel] {m.summary()}; {len(self.free_skin)} free skin columns ({src}); "
                  f"G_lat {self.G_lat.numel() * self.G_lat.element_size() / 1e6:.0f} MB on {self.device}")

    # ------------------------------------------------------------------ build
    def _bone_top_map(self, patient_data) -> np.ndarray:
        """Bone top height per (z,x) column of the CT (nan where no bone), for tissue thickness."""
        lbl = patient_data.get("label_volume")
        ct = np.asarray(patient_data["ct_volume"])
        bone = ct > 250.0
        if lbl is not None:
            bone = bone | (np.asarray(lbl) == 1)
        ny = bone.shape[1]
        top = ny - 1 - np.argmax(bone[:, ::-1, :], axis=1)
        oy, sy = float(patient_data["origin_mm"][1]), float(patient_data["spacing_mm"][1])
        return np.where(bone.any(axis=1), oy + top * sy, np.nan)

    def _precompute(self) -> torch.Tensor:
        m = self.mesh
        dev = self.device
        t0 = time.perf_counter()
        import scipy.sparse as sps
        corners = CUBE_CORNERS.astype(np.float64) * self.h
        Kref = np.stack([_tet_stiffness(corners[k], 1.0, self.nu) for k in KUHN_TETS])  # (6,12,12), E=1
        E_cell = np.where(m.cell_hu < self.fat_hu, self.e_fat, self.e_muscle) * 1e-3     # kPa -> N/mm^2
        Ke = Kref[m.tet_ref] * E_cell[m.tet_cell][:, None, None]
        dofs = (m.tets[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
        rows = np.repeat(dofs, 12, axis=1).ravel()
        cols = np.tile(dofs, (1, 12)).ravel()
        n = 3 * m.num_nodes
        K = sps.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
        free_dof = np.repeat(~m.fixed, 3)
        fidx = np.nonzero(free_dof)[0]
        Kff = K[fidx][:, fidx].tocsr()
        Kff.sort_indices()
        nf = Kff.shape[0]
        A = torch.sparse_csr_tensor(torch.from_numpy(Kff.indptr.astype(np.int64)),
                                    torch.from_numpy(Kff.indices.astype(np.int64)),
                                    torch.from_numpy(Kff.data.astype(np.float32)), size=(nf, nf)).to(dev)
        diag = torch.from_numpy(Kff.diagonal().astype(np.float32)).to(dev)
        self.timings["assembly_s"] = time.perf_counter() - t0
        self.timings["free_dofs"] = float(nf)
        self.timings["nnz"] = float(Kff.nnz)

        # free-dof row of each node's y dof
        dof_to_free = -np.ones(n, dtype=np.int64)
        dof_to_free[fidx] = np.arange(nf)
        free_skin = np.nonzero(~self.skin_fixed)[0]
        self.free_skin = free_skin
        sk_nodes = m.skin_nodes[free_skin]
        # one unit-force RHS per (free skin node t, direction j), ordered 3 t + j (x, y, z of frame P)
        rhs_rows = dof_to_free[(sk_nodes[:, None] * 3 + np.arange(3)[None, :]).ravel()]

        # lattice gather map: lattice node -> mesh node (itself / column skin node above skin / -1)
        NX, NY, NZ = self.NX, self.NY, self.NZ
        lat_src = -np.ones((NX, NY, NZ), dtype=np.int64)
        ijk = m.node_ijk
        lat_src[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(m.num_nodes)
        jj = np.arange(NY)[None, :, None]
        skin_j = -np.ones((NX, NZ), dtype=np.int64)
        skin_node_of_col = -np.ones((NX, NZ), dtype=np.int64)
        sik = m.skin_column_ik
        skin_j[sik[:, 0], sik[:, 1]] = ijk[m.skin_nodes, 1]
        skin_node_of_col[sik[:, 0], sik[:, 1]] = m.skin_nodes
        above = (jj > skin_j[:, None, :]) & (skin_j[:, None, :] >= 0)
        lat_src = np.where(above, np.broadcast_to(skin_node_of_col[:, None, :], lat_src.shape), lat_src)
        # order (z, y, x) for grid_sample (D=z, H=y, W=x)
        lat_src_zyx = torch.from_numpy(np.ascontiguousarray(lat_src.transpose(2, 1, 0)).ravel()).to(dev)
        valid = lat_src_zyx >= 0
        src_c = lat_src_zyx.clamp_min(0)
        free_of_node_dof = torch.from_numpy(dof_to_free).to(dev)

        Sf3 = 3 * len(free_skin)
        Nlat = NX * NY * NZ
        G_lat = torch.zeros(Sf3, 3 * Nlat, dtype=torch.float16)
        C = np.zeros((Sf3, Sf3), dtype=np.float64)
        rows_t = torch.from_numpy(rhs_rows).to(dev)
        t1 = time.perf_counter()
        iters, worst = [], 0.0
        self_batch = max(1, int(self._pcg_batch))
        for s0 in range(0, Sf3, self_batch):
            sl = slice(s0, min(Sf3, s0 + self_batch))
            nb = sl.stop - sl.start
            B = torch.zeros(nf, nb, device=dev)
            B[rows_t[sl], torch.arange(nb, device=dev)] = 1.0       # unit force along +x / +y / +z
            X, it, rel = _batched_pcg(A, diag, B, rtol=self._pcg_rtol, maxit=5000)
            iters.append(it)
            worst = max(worst, rel)
            # skin compliance C[3t+j, 3s+i] = displacement i of skin node s per unit force j at t
            C[sl, :] = X[rows_t, :].t().double().cpu().numpy()
            # full nodal field (3 comps) -> lattice
            Xp = torch.cat([X, torch.zeros(1, nb, device=dev)], 0)   # last row = zero (fixed dof)
            for c in range(3):
                fr = free_of_node_dof[src_c * 3 + c]
                fr = torch.where(valid & (fr >= 0), fr, torch.full_like(fr, nf))
                G_lat[sl, c * Nlat:(c + 1) * Nlat] = Xp[fr, :].t().half().cpu()
            del X, B, Xp
        if dev.type == "cuda":
            torch.cuda.synchronize()
        self.timings["pcg_s"] = time.perf_counter() - t1
        self.timings["pcg_iters_max"] = float(max(iters) if iters else 0)
        self.timings["pcg_worst_rel_residual"] = worst
        # Maxwell-Betti: K symmetric -> C symmetric; the residual asymmetry measures assembly + PCG error
        self.timings["compliance_asymmetry"] = float(np.linalg.norm(C - C.T) / np.linalg.norm(C))
        self.C_base = 0.5 * (C + C.T)
        del A
        return G_lat

    # --------------------------------------------------------------- params
    def set_parameters(self, e_scale: Optional[float] = None, beta_per_mm: Optional[float] = None):
        if e_scale is not None:
            self.e_scale = float(e_scale)
        if beta_per_mm is not None:
            self.beta = float(beta_per_mm)
        self._cache_last = None

    @property
    def effective_moduli_kpa(self) -> Tuple[float, float]:
        return self.e_fat * self.e_scale, self.e_muscle * self.e_scale

    def first_contact_indentation(self, probe_pose_P: np.ndarray, iters: int = 4) -> float:
        """Indentation (mm along the beam, may be negative) at which the face first touches the
        undeformed skin.  Negative where the face bridges a concavity (e.g. a transverse probe over
        the midline groove touches the paraspinal bulges before the skin under its centre)."""
        T = np.asarray(probe_pose_P, dtype=np.float64)
        o = T[:3, 3]
        ex = T[:3, 0] / np.linalg.norm(T[:3, 0])
        b = T[:3, 1] / np.linalg.norm(T[:3, 1])
        ez = T[:3, 2] / np.linalg.norm(T[:3, 2])
        if b[1] > -0.2:
            return float("nan")
        ls, es, _ = self._face_samples()
        d = 0.0
        for _ in range(iters):
            q = (o + d * b)[None, :] + ls[:, None] * ex[None, :] + es[:, None] * ez[None, :]
            g, _ = self._normal_gap(q, b)
            d -= float(g.max())
        return d

    def _normal_gap(self, q: np.ndarray, b: np.ndarray, iters: int = 3):
        """Overlap of the undeformed skin with face samples `q`, measured along the beam b (the face normal):
        g = t such that q - t b lies on the skin (t > 0: the sample is inside the tissue).  Returns (g, p) with
        p = q - g b the skin point the sample presses on.  Fixed-point on the height map; exact for flat skin
        after one step, and it converges for skin slope x tan(tilt) < 1."""
        cb = -b[1]
        t = (self.skin_height(q[:, 0], q[:, 2]) - q[:, 1]) / cb
        for _ in range(iters):
            p = q - t[:, None] * b[None, :]
            t = (self.skin_height(p[:, 0], p[:, 2]) - q[:, 1]) / cb
        return t, q - t[:, None] * b[None, :]

    def tissue_thickness(self, probe_pose_P: np.ndarray) -> float:
        """Soft-tissue thickness (mm) along the beam from the skin under the probe centre to the first
        bone below it (vertical bone-top map / cos tilt); nan where no bone lies below."""
        T = np.asarray(probe_pose_P, dtype=np.float64)
        o, b = T[:3, 3], T[:3, 1] / np.linalg.norm(T[:3, 1])
        bt = self._bone_top_at(o[0], o[2])
        return float((o[1] - bt) / max(-b[1], 1e-3)) if np.isfinite(bt) else float("nan")

    def max_indentation(self, probe_pose_P: np.ndarray, default_mm: float = 30.0) -> float:
        """Upper indentation limit for the admittance controller (same convention as the Kelvinlet
        backend): `max_indent_fraction` x tissue thickness under the probe centre (small-strain range of
        the linear model), never below first contact + 1 mm so contact stays reachable."""
        t = self.tissue_thickness(probe_pose_P)
        lim = self.max_indent_fraction * t if np.isfinite(t) else default_mm
        d0 = self.first_contact_indentation(probe_pose_P)
        return float(max(lim, d0 + 1.0)) if np.isfinite(d0) else float(lim)

    def calibrate_e_scale(self, probe_poses_P, k0_n_per_mm: float = 0.5, beta_per_mm: float = 0.15,
                          force_range_n: Tuple[float, float] = (2.0, 8.0), n_points: int = 7) -> float:
        """Set e_scale (and beta) so that, averaged over `probe_poses_P` (one 4x4 or a list), F(d0 + d)
        follows the literature curve F = (k0/beta)(exp(beta d) - 1) in a log-least-squares sense over
        the d giving `force_range_n` on that curve; d0 = per-pose first-contact indentation (the
        literature curve is defined from first contact).  Only the global modulus scale is fitted, so
        the spread between poses is anatomy (tissue thickness, bone, skin shape)."""
        poses = [probe_poses_P] if np.ndim(probe_poses_P) == 2 else list(probe_poses_P)
        self.set_parameters(beta_per_mm=beta_per_mm)
        f_lo, f_hi = force_range_n
        d_lo, d_hi = (np.log1p(beta_per_mm * f / k0_n_per_mm) / beta_per_mm for f in (f_lo, f_hi))
        ds = np.linspace(d_lo, d_hi, n_points)
        target = k0_n_per_mm / beta_per_mm * np.expm1(beta_per_mm * ds)
        d0s = [self.first_contact_indentation(T) for T in poses]

        def log_err():
            Fm = np.array([[self.force_at(T, d0 + d) for d in ds] for T, d0 in zip(poses, d0s)])
            if np.any(Fm <= 0):
                raise RuntimeError("a calibration pose is not in contact")
            return np.log(target)[None, :] - np.log(Fm)

        for _ in range(10):
            corr = float(np.exp(log_err().mean()))
            self.set_parameters(e_scale=self.e_scale * corr)
            if abs(corr - 1.0) < 1e-4:
                break
        err = log_err()
        self.calibration = {"first_contact_mm": d0s, "k0": k0_n_per_mm, "beta": beta_per_mm,
                            "e_scale": self.e_scale, "indentations_mm": ds.tolist(),
                            "rms_log_error_per_pose": np.sqrt((err ** 2).mean(1)).tolist(),
                            "mean_log_error_per_pose": err.mean(1).tolist()}
        return self.e_scale

    def window_poses(self, c1_bounds=(-32.0, 32.0), c2_bounds=(-55.0, -10.0), n: int = 3):
        """n x n untilted transverse poses on the skin over a scan window (default: env L4 window)."""
        xs = np.linspace(c1_bounds[0], c1_bounds[1], n + 2)[1:-1]
        zs = np.linspace(c2_bounds[0], c2_bounds[1], n + 2)[1:-1]
        return [self.dorsal_pose(x, z) for z in zs for x in xs]

    @classmethod
    def from_patient_data(cls, patient_data: Dict[str, Any], calibrate: bool = True,
                          c1_bounds=(-32.0, 32.0), c2_bounds=(-55.0, -10.0), k0_n_per_mm: float = 0.5,
                          **kwargs) -> "FEMTissueModel":
        """Build (or load from cache) and, by default, calibrate the global modulus on a 3x3 grid of
        untilted poses over the scan window so the window on average follows the literature curve.
        Unless given, the mesh ROI is `default_roi` around this scan window (30 mm footprint margin)."""
        if kwargs.get("roi_min_mm") is None or kwargs.get("roi_max_mm") is None:
            lo, hi = default_roi(patient_data, c1_bounds=tuple(c1_bounds), c2_bounds=tuple(c2_bounds))
            kwargs["roi_min_mm"] = lo if kwargs.get("roi_min_mm") is None else kwargs["roi_min_mm"]
            kwargs["roi_max_mm"] = hi if kwargs.get("roi_max_mm") is None else kwargs["roi_max_mm"]
        model = cls(patient_data, **kwargs)
        if calibrate:
            with _BlasScope():
                model.calibrate_e_scale(model.window_poses(c1_bounds, c2_bounds), k0_n_per_mm=k0_n_per_mm,
                                        beta_per_mm=model.beta)
        return model

    # ------------------------------------------------------------- geometry
    def skin_height(self, x, z):
        return bilinear_height(self.hmap, self.hmap_o, self.hmap_s, x, z)

    def dorsal_pose(self, x: float, z: float, yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
        """Probe pose on the undeformed skin at (x, z), same convention as env._build_probe_matrix."""
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        r_base = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        T = np.eye(4)
        T[:3, :3] = r_yaw @ r_pitch @ r_base
        T[:3, 3] = [x, float(self.skin_height(x, z)), z]
        return T

    def _face_samples(self):
        L, W = self.face_dims
        nl = max(2, int(round(L / self.face_sample)))
        ne = max(2, int(round(W / self.face_sample)))
        ls = (np.arange(nl) + 0.5) * (L / nl) - L / 2
        es = (np.arange(ne) + 0.5) * (W / ne) - W / 2
        LL, EE = np.meshgrid(ls, es, indexing="ij")
        return LL.ravel(), EE.ravel(), (L / nl) * (W / ne)

    # ---------------------------------------------------------------- solve
    def _contact(self, probe_pose_P: np.ndarray, indentation_mm: float) -> Dict[str, Any]:
        T = np.asarray(probe_pose_P, dtype=np.float64)
        o = T[:3, 3]
        ex = T[:3, 0] / np.linalg.norm(T[:3, 0])
        b = T[:3, 1] / np.linalg.norm(T[:3, 1])
        ez = T[:3, 2] / np.linalg.norm(T[:3, 2])
        d = float(indentation_mm)
        out = {"o": o, "b": b, "d": d, "f": np.zeros(0), "act": np.zeros(0, dtype=np.int64)}
        if b[1] > -0.2:      # beam not pointing into the dorsal skin
            return out
        c = o + d * b
        ls, es, da = self._face_samples()
        q = c[None, :] + ls[:, None] * ex[None, :] + es[:, None] * ez[None, :]
        g, q = self._normal_gap(q, b)                            # q <- skin point under each sample
        pen = g > 0
        if not pen.any():
            return out
        q, g = q[pen], g[pen]
        a_j = da                                                 # face-plane area (force is along the normal)
        m = self.mesh
        fi = (q[:, 0] - m.lattice_origin[0]) / self.h
        fk = (q[:, 2] - m.lattice_origin[2]) / self.h
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, self.NX - 2)
        k0 = np.clip(np.floor(fk).astype(np.int64), 0, self.NZ - 2)
        ti = np.clip(fi - i0, 0, 1)
        tk = np.clip(fk - k0, 0, 1)
        nodes, wts = [], []
        for di, dk, w in ((0, 0, (1 - ti) * (1 - tk)), (1, 0, ti * (1 - tk)), (0, 1, (1 - ti) * tk), (1, 1, ti * tk)):
            nodes.append(m.column_skin[i0 + di, k0 + dk])
            wts.append(w)
        nodes = np.stack(nodes, 1)
        wts = np.stack(wts, 1) * (nodes >= 0)
        wsum = wts.sum(1, keepdims=True)
        ok = wsum[:, 0] > 1e-9
        wts = np.where(ok[:, None], wts / np.maximum(wsum, 1e-12), 0.0)
        # sample -> candidate-node interpolation matrix N (n_samples x n_cand)
        ns = len(g)
        nodes_f = nodes.ravel()
        w_f = wts.ravel()
        keep = (nodes_f >= 0) & (w_f > 0)
        cand, inv = np.unique(nodes_f[keep], return_inverse=True)
        Nm = np.zeros((ns, len(cand)))
        np.add.at(Nm, (np.repeat(np.arange(ns), 4)[keep], inv), w_f[keep])
        has = Nm.sum(1) > 1e-12
        Nm, g = Nm[has], g[has]
        # normal compliance among candidates, Cb[a,c] = b^T C_ac b (fixed skin nodes have zero compliance)
        fr = self.skin_to_free[cand]
        Cc = np.zeros((len(cand), len(cand)))
        okf = fr >= 0
        if okf.any():
            r3 = (3 * fr[okf][:, None] + np.arange(3)[None, :]).ravel()
            n = int(okf.sum())
            sub = self.C_base[np.ix_(r3, r3)].reshape(n, 3, n, 3)
            Cc[np.ix_(okf, okf)] = np.einsum("i,aicj,j->ac", b, sub, b) / self.e_scale
        kpa = self.k_p * a_j
        # Frictionless penalty contact per face sample j, along the face normal b:
        #   lambda_j = k_p a_j (g_j - N_j u)^+,  nodal forces f = N^T lambda (along b),  u = b.(nodal disp) = Cb f.
        # On a fixed active sample set A:  (I + N_A^T k N_A C) f = N_A^T k g_A.  Iterate the set until
        # consistent (semi-smooth Newton on a monotone piecewise-linear problem; converges in a few steps).
        act = np.ones(len(g), dtype=bool)
        I = np.eye(len(cand))
        f = np.zeros(len(cand))
        for _ in range(30):
            NA = Nm[act]
            f = _gen_solve(I + (kpa * (NA.T @ NA)) @ Cc, kpa * (NA.T @ g[act]))
            new_act = g - Nm @ (Cc @ f) > 0
            if np.array_equal(new_act, act):
                break
            act = new_act
        f = np.maximum(f, 0.0)
        out.update(cand=cand, f=f, act=act, Cc=Cc, Nm=Nm, kpa=kpa, ex=ex, ez=ez, c=c)
        return out

    def _forces(self, ct: Dict[str, Any], f: np.ndarray) -> Dict[str, Any]:
        b = ct["b"]
        Fs = float(f.sum())
        if Fs <= 0:
            return {"F": 0.0, "Flin": 0.0, "phi": 1.0, "comp": 0.0, "Fvec": np.zeros(3), "tau": np.zeros(3)}
        u = ct["Cc"] @ f
        comp = float(f @ u / Fs)
        bu = self.beta * np.maximum(u, 0.0)
        w = np.where(bu > 1e-9, np.expm1(bu) / np.maximum(bu, 1e-12), 1.0)   # per-node stiffening
        fw = f * w
        Fw = float(fw.sum())
        phi = Fw / Fs
        Flin = Fs
        Fvec = -b * Fw                                             # tissue pushes the probe back along -b
        r = self.skin_ref[ct["cand"]] - ct["c"][None, :]
        tau = np.cross(r, -fw[:, None] * b[None, :]).sum(0)
        return {"F": Flin * phi, "Flin": Flin, "phi": phi, "comp": comp, "Fvec": Fvec, "tau": tau}

    def _stiffness(self, ct: Dict[str, Any], F0: float) -> float:
        """dF/d(indentation) with the active sample set frozen; the normal gap grows 1:1, dg_j/dd = 1."""
        if len(ct["f"]) == 0 or F0 <= 0:
            return 0.0
        NA = ct["Nm"][ct["act"]]
        I = np.eye(len(ct["f"]))
        D = ct["kpa"] * (NA.T @ NA)
        df = _gen_solve(I + D @ ct["Cc"], ct["kpa"] * (NA.T @ np.ones(len(NA))))
        eps = 1e-3
        return (self._forces(ct, ct["f"] + eps * df)["F"] - F0) / eps

    @staticmethod
    def blas_scope() -> _BlasScope:
        """Context limiting OpenBLAS to one thread (see `_BlasScope`); wrap batches of queries in it."""
        return _BlasScope()

    def force_at(self, probe_pose_P: np.ndarray, indentation_mm: float) -> float:
        """Reaction force (N, along the probe axis); CPU only, no field evaluation."""
        key = np.asarray(probe_pose_P, dtype=np.float64).tobytes()
        if self._cache_last is not None and self._cache_last[0] == key and self._cache_last[1] == float(indentation_mm):
            return self._cache_last[2].reaction_force_n
        with _BlasScope():
            ct = self._contact(probe_pose_P, indentation_mm)
            if len(ct["f"]) == 0:
                return 0.0
            return self._forces(ct, ct["f"])["F"]

    def solve(self, probe_pose_P: np.ndarray, indentation_mm: float) -> FEMTissueState:
        with _BlasScope():
            return self._solve(probe_pose_P, indentation_mm)

    def _solve(self, probe_pose_P: np.ndarray, indentation_mm: float) -> FEMTissueState:
        T = np.asarray(probe_pose_P, dtype=np.float64)
        key = T.tobytes()
        if self._cache_last is not None and self._cache_last[0] == key and self._cache_last[1] == float(indentation_mm):
            return self._cache_last[2]
        ct = self._contact(T, indentation_mm)
        o, b = ct["o"], ct["b"]
        cos_t = tilt_cosine(b, self.skin_surface.normal(o[0], o[2]))
        bone_depth = self.tissue_thickness(T)
        f = ct["f"]
        if len(f) == 0 or f.sum() <= 0:
            st = FEMTissueState(indentation_mm=float(indentation_mm), reaction_force_n=0.0, in_contact=False,
                                contact_point_mm=o.copy(), beam_dir=b.copy(), stiffness_n_per_mm=0.0,
                                bone_depth_mm=bone_depth, cos_tilt=cos_t, warp_iterations=self.warp_iterations)
            self._cache_last = (key, float(indentation_mm), st)
            return st
        fo = self._forces(ct, f)
        K = self._stiffness(ct, fo["F"])
        # displacement field (GPU GEMV over the active free skin columns)
        cand = ct["cand"]
        fr = self.skin_to_free[cand]
        use = (fr >= 0) & (f > 0)
        rows = torch.from_numpy((3 * fr[use][:, None] + np.arange(3)[None, :]).ravel()).to(self.device)
        w3 = (f[use][:, None] * b[None, :]).ravel() / self.e_scale       # nodal force vectors f_c b
        fw = torch.from_numpy(w3).to(self.device, self.field_dtype)
        u_flat = fw @ self.G_lat.index_select(0, rows)
        u_grid = u_flat.float().view(1, 3, self.NZ, self.NY, self.NX)
        disp = self._make_displacement_fn(u_grid)
        st = FEMTissueState(
            indentation_mm=float(indentation_mm), reaction_force_n=float(fo["F"]), in_contact=True,
            contact_point_mm=o.copy(), beam_dir=b.copy(), stiffness_n_per_mm=float(K), bone_depth_mm=bone_depth,
            cos_tilt=cos_t, _displacement_fn=disp, warp_iterations=self.warp_iterations,
            contact_stiffness_n_per_mm=float(K), force_vec_n=fo["Fvec"], torque_nmm=fo["tau"],
            linear_force_n=float(fo["Flin"]), stiffening_factor=float(fo["phi"]), compression_mm=float(fo["comp"]),
            num_active_nodes=int((f > 0).sum()), active_skin_nodes=cand[f > 0],
            u_grid=u_grid, grid_origin_mm=self.mesh.lattice_origin.copy(), grid_spacing_mm=self.h,
        )
        self._cache_last = (key, float(indentation_mm), st)
        return st

    def _bone_top_at(self, x: float, z: float) -> float:
        nz, nx = self._bone_top.shape
        ix = int(np.clip(np.rint((x - self.hmap_o[0]) / self.hmap_s[0]), 0, nx - 1))
        iz = int(np.clip(np.rint((z - self.hmap_o[1]) / self.hmap_s[1]), 0, nz - 1))
        return float(self._bone_top[iz, ix])

    def _make_displacement_fn(self, u_grid: torch.Tensor):
        org, ext = self._lat_origin_t, self._lat_ext_t

        def disp(X: torch.Tensor) -> torch.Tensor:
            X = torch.as_tensor(X).to(self.device, torch.float32)
            shp = X.shape
            g = (2.0 * (X.reshape(-1, 3) - org) / ext - 1.0).view(1, -1, 1, 1, 3)
            u = F.grid_sample(u_grid, g, mode="bilinear", padding_mode="border", align_corners=True)
            return u.view(3, -1).t().reshape(shp)

        return disp

    # ------------------------------------------------------------ rendering
    def skin_surface_mesh(self, state: Optional[TissueState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(vertices (S,3) P mm, triangles (T,3)) of the dorsal skin patch, deformed by `state` if given."""
        V = self.skin_ref
        if state is not None and state.in_contact:
            V = state.displace(torch.from_numpy(V).float().to(self.device)).cpu().numpy().astype(np.float64)
        return V, self.mesh.skin_tris
