"""
Soft-tissue model interface (contract C1) and the regularized-Kelvinlet backend.

Frames & units
--------------
All quantities live in the simulation patient frame P (the CT array frame as loaded by
`PatientAssetGenerator`; +Y points toward the dorsal skin, +Z cranio-caudal, +X lateral).
P is NOT true RAS (see contract C7).  Units: mm, N, kPa.

A probe pose `probe_pose_P` is a 4x4 T_P<-U whose origin is the UNDEFORMED skin contact point
under the probe centre, with columns  e_x = lateral array axis, b = +Y_probe = beam direction
(pointing into the body), e_z = elevation axis.  The probe face is pushed `indentation_mm`
along b beyond that point; the image plane therefore starts at origin + indentation * b.

Kelvinlet backend (KelvinletTissueModel)
----------------------------------------
* Field shape: `num_kelvinlets` regularized Kelvinlets (exact de Goes & James 2017 kernel, with
  a Boussinesq free-surface factor, see kelvinlet_deformation.py) spread along the probe aperture
  to mimic a line/rectangular punch, all loading along b.
* Bone rigidity: the first bone voxel along the beam is found by ray-marching the bone mask
  (label == 1, or CT > 300 HU if no labels) from every point of a probe-frame grid.  The soft layer
  above the bone is treated as bonded to a rigid substrate using Steinbrenner's layered-settlement
  approximation:  u(X) = u_hs(X) - u_hs(X_b), X_b = X moved along b to the bone, and u = 0 at and
  beyond the bone.  Bone voxels therefore never move, and thin layers (e.g. over the spinous
  process) are stiffer than thick ones.
* Force: the linear contact stiffness k_lin = 1 / C, where C is the mean beam-axis face
  displacement per newton of the bone-corrected multi-Kelvinlet field (depends on mu, nu, the
  probe footprint and the local tissue thickness).  Fat/muscle strain stiffening is modelled with
  the exponential law recommended for paraspinal tissue (research_literature.md, section 2):
      F(d_n) = (k_lin / beta) (exp(beta d_n) - 1),   d_n = d cos(tilt)
  where tilt is the angle between the beam and the local inward skin normal.
  With the default mu the reference configuration (35 mm of tissue over bone, no tilt) has
  k_lin ~= 0.5 N/mm, giving ~3 mm at 2 N, ~6 mm at 5 N and ~9 mm at 10 N.
* Indentation is limited to `max_indent_fraction` (0.5) of the local tissue thickness.
* Inverse warp: x = X + u(X) is inverted with `warp_iterations` fixed-point iterations on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from ReconstructionPipeline.simulation.kelvinlet_deformation import (
    kelvinlet_constants,
    regularized_kelvinlet,
)

_NO_BONE_MM = 1.0e4  # sentinel depth meaning "no bone along this ray"


# ---------------------------------------------------------------------------
# Contract C1
# ---------------------------------------------------------------------------
@dataclass
class TissueState:
    indentation_mm: float
    reaction_force_n: float
    in_contact: bool
    contact_point_mm: np.ndarray
    beam_dir: np.ndarray
    stiffness_n_per_mm: float = float("nan")   # tangent stiffness dF/dd at the solution
    bone_depth_mm: float = float("nan")        # tissue thickness along the beam under the probe centre
    cos_tilt: float = 1.0
    _displacement_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = field(default=None, repr=False)
    warp_iterations: int = 5

    def displace(self, X_ref: torch.Tensor) -> torch.Tensor:
        """(N,3) reference points -> (N,3) deformed positions x = X + u(X)."""
        if not self.in_contact or self._displacement_fn is None:
            return X_ref
        return X_ref + self._displacement_fn(X_ref)

    def warp_to_reference(self, x_deformed: torch.Tensor) -> torch.Tensor:
        """(N,3) deformed points -> (N,3) reference points, inverting x = X + u(X) by fixed point."""
        if not self.in_contact or self._displacement_fn is None:
            return x_deformed
        X = x_deformed
        for _ in range(self.warp_iterations):
            X = x_deformed - self._displacement_fn(X)
        return X

    def probe_face_pose(self, probe_pose_P: np.ndarray) -> np.ndarray:
        """Pose of the (indented) probe face = image-plane pose used for slicing/back-projection."""
        T = np.array(probe_pose_P, dtype=np.float64, copy=True)
        T[:3, 3] = T[:3, 3] + self.indentation_mm * T[:3, 1]
        return T


@runtime_checkable
class TissueModel(Protocol):
    def solve(self, probe_pose_P: np.ndarray, indentation_mm: float) -> TissueState: ...

    def force_at(self, probe_pose_P: np.ndarray, indentation_mm: float) -> float: ...


# ---------------------------------------------------------------------------
# Skin surface from the CT body mask
# ---------------------------------------------------------------------------
class SkinSurface:
    """
    Dorsal skin height map y_skin(x, z) in frame P (+Y toward the dorsal skin), bilinear lookup and
    outward normals from the map gradient.
    """

    def __init__(
        self,
        height_map: np.ndarray,
        origin_xz: Tuple[float, float],
        spacing_xz: Tuple[float, float],
        normal_smoothing_mm: float = 2.0,
    ):
        self.h = np.asarray(height_map, dtype=np.float64)  # (nz, nx)
        self.ox, self.oz = float(origin_xz[0]), float(origin_xz[1])
        self.sx, self.sz = float(spacing_xz[0]), float(spacing_xz[1])
        hs = self.h
        if normal_smoothing_mm > 0:
            from scipy.ndimage import gaussian_filter
            hs = gaussian_filter(self.h, sigma=(normal_smoothing_mm / self.sz, normal_smoothing_mm / self.sx))
        dz, dx = np.gradient(hs, self.sz, self.sx)
        self._dhdx, self._dhdz = dx, dz

    @classmethod
    def from_patient_data(cls, patient_data: Dict[str, Any]) -> "SkinSurface":
        if "skin_height_map" in patient_data:
            return cls(
                patient_data["skin_height_map"],
                patient_data["skin_map_origin_xz"],
                patient_data["skin_map_spacing_xz"],
            )
        # Flat fallback (legacy dicts without a height map)
        y = float(patient_data.get("skin_y_mm", 55.0))
        ox, oy, oz = patient_data["origin_mm"]
        return cls(np.full((2, 2), y), (ox, oz), (1e4, 1e4), normal_smoothing_mm=0.0)

    def _interp(self, arr: np.ndarray, x, z):
        nz, nx = arr.shape
        fx = np.clip((np.asarray(x, dtype=np.float64) - self.ox) / self.sx, 0, nx - 1)
        fz = np.clip((np.asarray(z, dtype=np.float64) - self.oz) / self.sz, 0, nz - 1)
        x0 = np.clip(np.floor(fx).astype(int), 0, max(nx - 2, 0))
        z0 = np.clip(np.floor(fz).astype(int), 0, max(nz - 2, 0))
        x1 = np.minimum(x0 + 1, nx - 1)
        z1 = np.minimum(z0 + 1, nz - 1)
        tx, tz = fx - x0, fz - z0
        v = (arr[z0, x0] * (1 - tx) * (1 - tz) + arr[z0, x1] * tx * (1 - tz)
             + arr[z1, x0] * (1 - tx) * tz + arr[z1, x1] * tx * tz)
        return v

    def height(self, x, z):
        v = self._interp(self.h, x, z)
        return float(v) if np.ndim(v) == 0 else v

    def normal(self, x: float, z: float) -> np.ndarray:
        """Outward unit skin normal (pointing out of the body, ~+Y on the back)."""
        gx = float(self._interp(self._dhdx, x, z))
        gz = float(self._interp(self._dhdz, x, z))
        n = np.array([-gx, 1.0, -gz])
        return n / np.linalg.norm(n)


def tilt_cosine(beam_dir: np.ndarray, outward_normal: np.ndarray) -> float:
    """cos of the angle between the beam and the inward skin normal (1 = perpendicular insonation)."""
    b = np.asarray(beam_dir, dtype=np.float64)
    b = b / np.linalg.norm(b)
    return float(np.clip(-np.dot(b, outward_normal), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Kelvinlet backend
# ---------------------------------------------------------------------------
class KelvinletTissueModel:
    """Regularized-Kelvinlet probe-tissue model implementing `TissueModel` (see module docstring)."""

    def __init__(
        self,
        bone_mask_zyx: np.ndarray,
        spacing_mm: Tuple[float, float, float],
        origin_mm: Tuple[float, float, float],
        skin_surface: Optional[SkinSurface] = None,
        shear_modulus_kpa: Optional[float] = None,
        poissons_ratio: float = 0.45,
        beta_per_mm: float = 0.15,
        reference_k0_n_per_mm: float = 0.5,
        reference_depth_mm: float = 35.0,
        aperture_mm: float = 40.0,
        num_kelvinlets: int = 5,
        eps_mm: float = 6.0,
        max_indent_fraction: float = 0.5,
        max_ray_depth_mm: float = 90.0,
        ray_step_mm: float = 0.5,
        warp_iterations: int = 5,
        field_grid: Tuple[float, float, float] = (50.0, 26.0, 2.0),  # lateral half-extent, elevational half-extent, step
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bone_mask = np.ascontiguousarray(np.asarray(bone_mask_zyx, dtype=bool))
        self.spacing = np.asarray(spacing_mm, dtype=np.float64)
        self.origin = np.asarray(origin_mm, dtype=np.float64)
        self.skin = skin_surface
        self.nu = float(poissons_ratio)
        self.beta = float(beta_per_mm)
        self.aperture_mm = float(aperture_mm)
        self.num_kelvinlets = int(num_kelvinlets)
        self.eps = float(eps_mm)
        self.max_indent_fraction = float(max_indent_fraction)
        self.max_ray_depth = float(max_ray_depth_mm)
        self.ray_step = float(ray_step_mm)
        self.warp_iterations = int(warp_iterations)
        self.field_grid = field_grid
        self.lateral_offsets = (
            np.linspace(-self.aperture_mm / 2.0, self.aperture_mm / 2.0, self.num_kelvinlets)
            if self.num_kelvinlets > 1 else np.zeros(1)
        )
        self._ray_s = np.arange(0.0, self.max_ray_depth + 1e-9, self.ray_step)

        if shear_modulus_kpa is None:
            # Calibrate mu so that the reference configuration has the literature k0.
            shear_modulus_kpa = self.calibrate_shear_modulus(reference_k0_n_per_mm, reference_depth_mm)
        self.set_shear_modulus(shear_modulus_kpa)

        # GPU copy of the bone mask for the field ray-march: (1,1,D,H,W)
        self._bone_t = torch.from_numpy(self.bone_mask.astype(np.float32)).to(self.device)[None, None]

    # ----------------------------------------------------------------- params
    def set_shear_modulus(self, shear_modulus_kpa: float):
        self.shear_modulus_kpa = float(shear_modulus_kpa)
        self.a, self.b = kelvinlet_constants(self.shear_modulus_kpa, self.nu)

    def calibrate_shear_modulus(self, k0_n_per_mm: float, depth_mm: float) -> float:
        """mu (kPa) such that a flat layer of `depth_mm` over bone, untilted, has k_lin = k0."""
        a1, b1 = kelvinlet_constants(1.0, self.nu)
        ex = np.array([1.0, 0.0, 0.0])
        c1 = self._face_compliance(self.lateral_offsets[:, None] * ex[None, :], np.array([0.0, -1.0, 0.0]),
                                   np.full(self.num_kelvinlets, float(depth_mm)), a1, b1)
        # compliance scales as 1/mu  ->  k = mu / c1(mu=1)
        return float(k0_n_per_mm * c1)

    @classmethod
    def from_patient_data(cls, patient_data: Dict[str, Any], **kwargs) -> "KelvinletTissueModel":
        lbl = patient_data.get("label_volume")
        if lbl is not None:
            bone = np.asarray(lbl) == 1
        else:
            bone = np.asarray(patient_data["ct_volume"]) > 300.0
        skin = kwargs.pop("skin_surface", None) or SkinSurface.from_patient_data(patient_data)
        return cls(bone, patient_data["spacing_mm"], patient_data["origin_mm"], skin_surface=skin, **kwargs)

    @staticmethod
    def sample_parameter_ranges(rng: np.random.Generator) -> Dict[str, float]:
        """Patient randomisation ranges from the literature: k0 in [0.3, 0.8] N/mm, beta in [0.10, 0.20] /mm."""
        return {
            "reference_k0_n_per_mm": float(rng.uniform(0.3, 0.8)),
            "beta_per_mm": float(rng.uniform(0.10, 0.20)),
        }

    # -------------------------------------------------------------- geometry
    @staticmethod
    def _frame(probe_pose_P: np.ndarray):
        T = np.asarray(probe_pose_P, dtype=np.float64)
        ex = T[:3, 0] / np.linalg.norm(T[:3, 0])
        b = T[:3, 1] / np.linalg.norm(T[:3, 1])
        ez = T[:3, 2] / np.linalg.norm(T[:3, 2])
        return T[:3, 3].copy(), ex, b, ez

    def _bone_depths_np(self, starts: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Distance (mm) along b from each start point to the first bone voxel; _NO_BONE_MM if none."""
        pts = starts[:, None, :] + self._ray_s[None, :, None] * b[None, None, :]
        idx = np.rint((pts - self.origin) / self.spacing).astype(np.int64)  # (N,S,3) as (ix,iy,iz)
        D, H, W = self.bone_mask.shape
        inb = ((idx[..., 0] >= 0) & (idx[..., 0] < W) & (idx[..., 1] >= 0) & (idx[..., 1] < H)
               & (idx[..., 2] >= 0) & (idx[..., 2] < D))
        ix = np.clip(idx[..., 0], 0, W - 1)
        iy = np.clip(idx[..., 1], 0, H - 1)
        iz = np.clip(idx[..., 2], 0, D - 1)
        hit = self.bone_mask[iz, iy, ix] & inb
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        depth = np.where(any_hit, self._ray_s[first], _NO_BONE_MM)
        return depth

    def _face_compliance(self, sources_rel, b, h_src, a, b_const) -> float:
        """Mean beam-axis displacement of the face points per newton of total load."""
        n = len(h_src)
        bt = torch.tensor(b, dtype=torch.float64)
        f = bt / n
        src = torch.tensor(sources_rel, dtype=torch.float64)
        r = src[:, None, :] - src[None, :, :]  # (j, i, 3): face point j minus source i
        u = regularized_kelvinlet(r, f, a, b_const, self.eps, self.nu, bt).sum(dim=1)  # (j,3)
        h = torch.tensor(h_src, dtype=torch.float64)
        has = h < _NO_BONE_MM * 0.5
        rb = r + (h[:, None, None] * bt)
        ub = regularized_kelvinlet(rb, f, a, b_const, self.eps, self.nu, bt).sum(dim=1)
        u = torch.where(has[:, None], u - ub, u)
        return float(torch.mean(u @ bt))

    def _cos_tilt(self, x_c: np.ndarray, b: np.ndarray) -> float:
        if self.skin is None:
            return 1.0
        return tilt_cosine(b, self.skin.normal(x_c[0], x_c[2]))

    def contact_mechanics(self, probe_pose_P: np.ndarray) -> Dict[str, float]:
        """Linear stiffness, tissue thickness under the probe and tilt for a pose (cheap, CPU)."""
        x_c, ex, b, ez = self._frame(probe_pose_P)
        src = x_c[None, :] + self.lateral_offsets[:, None] * ex[None, :]
        h = self._bone_depths_np(src, b)
        comp = self._face_compliance(self.lateral_offsets[:, None] * ex[None, :], b, h, self.a, self.b)
        finite = h[h < _NO_BONE_MM * 0.5]
        h_c = float(np.min(finite)) if len(finite) else float("inf")
        max_d = self.max_indent_fraction * (h_c if np.isfinite(h_c) else self.max_ray_depth)
        return {
            "k_lin": 1.0 / max(comp, 1e-9),
            "compliance_mm_per_n": comp,
            "bone_depth_mm": h_c,
            "max_indentation_mm": float(max_d),
            "cos_tilt": self._cos_tilt(x_c, b),
        }

    def force_law(self, k_lin: float, indentation_mm: float, cos_tilt: float) -> Tuple[float, float]:
        """Returns (force N, tangent stiffness N/mm along the beam)."""
        d_n = max(0.0, float(indentation_mm)) * cos_tilt
        if self.beta <= 1e-9:
            return k_lin * d_n, k_lin * cos_tilt
        e = np.exp(self.beta * d_n)
        return float(k_lin / self.beta * (e - 1.0)), float(k_lin * e * cos_tilt)

    def max_indentation(self, probe_pose_P: np.ndarray) -> float:
        return self.contact_mechanics(probe_pose_P)["max_indentation_mm"]

    def force_at(self, probe_pose_P: np.ndarray, indentation_mm: float) -> float:
        m = self.contact_mechanics(probe_pose_P)
        return self.force_law(m["k_lin"], indentation_mm, m["cos_tilt"])[0]

    def indentation_for_force(self, probe_pose_P: np.ndarray, force_n: float) -> float:
        """Analytic inverse of the force law (used by the legacy force-driven API)."""
        m = self.contact_mechanics(probe_pose_P)
        if force_n <= 0:
            return 0.0
        k, c = m["k_lin"], max(m["cos_tilt"], 1e-3)
        d_n = np.log1p(self.beta * force_n / k) / self.beta if self.beta > 1e-9 else force_n / k
        return float(min(d_n / c, m["max_indentation_mm"]))

    # ------------------------------------------------------------------ solve
    def _bone_depth_grid(self, x_c, ex, b, ez) -> torch.Tensor:
        """(Ne, Nl) bone depth along b from the plane through x_c perpendicular to b (GPU ray-march)."""
        L, E, step = self.field_grid
        ls = torch.arange(-L, L + 1e-6, step, device=self.device, dtype=torch.float32)
        es = torch.arange(-E, E + 1e-6, step, device=self.device, dtype=torch.float32)
        ss = torch.from_numpy(self._ray_s.astype(np.float32)).to(self.device)
        xc = torch.tensor(x_c, dtype=torch.float32, device=self.device)
        ext = torch.tensor(ex, dtype=torch.float32, device=self.device)
        bt = torch.tensor(b, dtype=torch.float32, device=self.device)
        ezt = torch.tensor(ez, dtype=torch.float32, device=self.device)
        start = xc + ls[None, :, None] * ext + es[:, None, None] * ezt  # (Ne,Nl,3)
        pts = start[:, :, None, :] + ss[None, None, :, None] * bt       # (Ne,Nl,S,3)
        org = torch.tensor(self.origin, dtype=torch.float32, device=self.device)
        spc = torch.tensor(self.spacing, dtype=torch.float32, device=self.device)
        vox = (pts - org) / spc
        D, H, W = self.bone_mask.shape
        g = torch.stack([2 * vox[..., 0] / (W - 1) - 1, 2 * vox[..., 1] / (H - 1) - 1,
                         2 * vox[..., 2] / (D - 1) - 1], dim=-1)
        Ne, Nl, S, _ = g.shape
        hit = F.grid_sample(self._bone_t, g.view(1, Ne * Nl, S, 1, 3), mode="nearest",
                            padding_mode="zeros", align_corners=True).view(Ne, Nl, S) > 0.5
        first = torch.argmax(hit.to(torch.uint8), dim=-1)
        depth = torch.where(hit.any(dim=-1), ss[first], torch.full_like(ss[first], _NO_BONE_MM))
        return depth

    def solve(self, probe_pose_P: np.ndarray, indentation_mm: float) -> TissueState:
        x_c, ex, b, ez = self._frame(probe_pose_P)
        m = self.contact_mechanics(probe_pose_P)
        d = float(np.clip(indentation_mm, 0.0, m["max_indentation_mm"]))
        force, k_t = self.force_law(m["k_lin"], d, m["cos_tilt"])
        if d <= 1e-6:
            return TissueState(0.0, 0.0, False, x_c, b, stiffness_n_per_mm=k_t,
                               bone_depth_mm=m["bone_depth_mm"], cos_tilt=m["cos_tilt"])

        dev = self.device
        depth_grid = self._bone_depth_grid(x_c, ex, b, ez)  # (Ne,Nl)
        L, E, _ = self.field_grid
        xc_t = torch.tensor(x_c, dtype=torch.float32, device=dev)
        ex_t = torch.tensor(ex, dtype=torch.float32, device=dev)
        b_t = torch.tensor(b, dtype=torch.float32, device=dev)
        ez_t = torch.tensor(ez, dtype=torch.float32, device=dev)
        src = xc_t + torch.tensor(self.lateral_offsets, dtype=torch.float32, device=dev)[:, None] * ex_t
        a, bc, eps, nu, n = self.a, self.b, self.eps, self.nu, self.num_kelvinlets
        f_unit = b_t / n
        grid_img = depth_grid[None, None]  # (1,1,Ne,Nl)
        org_t = torch.tensor(self.origin, dtype=torch.float32, device=dev)
        spc_t = torch.tensor(self.spacing, dtype=torch.float32, device=dev)
        Dv, Hv, Wv = self.bone_mask.shape

        def u_unit(X: torch.Tensor) -> torch.Tensor:
            rel = X - xc_t
            l = rel @ ex_t
            s = rel @ b_t
            e = rel @ ez_t
            gq = torch.stack([l / L, e / E], dim=-1).view(1, 1, -1, 2)
            h = F.grid_sample(grid_img, gq, mode="nearest", padding_mode="border",
                              align_corners=True).view(-1)
            r = X[:, None, :] - src[None, :, :]
            u = regularized_kelvinlet(r, f_unit, a, bc, eps, nu, b_t).sum(dim=1)
            has_bone = h < _NO_BONE_MM * 0.5
            rb = r + ((h - s)[:, None, None] * b_t)
            ub = regularized_kelvinlet(rb, f_unit, a, bc, eps, nu, b_t).sum(dim=1)
            u = torch.where(has_bone[:, None], u - ub, u)
            u = torch.where((has_bone & (s >= h))[:, None], torch.zeros_like(u), u)
            # Bone voxels themselves are rigid (guards against the coarse depth-grid lookup at bone edges)
            vox = (X - org_t) / spc_t
            gb = torch.stack([2 * vox[:, 0] / (Wv - 1) - 1, 2 * vox[:, 1] / (Hv - 1) - 1,
                              2 * vox[:, 2] / (Dv - 1) - 1], dim=-1).view(1, 1, 1, -1, 3)
            in_bone = F.grid_sample(self._bone_t, gb, mode="nearest", padding_mode="zeros",
                                    align_corners=True).view(-1) > 0.5
            return torch.where(in_bone[:, None], torch.zeros_like(u), u)

        # Amplitude so that the mean beam-axis displacement of the face points equals d
        face_u = (u_unit(src) @ b_t).mean()
        scale = d / float(torch.clamp(face_u, min=1e-9))

        def disp(X: torch.Tensor) -> torch.Tensor:
            X = X.to(dev, torch.float32)
            shp = X.shape
            return (scale * u_unit(X.reshape(-1, 3))).reshape(shp)

        return TissueState(
            indentation_mm=d,
            reaction_force_n=float(force),
            in_contact=True,
            contact_point_mm=x_c,
            beam_dir=b,
            stiffness_n_per_mm=k_t,
            bone_depth_mm=m["bone_depth_mm"],
            cos_tilt=m["cos_tilt"],
            _displacement_fn=disp,
            warp_iterations=self.warp_iterations,
        )
