"""
Regularized Kelvinlet kernels (de Goes & James, "Regularized Kelvinlets: Sculpting Brushes
based on Fundamental Solutions of Elasticity", ACM TOG 36(4), SIGGRAPH 2017).

For a point load f at the origin, r = x - x0, shear modulus mu, Poisson ratio nu:

    r_e = sqrt(|r|^2 + e^2)
    a   = 1 / (4 pi mu)
    b   = a / (4 (1 - nu))
    u(r) = [ (a - b)/r_e * I  +  b/r_e^3 * r r^T  +  (a/2) e^2/r_e^3 * I ] f

This is the *exact* linear-elastic solution in an unbounded medium for the smoothed force
density f * 15 e^4 / (8 pi r_e^7).  At r = 0: u(0) = a (5 - 6 nu) / (4 (1 - nu) e) * f.

Free-surface (half-space) correction
------------------------------------
A probe presses on the *surface* of the body, not inside an infinite medium.  The classical
half-space solution for a normal point load is Boussinesq's.  Comparing the beam-parallel
displacement of Boussinesq,  u_z = F/(4 pi mu R) [2(1-nu) + cos^2(t)],  with the Kelvin
full-space solution,  u_par = F/(16 pi mu (1-nu) R) [(3-4nu) + cos^2(t)]  (t = angle between r and
the load direction), gives a direction-dependent ratio

    c_hs(t) = 4 (1-nu) [2(1-nu) + cos^2 t] / [(3-4nu) + cos^2 t]

which is ~2 for nearly incompressible tissue (exactly 2 for nu = 0.5, 2.0-2.1 for nu = 0.45-0.46).
We multiply the regularized Kelvinlet by c_hs, using a regularized cos^2 t = ((r.n)^2 + e^2)/r_e^2
so that the factor is finite at r = 0 and equals the on-axis value there.
Validation (see tests): with this factor the centre displacement of a single regularized
Kelvinlet of radius e equals the Boussinesq centre displacement of a uniform circular load of
radius a = e, w0 = (1 - nu) F / (pi mu a), to within 1%.
The correction is applied to the whole vector (an approximation for the tangential components,
which in Boussinesq are small, O(1 - 2 nu), for nearly incompressible tissue).
"""

from typing import Optional, Union
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def kelvinlet_constants(shear_modulus_kpa: float, poissons_ratio: float):
    """Returns (a, b) of de Goes & James in mm/N units (mu converted from kPa to N/mm^2)."""
    mu = float(shear_modulus_kpa) * 1e-3  # 1 kPa = 1e-3 N/mm^2
    a = 1.0 / (4.0 * np.pi * mu)
    b = a / (4.0 * (1.0 - float(poissons_ratio)))
    return a, b


def half_space_factor(cos2: torch.Tensor, nu: float) -> torch.Tensor:
    """Boussinesq / Kelvin displacement ratio c_hs(t) for a normal load on a half-space."""
    return 4.0 * (1.0 - nu) * (2.0 * (1.0 - nu) + cos2) / ((3.0 - 4.0 * nu) + cos2)


def regularized_kelvinlet(
    r: torch.Tensor,
    f: torch.Tensor,
    a: float,
    b: float,
    eps: float,
    nu: Optional[float] = None,
    load_dir: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact regularized Kelvinlet displacement (de Goes & James 2017, eq. with (a/2) e^2 term).

    Args:
        r: (..., 3) offsets x - x0 in mm.
        f: (3,) or broadcastable (..., 3) force in N.
        a, b: elastic constants from `kelvinlet_constants` (mm/N).
        eps: regularization radius (mm).
        nu, load_dir: if both given, applies the Boussinesq free-surface factor c_hs(t) with t the
            angle between r and the unit load direction `load_dir` (surface normal pointing inward).
    Returns:
        (..., 3) displacement in mm.
    """
    r2 = torch.sum(r * r, dim=-1, keepdim=True)
    re2 = r2 + eps * eps
    re = torch.sqrt(re2)
    re3 = re * re2
    r_dot_f = torch.sum(r * f, dim=-1, keepdim=True)
    u = ((a - b) / re + 0.5 * a * eps * eps / re3) * f + (b / re3) * r_dot_f * r
    if nu is not None and load_dir is not None:
        rn = torch.sum(r * load_dir, dim=-1, keepdim=True)
        cos2 = (rn * rn + eps * eps) / re2
        u = u * half_space_factor(cos2, nu)
    return u


class RegularizedKelvinletDeformation:
    """
    Thin convenience wrapper around `regularized_kelvinlet` for a single point load.

    The full probe-tissue model (multiple Kelvinlets along the probe face, bone rigidity,
    force-indentation law, inverse warping) lives in `tissue_model.KelvinletTissueModel`.
    """

    def __init__(
        self,
        shear_modulus_kpa: float = 6.0,
        poissons_ratio: float = 0.45,
        probe_radius_mm: float = 6.0,
        max_indentation_mm: float = 30.0,
        half_space: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shear_modulus_kpa = float(shear_modulus_kpa)
        self.mu_n_per_mm2 = self.shear_modulus_kpa * 1e-3
        self.poissons_ratio = float(poissons_ratio)
        self.epsilon = float(probe_radius_mm)
        self.max_indentation_mm = float(max_indentation_mm)
        self.half_space = bool(half_space)
        self.const_a, self.const_b = kelvinlet_constants(self.shear_modulus_kpa, self.poissons_ratio)

    def _t(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)

    def compute_displacement(
        self,
        query_points: ArrayLike,
        contact_point: ArrayLike,
        force_vector: ArrayLike,
        bone_depth_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Displacement u(x) (mm) at `query_points` (..., 3) due to a point load `force_vector` (N)
        applied at `contact_point` on the tissue surface (load direction = inward normal).

        If `bone_depth_mm` is given, a rigid substrate is placed at that depth along the load
        direction below the contact point.  The displacement is then the compression of the
        soft layer only (Steinbrenner layered-settlement approximation):
            u_layer(X) = u_hs(X) - u_hs(X_b),   X_b = projection of X onto the bone plane,
        and u = 0 at and beyond the bone plane.
        """
        pts = self._t(query_points)
        cp = self._t(contact_point)
        f_vec = self._t(force_vector)
        f_norm = torch.norm(f_vec)
        if float(f_norm) < 1e-9:
            return torch.zeros_like(pts)
        n = f_vec / f_norm
        nu = self.poissons_ratio if self.half_space else None
        nd = n if self.half_space else None

        r = pts - cp
        u = regularized_kelvinlet(r, f_vec, self.const_a, self.const_b, self.epsilon, nu, nd)
        if bone_depth_mm is not None and bone_depth_mm > 0:
            s = torch.sum(r * n, dim=-1, keepdim=True)
            r_b = r + (bone_depth_mm - s) * n
            u_b = regularized_kelvinlet(r_b, f_vec, self.const_a, self.const_b, self.epsilon, nu, nd)
            u = torch.where(s < bone_depth_mm, u - u_b, torch.zeros_like(u))

        u_mag = torch.norm(u, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_indentation_mm / (u_mag + 1e-9), max=1.0)
        return u * scale

    def center_compliance_mm_per_n(self, bone_depth_mm: Optional[float] = None) -> float:
        """Beam-axis displacement at the load point per newton (mm/N)."""
        a, b, e, nu = self.const_a, self.const_b, self.epsilon, self.poissons_ratio
        c0 = a * (5.0 - 6.0 * nu) / (4.0 * (1.0 - nu) * e)
        if self.half_space:
            c0 *= float(half_space_factor(torch.tensor(1.0), nu))
        if bone_depth_mm is not None and bone_depth_mm > 0:
            h = float(bone_depth_mm)
            re = np.sqrt(h * h + e * e)
            ch = (a - b) / re + b * h * h / re ** 3 + 0.5 * a * e * e / re ** 3
            if self.half_space:
                ch *= float(half_space_factor(torch.tensor(1.0), nu))
            c0 -= ch
        return float(c0)

    def get_contact_indentation(self, normal_force_n: float, bone_depth_mm: Optional[float] = None) -> float:
        """Linear-elastic centre indentation (mm) under `normal_force_n` (N)."""
        if normal_force_n <= 0.0:
            return 0.0
        return float(min(normal_force_n * self.center_compliance_mm_per_n(bone_depth_mm), self.max_indentation_mm))
