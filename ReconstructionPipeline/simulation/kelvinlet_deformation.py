"""
GPU-Accelerated Regularized Kelvinlet Soft-Tissue Deformation Module.
Implements closed-form, singularity-free Green's functions of linear elasticity
for real-time probe-tissue contact simulation in robotic ultrasound.

Grounded in:
- De Goes & James, "Regularized Kelvinlets: Sculpting with Green's Functions", ACM TOG (SIGGRAPH 2017).
- "Neural-Augmented Kelvinlets for Real-Time Soft Tissue Simulation" (2024-2025).
- Compatible with backward Eulerian coordinate warping (< 0.5 ms on GPU).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


class RegularizedKelvinletDeformation(nn.Module):
    """
    Computes smooth, closed-form 3D soft tissue displacement fields u(x)
    under localized probe contact forces F_N without tetrahedral mesh inversion.
    """

    def __init__(
        self,
        shear_modulus_kpa: float = 4.5,        # mu: shear modulus for compliant flank/paraspinal soft tissue (kPa)
        poissons_ratio: float = 0.46,          # nu: nearly incompressible soft tissue
        probe_radius_mm: float = 14.0,         # epsilon: regularized probe contact radius (mm)
        max_indentation_mm: float = 15.0,      # Maximum physiological compression clamp
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            shear_modulus_kpa: Shear modulus mu in kPa (default 4.5 kPa for compliant flank/paraspinal tissue).
            poissons_ratio: Poisson's ratio nu (default 0.48 for biological tissue).
            probe_radius_mm: Regularization scale parameter epsilon (mm) matching probe footprint.
            max_indentation_mm: Clamp on maximum indentation to prevent non-physical distortion.
            device: torch.device ('cuda' or 'cpu').
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.shear_modulus_kpa = shear_modulus_kpa
        # 1 kPa = 1e-3 N / mm^2
        self.mu_n_per_mm2 = shear_modulus_kpa * 1e-3
        self.poissons_ratio = poissons_ratio
        self.epsilon = probe_radius_mm
        self.max_indentation_mm = max_indentation_mm

        # Elasticity constants a and b
        # a = (3 - 4*nu) / (2 * (1 - nu))
        # b = 1 / (4 * (1 - nu))
        nu = self.poissons_ratio
        self.const_a = (3.0 - 4.0 * nu) / (2.0 * (1.0 - nu))
        self.const_b = 1.0 / (4.0 * (1.0 - nu))
        self.prefactor = 1.0 / (4.0 * np.pi * self.mu_n_per_mm2)

    def compute_displacement(
        self,
        query_points: Union[torch.Tensor, np.ndarray],
        contact_point: Union[torch.Tensor, np.ndarray],
        force_vector: Union[torch.Tensor, np.ndarray],
        bone_depth_mm: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Computes 3D displacement vector u(x) for an array of query points in physical coordinates (mm).

        Formula:
            r = x - x_c
            r_eps = sqrt(||r||^2 + epsilon^2)
            u(x) = (1 / 4*pi*mu) * [ (a / r_eps) * F + (b / r_eps^3) * (r . F) * r ]

        Args:
            query_points: Tensor or ndarray of shape (..., 3) in mm.
            contact_point: Tensor or ndarray of shape (3,) in mm.
            force_vector: Force vector F = [Fx, Fy, Fz] in Newtons.
            bone_depth_mm: Optional depth of rigid bone boundary beneath contact point.
                           If provided, displacement decays smoothly to 0 at bone depth.

        Returns:
            displacement: Tensor of shape (..., 3) matching query_points shape.
        """
        if isinstance(query_points, np.ndarray):
            pts = torch.from_numpy(query_points).float().to(self.device)
        else:
            pts = query_points.float().to(self.device)

        if isinstance(contact_point, np.ndarray):
            cp = torch.from_numpy(contact_point).float().to(self.device)
        else:
            cp = contact_point.float().to(self.device)

        if isinstance(force_vector, np.ndarray):
            f_vec = torch.from_numpy(force_vector).float().to(self.device)
        else:
            f_vec = force_vector.float().to(self.device)

        # Force magnitude
        f_norm = torch.norm(f_vec)
        if f_norm < 1e-6:
            return torch.zeros_like(pts)

        # Vector r = x - x_c (shape: ..., 3)
        r = pts - cp

        # Regularized radius r_eps = sqrt(||r||^2 + epsilon^2)
        r_sq = torch.sum(r ** 2, dim=-1, keepdim=True) # (..., 1)
        r_eps = torch.sqrt(r_sq + self.epsilon ** 2)   # (..., 1)
        r_eps_cubed = r_eps ** 3                       # (..., 1)

        # Dot product (r . F) (shape: ..., 1)
        r_dot_f = torch.sum(r * f_vec, dim=-1, keepdim=True)

        # Term 1: (a / r_eps) * F
        term1 = (self.const_a / r_eps) * f_vec

        # Term 2: (b / r_eps^3) * (r . F) * r
        term2 = (self.const_b / r_eps_cubed) * r_dot_f * r

        # Total regularized Kelvinlet displacement
        u = self.prefactor * (term1 + term2)

        # Physiological clamping
        u_mag = torch.norm(u, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_indentation_mm / (u_mag + 1e-6), max=1.0)
        u = u * scale

        # Rigid bone boundary attenuation: soft tissue cannot displace through rigid bone cortex
        if bone_depth_mm is not None and bone_depth_mm > 0:
            # Normalized depth along force direction
            f_dir = f_vec / (f_norm + 1e-6)
            depth_along_beam = torch.sum(r * f_dir, dim=-1, keepdim=True)
            # Damping factor: 1.0 at skin, drops linearly to 0 at bone depth
            bone_damping = torch.clamp(1.0 - (depth_along_beam / (bone_depth_mm + 1e-6)), 0.0, 1.0)
            # Only damp points beneath the contact plane
            beneath_mask = (depth_along_beam > 0.0).float()
            u = u * (1.0 - beneath_mask * (1.0 - bone_damping))

        return u

    def get_contact_indentation(self, normal_force_n: float) -> float:
        """
        Calculates theoretical indentation depth (mm) at contact center (r = 0).
        u(0) = (a / (4 * pi * mu * epsilon)) * F_N
        """
        if normal_force_n <= 0.0:
            return 0.0
        u0 = (self.const_a / (4.0 * np.pi * self.mu_n_per_mm2 * self.epsilon)) * normal_force_n
        return float(min(u0, self.max_indentation_mm))
