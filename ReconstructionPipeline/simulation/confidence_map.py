"""
Ultrasound Confidence Mapping and Acoustic Reliability Estimator.
Quantifies signal reliability based on wave attenuation, bone acoustic shadowing,
and angle of incidence to guide Next-Best-View (NBV) planning and filter 3D point cloud back-projection.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


class UltrasoundConfidenceEstimator:
    """
    Estimates a per-pixel confidence map C(x, y) in [0, 1] representing the geometric reliability
    of detected ultrasound reflections.
    """
    def __init__(
        self,
        min_incidence_cos: float = 0.85, # cos(31 deg) ~ 0.85 threshold for specular bone
        attenuation_decay_rate: float = 2.5,
        shadow_threshold: float = 0.15,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_incidence_cos = min_incidence_cos
        self.attenuation_decay_rate = attenuation_decay_rate
        self.shadow_threshold = shadow_threshold

    def compute_confidence_map(
        self,
        bmode_image: torch.Tensor,
        energy_map: Optional[torch.Tensor] = None,
        estimated_normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the 2D confidence map.

        Args:
            bmode_image: (H, W) or (B, H, W) B-mode ultrasound intensity in [0, 1].
            energy_map: (H, W) or (B, H, W) acoustic energy map E(u).
            estimated_normals: Optional (H, W, 2) or (B, H, W, 2) normalized surface gradients [nx, ny].

        Returns:
            confidence_map: Tensor of same shape as bmode_image with values in [0, 1].
        """
        img = bmode_image.to(self.device)
        is_single = img.ndim == 2
        if is_single:
            img = img.unsqueeze(0)

        batch_size, h, w = img.shape

        # 1. Depth transmission factor: signal decay with axial depth y
        depth_coords = torch.linspace(0.0, 1.0, h, device=self.device).view(1, h, 1).expand(batch_size, -1, w)
        depth_penalty = torch.exp(-self.attenuation_decay_rate * depth_coords)

        # 2. Cumulative shadow detection along axial beam (y axis):
        # High intensity bone reflection attenuates all sound below it
        # Compute cumulative max or exponential running shadow
        cumulative_attenuator = torch.cumsum(img * 0.1, dim=1)
        shadow_factor = torch.clamp(1.0 - cumulative_attenuator, min=0.0, max=1.0)

        # 3. Energy factor
        if energy_map is not None:
            e_map = energy_map.to(self.device)
            if e_map.ndim == 2:
                e_map = e_map.unsqueeze(0)
            energy_factor = torch.clamp(e_map, 0.0, 1.0)
        else:
            energy_factor = depth_penalty

        # 4. Incidence angle factor
        if estimated_normals is not None:
            # Beam direction is [0, 1] (downwards +y)
            # ny is component along acoustic ray
            ny = estimated_normals[..., 1].to(self.device)
            if ny.ndim == 2:
                ny = ny.unsqueeze(0)
            cos_theta = torch.clamp(torch.abs(ny), 0.0, 1.0)
            incidence_factor = torch.where(
                cos_theta >= self.min_incidence_cos,
                cos_theta,
                cos_theta * 0.3 # Heavy penalty for non-orthogonal surfaces
            )
        else:
            incidence_factor = torch.ones_like(img)

        # Total Confidence = Energy * Shadow * Incidence * (Intensity presence)
        confidence = energy_factor * shadow_factor * incidence_factor
        confidence = torch.clamp(confidence, 0.0, 1.0)

        return confidence.squeeze(0) if is_single else confidence

    def extract_reliable_bone_points(
        self,
        bmode_image: torch.Tensor,
        confidence_map: torch.Tensor,
        intensity_threshold: float = 0.55,
        confidence_threshold: float = 0.40,
    ) -> torch.Tensor:
        """
        Filters the 2D ultrasound frame to return only high-confidence bone boundary coordinates.

        Returns:
            points_2d: (N, 2) tensor of pixel coordinates (x, y) satisfying bone criteria.
        """
        mask = (bmode_image >= intensity_threshold) & (confidence_map >= confidence_threshold)
        # Find indices: y is dim 0, x is dim 1
        indices = torch.nonzero(mask, as_tuple=False) # (N, 2) where col 0 is y, col 1 is x
        # Return as (x, y)
        if len(indices) == 0:
            return torch.empty((0, 2), device=self.device)
        return torch.stack([indices[:, 1], indices[:, 0]], dim=-1).float()
