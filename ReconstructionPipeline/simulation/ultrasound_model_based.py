"""
Model-Based (Acoustic Ray-Tracing) Ultrasound Simulator.
Faithfully implements acoustic wave reflection, depth attenuation, specular incidence penalties,
bone acoustic shadowing, Point Spread Function (PSF) convolution, and speckle backscattering.
Based on SonoGym (Ao et al., 2025) and Salehi et al. (MICCAI 2015).
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np


class ModelBasedUltrasoundSimulator:
    """
    GPU/CPU vectorized model-based ultrasound simulator generating B-mode US images
    from 2D CT slices and anatomical label maps.
    """
    def __init__(
        self,
        frequency_mhz: float = 5.0,
        initial_energy: float = 1.0,
        spatial_resolution_mm: float = 0.5,
        psf_sigma_axial: float = 2.0,
        psf_sigma_lateral: float = 1.5,
        kernel_size: int = 15,
        specular_exponent: float = 4.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            frequency_mhz: Acoustic transducer center frequency in MHz.
            initial_energy: Transmit acoustic energy E0.
            spatial_resolution_mm: Pixel spacing in mm.
            psf_sigma_axial: Standard deviation for PSF along beam axis (y) in pixels.
            psf_sigma_lateral: Standard deviation for PSF across beam (x) in pixels.
            kernel_size: Size of 2D Gaussian convolution kernel (must be odd).
            specular_exponent: Exponent sharpening specular angle-dependence.
            device: torch.device ('cuda' or 'cpu').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.frequency = frequency_mhz
        self.initial_energy = initial_energy
        self.delta_y_cm = spatial_resolution_mm / 10.0  # mm to cm for dB/(cm*MHz)
        self.specular_exponent = specular_exponent

        # Construct 2D anisotropic Gaussian Point Spread Function (PSF) kernel
        self.psf_kernel = self._build_psf_kernel(kernel_size, psf_sigma_axial, psf_sigma_lateral)

        # Standard attenuation coefficients alpha in dB / (cm * MHz)
        # Scaled for natural exponential attenuation
        self.alpha_soft_tissue = 0.06
        self.alpha_bone = 0.95   # Strong acoustic attenuation causing bone shadow
        self.alpha_fluid = 0.005

    def _build_psf_kernel(self, ksize: int, sigma_y: float, sigma_x: float) -> torch.Tensor:
        """Creates an oriented 2D Gaussian convolution kernel."""
        ax = torch.arange(ksize, device=self.device) - ksize // 2
        ay = torch.arange(ksize, device=self.device) - ksize // 2
        y, x = torch.meshgrid(ay, ax, indexing="ij")

        kernel = torch.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
        kernel = kernel / torch.sum(kernel)
        # Reshape to (1, 1, ksize, ksize) for F.conv2d
        return kernel.unsqueeze(0).unsqueeze(0)

    def simulate_bmode(
        self,
        ct_slice: torch.Tensor,
        label_slice: Optional[torch.Tensor] = None,
        bone_label_id: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulates a B-mode ultrasound image from a 2D CT slice and optional anatomical label map.

        Args:
            ct_slice: 2D tensor (H, W) or 3D tensor (B, H, W) of CT Hounsfield Units.
            label_slice: Optional 2D/3D tensor of integer tissue segmentation labels.
            bone_label_id: Class index corresponding to cortical/trabecular bone.

        Returns:
            Dictionary containing:
                'bmode': Simulated ultrasound image (B, H, W) normalized to [0, 1]
                'reflection': Specular reflection component R (B, H, W)
                'backscatter': Backscattering speckle component B (B, H, W)
                'energy': Remaining acoustic energy map E (B, H, W)
        """
        if ct_slice.ndim == 2:
            ct = ct_slice.unsqueeze(0).to(self.device) # (1, H, W)
            labels = label_slice.unsqueeze(0).to(self.device) if label_slice is not None else None
            is_single = True
        else:
            ct = ct_slice.to(self.device)
            labels = label_slice.to(self.device) if label_slice is not None else None
            is_single = False

        batch_size, h, w = ct.shape

        # 1. Acoustic Impedance Z: proportional to CT intensity (shifted HU)
        # Clamp HU range: air (-1000) -> 0, soft tissue (0-100) -> 1.5, bone (1000-2000) -> 6-8
        ct_clamped = torch.clamp(ct, min=-1000.0, max=2500.0)
        impedance = (ct_clamped + 1000.0) / 1000.0 + 0.1  # (B, H, W)

        # 2. Local Attenuation Map alpha:
        if labels is not None:
            bone_mask = (labels == bone_label_id).float()
            alpha_map = self.alpha_soft_tissue * (1.0 - bone_mask) + self.alpha_bone * bone_mask
        else:
            # Estimate bone directly from high CT attenuation (HU > 350)
            bone_mask = (ct > 350.0).float()
            alpha_map = self.alpha_soft_tissue * (1.0 - bone_mask) + self.alpha_bone * bone_mask

        # 3. Cumulative Acoustic Energy E(u): ray propagates along axis +y (downwards)
        # E(x, y) = E0 * exp( -f * sum_{v=0}^y alpha(x, v) * delta_y )
        # Cumulative sum along dim 1 (axial height)
        cumulative_attenuation = torch.cumsum(alpha_map * self.delta_y_cm, dim=1)
        energy = self.initial_energy * torch.exp(-self.frequency * cumulative_attenuation)

        # 4. Tissue Boundary Gradient & Incidence Angle Theta
        # Compute spatial derivatives: dZ/dy (axial) and dZ/dx (lateral)
        # Ray direction is [0, 1] (along +y)
        dZ_dy = torch.zeros_like(impedance)
        dZ_dx = torch.zeros_like(impedance)
        dZ_dy[:, 1:, :] = impedance[:, 1:, :] - impedance[:, :-1, :]
        dZ_dx[:, :, 1:] = impedance[:, :, 1:] - impedance[:, :, :-1]

        grad_norm = torch.sqrt(dZ_dy ** 2 + dZ_dx ** 2 + 1e-6)
        
        # Incidence angle cosine: cos(Theta) = |dZ/dy| / grad_norm
        # Perpendicular incidence has normal along y axis -> cos(Theta) = 1.0
        cos_theta = torch.clamp(torch.abs(dZ_dy) / grad_norm, 0.0, 1.0)
        specular_factor = torch.pow(cos_theta, self.specular_exponent)

        # Boundary indicator G(u): high impedance gradient
        boundary_indicator = torch.sigmoid((grad_norm - 0.2) * 10.0)

        # Acoustic reflection coefficient: (Z2 - Z1) / (Z2 + Z1)
        # Forward difference along axial axis y
        z_curr = impedance[:, :-1, :]
        z_next = impedance[:, 1:, :]
        refl_coeff = torch.zeros_like(impedance)
        refl_coeff[:, :-1, :] = torch.abs((z_next - z_curr) / (z_next + z_curr + 1e-6))

        # Raw Reflection R_raw = E * cos(Theta) * refl_coeff * G
        reflection_raw = energy * specular_factor * refl_coeff * boundary_indicator

        # 5. Backscattering (Speckle Texture) B
        # Sample stochastic Gaussian noise maps N0, N1
        n0 = torch.randn_like(impedance)
        n1 = torch.rand_like(impedance)
        scatter_mask = (n1 < 0.75).float()
        speckle_texture = (0.2 + 0.3 * torch.abs(n0)) * scatter_mask
        # Soft tissue has speckle; bone internal cavity attenuated; fluid has minimal speckle
        tissue_scatter = torch.where(bone_mask > 0.5, 0.05, 0.35)
        backscatter_raw = energy * tissue_scatter * speckle_texture

        # 6. Convolution with Point Spread Function (PSF)
        # Add channel dim: (B, 1, H, W)
        pad = self.psf_kernel.shape[-1] // 2
        r_convolved = F.conv2d(reflection_raw.unsqueeze(1), self.psf_kernel, padding=pad).squeeze(1)
        b_convolved = F.conv2d(backscatter_raw.unsqueeze(1), self.psf_kernel, padding=pad).squeeze(1)

        # 7. Total Image Synthesis: I = R + B with logarithmic dynamic range compression
        total_intensity = r_convolved * 3.5 + b_convolved
        # Standard ultrasound log compression: log(1 + c * I)
        bmode_log = torch.log1p(15.0 * total_intensity) / torch.log1p(torch.tensor(15.0, device=self.device))
        bmode = torch.clamp(bmode_log, 0.0, 1.0)

        if is_single:
            return {
                "bmode": bmode.squeeze(0),
                "reflection": r_convolved.squeeze(0),
                "backscatter": b_convolved.squeeze(0),
                "energy": energy.squeeze(0),
                "bone_mask": bone_mask.squeeze(0),
            }

        return {
            "bmode": bmode,
            "reflection": r_convolved,
            "backscatter": b_convolved,
            "energy": energy,
            "bone_mask": bone_mask,
        }
