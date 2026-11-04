"""
Image-only Ultrasound Confidence Map.

Fast approximation of Karamalis et al., "Ultrasound confidence maps using random walks"
(MedIA 16(6), 2012): confidence along a scanline decays with the acoustic energy already
reflected/absorbed above a pixel. Instead of solving the random-walk linear system, the
transmitted fraction is approximated from the B-mode image alone:

    C(y, x) = exp(-k * sum_{y' < y - g} e(y', x) * dy_mm) * exp(-depth_mm / L)

where e = clip((I - t) / (1 - t), 0, 1) is the "excess" echo intensity above a speckle level t
(strong specular reflectors such as bone remove most of the energy, so pixels distal to them lie
in an acoustic shadow), g is a guard of ~1.5 mm so the reflecting ridge itself keeps its confidence
(the shadow starts BELOW the peak, not at it), and L is a mild depth-decay length. The result is
smoothed laterally (mimicking the lateral coupling of the random walk).

Only the B-mode image is used. The simulator's privileged `energy` / `effective_energy` /
`bone_mask` outputs must not be passed in; `energy_map` is accepted for backward compatibility
and ignored.
"""

from typing import Optional
import torch
import torch.nn.functional as F


class UltrasoundConfidenceEstimator:
    """Per-pixel confidence C(x, y) in [0, 1] computed purely from the B-mode image."""

    def __init__(
        self,
        min_incidence_cos: float = 0.85,
        attenuation_decay_rate: float = 1.6,
        shadow_threshold: float = 0.70,
        axial_resolution_mm: float = 0.375,
        guard_mm: float = 1.5,
        depth_decay_mm: float = 150.0,
        lateral_smoothing_px: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            min_incidence_cos: Threshold on |n_y| for the optional incidence term.
            attenuation_decay_rate: k, per (mm * unit excess intensity).
            shadow_threshold: t, intensity above which echoes count as strong reflectors.
            axial_resolution_mm: Pixel spacing along the beam (rows) [mm].
            guard_mm: Distance below a reflector before its shadow starts.
            depth_decay_mm: L, mild depth decay length (generic attenuation proxy).
            lateral_smoothing_px: Width of the lateral box filter (odd, 1 disables).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_incidence_cos = min_incidence_cos
        self.attenuation_decay_rate = attenuation_decay_rate
        self.shadow_threshold = shadow_threshold
        self.axial_resolution_mm = axial_resolution_mm
        self.guard_mm = guard_mm
        self.depth_decay_mm = depth_decay_mm
        self.lateral_smoothing_px = max(1, int(lateral_smoothing_px) | 1)

    def compute_confidence_map(
        self,
        bmode_image: torch.Tensor,
        energy_map: Optional[torch.Tensor] = None,
        estimated_normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bmode_image: (H, W) or (B, H, W) B-mode intensity in [0, 1] (rows = depth).
            energy_map: Ignored (kept for backward compatibility; confidence is image-only).
            estimated_normals: Optional (H, W, 2) / (B, H, W, 2) surface normals [nx, ny].

        Returns:
            Confidence map with the same shape as bmode_image, values in [0, 1].
        """
        img = torch.as_tensor(bmode_image, dtype=torch.float32).to(self.device)
        is_single = img.ndim == 2
        if is_single:
            img = img.unsqueeze(0)
        _, h, _ = img.shape
        dy = self.axial_resolution_mm

        # 1. Excess echo intensity above the speckle level (strong reflectors)
        t = self.shadow_threshold
        excess = torch.clamp((img - t) / (1.0 - t), 0.0, 1.0)

        # 2. Integrated reflected energy above each pixel, shifted by a guard so the ridge keeps confidence
        cum = torch.cumsum(excess, dim=1) * dy
        g = max(1, int(round(self.guard_mm / dy)))
        cum_shift = torch.zeros_like(cum)
        if g < h:
            cum_shift[:, g:, :] = cum[:, :-g, :]
        transmitted = torch.exp(-self.attenuation_decay_rate * cum_shift)

        # 3. Mild generic depth decay
        depth_mm = torch.arange(h, device=self.device, dtype=torch.float32).view(1, -1, 1) * dy
        conf = transmitted * torch.exp(-depth_mm / self.depth_decay_mm)

        # 4. Lateral smoothing (random-walk-like lateral coupling)
        k = self.lateral_smoothing_px
        if k > 1:
            conf = F.avg_pool2d(conf.unsqueeze(1), kernel_size=(1, k), stride=1,
                                padding=(0, k // 2), count_include_pad=False).squeeze(1)

        # 5. Optional incidence-angle term
        if estimated_normals is not None:
            ny = estimated_normals[..., 1].to(self.device)
            if ny.ndim == 2:
                ny = ny.unsqueeze(0)
            cos_theta = torch.clamp(torch.abs(ny), 0.0, 1.0)
            conf = conf * torch.where(cos_theta >= self.min_incidence_cos, cos_theta, cos_theta * 0.3)

        conf = torch.clamp(conf, 0.0, 1.0)
        return conf.squeeze(0) if is_single else conf

    def extract_reliable_bone_points(
        self,
        bmode_image: torch.Tensor,
        confidence_map: torch.Tensor,
        intensity_threshold: float = 0.55,
        confidence_threshold: float = 0.40,
    ) -> torch.Tensor:
        """Returns (N, 2) float [x, y] pixel coordinates that are bright and high-confidence."""
        mask = (bmode_image >= intensity_threshold) & (confidence_map >= confidence_threshold)
        indices = torch.nonzero(mask, as_tuple=False)
        if len(indices) == 0:
            return torch.empty((0, 2), device=self.device)
        return torch.stack([indices[:, 1], indices[:, 0]], dim=-1).float()
