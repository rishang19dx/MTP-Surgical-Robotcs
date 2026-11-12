"""
Model-Based (Acoustic Ray-Tracing) Ultrasound Simulator.

Implements, per scanline (image column, beam along +y):
  * depth attenuation with frequency-dependent coefficients alpha [dB/(cm*MHz)], converted to nepers
    (1 dB = 0.1151 Np for amplitude) and integrated over the physical pixel spacing (round trip),
  * interface transmission losses (1 - R^2 per crossed interface, round trip),
  * specular reflection R = |Z2 - Z1| / (Z2 + Z1) weighted by an incidence-angle term,
  * Rayleigh speckle: complex Gaussian scatterers convolved with an anisotropic Gaussian PSF
    whose sigmas are given in millimetres (converted per axis to pixels), envelope = |.|,
  * additive Rayleigh electronic noise (so acoustic shadows are dark but not exactly zero),
  * a fixed nominal Time Gain Compensation (TGC) ramp in dB/cm that does NOT depend on the
    tissue actually present (as on a real scanner),
  * log compression to a fixed display dynamic range.

Based on SonoGym (Ao et al., 2025) and Salehi et al. (MICCAI 2015).

Attenuation constants (amplitude, dB/(cm*MHz)), standard literature values
(Duck, "Physical Properties of Tissue", 1990; Szabo, "Diagnostic Ultrasound Imaging", 2004):
  soft tissue 0.6 (range 0.5-0.7), cortical bone 20 (range ~10-20+, here the upper end since
  it also stands in for scattering losses in trabecular bone).
"""

from typing import Optional, Dict
import math
import warnings

import torch
import torch.nn.functional as F

DB_TO_NEPER = math.log(10.0) / 20.0  # 0.1151 Np per dB (amplitude)

# Superficial fascia (thoracolumbar fascia over the erector spinae), DEC-055. A CT cannot resolve this thin
# dense-collagen layer, so a CT-derived B-mode shows no fascia echo (DEC-051). Literature values:
#   thickness 1.65 mm: healthy adults at L3, 2-3 cm lateral to the spinous process, transverse axis,
#     1.6 +- 0.4 mm (right) and 1.7 +- 0.51 mm (left) (Pirri et al. 2023, Diagnostics 13:1436, Table 3);
#   sound speed across collagen fibres 1650 +- 5 m/s and attenuation 11.3 dB/cm at 7 MHz
#     (Miles 1996, J Acoust Soc Am 99:3225, equine deep digital flexor tendon, "C" direction, 0 C);
#   density 1.11 g/cm^3 (tendon, ~1.10-1.12 as used in the tendon literature).
# Impedance = 1.650 km/s x 1.11 g/cm^3 = 1.83 MRayl.
FASCIA_THICKNESS_MM = 1.65
FASCIA_IMPEDANCE_MRAYL = 1.650 * 1.11
FASCIA_ALPHA_DB = 11.3 / 7.0          # dB / (cm MHz), amplitude


def superficial_fascia_mask(muscle: torch.Tensor, axial_resolution_mm: float,
                            thickness_mm: float = FASCIA_THICKNESS_MM) -> torch.Tensor:
    """(B, H, W) or (H, W) fractional fascia occupancy in [0, 1]: a layer `thickness_mm` thick directly above (on the
    probe side of) the first muscle pixel of every image column, measured along the beam. Columns without muscle get
    none. `muscle` is the (warped) muscle label slice, so the layer moves with the tissue."""
    m = muscle.bool()
    single = m.ndim == 2
    if single:
        m = m.unsqueeze(0)
    h = m.shape[1]
    has = m.any(1)                                                      # (B, W)
    r0 = torch.argmax(m.to(torch.uint8), dim=1).to(torch.float32)       # first muscle row per column
    tp = float(thickness_mm) / float(axial_resolution_mm)
    y = torch.arange(h, device=m.device, dtype=torch.float32).view(1, -1, 1)
    lo, hi = (r0 - tp).unsqueeze(1), r0.unsqueeze(1)
    cover = torch.clamp(torch.minimum(y + 1.0, hi) - torch.maximum(y, lo), 0.0, 1.0)
    cover = cover * has.unsqueeze(1)
    return cover[0] if single else cover


class ModelBasedUltrasoundSimulator:
    """
    GPU/CPU vectorized model-based ultrasound simulator generating B-mode US images
    from 2D CT slices and anatomical label maps.

    Image convention: rows = axial depth (beam along +row), columns = lateral position.
    """

    def __init__(
        self,
        frequency_mhz: float = 5.0,
        initial_energy: float = 1.0,
        axial_resolution_mm: float = 0.375,
        lateral_resolution_mm: float = 0.333,
        psf_sigma_axial_mm: float = 0.35,
        psf_sigma_lateral_mm: float = 0.7,
        alpha_soft_tissue_db: float = 0.6,
        alpha_bone_db: float = 20.0,
        tgc_db_per_cm: float = 5.0,
        tgc_max_db: float = 40.0,
        dynamic_range_db: float = 60.0,
        reference_amplitude: float = 1.0,
        scatter_amplitude: float = 0.008,
        noise_amplitude: float = 0.002,
        specular_exponent: float = 2.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        spatial_resolution_mm: Optional[float] = None,
        psf_sigma_axial: Optional[float] = None,
        psf_sigma_lateral: Optional[float] = None,
        kernel_size: Optional[int] = None,
        alpha_fat_db: Optional[float] = None,
        alpha_muscle_db: Optional[float] = None,
        fat_hu_threshold: float = -20.0,
        receiver_noise_amplitude: Optional[float] = None,
    ):
        """
        Args:
            frequency_mhz: Transducer centre frequency f [MHz].
            initial_energy: Transmit amplitude scale E0.
            axial_resolution_mm: Pixel spacing along the beam (rows) [mm]. Use depth_mm / (H - 1).
            lateral_resolution_mm: Pixel spacing across the beam (columns) [mm]. Use width_mm / (W - 1).
            psf_sigma_axial_mm / psf_sigma_lateral_mm: Gaussian PSF standard deviations [mm].
            alpha_soft_tissue_db / alpha_bone_db: Amplitude attenuation [dB/(cm*MHz)].
            tgc_db_per_cm: Fixed nominal TGC slope [dB/cm of depth] (independent of the tissue).
            tgc_max_db: TGC gain cap [dB].
            dynamic_range_db: Displayed log-compression dynamic range [dB].
            reference_amplitude: Echo amplitude mapped to full brightness (1.0).
            scatter_amplitude: Diffuse backscatter amplitude of soft tissue (relative to E0).
            noise_amplitude: Rayleigh electronic noise floor added after TGC (sets shadow brightness).
            specular_exponent: Exponent n in cos(theta)^n incidence weighting.
            seed: Optional seed for the internal speckle generator (reproducible frames).
            device: torch.device ('cuda' or 'cpu').
            spatial_resolution_mm: DEPRECATED alias that sets both axial and lateral spacing.
            psf_sigma_axial / psf_sigma_lateral: DEPRECATED PSF sigmas in pixels (override the mm values).
            kernel_size: DEPRECATED fixed PSF kernel size (now derived from the sigmas, 3 sigma).
            alpha_fat_db / alpha_muscle_db: per-tissue soft-tissue attenuation [dB/(cm*MHz)] (DEC-042). When both
                are given, soft tissue with HU < `fat_hu_threshold` uses the fat value and the rest the muscle value;
                otherwise `alpha_soft_tissue_db` is used everywhere (legacy, the single-env default).
            receiver_noise_amplitude: Rayleigh noise added at the receiver INPUT, i.e. amplified by the TGC with
                depth (DEC-042).  None = legacy (only the post-TGC `noise_amplitude`).  See
                `receiver_noise_from_penetration` for the literature-anchored value.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if spatial_resolution_mm is not None:
            warnings.warn(
                "spatial_resolution_mm is deprecated; pass axial_resolution_mm and lateral_resolution_mm",
                DeprecationWarning, stacklevel=2,
            )
            axial_resolution_mm = lateral_resolution_mm = float(spatial_resolution_mm)

        self.frequency = float(frequency_mhz)
        self.initial_energy = float(initial_energy)
        self.axial_resolution_mm = float(axial_resolution_mm)
        self.lateral_resolution_mm = float(lateral_resolution_mm)
        self.delta_y_cm = self.axial_resolution_mm / 10.0
        self.specular_exponent = float(specular_exponent)

        self.alpha_soft_tissue_db = float(alpha_soft_tissue_db)
        self.alpha_fat_db = None if alpha_fat_db is None else float(alpha_fat_db)
        self.alpha_muscle_db = None if alpha_muscle_db is None else float(alpha_muscle_db)
        self.fat_hu_threshold = float(fat_hu_threshold)
        self.receiver_noise_amplitude = None if receiver_noise_amplitude is None else float(receiver_noise_amplitude)
        self.alpha_bone_db = float(alpha_bone_db)
        self.tgc_db_per_cm = float(tgc_db_per_cm)
        self.tgc_max_db = float(tgc_max_db)
        self.dynamic_range_db = float(dynamic_range_db)
        self.reference_amplitude = float(reference_amplitude)
        self.scatter_amplitude = float(scatter_amplitude)
        self.noise_amplitude = float(noise_amplitude)

        # PSF sigmas in pixels per axis
        sig_ax_px = psf_sigma_axial if psf_sigma_axial is not None else psf_sigma_axial_mm / self.axial_resolution_mm
        sig_lat_px = psf_sigma_lateral if psf_sigma_lateral is not None else psf_sigma_lateral_mm / self.lateral_resolution_mm
        self.psf_sigma_axial_px = float(sig_ax_px)
        self.psf_sigma_lateral_px = float(sig_lat_px)
        self.psf_axial = self._gauss_1d(sig_ax_px, kernel_size).view(1, 1, -1, 1)
        self.psf_lateral = self._gauss_1d(sig_lat_px, kernel_size).view(1, 1, 1, -1)

        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(seed))

    @staticmethod
    def receiver_noise_from_penetration(frequency_mhz: float = 5.0, scatter_amplitude: float = 0.008,
                                        alpha_avg_db: float = 0.54, wavelengths: float = 500.0) -> float:
        """Receiver-input noise amplitude at which average soft-tissue scatter reaches SNR = 1 at the textbook
        maximum imaging depth of ~`wavelengths` wavelengths (DEC-042): D = wavelengths x 1.54/f mm,
        n0 = scatter x 10^(-2 alpha f D[cm] / 20).  5 MHz -> D = 154 mm, n0 = 5.56e-7 for scatter 0.008."""
        depth_cm = wavelengths * 1.54 / frequency_mhz / 10.0
        return float(scatter_amplitude * 10.0 ** (-2.0 * alpha_avg_db * frequency_mhz * depth_cm / 20.0))

    @classmethod
    def depth_physics(cls, frequency_mhz: float = 5.0, image_depth_mm: float = 75.0,
                      **kwargs) -> "ModelBasedUltrasoundSimulator":
        """Simulator with the DEC-042 depth-dependent physics: fat 0.48 / muscle 1.09 dB/cm/MHz (arXiv 1710.05065)
        and receiver noise before the TGC anchored at 500-wavelength penetration; plus the DEC-043 fixed
        muscle-appropriate TGC preset: slope 2 x alpha_muscle x f dB/cm (10.9 dB/cm at 5 MHz), capped at the image
        depth.  The preset is fixed (the same for every patient and frame), independent of the tissue present."""
        alpha_fat, alpha_muscle = 0.48, 1.09
        scatter = kwargs.get("scatter_amplitude", 0.008)
        tgc = 2.0 * alpha_muscle * frequency_mhz
        kwargs.setdefault("tgc_db_per_cm", tgc)
        kwargs.setdefault("tgc_max_db", tgc * image_depth_mm / 10.0)
        return cls(frequency_mhz=frequency_mhz, alpha_fat_db=alpha_fat, alpha_muscle_db=alpha_muscle,
                   receiver_noise_amplitude=cls.receiver_noise_from_penetration(frequency_mhz, scatter), **kwargs)

    def _gauss_1d(self, sigma_px: float, ksize: Optional[int]) -> torch.Tensor:
        """Peak-normalised 1D Gaussian (a single-pixel reflector keeps its amplitude)."""
        sigma_px = max(float(sigma_px), 1e-3)
        half = int(math.ceil(3.0 * sigma_px))
        if ksize is not None:
            half = min(half, ksize // 2)
        r = torch.arange(-half, half + 1, device=self.device, dtype=torch.float32)
        return torch.exp(-0.5 * (r / sigma_px) ** 2)

    def _psf(self, x: torch.Tensor) -> torch.Tensor:
        """Separable PSF convolution. x: (N, C, H, W) -> same shape."""
        c = x.shape[1]
        ka = self.psf_axial.expand(c, 1, -1, -1)
        kl = self.psf_lateral.expand(c, 1, -1, -1)
        x = F.conv2d(x, ka, padding=(ka.shape[2] // 2, 0), groups=c)
        return F.conv2d(x, kl, padding=(0, kl.shape[3] // 2), groups=c)

    def simulate_bmode(
        self,
        ct_slice: torch.Tensor,
        label_slice: Optional[torch.Tensor] = None,
        bone_label_id: int = 1,
        generator: Optional[torch.Generator] = None,
        fascia_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulates a B-mode ultrasound image from a 2D CT slice and optional anatomical label map.

        Args:
            ct_slice: (H, W) or (B, H, W) CT Hounsfield Units.
            label_slice: Optional (H, W) / (B, H, W) integer tissue labels.
            bone_label_id: Label value of bone.
            generator: Optional torch.Generator (on self.device) for the speckle; falls back to the
                       internal seeded generator, then to the global torch RNG.
            fascia_mask: Optional (H, W) / (B, H, W) fractional fascia occupancy (`superficial_fascia_mask`).
                       Inside it the impedance is FASCIA_IMPEDANCE_MRAYL and the attenuation FASCIA_ALPHA_DB
                       (DEC-055). None (default) leaves the simulator unchanged.

        Returns dict (privileged keys `energy`, `effective_energy`, `bone_mask` are simulator internals
        and must not be consumed by the perception pipeline):
            'bmode': (B, H, W) log-compressed image in [0, 1]
            'reflection': PSF-convolved specular echo amplitude (after TGC)
            'backscatter': speckle envelope amplitude (after TGC)
            'energy': round-trip transmitted amplitude reaching each pixel (no TGC)
            'effective_energy': energy * TGC gain, clamped to [0, 1]
            'bone_mask': bone indicator used by the simulator
        """
        ct = ct_slice.to(self.device).float()
        labels = label_slice.to(self.device) if label_slice is not None else None
        is_single = ct.ndim == 2
        if is_single:
            ct = ct.unsqueeze(0)
            labels = labels.unsqueeze(0) if labels is not None else None
        b, h, w = ct.shape

        # 1. Bone mask
        bone_mask = ct > 300.0
        if labels is not None:
            bone_mask = bone_mask | (labels == bone_label_id)
        bone_f = bone_mask.float()

        # 2. Acoustic impedance [MRayl]: soft tissue ~1.54 (fat ~1.38), cortical bone ~6-8
        ct_c = torch.clamp(ct, -1000.0, 2500.0)
        z_tissue = 1.54 + torch.clamp(ct_c - 40.0, -500.0, 200.0) / 1000.0
        z_bone = 6.5 + torch.clamp(ct_c - 300.0, 0.0, 1500.0) / 750.0
        impedance = torch.where(bone_mask, z_bone, z_tissue)
        fas = None
        if fascia_mask is not None:
            fas = fascia_mask.to(self.device).float()
            fas = (fas.unsqueeze(0) if fas.ndim == 2 else fas).clamp(0.0, 1.0) * (1.0 - bone_mask.float())
            impedance = impedance * (1.0 - fas) + FASCIA_IMPEDANCE_MRAYL * fas

        # 3. Interface reflection coefficient at the boundary between row y and y+1 (assigned to row y+1,
        #    i.e. the first pixel of the deeper medium where the echo returns from).
        refl = torch.zeros_like(impedance)
        z_prev, z_next = impedance[:, :-1, :], impedance[:, 1:, :]
        refl[:, 1:, :] = torch.abs((z_next - z_prev) / (z_next + z_prev))

        # 4. Round-trip amplitude attenuation in nepers, integrated over physical depth (exclusive of the
        #    current pixel so a reflector is not attenuated by the medium beneath it).
        if self.alpha_fat_db is not None and self.alpha_muscle_db is not None:
            alpha_soft = torch.where(ct < self.fat_hu_threshold, self.alpha_fat_db, self.alpha_muscle_db)
        else:
            alpha_soft = self.alpha_soft_tissue_db
        if fas is not None:
            alpha_soft = alpha_soft * (1.0 - fas) + FASCIA_ALPHA_DB * fas
        alpha_np_per_cm = DB_TO_NEPER * self.frequency * (alpha_soft * (1.0 - bone_f) + self.alpha_bone_db * bone_f)
        path_np = 2.0 * alpha_np_per_cm * self.delta_y_cm
        att_np = torch.cumsum(path_np, dim=1) - path_np
        # Interface transmission losses (round trip amplitude factor 1 - R^2), exclusive of current pixel
        log_t = torch.log(torch.clamp(1.0 - refl ** 2, min=1e-6))
        trans_np = torch.cumsum(log_t, dim=1) - log_t
        energy = self.initial_energy * torch.exp(-att_np + trans_np)

        # 5. Fixed nominal TGC: depth ramp in dB/cm, independent of the actual tissue
        depth_cm = torch.arange(h, device=self.device, dtype=torch.float32).view(1, -1, 1) * self.delta_y_cm
        tgc_db = torch.clamp(self.tgc_db_per_cm * depth_cm, max=self.tgc_max_db)
        tgc_gain = torch.pow(10.0, tgc_db / 20.0)
        effective_energy = torch.clamp(energy * tgc_gain, 0.0, 1.0)

        # 6. Incidence angle from the impedance gradient (physical units): beam along +y
        dz_dy = torch.zeros_like(impedance)
        dz_dx = torch.zeros_like(impedance)
        dz_dy[:, 1:, :] = (impedance[:, 1:, :] - impedance[:, :-1, :]) / self.axial_resolution_mm
        dz_dx[:, :, 1:-1] = (impedance[:, :, 2:] - impedance[:, :, :-2]) / (2.0 * self.lateral_resolution_mm)
        grad_norm = torch.sqrt(dz_dy ** 2 + dz_dx ** 2 + 1e-12)
        cos_theta = torch.clamp(torch.abs(dz_dy) / grad_norm, 0.0, 1.0)
        specular = torch.pow(cos_theta, self.specular_exponent)

        amp = energy * tgc_gain
        reflection_raw = amp * refl * specular

        # 7. Rayleigh speckle: complex Gaussian scatterers -> PSF -> envelope
        gen = generator if generator is not None else self.generator
        rnd = torch.randn((b, 4, h, w), generator=gen, device=self.device, dtype=torch.float32)
        scat, noise = rnd[:, :2], rnd[:, 2:]
        scatter_strength = torch.where(bone_mask, 0.5, 1.0) * self.scatter_amplitude
        scat = scat * (amp * scatter_strength).unsqueeze(1)

        # 8. PSF (one grouped separable conv over [reflection, scatter_re, scatter_im])
        stacked = torch.cat([reflection_raw.unsqueeze(1), scat], dim=1)
        conv = self._psf(stacked)
        r_conv = conv[:, 0]
        b_env = torch.sqrt(conv[:, 1] ** 2 + conv[:, 2] ** 2)

        # 9. Envelope (incoherent sum) and log compression to a fixed dynamic range
        noise_amp = self.noise_amplitude
        if self.receiver_noise_amplitude is not None:   # receiver-input noise is amplified by the TGC (DEC-042)
            noise_amp = noise_amp + self.receiver_noise_amplitude * tgc_gain
        noise_env = noise_amp * torch.sqrt(noise[:, 0] ** 2 + noise[:, 1] ** 2)
        envelope = r_conv + b_env + noise_env
        db = 20.0 * torch.log10(envelope / self.reference_amplitude + 1e-12)
        bmode = torch.clamp(1.0 + db / self.dynamic_range_db, 0.0, 1.0)

        out = {
            "bmode": bmode,
            "reflection": r_conv,
            "backscatter": b_env,
            "energy": energy,
            "effective_energy": effective_energy,
            "bone_mask": bone_f,
        }
        if is_single:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out
