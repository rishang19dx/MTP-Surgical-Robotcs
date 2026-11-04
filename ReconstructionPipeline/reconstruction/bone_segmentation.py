"""
2D Bone Surface Segmentation for Ultrasound Frames (ground-truth free).

`segment_from_bmode` is a vectorized classical bone-surface extractor in the spirit of
shadow-peak / confidence-map methods (Karamalis et al. 2012; Pandey et al.): per scanline it keeps
the last bright ridge that is followed by a sustained distal acoustic shadow, then removes
isolated responses with a column-continuity (run-length) filter. At most one surface pixel is
returned per scanline.

`segment_from_label_map` extracts the first-hit bone surface from a label slice; it is an
evaluation oracle only and must not feed the reconstruction pipeline.
"""

from typing import Optional
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d


def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a


class BoneSurfaceSegmenter:
    """Extracts 2D bone boundary pixel coordinates [x_px, y_px] from ultrasound frames."""

    def __init__(
        self,
        dropout_probability: float = 0.0,
        intensity_threshold: float = 0.75,
        shadow_max_intensity: float = 0.35,
        min_contrast: float = 0.40,
        min_confidence: float = 0.20,
        min_depth_y: int = 20,
        axial_resolution_mm: float = 0.375,
        shadow_start_mm: float = 2.5,
        shadow_end_mm: float = 10.0,
        max_bright_fraction: float = 0.15,
        max_jump_px: int = 3,
        min_run_columns: int = 5,
        lateral_smoothing_px: int = 3,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        # Deprecated, unused (kept so old call sites do not break)
        gradient_threshold: Optional[float] = None,
        shadow_contrast_ratio: Optional[float] = None,
    ):
        """
        Args:
            dropout_probability: Optional random removal of detected points (augmentation). Default 0:
                the simulator already produces physical dropout (incidence, shadowing).
            intensity_threshold: Minimum ridge intensity.
            shadow_max_intensity: Maximum mean intensity of the distal window (sustained shadow).
            min_contrast: Minimum (ridge - distal mean).
            min_confidence: Minimum confidence at the ridge (if a confidence map is given).
            min_depth_y: Rows to skip in the near field (ring-down / skin).
            axial_resolution_mm: Row spacing [mm], used to convert the shadow window to pixels.
            shadow_start_mm / shadow_end_mm: Distal window below the ridge that must be shadowed.
            max_bright_fraction: Max fraction of distal-window pixels brighter than the ridge threshold
                minus 0.25 (rejects ridges followed by textured tissue).
            max_jump_px: Max axial jump between laterally adjacent detections in one run.
            min_run_columns: Runs shorter than this (in columns) are dropped as isolated responses.
            lateral_smoothing_px: Lateral box filter width applied before ridge detection.
            rng: numpy Generator for dropout. If None, one is created from `seed`.
        """
        self.dropout_prob = dropout_probability
        self.intensity_thresh = intensity_threshold
        self.shadow_max = shadow_max_intensity
        self.min_contrast = min_contrast
        self.min_confidence = min_confidence
        self.min_depth_y = min_depth_y
        self.axial_resolution_mm = axial_resolution_mm
        self.shadow_start_mm = shadow_start_mm
        self.shadow_end_mm = shadow_end_mm
        self.max_bright_fraction = max_bright_fraction
        self.max_jump_px = max_jump_px
        self.min_run_columns = min_run_columns
        self.lateral_smoothing_px = lateral_smoothing_px
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def segment_from_bmode(
        self,
        bmode_image: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Args:
            bmode_image: (H, W) normalized intensity in [0, 1], rows = depth.
            confidence_map: Optional (H, W) image-derived confidence.
            rng: Optional numpy Generator for dropout (overrides the instance generator).

        Returns:
            (N, 2) int32 array of [x_px, y_px], at most one point per column.
        """
        img = np.asarray(_to_numpy(bmode_image), dtype=np.float32)
        conf = _to_numpy(confidence_map)
        h, w = img.shape
        if self.lateral_smoothing_px > 1:
            img = uniform_filter1d(img, size=self.lateral_smoothing_px, axis=1, mode="nearest")

        s0 = max(1, int(round(self.shadow_start_mm / self.axial_resolution_mm)))
        s1 = max(s0 + 2, int(round(self.shadow_end_mm / self.axial_resolution_mm)))
        y_lo = max(1, self.min_depth_y)
        y_hi = h - 1 - s0 - (s1 - s0) // 2  # need at least half of the distal window inside the image
        if y_hi <= y_lo:
            return np.empty((0, 2), dtype=np.int32)

        # 1. Ridge candidates: strict rise, non-strict fall (first pixel of a plateau)
        c = img[1:-1]
        ridge = (c >= self.intensity_thresh) & (c > img[:-2]) & (c >= img[2:])
        ridge = np.vstack([np.zeros((1, w), bool), ridge, np.zeros((1, w), bool)])
        ridge[:y_lo] = False
        ridge[y_hi:] = False
        if conf is not None:
            ridge &= np.asarray(conf) >= self.min_confidence

        # 2. Distal window statistics via prefix sums (window rows [y+s0, y+s1), clipped to image)
        pad = np.zeros((1, w), np.float32)
        csum = np.vstack([pad, np.cumsum(img, axis=0)])
        bright = (img > self.intensity_thresh - 0.25).astype(np.float32)
        bsum = np.vstack([pad, np.cumsum(bright, axis=0)])
        ys = np.arange(h)
        a = np.minimum(ys + s0, h)
        b = np.minimum(ys + s1, h)
        n = np.maximum(b - a, 1).astype(np.float32)[:, None]
        distal_mean = (csum[b] - csum[a]) / n
        distal_bright = (bsum[b] - bsum[a]) / n

        contrast = img - distal_mean
        valid = (ridge & (distal_mean <= self.shadow_max) & (contrast >= self.min_contrast)
                 & (distal_bright <= self.max_bright_fraction))
        score = np.where(valid, contrast * (conf if conf is not None else 1.0), -np.inf)

        # 3. One surface per scanline
        y_best = np.argmax(score, axis=0)
        has = np.isfinite(score[y_best, np.arange(w)])
        xs = np.nonzero(has)[0]
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        yb = y_best[xs]

        # Centre on saturated / flat ridge plateaus (the PSF spreads a reflector symmetrically)
        k = np.arange(1, 9)
        rows = np.minimum(yb[:, None] + k[None, :], h - 1)
        flat = img[rows, xs[:, None]] >= (img[yb, xs] - 0.02)[:, None]
        plateau = np.cumprod(flat, axis=1).sum(axis=1)
        yb = yb + (plateau + 1) // 2

        # 4. Column continuity: split into runs at lateral gaps or axial jumps, drop short runs
        brk = np.ones(xs.size, bool)
        brk[1:] = (np.diff(xs) > 1) | (np.abs(np.diff(yb)) > self.max_jump_px)
        run_id = np.cumsum(brk) - 1
        run_len = np.bincount(run_id)
        keep = run_len[run_id] >= self.min_run_columns
        xs, yb = xs[keep], yb[keep]

        # 5. Optional stochastic dropout (explicit generator, no global RNG)
        if self.dropout_prob > 0.0 and xs.size:
            gen = rng if rng is not None else self.rng
            m = gen.random(xs.size) >= self.dropout_prob
            xs, yb = xs[m], yb[m]

        return np.stack([xs, yb], axis=1).astype(np.int32)

    def segment_from_label_map(
        self,
        label_slice: np.ndarray,
        bone_label_id: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        EVALUATION ORACLE: first-hit (topmost along the beam) bone pixel per column of a label slice.
        Applies the same optional dropout as `segment_from_bmode`.
        """
        lab = np.asarray(_to_numpy(label_slice))
        bone = lab == bone_label_id
        has = bone.any(axis=0)
        xs = np.nonzero(has)[0]
        ys = np.argmax(bone, axis=0)[xs]
        if self.dropout_prob > 0.0 and xs.size:
            gen = rng if rng is not None else self.rng
            m = gen.random(xs.size) >= self.dropout_prob
            xs, ys = xs[m], ys[m]
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.stack([xs, ys], axis=1).astype(np.int32)

    def segment(
        self,
        bmode_image: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        label_slice: Optional[np.ndarray] = None,
        source: str = "bmode",
    ) -> np.ndarray:
        """Unified interface. Defaults to B-mode; source='label' uses the oracle."""
        if source == "label" and label_slice is not None:
            return self.segment_from_label_map(label_slice)
        return self.segment_from_bmode(bmode_image, confidence_map=confidence_map)
