"""
2D Bone Surface Segmentation and Boundary Extractor for Ultrasound Frames.
Supports robust hyperechoic peak and acoustic shadow detection directly from
raw B-mode ultrasound images, as well as ground-truth label map extraction for comparison.
"""

from typing import Optional, Tuple
import numpy as np
import torch


class BoneSurfaceSegmenter:
    """
    Extracts 2D bone boundary pixel coordinates from ultrasound frames.
    Prioritizes direct clinical B-mode ultrasound signal processing over CT ground-truth labels.
    """
    def __init__(
        self,
        dropout_probability: float = 0.15,
        intensity_threshold: float = 0.18,
        gradient_threshold: float = 0.03,
        shadow_contrast_ratio: float = 1.25,
        min_confidence: float = 0.20,
    ):
        """
        Args:
            dropout_probability: Probability of missing points to simulate stochastic acoustic dropout.
            intensity_threshold: Minimum normalized B-mode intensity for hyperechoic bone reflection.
            gradient_threshold: Minimum vertical intensity gradient for rising edge detection.
            shadow_contrast_ratio: Ratio of peak bone echo to underlying acoustic shadow intensity.
            min_confidence: Minimum acoustic confidence required to accept a bone candidate.
        """
        self.dropout_prob = dropout_probability
        self.intensity_thresh = intensity_threshold
        self.grad_thresh = gradient_threshold
        self.shadow_contrast = shadow_contrast_ratio
        self.min_confidence = min_confidence

    def segment_from_bmode(
        self,
        bmode_image: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extracts bone boundary pixels directly from a B-mode ultrasound image by detecting
        prominent hyperechoic reflection peaks followed by distal acoustic shadows.

        Args:
            bmode_image: (H, W) array of normalized ultrasound intensity [0, 1].
            confidence_map: Optional (H, W) array of acoustic confidence scores.

        Returns:
            contour_pixels: (N, 2) array of [x, y] pixel coordinates.
        """
        if isinstance(bmode_image, torch.Tensor):
            bmode_image = bmode_image.detach().cpu().numpy()
        if isinstance(confidence_map, torch.Tensor):
            confidence_map = confidence_map.detach().cpu().numpy()

        h, w = bmode_image.shape
        points = []

        # Analyze each A-line (scan column x) from near-field to far-field (axial y)
        for x in range(w):
            col = bmode_image[:, x]
            conf_col = confidence_map[:, x] if confidence_map is not None else np.ones(h)

            # Skip columns with complete acoustic decoupling / zero confidence
            if np.max(conf_col) < self.min_confidence:
                continue

            peaks = []
            # Bone interfaces typically appear beyond the superficial transducer ring-down zone (y >= 6)
            for y in range(6, h - 16):
                val = col[y]
                # 1. Local hyperechoic peak condition
                if val >= self.intensity_thresh and val > col[y - 1] and val >= col[y + 1]:
                    # 2. Confidence check at the peak
                    if conf_col[y] >= self.min_confidence:
                        # 3. Acoustic shadow check: distal tissue beneath bone reflects significantly less energy
                        distal_shadow = float(np.mean(col[y + 4 : min(h, y + 24)]))
                        if val >= distal_shadow * self.shadow_contrast or val >= 0.35:
                            peaks.append((val, y))

            if len(peaks) > 0:
                # Stochastic dropout (simulating acoustic specular drop-out at steep angles)
                if np.random.rand() > self.dropout_prob:
                    # Pick the shallowest prominent reflection (first bone cortex encountered by the sound wave)
                    best_y = max(peaks[:2], key=lambda item: item[0])[1]
                    points.append([x, best_y])

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(points, dtype=np.int32)

    def segment_from_label_map(
        self,
        label_slice: np.ndarray,
        bone_label_id: int = 1,
    ) -> np.ndarray:
        """
        Extracts the topmost bone surface contour from a ground-truth label slice (simulation baseline).
        """
        if isinstance(label_slice, torch.Tensor):
            label_slice = label_slice.detach().cpu().numpy()

        h, w = label_slice.shape
        points = []

        for x in range(w):
            col = label_slice[:, x]
            bone_indices = np.where(col == bone_label_id)[0]
            if len(bone_indices) > 0:
                y_first = bone_indices[0]
                if np.random.rand() > self.dropout_prob:
                    points.append([x, y_first])

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(points, dtype=np.int32)

    def segment(
        self,
        bmode_image: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        label_slice: Optional[np.ndarray] = None,
        source: str = "bmode",
    ) -> np.ndarray:
        """
        Unified segmentation interface. Defaults to extracting directly from B-mode ultrasound.
        """
        if source == "bmode":
            return self.segment_from_bmode(bmode_image, confidence_map=confidence_map)
        elif source == "label" and label_slice is not None:
            return self.segment_from_label_map(label_slice)
        else:
            return self.segment_from_bmode(bmode_image, confidence_map=confidence_map)
