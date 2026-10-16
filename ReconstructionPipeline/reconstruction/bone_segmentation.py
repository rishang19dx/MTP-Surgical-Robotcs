"""
2D Bone Surface Segmentation and Boundary Extractor for Ultrasound Frames.
Supports both ground-truth label map contour extraction with acoustic dropout (simulation mode)
and peak acoustic reflection boundary detection from B-mode frames.
"""

from typing import Optional, Tuple
import numpy as np
import torch


class BoneSurfaceSegmenter:
    """
    Extracts 2D bone boundary pixel coordinates from ultrasound frames.
    """
    def __init__(
        self,
        dropout_probability: float = 0.15,
        intensity_threshold: float = 0.50,
        gradient_threshold: float = 0.20,
    ):
        """
        Args:
            dropout_probability: Probability of missing points to simulate ultrasound signal dropout.
            intensity_threshold: Minimum normalized B-mode intensity for bone boundary.
            gradient_threshold: Minimum vertical intensity gradient for edge detection.
        """
        self.dropout_prob = dropout_probability
        self.intensity_thresh = intensity_threshold
        self.grad_thresh = gradient_threshold

    def segment_from_label_map(
        self,
        label_slice: np.ndarray,
        bone_label_id: int = 1,
    ) -> np.ndarray:
        """
        Extracts the topmost bone surface contour from a ground-truth label slice,
        applying stochastic acoustic dropout.

        Args:
            label_slice: (H, W) 2D array of tissue class integers.
            bone_label_id: Integer identifier for bone tissue.

        Returns:
            contour_pixels: (N, 2) array of [x, y] pixel coordinates.
        """
        h, w = label_slice.shape
        points = []

        # Find the first bone pixel encountered from top to bottom along each scan column x
        for x in range(w):
            col = label_slice[:, x]
            bone_indices = np.where(col == bone_label_id)[0]
            if len(bone_indices) > 0:
                # Topmost bone boundary
                y_first = bone_indices[0]
                
                # Apply stochastic acoustic dropout
                if np.random.rand() > self.dropout_prob:
                    points.append([x, y_first])

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(points, dtype=np.int32)

    def segment_from_bmode(
        self,
        bmode_image: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extracts bone boundary pixels from a B-mode ultrasound image by detecting
        intense specular reflection peaks with an underlying shadow.

        Args:
            bmode_image: (H, W) array of normalized ultrasound intensity [0, 1].
            confidence_map: Optional (H, W) array of acoustic confidence scores.

        Returns:
            contour_pixels: (N, 2) array of [x, y] pixel coordinates.
        """
        h, w = bmode_image.shape
        points = []

        # Search top-to-bottom for prominent reflection peaks
        for x in range(w):
            col = bmode_image[:, x]
            conf_col = confidence_map[:, x] if confidence_map is not None else np.ones(h)

            # Compute vertical gradient (rising edge of bone interface)
            dy = np.diff(col, prepend=col[0])

            # Candidate positions: high intensity + positive rising edge + high confidence
            candidates = np.where(
                (col >= self.intensity_thresh) & 
                (dy >= self.grad_thresh) & 
                (conf_col >= 0.35)
            )[0]

            if len(candidates) > 0:
                y_peak = candidates[0]
                points.append([x, y_peak])

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(points, dtype=np.int32)
