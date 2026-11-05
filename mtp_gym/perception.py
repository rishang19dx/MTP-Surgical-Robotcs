"""
Batched, image-only perception for the vector env (DEC-041, task T5).

`segment_bmode_batch` is a torch port of `BoneSurfaceSegmenter.segment_from_bmode`
(ReconstructionPipeline/reconstruction/bone_segmentation.py, dropout 0).  It runs on (B, H, W) images and returns
a dense per-column result: `has` (B, W) bool and `y` (B, W) row index, at most one surface pixel per scanline,
the same points as the numpy version (tested for equality in `tests/test_vec_reconstruction.py`).

Steps (as the numpy version):
  1. lateral box filter, ridge candidates (strict rise, non-strict fall) in the depth band, confidence gate;
  2. distal-window mean / bright fraction by prefix sums -> sustained-shadow and contrast tests;
  3. best ridge per column by contrast x confidence, then the plateau centring;
  4. run-length continuity along columns: a run breaks at a column without a detection or an axial jump
     > max_jump_px; runs shorter than min_run_columns are dropped.

Only the B-mode image and the image-only confidence map are inputs (AGENTS.md invariant 5).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter


def _uniform_filter_lateral(img: torch.Tensor, size: int) -> torch.Tensor:
    """scipy.ndimage.uniform_filter1d(axis=1 of (H, W), mode='nearest') on (B, H, W)."""
    left = size // 2
    right = size - 1 - left
    x = F.pad(img.unsqueeze(1), (left, right, 0, 0), mode="replicate")
    k = torch.full((1, 1, 1, size), 1.0 / size, dtype=img.dtype, device=img.device)
    return F.conv2d(x, k).squeeze(1)


@torch.no_grad()
def segment_bmode_batch(img: torch.Tensor, conf: torch.Tensor, seg: BoneSurfaceSegmenter
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B,H,W) B-mode in [0,1] and confidence -> (has (B,W) bool, y (B,W) long row of the surface)."""
    if seg.dropout_prob > 0.0:
        raise NotImplementedError("stochastic dropout is not supported in the batched segmenter")
    img = img.float()
    B, h, w = img.shape
    dev = img.device
    if seg.lateral_smoothing_px > 1:
        img = _uniform_filter_lateral(img, seg.lateral_smoothing_px)
    s0 = max(1, int(round(seg.shadow_start_mm / seg.axial_resolution_mm)))
    s1 = max(s0 + 2, int(round(seg.shadow_end_mm / seg.axial_resolution_mm)))
    y_lo = max(1, seg.min_depth_y)
    y_hi = h - 1 - s0 - (s1 - s0) // 2
    empty = (torch.zeros(B, w, dtype=torch.bool, device=dev), torch.zeros(B, w, dtype=torch.long, device=dev))
    if y_hi <= y_lo:
        return empty
    c = img[:, 1:-1]
    ridge = torch.zeros_like(img, dtype=torch.bool)
    ridge[:, 1:-1] = (c >= seg.intensity_thresh) & (c > img[:, :-2]) & (c >= img[:, 2:])
    ridge[:, :y_lo] = False
    ridge[:, y_hi:] = False
    if conf is not None:
        ridge &= conf >= seg.min_confidence
    pad = torch.zeros(B, 1, w, device=dev)
    csum = torch.cat([pad, img.cumsum(1)], 1)
    bright = (img > seg.intensity_thresh - 0.25).float()
    bsum = torch.cat([pad, bright.cumsum(1)], 1)
    ys = torch.arange(h, device=dev)
    a = (ys + s0).clamp_max(h)
    b = (ys + s1).clamp_max(h)
    n = (b - a).clamp_min(1).float()[None, :, None]
    distal_mean = (csum[:, b] - csum[:, a]) / n
    distal_bright = (bsum[:, b] - bsum[:, a]) / n
    contrast = img - distal_mean
    valid = ridge & (distal_mean <= seg.shadow_max) & (contrast >= seg.min_contrast) \
        & (distal_bright <= seg.max_bright_fraction)
    weight = conf if conf is not None else torch.ones_like(img)
    score = torch.where(valid, contrast * weight, torch.full_like(img, float("-inf")))
    best, y_best = score.max(1)                                   # first max, as np.argmax
    has = torch.isfinite(best)
    # plateau centring
    k = torch.arange(1, 9, device=dev)
    rows = (y_best[:, :, None] + k).clamp_max(h - 1)              # (B, W, 8)
    cols = torch.arange(w, device=dev)[None, :, None].expand(B, w, 8)
    vals = img[torch.arange(B, device=dev)[:, None, None], rows, cols]
    ridge_v = img.gather(1, y_best[:, None, :])[:, 0]             # (B, W)
    flat = vals >= (ridge_v - 0.02)[..., None]
    plateau = torch.cumprod(flat.long(), dim=2).sum(2)
    yb = y_best + (plateau + 1) // 2
    # run-length continuity over columns
    prev_has = torch.zeros_like(has)
    prev_has[:, 1:] = has[:, :-1]
    jump = torch.zeros_like(has)
    jump[:, 1:] = (yb[:, 1:] - yb[:, :-1]).abs() > seg.max_jump_px
    brk = has & (~prev_has | jump)
    run_id = brk.long().cumsum(1) - 1                             # valid where has
    rid = torch.where(has, run_id, torch.full_like(run_id, w))    # park non-detections in slot w
    run_len = torch.zeros(B, w + 1, dtype=torch.long, device=dev).scatter_add_(1, rid, has.long())
    keep = has & (run_len.gather(1, rid) >= seg.min_run_columns)
    return keep, yb


def backproject(has: torch.Tensor, y: torch.Tensor, T_world_probe: torch.Tensor, image_size, physical_size_mm
                ) -> torch.Tensor:
    """Per-column surface pixels -> (B, W, 3) frame-P points (NaN where `has` is False).
    Same pixel -> probe-frame map as `PointCloudAccumulator.pixels_to_probe_coords`."""
    h, w = image_size
    depth_mm, width_mm = physical_size_mm
    sx, sy = width_mm / max(w - 1, 1), depth_mm / max(h - 1, 1)
    B = has.shape[0]
    xs = (torch.arange(w, device=has.device, dtype=torch.float64) - (w - 1) / 2.0) * sx
    p = torch.stack([xs.expand(B, w), y.double() * sy, torch.zeros(B, w, dtype=torch.float64, device=has.device)], -1)
    T = T_world_probe.double()
    world = p @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
    return torch.where(has[..., None], world, torch.full_like(world, float("nan")))
