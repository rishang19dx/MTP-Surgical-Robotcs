"""
Lightweight PointNet encoder for the partial reconstructed bone point cloud (Qi et al., CVPR 2017),
with the DP3-style per-point LayerNorm (Ze et al., RSS 2024) instead of BatchNorm.

BatchNorm made the encoder batch-dependent: PPO rollouts (batch 1, train mode) and PPO updates (mini-batch)
produced different features for the same cloud, so the importance ratio was biased; and BC policies
evaluated in eval mode used running statistics that never matched single-cloud inference. LayerNorm over
channels is per point, so train/eval and batch-1/batch-B outputs are identical.

Coordinates are in frame P millimetres (tens of mm). The encoder normalises them with fixed buffers
`point_center` (3,) and `point_scale` (scalar), fitted from training data by the training scripts and saved
in the state dict, so every consumer (benchmark, server) gets the same normalisation automatically.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def prepare_points(points: np.ndarray, max_points: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Fixed-size (max_points, 3) float32 cloud: random subsample without replacement if N > max_points,
    resample with replacement if 0 < N < max_points (max-pooling is invariant to duplicates).
    An empty cloud returns an all-zero (max_points, 3) array; callers that care must check N themselves.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.zeros((max_points, 3), dtype=np.float32)
    if n >= max_points:
        return pts[rng.choice(n, max_points, replace=False)]
    return np.vstack([pts, pts[rng.choice(n, max_points - n, replace=True)]])


class PointNetEncoder(nn.Module):
    """Maps (B, N, 3) or (N, 3) clouds in mm to a permutation-invariant (B, latent_dim) feature."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("point_center", torch.zeros(in_channels))
        self.register_buffer("point_scale", torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.fc_latent = nn.Linear(256, latent_dim)

    @torch.no_grad()
    def set_normalization(self, center, scale: float) -> None:
        self.point_center.copy_(torch.as_tensor(center, dtype=torch.float32))
        self.point_scale.fill_(float(scale))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        is_single = points.ndim == 2
        if is_single:
            points = points.unsqueeze(0)
        x = (points - self.point_center) / self.point_scale
        x = self.mlp(x)                       # (B, N, 256), per-point
        latent = torch.relu(self.fc_latent(x.max(dim=1)[0]))
        return latent.squeeze(0) if is_single else latent


def fit_point_normalization(clouds) -> tuple:
    """(center (3,), scalar scale) = mean and RMS spread of all points in a list of (N, 3) clouds."""
    allp = np.concatenate([np.asarray(c, np.float32).reshape(-1, 3) for c in clouds if len(c)], axis=0)
    center = allp.mean(axis=0)
    scale = float(np.sqrt(((allp - center) ** 2).sum(axis=1).mean() / 3.0))
    return center, max(scale, 1e-3)


def pose_normalization(env) -> tuple:
    """Fixed probe-pose normalisation from the env geometry: centre/half-width of the scan window, angle limits."""
    c1, c2 = env.c1_bounds, env.c2_bounds
    center = np.array([0.5 * (c1[0] + c1[1]), 0.5 * (c2[0] + c2[1]), 0.0, 0.0], np.float32)
    scale = np.array([0.5 * (c1[1] - c1[0]), 0.5 * (c2[1] - c2[0]), env.max_yaw, env.max_pitch], np.float32)
    return center, scale
