"""
Lightweight PointNet Encoder for Unstructured Partial Point Clouds.
Encodes the current reconstructed bone surface point cloud into a compact latent vector
to feed the active view planning policy without heavy 3D volumetric CNNs.
Based on Qi et al. (CVPR 2017) and active perception literature.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Lightweight PointNet feature extractor mapping (B, N, 3) point coordinates
    to a global latent representation (B, latent_dim).
    Invariant to point order permutations via symmetric max-pooling.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared multi-layer perceptron 1: 3 -> 64 -> 128
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        # Shared multi-layer perceptron 2: 128 -> 256 -> latent_dim
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc_latent = nn.Linear(256, latent_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Tensor of shape (B, N, 3) or (N, 3) representing (x, y, z) coordinates.

        Returns:
            global_feature: Tensor of shape (B, latent_dim) invariant to permutation.
        """
        is_single = points.ndim == 2
        if is_single:
            points = points.unsqueeze(0) # (1, N, 3)

        # PointNet expects (B, channels, N)
        x = points.transpose(1, 2) # (B, 3, N)

        # Multi-layer perceptrons over each point independently
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 256, N)

        # Symmetric max pooling across all N points
        # Global feature is invariant to point order
        global_pool = torch.max(x, dim=2)[0] # (B, 256)

        # Linear projection to latent representation
        latent = F.relu(self.fc_latent(global_pool)) # (B, latent_dim)

        return latent.squeeze(0) if is_single else latent
