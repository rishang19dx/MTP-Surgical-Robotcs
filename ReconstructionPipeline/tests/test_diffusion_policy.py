"""
Unit and integration tests for the SOTA 3D Diffusion Policy.
Verifies EDM continuous-time loss, Heun/Euler/Consistency samplers, and PointNet integration.
"""

import pytest
import torch
import numpy as np

from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy


def test_diffusion_policy_initialization():
    policy = SOTADiffusionPolicy(
        action_dim=4,
        horizon=8,
        pointnet_latent_dim=64,
        pose_dim=4,
        hidden_dim=128,
        num_layers=2,
    )
    assert policy.action_dim == 4
    assert policy.horizon == 8


def test_diffusion_policy_loss_and_backprop():
    policy = SOTADiffusionPolicy(
        action_dim=4,
        horizon=8,
        pointnet_latent_dim=64,
        pose_dim=4,
        hidden_dim=128,
        num_layers=2,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    B = 2
    N = 30
    point_cloud = torch.randn(B, N, 3)
    probe_pose = torch.tensor([[10.0, 5.0, 0.1, -0.05], [12.0, 4.0, 0.0, 0.02]], dtype=torch.float32)
    clean_actions = torch.randn(B, 8, 4)

    optimizer.zero_grad()
    loss = policy.compute_loss(clean_actions, point_cloud, probe_pose)

    assert torch.is_tensor(loss)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    # Ensure gradients reached PointNet and backbone
    for p in policy.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_diffusion_policy_sampling_methods():
    policy = SOTADiffusionPolicy(
        action_dim=4,
        horizon=8,
        pointnet_latent_dim=64,
        pose_dim=4,
        hidden_dim=128,
        num_layers=2,
    )

    B = 2
    N = 25
    point_cloud = torch.randn(B, N, 3)
    probe_pose = torch.tensor([[5.0, 2.0, 0.0, 0.0], [6.0, 3.0, 0.1, -0.1]], dtype=torch.float32)

    # 1. 2nd-order Heun solver (5 steps)
    actions_heun = policy.sample_actions(point_cloud, probe_pose, num_steps=5, method="heun")
    assert actions_heun.shape == (B, 8, 4)
    assert (actions_heun >= -1.0).all() and (actions_heun <= 1.0).all()

    # 2. 1st-order Euler solver (3 steps)
    actions_euler = policy.sample_actions(point_cloud, probe_pose, num_steps=3, method="euler")
    assert actions_euler.shape == (B, 8, 4)
    assert (actions_euler >= -1.0).all() and (actions_euler <= 1.0).all()

    # 3. 1-step Fast Consistency mode
    actions_fast = policy.sample_actions(point_cloud, probe_pose, num_steps=1, method="consistency")
    assert actions_fast.shape == (B, 8, 4)
    assert (actions_fast >= -1.0).all() and (actions_fast <= 1.0).all()
