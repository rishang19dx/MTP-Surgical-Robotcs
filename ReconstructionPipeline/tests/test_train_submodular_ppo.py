"""
Unit and integration tests for Submodular PPO training pipeline.
Verifies PointNet feature fusion, Actor-Critic distributions, GAE computation, and PPO loss updates.
"""

import pytest
import torch
import numpy as np

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.train_submodular_ppo import (
    SubmodularFeatureExtractor,
    SubmodularActorCritic,
    PPORolloutBuffer,
    SubmodularPPOTrainer,
)


def test_feature_extractor_shapes():
    extractor = SubmodularFeatureExtractor(
        pointnet_latent_dim=64,
        pose_dim=4,
        out_features=128,
        max_points=50,
    )

    B = 2
    # Test case 1: fewer than max_points (tests padding)
    points_small = torch.randn(B, 10, 3)
    pose = torch.randn(B, 4)
    feat1 = extractor(points_small, pose)
    assert feat1.shape == (B, 128)

    # Test case 2: more than max_points (tests subsampling)
    points_large = torch.randn(B, 100, 3)
    feat2 = extractor(points_large, pose)
    assert feat2.shape == (B, 128)


def test_actor_critic_forward():
    ac = SubmodularActorCritic(action_dim=4, latent_dim=128, hidden_dim=64)

    B = 3
    points = torch.randn(B, 30, 3)
    pose = torch.randn(B, 4)

    # Sample action
    action, log_prob, entropy, value = ac.get_action_and_value(points, pose)
    assert action.shape == (B, 4)
    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)
    assert value.shape == (B,)
    assert (action >= -1.0).all() and (action <= 1.0).all()

    # Evaluate given action
    fixed_action = torch.zeros(B, 4)
    _, log_prob_eval, _, _ = ac.get_action_and_value(points, pose, action=fixed_action)
    assert log_prob_eval.shape == (B,)


def test_rollout_buffer_and_gae():
    buffer = PPORolloutBuffer(buffer_size=8, action_dim=4)

    for i in range(8):
        pts = np.random.randn(20, 3).astype(np.float32)
        pose = np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
        act = torch.randn(4)
        logp = torch.tensor(-1.0)
        reward = 1.0 if i % 2 == 0 else 0.5
        done = (i == 7)
        val = torch.tensor(2.0)
        buffer.add(pts, pose, act, logp, reward, done, val)

    last_val = torch.tensor(2.0)
    buffer.compute_gae(last_val, gamma=0.99, gae_lambda=0.95)

    assert buffer.advantages.shape == (8,)
    assert buffer.returns.shape == (8,)
    assert not torch.isnan(buffer.advantages).any()
    assert not torch.isnan(buffer.returns).any()


def test_ppo_dry_run_rollout():
    """Smoke test: runs a minimal 4-step rollout to verify complete end-to-end interaction."""
    device = torch.device("cpu")
    env = UltrasoundReconstructionEnv(device=device)
    ac = SubmodularActorCritic(action_dim=4, latent_dim=128, hidden_dim=64).to(device)

    trainer = SubmodularPPOTrainer(
        env=env,
        actor_critic=ac,
        lr=1e-3,
        device=device,
    )

    # Test minimal 4 steps (takes ~1-2 seconds)
    trainer.train(
        total_timesteps=4,
        rollout_steps=4,
        num_epochs=1,
        batch_size=4,
        save_path=None,
    )
