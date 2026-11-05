"""
Tests for the Submodular PPO training pipeline: feature fusion, batch-independent PointNet, exact log-probs,
GAE, coverage reward shaping, time-limit bootstrapping from the true next observation, env API smoke test.
"""

import numpy as np
import torch

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.pointnet_encoder import PointNetEncoder, prepare_points
from ReconstructionPipeline.planning.train_submodular_ppo import (
    PPORolloutBuffer,
    SubmodularActorCritic,
    SubmodularFeatureExtractor,
    SubmodularPPOTrainer,
    shaped_reward,
)


def test_feature_extractor_shapes():
    extractor = SubmodularFeatureExtractor(pointnet_latent_dim=64, pose_dim=4, out_features=128, max_points=50)
    pose = torch.randn(2, 4)
    assert extractor(torch.randn(2, 10, 3), pose).shape == (2, 128)
    assert extractor(torch.randn(2, 100, 3), pose).shape == (2, 128)


def test_prepare_points_fixed_size():
    rng = np.random.default_rng(0)
    assert prepare_points(np.random.randn(1000, 3), 256, rng).shape == (256, 3)
    small = np.random.randn(5, 3).astype(np.float32)
    out = prepare_points(small, 256, rng)
    assert out.shape == (256, 3) and set(map(tuple, out)) <= set(map(tuple, small))


def test_pointnet_is_batch_independent():
    """LayerNorm encoder: a cloud's feature must not depend on the other clouds in the batch or train/eval mode."""
    enc = PointNetEncoder(latent_dim=32)
    enc.set_normalization([1.0, 2.0, 3.0], 10.0)
    clouds = torch.randn(4, 64, 3) * 20
    enc.train()
    batched = enc(clouds)
    enc.eval()
    single = torch.stack([enc(c) for c in clouds])
    assert torch.allclose(batched, single, atol=1e-5)
    assert "point_center" in enc.state_dict() and "point_scale" in enc.state_dict()


def test_actor_critic_forward_and_exact_log_prob():
    ac = SubmodularActorCritic(action_dim=4, latent_dim=128, hidden_dim=64)
    points, pose = torch.randn(3, 30, 3), torch.randn(3, 4)
    action, log_prob, entropy, value = ac.get_action_and_value(points, pose)
    assert action.shape == (3, 4) and log_prob.shape == entropy.shape == value.shape == (3,)
    # The stored (unclipped) action must re-evaluate to the same log-prob in the update.
    _, log_prob_eval, _, _ = ac.get_action_and_value(points, pose, action=action)
    assert torch.allclose(log_prob, log_prob_eval, atol=1e-5)
    mean_action, _, _, _ = ac.get_action_and_value(points, pose, deterministic=True)
    assert (mean_action.abs() <= 1.0).all()


def test_rollout_buffer_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    buf = PPORolloutBuffer(buffer_size=4, action_dim=4)
    rewards, values, dones = [1.0, 0.0, 2.0, 1.0], [0.5, 0.2, 0.1, 0.3], [False, True, False, False]
    for r, v, d in zip(rewards, values, dones):
        buf.add(np.zeros((8, 3), np.float32), np.zeros(4, np.float32), torch.zeros(4), torch.tensor(0.0), r, d,
                torch.tensor(v))
    last_v = 0.7
    buf.compute_gae(torch.tensor(last_v), gamma=gamma, gae_lambda=lam)
    adv, gae = np.zeros(4), 0.0
    for t in reversed(range(4)):
        nv = last_v if t == 3 else values[t + 1]
        nnt = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * nv * nnt - values[t]
        gae = delta + gamma * lam * nnt * gae
        adv[t] = gae
    assert np.allclose(buf.advantages.numpy(), adv, atol=1e-6)
    assert np.allclose(buf.returns.numpy(), adv + np.array(values), atol=1e-6)


def test_shaped_reward_uses_coverage_gain_and_keeps_penalties():
    info = {"marginal_gain": 40.0, "coverage_ratio": 0.30}
    env_r = 0.05 * 40.0 - 0.25   # area term minus penalties
    assert np.isclose(shaped_reward(env_r, info, prev_coverage=0.25, mode="coverage"), 100 * 0.05 - 0.25)
    assert shaped_reward(env_r, info, 0.25, mode="env") == env_r


class _FakeEnv:
    """Two-step episodes; the truncating step returns a next observation tagged with pose c1 = 9."""
    c1_bounds, c2_bounds, max_yaw, max_pitch = (-10.0, 10.0), (-10.0, 10.0), 0.5, 0.5

    def __init__(self):
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return {"reconstructed_points": np.random.randn(20, 3).astype(np.float32),
                "probe_pose": np.zeros(4, np.float32), "coverage_ratio": 0.0}

    def step(self, action):
        self.t += 1
        obs = {"reconstructed_points": np.random.randn(20, 3).astype(np.float32),
               "probe_pose": np.array([9.0 if self.t == 2 else 1.0, 0, 0, 0], np.float32),
               "coverage_ratio": 0.1 * self.t}
        info = {"marginal_gain": 0.0, "coverage_ratio": 0.1 * self.t, "normal_force": 4.0}
        return obs, 0.0, False, self.t >= 2, info


def test_truncation_bootstraps_from_true_next_observation():
    ac = SubmodularActorCritic(action_dim=4, latent_dim=64, hidden_dim=32)
    trainer = SubmodularPPOTrainer(env=_FakeEnv(), actor_critic=ac, device=torch.device("cpu"))
    seen = []
    orig = ac.get_value

    def spy(points, pose):
        seen.append(float(pose.reshape(-1, 4)[0, 0]))
        return orig(points, pose)

    ac.get_value = spy
    episodes = trainer.train(total_timesteps=4, rollout_steps=4, num_epochs=1, batch_size=4,
                             normalization_warmup_steps=0)
    assert 9.0 in seen, "truncation bootstrap must evaluate V on the pre-reset next observation"
    assert len(episodes) == 2 and all(e["length"] == 2 for e in episodes)
    assert np.isclose(episodes[0]["return"], 100 * 0.2)   # logged return excludes the bootstrap value


def test_ppo_dry_run_on_env():
    """Two real env steps end to end (reset(seed=, options=), info keys, update)."""
    env = UltrasoundReconstructionEnv(device=torch.device("cpu"), max_steps=2, seed=0)
    ac = SubmodularActorCritic(action_dim=4, latent_dim=128, hidden_dim=64)
    trainer = SubmodularPPOTrainer(env=env, actor_critic=ac, lr=1e-3, device=torch.device("cpu"))
    episodes = trainer.train(total_timesteps=4, rollout_steps=4, num_epochs=1, batch_size=4,
                             normalization_warmup_steps=2)
    assert len(episodes) == 2
    assert all(0.0 <= e["coverage"] <= 1.0 for e in episodes)
    assert float(ac.feature_extractor.pointnet.point_scale) != 1.0   # normalisation was fitted
