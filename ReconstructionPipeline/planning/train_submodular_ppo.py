"""
Submodular PPO for active robotic ultrasound scanning.

- Reward (default `--reward coverage`): 100 x the gain in target-ROI coverage (the benchmark's Cov@3,
  a monotone submodular set function of the frames taken) plus the env's path / rotation / force penalties.
  `--reward env` uses the env's raw reward (0.05 x reconstructed-area gain, which also rewards echoes from
  neighbouring vertebrae) instead.
- State: PointNet (per-point LayerNorm) encoding of the accumulated bone cloud, subsampled once to
  `max_points` per step and stored, so the rollout and the update see the SAME cloud, plus the normalised
  probe pose [c1, c2, yaw, pitch].
- Clipped-surrogate PPO with GAE(lambda). Time-limit truncation bootstraps with gamma * V(s_{T+1}) of the
  true next observation (Pardo et al., 2018); termination (coverage >= target or force > 20 N) does not.
- Episodes start from the wide start-pose distribution of `planning.start_pose` (same as the benchmark).
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.pointnet_encoder import (
    PointNetEncoder, fit_point_normalization, pose_normalization, prepare_points)
from ReconstructionPipeline.planning.start_pose import reset_wide


# =============================================================================
# 1. State feature extractor
# =============================================================================
class SubmodularFeatureExtractor(nn.Module):
    """PointNet(cloud) ++ MLP(normalised pose) -> state embedding. Clouds should come from `prepare_points`."""

    def __init__(
        self,
        pointnet_latent_dim: int = 128,
        pose_dim: int = 4,
        out_features: int = 192,
        max_points: int = 256,
    ):
        super().__init__()
        self.max_points = max_points
        self.pointnet = PointNetEncoder(latent_dim=pointnet_latent_dim)
        self.register_buffer("pose_center", torch.zeros(pose_dim))
        self.register_buffer("pose_scale", torch.ones(pose_dim))
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.Mish(),
            nn.Linear(64, 64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(pointnet_latent_dim + 64, out_features),
            nn.Mish(),
        )

    @torch.no_grad()
    def set_pose_normalization(self, center, scale) -> None:
        self.pose_center.copy_(torch.as_tensor(center, dtype=torch.float32))
        self.pose_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """points (B, N, 3) or (N, 3) in mm; pose (B, 4) or (4,). Returns (B, out_features)."""
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        pc_feat = self.pointnet(points)
        pose_feat = self.pose_mlp((pose - self.pose_center) / self.pose_scale)
        return self.fusion(torch.cat([pc_feat, pose_feat], dim=-1))


# =============================================================================
# 2. Actor-critic
# =============================================================================
class SubmodularActorCritic(nn.Module):
    """Gaussian policy (tanh-bounded mean, state-independent log-std) and value head for the 4-DoF action."""

    def __init__(
        self,
        action_dim: int = 4,
        latent_dim: int = 192,
        hidden_dim: int = 128,
        init_log_std: float = -0.5,
        max_points: int = 256,
    ):
        super().__init__()
        self.feature_extractor = SubmodularFeatureExtractor(out_features=latent_dim, max_points=max_points)
        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def get_value(self, points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        return self.critic(self.feature_extractor(points, pose)).squeeze(-1)

    def get_action_and_value(
        self,
        points: torch.Tensor,
        pose: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (action, log_prob, entropy, value). The sampled action is NOT clipped, so its log-prob is
        exact; the env clips to [-1, 1]. `deterministic` returns the mean (used for evaluation).
        """
        features = self.feature_extractor(points, pose)
        mean = self.actor_mean(features)
        dist = Normal(mean, self.actor_log_std.exp().expand_as(mean))
        if action is None:
            action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value


# =============================================================================
# 3. Rollout buffer with GAE(lambda)
# =============================================================================
class PPORolloutBuffer:
    """On-policy rollout storage. Points must already be fixed-size (`prepare_points`)."""

    def __init__(self, buffer_size: int, action_dim: int = 4, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.points_list: List[np.ndarray] = []
        self.poses_list: List[np.ndarray] = []
        z = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.actions = z(self.buffer_size, self.action_dim)
        self.log_probs, self.rewards, self.dones = z(self.buffer_size), z(self.buffer_size), z(self.buffer_size)
        self.values, self.advantages, self.returns = z(self.buffer_size), z(self.buffer_size), z(self.buffer_size)
        self.pos = 0

    def add(self, points, pose, action, log_prob, reward: float, done: bool, value):
        self.points_list.append(points)
        self.poses_list.append(pose)
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = 1.0 if done else 0.0
        self.values[self.pos] = value
        self.pos += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """dones[t] = 1 cuts bootstrapping after step t (truncation value is already folded into the reward)."""
        last_gae = 0.0
        n = self.pos
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values   # entries >= pos stay unused


def shaped_reward(env_reward: float, info: Dict, prev_coverage: float, mode: str = "coverage") -> float:
    """`coverage`: 100 x coverage gain + the env's penalty terms (env reward minus its 0.05 x area-gain term)."""
    if mode == "env":
        return float(env_reward)
    penalties = float(env_reward) - 0.05 * float(info["marginal_gain"])
    return 100.0 * (float(info["coverage_ratio"]) - prev_coverage) + penalties


# =============================================================================
# 4. Trainer
# =============================================================================
class SubmodularPPOTrainer:
    def __init__(
        self,
        env: UltrasoundReconstructionEnv,
        actor_critic: SubmodularActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        reward_mode: str = "coverage",
        reward_scale: float = 0.1,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.gamma, self.gae_lambda, self.clip_ratio = gamma, gae_lambda, clip_ratio
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.reward_mode, self.reward_scale = reward_mode, reward_scale
        self.max_points = self.ac.feature_extractor.max_points
        self.rng = np.random.default_rng(seed)
        self.episode_seed = 100_000 + 1000 * seed   # disjoint from the benchmark seeds (0..~7000)

    def _tensors(self, pts: np.ndarray, pose: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.as_tensor(pts, dtype=torch.float32, device=self.device),
                torch.as_tensor(pose, dtype=torch.float32, device=self.device))

    def _reset(self) -> Dict:
        self.episode_seed += 1
        return reset_wide(self.env, self.episode_seed)

    def fit_normalization(self, warmup_steps: int = 144) -> None:
        """Pose normalisation from the scan window; point normalisation from a short random-action warm-up."""
        c, sc = pose_normalization(self.env)
        self.ac.feature_extractor.set_pose_normalization(c, sc)
        clouds, obs = [], self._reset()
        for _ in range(warmup_steps):
            obs, _, term, trunc, _ = self.env.step(self.rng.uniform(-1.0, 1.0, 4))
            clouds.append(obs["reconstructed_points"])
            if term or trunc:
                obs = self._reset()
        center, scale = fit_point_normalization(clouds)
        self.ac.feature_extractor.pointnet.set_normalization(center, scale)
        print(f"[norm] pose centre {c} scale {sc}; points centre {center} scale {scale:.2f} mm")

    def train_step(self, buffer: PPORolloutBuffer, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        n = buffer.pos
        idx = np.arange(n)
        adv = buffer.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        b_pts = torch.as_tensor(np.stack(buffer.points_list), dtype=torch.float32, device=self.device)
        b_pose = torch.as_tensor(np.stack(buffer.poses_list), dtype=torch.float32, device=self.device)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(num_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, batch_size):
                mb = idx[start:start + batch_size]
                _, new_logp, entropy, new_val = self.ac.get_action_and_value(b_pts[mb], b_pose[mb], buffer.actions[mb])
                log_ratio = new_logp - buffer.log_probs[mb]
                ratio = log_ratio.exp()
                surr = torch.min(ratio * adv[mb],
                                 torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[mb])
                policy_loss = -surr.mean()
                value_loss = 0.5 * ((new_val - buffer.returns[mb]) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                    stats["clip_frac"].append(float(((ratio - 1).abs() > self.clip_ratio).float().mean()))
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(
        self,
        total_timesteps: int = 10000,
        rollout_steps: int = 256,
        num_epochs: int = 4,
        batch_size: int = 64,
        save_path: Optional[str] = None,
        log_csv: Optional[str] = None,
        max_minutes: Optional[float] = None,
        normalization_warmup_steps: int = 144,
    ) -> List[Dict[str, float]]:
        """Runs PPO; returns the per-episode log (step, return, coverage, length, terminated, max force).
        `normalization_warmup_steps` random-action steps fit the point normalisation first (0 = keep current)."""
        if normalization_warmup_steps > 0:
            self.fit_normalization(normalization_warmup_steps)
        buffer = PPORolloutBuffer(buffer_size=rollout_steps, action_dim=4, device=str(self.device))
        obs = self._reset()
        pts = prepare_points(obs["reconstructed_points"], self.max_points, self.rng)
        prev_cov, ep_ret, ep_len, ep_fmax = float(obs["coverage_ratio"]), 0.0, 0, 0.0
        episodes: List[Dict[str, float]] = []
        writer = None
        if log_csv:
            os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
            fh = open(log_csv, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=["step", "episode", "return", "coverage", "length",
                                                    "terminated", "max_force_n", "wall_s"])
            writer.writeheader()
        t0, global_step = time.time(), 0
        print(f"=== Submodular PPO: {total_timesteps} steps, rollout {rollout_steps}, device {self.device} ===")
        while global_step < total_timesteps:
            if max_minutes is not None and (time.time() - t0) / 60.0 > max_minutes:
                print(f"[time-box] stopping at step {global_step} after {max_minutes} min")
                break
            buffer.reset()
            n_roll = min(rollout_steps, total_timesteps - global_step)
            for _ in range(n_roll):
                global_step += 1
                pts_t, pose_t = self._tensors(pts, obs["probe_pose"])
                with torch.no_grad():
                    action, logp, _, val = self.ac.get_action_and_value(pts_t, pose_t)
                action, logp, val = action.squeeze(0), logp.squeeze(0), val.squeeze(0)
                next_obs, env_r, terminated, truncated, info = self.env.step(action.cpu().numpy())
                r = shaped_reward(env_r, info, prev_cov, self.reward_mode)
                prev_cov = float(info["coverage_ratio"])
                ep_ret, ep_len, ep_fmax = ep_ret + r, ep_len + 1, max(ep_fmax, float(info["normal_force"]))
                next_pts = prepare_points(next_obs["reconstructed_points"], self.max_points, self.rng)
                r_train = r * self.reward_scale
                if truncated and not terminated:
                    # Time-limit: bootstrap with the value of the TRUE next observation (before reset).
                    with torch.no_grad():
                        r_train += self.gamma * float(self.ac.get_value(*self._tensors(next_pts, next_obs["probe_pose"])))
                done = terminated or truncated
                buffer.add(pts, obs["probe_pose"], action, logp, r_train, done, val)
                obs, pts = next_obs, next_pts
                if done:
                    ep = {"step": global_step, "episode": len(episodes) + 1, "return": ep_ret,
                          "coverage": prev_cov, "length": ep_len, "terminated": int(terminated),
                          "max_force_n": ep_fmax, "wall_s": time.time() - t0}
                    episodes.append(ep)
                    if writer:
                        writer.writerow(ep)
                        fh.flush()
                    obs = self._reset()
                    pts = prepare_points(obs["reconstructed_points"], self.max_points, self.rng)
                    prev_cov, ep_ret, ep_len, ep_fmax = float(obs["coverage_ratio"]), 0.0, 0, 0.0
            with torch.no_grad():
                last_val = self.ac.get_value(*self._tensors(pts, obs["probe_pose"])).squeeze(0)
            buffer.compute_gae(last_val, gamma=self.gamma, gae_lambda=self.gae_lambda)
            m = self.train_step(buffer, num_epochs=num_epochs, batch_size=batch_size)
            recent = episodes[-10:]
            if recent:
                print(f"step {global_step:6d} | eps {len(episodes):4d} | ret(10) {np.mean([e['return'] for e in recent]):6.2f} "
                      f"| cov(10) {100 * np.mean([e['coverage'] for e in recent]):5.1f}% | vloss {m['value_loss']:.3f} "
                      f"| kl {m['approx_kl']:.4f} | std {self.ac.actor_log_std.exp().mean().item():.3f} "
                      f"| {time.time() - t0:5.0f}s", flush=True)
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save({"step": global_step, "state_dict": self.ac.state_dict(),
                            "optimizer": self.optimizer.state_dict(), "reward_mode": self.reward_mode}, save_path)
        if writer:
            fh.close()
        print(f"=== PPO done: {global_step} steps, {len(episodes)} episodes, {time.time() - t0:.0f}s ===")
        return episodes


# =============================================================================
# 5. CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Submodular PPO for active robotic ultrasound scanning")
    p.add_argument("--total-timesteps", type=int, default=50000)
    p.add_argument("--max-minutes", type=float, default=None, help="Wall-clock time box (stops after the current rollout)")
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--reward", choices=["coverage", "env"], default="coverage")
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=72, help="Episode length (benchmark step budget)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="checkpoints/submodular_ppo.pt")
    p.add_argument("--log-csv", type=str, default=None, help="Per-episode CSV log")
    p.add_argument("--patient-dir", type=str, default="data/sample_totalsegmentator_patient")
    p.add_argument("--target-vertebra", type=str, default="vertebrae_L4")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true", help="16 steps, 1 update, no checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    env = UltrasoundReconstructionEnv(patient_data=args.patient_dir or None, target_vertebra=args.target_vertebra,
                                      max_steps=args.max_steps, device=device, seed=args.seed)
    ac = SubmodularActorCritic(action_dim=4, latent_dim=192).to(device)
    trainer = SubmodularPPOTrainer(env=env, actor_critic=ac, lr=args.lr, gamma=args.gamma,
                                   gae_lambda=args.gae_lambda, clip_ratio=args.clip_ratio, ent_coef=args.ent_coef,
                                   vf_coef=args.vf_coef, reward_mode=args.reward, reward_scale=args.reward_scale,
                                   seed=args.seed, device=device)
    if args.dry_run:
        trainer.train(total_timesteps=16, rollout_steps=16, num_epochs=1, batch_size=8, save_path=None)
    else:
        trainer.train(total_timesteps=args.total_timesteps, rollout_steps=args.rollout_steps,
                      num_epochs=args.num_epochs, batch_size=args.batch_size, save_path=args.save_path,
                      log_csv=args.log_csv, max_minutes=args.max_minutes)
