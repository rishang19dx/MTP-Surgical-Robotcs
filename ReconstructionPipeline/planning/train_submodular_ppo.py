"""
Submodular Proximal Policy Optimization (PPO) for Active Robotic Ultrasound Scanning.

Grounded in:
1. Submodular MDP Formulation (Ao et al., SonoGym 2025; Zheng et al., 2021):
   - Monotone submodular reward based on greedy marginal surface area gain:
     r_t = alpha * Delta|M_{0:t}| - (w_path * Delta_path + w_rot * Delta_rot)
   - Bounded (1 - 1/e) theoretical approximation for next-best-view trajectory planning.
2. 3D Geometric State Representation (DP3 / PointNet):
   - Permutation-invariant PointNet encoder for accumulated 3D bone point clouds.
   - Fused with proprioceptive probe pose [x, z, yaw, pitch].
3. Clipped Surrogate PPO with Generalized Advantage Estimation (GAE-Lambda):
   - Continuous Gaussian policy with state-dependent mean and log-std.
   - Entropy bonus ensuring exhaustive exploratory coverage over complex spine anatomy.

Designed for scalable training on high-performance GPU workstations (e.g., RTX 5000 Blackwell Pro).
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.pointnet_encoder import PointNetEncoder


# =============================================================================
# 1. State Preprocessor & Feature Extractor
# =============================================================================
class SubmodularFeatureExtractor(nn.Module):
    """
    Extracts unified latent state representation from 3D point cloud and probe pose.
    """

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
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.Mish(),
            nn.Linear(64, 64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(pointnet_latent_dim + 64, out_features),
            nn.Mish(),
        )

    def forward(self, points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) point cloud
            pose: (B, 4) or (4,) probe pose
        Returns:
            (B, out_features) state embedding
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        # Uniform subsampling / padding to fixed point count
        B, N, _ = points.shape
        if N > self.max_points:
            indices = torch.randperm(N, device=points.device)[: self.max_points]
            points = points[:, indices, :]
        elif N < self.max_points:
            pad_size = self.max_points - N
            if N == 0:
                points = torch.zeros((B, self.max_points, 3), device=points.device)
            else:
                padding = points[:, :1, :].repeat(1, pad_size, 1)
                points = torch.cat([points, padding], dim=1)

        pc_feat = self.pointnet(points)
        pose_feat = self.pose_mlp(pose)
        fused = torch.cat([pc_feat, pose_feat], dim=-1)
        return self.fusion(fused)


# =============================================================================
# 2. Actor-Critic Network Architecture
# =============================================================================
class SubmodularActorCritic(nn.Module):
    """
    Actor-Critic Network for continuous 4-DoF active ultrasound scanning.
    """

    def __init__(
        self,
        action_dim: int = 4,
        latent_dim: int = 192,
        hidden_dim: int = 128,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.feature_extractor = SubmodularFeatureExtractor(out_features=latent_dim)

        # Actor (Policy) Head
        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(), # Actions bounded to [-1.0, 1.0]
        )
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # Critic (Value) Head
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def get_value(self, points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(points, pose)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(
        self,
        points: torch.Tensor,
        pose: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: action, log_prob, entropy, value
        """
        features = self.feature_extractor(points, pose)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value


# =============================================================================
# 3. PPO Rollout Buffer with GAE-Lambda
# =============================================================================
class PPORolloutBuffer:
    """Stores on-policy rollouts and computes GAE advantages."""

    def __init__(self, buffer_size: int, action_dim: int = 4, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.points_list: List[np.ndarray] = []
        self.poses_list: List[np.ndarray] = []
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.pos = 0

    def add(
        self,
        points: np.ndarray,
        pose: np.ndarray,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ):
        self.points_list.append(points)
        self.poses_list.append(pose)
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = 1.0 if done else 0.0
        self.values[self.pos] = value
        self.pos += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Computes Generalized Advantage Estimation (GAE-Lambda)."""
        last_gae = 0.0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values


# =============================================================================
# 4. Submodular PPO Trainer Engine
# =============================================================================
class SubmodularPPOTrainer:
    """
    Submodular PPO Agent for training Next-Best-View robotic ultrasound policies.
    """

    def __init__(
        self,
        env: UltrasoundReconstructionEnv,
        actor_critic: SubmodularActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def train_step(self, buffer: PPORolloutBuffer, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Performs PPO surrogate policy and value function update epochs."""
        total_steps = buffer.buffer_size
        indices = np.arange(total_steps)

        # Normalize advantages
        b_adv = buffer.advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Pre-convert points and poses to tensor batches
        max_pts = self.ac.feature_extractor.max_points
        padded_points = []
        for pts in buffer.points_list:
            N = len(pts)
            if N > max_pts:
                pts = pts[np.random.choice(N, max_pts, replace=False)]
            elif N < max_pts:
                if N == 0:
                    pts = np.zeros((max_pts, 3), dtype=np.float32)
                else:
                    pad = np.repeat(pts[:1], max_pts - N, axis=0)
                    pts = np.vstack([pts, pad])
            padded_points.append(pts)

        b_points = torch.tensor(np.array(padded_points), dtype=torch.float32, device=self.device)
        b_poses = torch.tensor(np.array(buffer.poses_list), dtype=torch.float32, device=self.device)
        b_actions = buffer.actions
        b_log_probs = buffer.log_probs
        b_returns = buffer.returns

        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                mb_pts = b_points[batch_idx]
                mb_pose = b_poses[batch_idx]
                mb_act = b_actions[batch_idx]
                mb_old_logp = b_log_probs[batch_idx]
                mb_adv = b_adv[batch_idx]
                mb_ret = b_returns[batch_idx]

                # Evaluate current policy on batch
                _, new_logp, entropy, new_val = self.ac.get_action_and_value(mb_pts, mb_pose, mb_act)

                # Ratio: r(theta) = exp(log_pi - log_pi_old)
                ratio = torch.exp(new_logp - mb_old_logp)

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def train(
        self,
        total_timesteps: int = 10000,
        rollout_steps: int = 128,
        num_epochs: int = 4,
        batch_size: int = 64,
        eval_interval: int = 1000,
        save_path: Optional[str] = None,
    ):
        """
        Executes on-policy Submodular PPO training loop.
        """
        buffer = PPORolloutBuffer(buffer_size=rollout_steps, action_dim=4, device=str(self.device))
        obs = self.env.reset()

        global_step = 0
        episode_rewards = []
        current_ep_reward = 0.0

        print(f"=== Starting Submodular PPO Training ({total_timesteps} total steps) ===")
        print(f"Device: {self.device} | Rollout size: {rollout_steps} | Batch size: {batch_size}")

        while global_step < total_timesteps:
            buffer.reset()

            # 1. Collect on-policy rollout
            for step_idx in range(rollout_steps):
                global_step += 1

                pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=self.device)
                pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    action, logp, _, val = self.ac.get_action_and_value(pts_t, pose_t)

                act_np = action.cpu().numpy().squeeze(0) if action.dim() > 1 else action.cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(act_np)
                done = terminated or truncated

                buffer.add(
                    points=obs["reconstructed_points"],
                    pose=obs["probe_pose"],
                    action=action.squeeze(0) if action.dim() > 1 else action,
                    log_prob=logp.squeeze(0) if logp.dim() > 0 else logp,
                    reward=reward,
                    done=done,
                    value=val.squeeze(0) if val.dim() > 0 else val,
                )

                current_ep_reward += reward
                obs = next_obs

                if done:
                    episode_rewards.append(current_ep_reward)
                    print(
                        f"Step {global_step:6d} | Ep Reward: {current_ep_reward:7.2f} | "
                        f"Coverage: {info['coverage_ratio']*100:5.1f}% | "
                        f"Path: {info['trajectory_length']:5.1f}mm | Points: {info['total_points']}"
                    )
                    current_ep_reward = 0.0
                    obs = self.env.reset()

            # 2. Compute GAE
            with torch.no_grad():
                pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=self.device)
                pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=self.device)
                last_val = self.ac.get_value(pts_t, pose_t)
            buffer.compute_gae(last_val, gamma=self.gamma, gae_lambda=self.gae_lambda)

            # 3. Optimize policy and value network
            metrics = self.train_step(buffer, num_epochs=num_epochs, batch_size=batch_size)

            # 4. Periodic Evaluation and Checkpointing
            if save_path and (global_step % eval_interval == 0 or global_step >= total_timesteps):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "step": global_step,
                        "state_dict": self.ac.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"[Checkpoint] Model saved to {save_path} at step {global_step}")

        print("=== Training Complete ===")


# =============================================================================
# 5. CLI Entrypoint
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Submodular PPO for Active Robotic Ultrasound Scanning")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total environment steps to train")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per PPO rollout buffer")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for gradient updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss weight")
    parser.add_argument("--save-path", type=str, default="checkpoints/submodular_ppo.pt", help="Checkpoint save path")
    parser.add_argument("--patient-dir", type=str, default="", help="Path to TotalSegmentator subject directory (ct.nii.gz & segmentations/)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 rollout and 1 update for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    patient_arg = args.patient_dir if args.patient_dir else None
    env = UltrasoundReconstructionEnv(patient_data=patient_arg, device=device)
    ac = SubmodularActorCritic(action_dim=4, latent_dim=192).to(device)

    trainer = SubmodularPPOTrainer(
        env=env,
        actor_critic=ac,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        device=device,
    )

    if args.dry_run:
        print("[Dry Run] Running minimal rollout (16 steps) to verify pipeline integrity...")
        trainer.train(
            total_timesteps=16,
            rollout_steps=16,
            num_epochs=1,
            batch_size=8,
            save_path=None,
        )
        print("[Dry Run] Successfully verified Submodular PPO training pipeline!")
    else:
        trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_path=args.save_path,
        )
