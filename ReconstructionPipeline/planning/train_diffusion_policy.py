"""
Training Script for SOTA 3D Diffusion Policy (DP3 / EDM) on Ultrasound Scanning.
Collects expert demonstration chunks (from heuristic & greedy exploration) on patient
anatomy (TotalSegmentator clinical volumes or procedural anatomy) and trains the
AdaLN-Zero Diffusion Transformer policy with Karras continuous-time preconditioning.
"""

import os
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy


class UltrasoundDemonstrationDataset(Dataset):
    """
    Trajectory action chunk dataset:
    Samples observation condition (PointNet reconstructed points + probe pose)
    paired with future continuous action chunks a_{t:t+H} in [-1, 1]^4.
    """

    def __init__(
        self,
        observations: List[Dict[str, np.ndarray]],
        action_trajectories: List[np.ndarray],
        horizon: int = 8,
        max_points: int = 256,
    ):
        self.samples = []
        self.horizon = horizon
        self.max_points = max_points

        for obs_seq, act_seq in zip(observations, action_trajectories):
            seq_len = len(act_seq)
            for t in range(seq_len):
                pts = obs_seq[t]["reconstructed_points"]
                pose = obs_seq[t]["probe_pose"]

                # Extract horizon chunk; pad with last action if near boundary
                chunk = act_seq[t : t + horizon]
                if len(chunk) < horizon:
                    pad_len = horizon - len(chunk)
                    pad = np.tile(chunk[-1:], (pad_len, 1)) if len(chunk) > 0 else np.zeros((pad_len, 4))
                    chunk = np.vstack([chunk, pad]) if len(chunk) > 0 else pad

                self.samples.append({
                    "points": pts.astype(np.float32),
                    "pose": pose.astype(np.float32),
                    "chunk": chunk.astype(np.float32),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        pts = s["points"]
        N = len(pts)
        if N > self.max_points:
            indices = np.random.choice(N, self.max_points, replace=False)
            pts = pts[indices]
        elif N < self.max_points:
            if N == 0:
                pts = np.zeros((self.max_points, 3), dtype=np.float32)
            else:
                pad_indices = np.random.choice(N, self.max_points - N, replace=True)
                pts = np.vstack([pts, pts[pad_indices]])

        return {
            "points": torch.tensor(pts, dtype=torch.float32),
            "pose": torch.tensor(s["pose"], dtype=torch.float32),
            "chunk": torch.tensor(s["chunk"], dtype=torch.float32),
        }


def collect_expert_demonstrations(
    env: UltrasoundReconstructionEnv,
    num_episodes: int = 20,
    horizon: int = 8,
) -> Tuple[List[List[Dict[str, np.ndarray]]], List[np.ndarray]]:
    """
    Collects demonstration trajectories using compliant raster and orbital sweeps.
    """
    planner = HeuristicScanPlanner()
    all_obs = []
    all_actions = []

    print(f"[*] Collecting {num_episodes} expert demonstration trajectories...")

    for ep in range(num_episodes):
        obs = env.reset()
        ep_obs = []
        ep_actions = []

        # Alternate between raster and orbital trajectories
        if ep % 2 == 0:
            lines = np.random.choice([5, 6, 7])
            pts = np.random.choice([10, 12, 14])
            traj = planner.generate_linear_raster_trajectory(num_sweep_lines=lines, points_per_line=pts)
        else:
            r1 = np.random.uniform(15.0, 20.0)
            r2 = np.random.uniform(25.0, 32.0)
            traj = planner.generate_orbital_trajectory(radii=[r1, r2], points_per_orbit=24)

        for target_p in traj:
            ep_obs.append(obs)
            curr_p = env.current_pose_params

            dx = np.clip((target_p[0] - curr_p[0]) / env.action_scale_trans, -1.0, 1.0)
            dz = np.clip((target_p[1] - curr_p[1]) / env.action_scale_trans, -1.0, 1.0)
            dyaw = np.clip((target_p[2] - curr_p[2]) / env.action_scale_yaw, -1.0, 1.0)
            dpitch = np.clip((target_p[3] - curr_p[3]) / env.action_scale_pitch, -1.0, 1.0)

            act = np.array([dx, dz, dyaw, dpitch], dtype=np.float32)
            ep_actions.append(act)

            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break

        if len(ep_actions) > horizon:
            all_obs.append(ep_obs)
            all_actions.append(np.array(ep_actions))

    total_chunks = sum(len(a) for a in all_actions)
    print(f"[+] Collected {len(all_actions)} valid trajectories ({total_chunks} total state-action transition steps).")
    return all_obs, all_actions


def train_diffusion_policy(
    patient_dir: Optional[str] = None,
    num_demo_episodes: int = 25,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 2e-4,
    horizon: int = 8,
    save_path: str = "checkpoints/diffusion_policy.pt",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    print(f"[*] Initializing Training Environment (Patient: {patient_dir or 'Synthetic Anatomy'}, Device: {device})...")
    env = UltrasoundReconstructionEnv(patient_data=patient_dir, max_steps=100, device=device)

    # 1. Collect Demonstrations
    obs_trajs, act_trajs = collect_expert_demonstrations(env, num_episodes=num_demo_episodes, horizon=horizon)
    dataset = UltrasoundDemonstrationDataset(obs_trajs, act_trajs, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Instantiate SOTA Diffusion Policy
    policy = SOTADiffusionPolicy(
        action_dim=4,
        horizon=horizon,
        pointnet_latent_dim=128,
        pose_dim=4,
        hidden_dim=256,
        num_layers=4,
        device=device,
    )
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print(f"[*] Starting Diffusion Training for {epochs} epochs ({len(dataset)} samples)...")
    start_time = time.time()

    policy.train()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch in dataloader:
            pts = batch["points"].to(device)
            pose = batch["pose"].to(device)
            chunks = batch["chunk"].to(device)

            optimizer.zero_grad()
            loss = policy.compute_loss(clean_actions=chunks, point_cloud=pts, probe_pose=pose)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                "epoch": epoch,
                "state_dict": policy.state_dict(),
                "loss": best_loss,
                "horizon": horizon,
            }, save_path)

        if epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d}/{epochs:02d} | EDM Loss: {mean_loss:.5f} (Best: {best_loss:.5f}) | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {elapsed:.1f}s")

    print(f"[✓] Training complete. Best checkpoint saved to: {save_path}")

    # 4. Quick Verification Rollout
    print("[*] Running verification test rollout with trained policy...")
    policy.eval()
    obs = env.reset()
    for s in range(30):
        pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=device)
        pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=device)
        with torch.no_grad():
            chunk = policy.sample_actions(pts_t, pose_t, num_steps=5, method="heun").squeeze(0).cpu().numpy()
        for act in chunk[:2]: # Execute first 2 actions
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        if term or trunc:
            break

    reconstructed_pts = env.accumulator.get_points()
    cov = float(info.get("coverage_ratio", 0.0)) * 100.0
    print(f"[✓] Verification Rollout Result: Points={len(reconstructed_pts)}, Coverage={cov:.1f}%, Mean Force={info.get('normal_force', 0.0):.2f}N")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOTA 3D Diffusion Policy for Autonomous Ultrasound")
    parser.add_argument("--patient-dir", type=str, default=None, help="Path to clinical TotalSegmentator patient directory")
    parser.add_argument("--num-demo-episodes", type=int, default=25, help="Number of demonstration episodes to collect")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--horizon", type=int, default=8, help="Action chunk prediction horizon")
    parser.add_argument("--save-path", type=str, default="checkpoints/diffusion_policy.pt", help="Checkpoint save path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    train_diffusion_policy(
        patient_dir=args.patient_dir,
        num_demo_episodes=args.num_demo_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        save_path=args.save_path,
        device_str=args.device,
    )
