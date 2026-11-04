"""
Behaviour cloning of the Flow Matching (CFM) and 3D Diffusion (EDM / DP3-style) action-chunk policies.

Expert: the elevational raster (best heuristic in results/benchmark_latest.md), randomised per episode so the
policy has to generalise rather than memorise one path:
  - start pose from the wide distribution of `planning.start_pose` (same as the benchmark),
  - lateral centre / half-width, number of lines, frames per line and cranio-caudal extent of the raster,
  - the raster starts at the corner nearest the start pose.
Imitation is capped by this expert; its own coverage is printed and stored with the dataset.

Observation = (accumulated bone cloud, probe pose); every observation's cloud is logged (subsampled to
`log_points`). Datasets saved with `--save-demos` and reloaded with `--demo-dataset` must contain these
logged clouds; there is no placeholder geometry.

Episodes are split into train / validation (`--val-frac`, default 15 %) BEFORE chunking, and the checkpoint
with the lowest validation loss (fixed noise seed, so the stochastic CFM / EDM losses are comparable across
epochs) is kept.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.flow_matching_policy import FlowMatchingPolicy
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.planning.pointnet_encoder import (
    PointNetEncoder, fit_point_normalization, pose_normalization, prepare_points)
from ReconstructionPipeline.planning.start_pose import reset_wide


# =============================================================================
# Dataset
# =============================================================================
class UltrasoundDemonstrationDataset(Dataset):
    """
    (cloud, pose) -> future action chunk a_{t:t+H} in [-1, 1]^4 (padded with the last action at the episode end).
    `episodes`: list of dicts with "points" (list of (N_t, 3) clouds), "poses" (T, 4), "actions" (T, 4).
    `deterministic=True` fixes each sample's point subsample (validation).
    """

    def __init__(self, episodes: List[Dict], horizon: int = 8, max_points: int = 256, deterministic: bool = False):
        self.horizon, self.max_points, self.deterministic = horizon, max_points, deterministic
        self.samples = []
        for ep in episodes:
            acts = np.asarray(ep["actions"], np.float32)
            for t in range(len(acts)):
                chunk = acts[t:t + horizon]
                if len(chunk) < horizon:
                    chunk = np.vstack([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)])
                self.samples.append((np.asarray(ep["points"][t], np.float32),
                                     np.asarray(ep["poses"][t], np.float32), chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pts, pose, chunk = self.samples[idx]
        rng = np.random.default_rng(idx) if self.deterministic else None
        return {"points": torch.from_numpy(prepare_points(pts, self.max_points, rng)),
                "pose": torch.from_numpy(pose), "chunk": torch.from_numpy(chunk)}


def save_demos(path: str, episodes: List[Dict], meta: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    flat = {"meta": json.dumps(meta), "num_episodes": len(episodes)}
    for i, ep in enumerate(episodes):
        flat[f"ep{i}_poses"] = np.asarray(ep["poses"], np.float32)
        flat[f"ep{i}_actions"] = np.asarray(ep["actions"], np.float32)
        flat[f"ep{i}_coverage"] = np.float32(ep["coverage"])
        flat[f"ep{i}_npts"] = np.array([len(p) for p in ep["points"]], np.int32)
        flat[f"ep{i}_points"] = np.concatenate([np.asarray(p, np.float32).reshape(-1, 3) for p in ep["points"]])
    np.savez_compressed(path, **flat)


def load_demos(path: str) -> Tuple[List[Dict], Dict]:
    """Loads a dataset written by `save_demos`; raises if the per-step point clouds were not logged."""
    data = np.load(path, allow_pickle=False)
    if "num_episodes" not in data or "ep0_points" not in data:
        raise ValueError(f"{path} has no logged per-step point clouds; re-collect with --save-demos.")
    episodes = []
    for i in range(int(data["num_episodes"])):
        split = np.cumsum(data[f"ep{i}_npts"])[:-1]
        episodes.append({"poses": data[f"ep{i}_poses"], "actions": data[f"ep{i}_actions"],
                         "coverage": float(data[f"ep{i}_coverage"]),
                         "points": np.split(data[f"ep{i}_points"], split)})
    return episodes, json.loads(str(data["meta"]))


# =============================================================================
# Expert
# =============================================================================
def randomized_raster(env: UltrasoundReconstructionEnv, start_pose: np.ndarray, rng: np.random.Generator,
                      budget: int) -> np.ndarray:
    """Elevational raster with randomised extent / spacing, starting at the corner nearest `start_pose`."""
    if env.is_l4_target:
        xc, hw = rng.uniform(-4.0, 4.0), rng.uniform(12.0, 18.0)
        z0, z1 = rng.uniform(-54.0, -48.0), rng.uniform(-16.0, -10.0)
    else:
        xc, hw = rng.uniform(-4.0, 4.0), rng.uniform(15.0, 22.0)
        z0, z1 = rng.uniform(-30.0, -24.0), rng.uniform(24.0, 30.0)
    lines = int(rng.choice([3, 4]))
    ppl = int(np.ceil(rng.uniform(1.0, 1.15) * budget / lines))   # >= budget: the expert acts for every step
    x_range = (xc - hw, xc + hw) if abs(start_pose[0] - (xc - hw)) <= abs(start_pose[0] - (xc + hw)) else (xc + hw, xc - hw)
    z_range = (z0, z1) if abs(start_pose[1] - z0) <= abs(start_pose[1] - z1) else (z1, z0)
    return HeuristicScanPlanner.generate_linear_raster_trajectory(x_range=x_range, z_range=z_range,
                                                                  num_sweep_lines=lines, points_per_line=ppl)


def waypoint_action(env: UltrasoundReconstructionEnv, target: np.ndarray) -> np.ndarray:
    p = env.current_pose_params
    return np.clip(np.array([(target[0] - p[0]) / env.action_scale_trans, (target[1] - p[1]) / env.action_scale_trans,
                             (target[2] - p[2]) / env.action_scale_yaw, (target[3] - p[3]) / env.action_scale_pitch],
                            np.float32), -1.0, 1.0)


def collect_expert_demonstrations(env: UltrasoundReconstructionEnv, num_episodes: int = 160, seed: int = 0,
                                  log_points: int = 512) -> List[Dict]:
    """Rolls out the randomised raster expert for `env.max_steps` steps per episode (the benchmark budget)."""
    rng = np.random.default_rng(seed)
    episodes = []
    t0 = time.time()
    for ep in range(num_episodes):
        obs = reset_wide(env, 200_000 + 1000 * seed + ep)
        traj = randomized_raster(env, obs["probe_pose"], rng, env.max_steps)
        pts, poses, acts = [], [], []
        for target in traj:
            pts.append(prepare_points(obs["reconstructed_points"], log_points, rng)
                       if len(obs["reconstructed_points"]) > log_points else np.asarray(obs["reconstructed_points"], np.float32))
            poses.append(obs["probe_pose"].copy())
            a = waypoint_action(env, target)
            acts.append(a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        episodes.append({"points": pts, "poses": np.array(poses), "actions": np.array(acts),
                         "coverage": float(info["coverage_ratio"])})
        if (ep + 1) % 20 == 0:
            c = [e["coverage"] for e in episodes]
            print(f"  expert ep {ep + 1}/{num_episodes} | Cov@3 {100 * np.mean(c):.1f} +- {100 * np.std(c):.1f}% "
                  f"| {time.time() - t0:.0f}s", flush=True)
    return episodes


def split_episodes(episodes: List[Dict], val_frac: float, seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    idx = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(round(val_frac * len(episodes))))
    return [episodes[i] for i in idx[n_val:]], [episodes[i] for i in idx[:n_val]]


# =============================================================================
# Training (shared loop: model-specific loss closure, val-loss checkpoint selection)
# =============================================================================
def _fit_loop(name: str, params, loss_fn, train_ds, val_ds, epochs: int, batch_size: int, lr: float,
              save_fn, device: torch.device, val_seed: int = 1234) -> List[Dict[str, float]]:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, history, t0 = float("inf"), [], time.time()
    for epoch in range(1, epochs + 1):
        losses = []
        for b in train_dl:
            opt.zero_grad()
            loss = loss_fn(b["points"].to(device), b["pose"].to(device), b["chunk"].to(device), train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(val_seed)   # same noise draws every epoch -> comparable validation loss
            vl = [loss_fn(b["points"].to(device), b["pose"].to(device), b["chunk"].to(device), train=False).item()
                  * len(b["chunk"]) for b in val_dl]
        val = float(np.sum(vl) / len(val_ds))
        tr = float(np.mean(losses))
        improved = val < best
        if improved:
            best = val
            save_fn(epoch, tr, val)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": val})
        print(f"{name} epoch {epoch:3d}/{epochs} | train {tr:.4f} | val {val:.4f}{' *' if improved else ''} "
              f"| {time.time() - t0:.0f}s", flush=True)
    print(f"[{name}] best val loss {best:.4f}")
    return history


def train_flow_matching_policy(train_ds, val_ds, pose_norm, point_norm, epochs: int = 60, batch_size: int = 64,
                               lr: float = 3e-4, horizon: int = 8, save_path: str = "checkpoints/flow_matching.pt",
                               device: torch.device = torch.device("cpu")) -> List[Dict[str, float]]:
    pointnet = PointNetEncoder(latent_dim=128).to(device)
    pointnet.set_normalization(*point_norm)
    policy = FlowMatchingPolicy(action_dim=4, chunk_horizon=horizon, condition_dim=132, device=device).to(device)
    policy.set_pose_normalization(*pose_norm)

    def loss_fn(pts, pose, chunk, train):
        pointnet.train(train)
        policy.train(train)
        return policy.compute_loss(expert_actions=chunk, condition=torch.cat([pointnet(pts), pose], dim=-1))

    def save_fn(epoch, tr, val):
        torch.save({"epoch": epoch, "pointnet_state_dict": pointnet.state_dict(),
                    "policy_state_dict": policy.state_dict(), "train_loss": tr, "val_loss": val,
                    "horizon": horizon}, save_path)

    params = list(pointnet.parameters()) + list(policy.parameters())
    return _fit_loop("CFM", params, loss_fn, train_ds, val_ds, epochs, batch_size, lr, save_fn, device)


def train_edm_diffusion_policy(train_ds, val_ds, pose_norm, point_norm, epochs: int = 60, batch_size: int = 64,
                               lr: float = 2e-4, horizon: int = 8, save_path: str = "checkpoints/diffusion_policy.pt",
                               device: torch.device = torch.device("cpu")) -> List[Dict[str, float]]:
    policy = SOTADiffusionPolicy(action_dim=4, horizon=horizon, pointnet_latent_dim=128, pose_dim=4,
                                 hidden_dim=256, num_layers=4, device=device)
    policy.pointnet.set_normalization(*point_norm)
    with torch.no_grad():
        policy.pose_center.copy_(torch.as_tensor(pose_norm[0]))
        policy.pose_scale.copy_(torch.as_tensor(pose_norm[1]))

    def loss_fn(pts, pose, chunk, train):
        policy.train(train)
        return policy.compute_loss(clean_actions=chunk, point_cloud=pts, probe_pose=pose)

    def save_fn(epoch, tr, val):
        torch.save({"epoch": epoch, "state_dict": policy.state_dict(), "train_loss": tr, "val_loss": val,
                    "horizon": horizon}, save_path)

    return _fit_loop("EDM", list(policy.parameters()), loss_fn, train_ds, val_ds, epochs, batch_size, lr, save_fn, device)


def main(args) -> Dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    env = UltrasoundReconstructionEnv(patient_data=args.patient_dir or None, target_vertebra=args.target_vertebra,
                                      max_steps=args.max_steps, device=device, seed=args.seed)
    if args.demo_dataset:
        episodes, meta = load_demos(args.demo_dataset)
        print(f"[*] Loaded {len(episodes)} expert episodes from {args.demo_dataset} ({meta})")
    else:
        print(f"[*] Collecting {args.num_demo_episodes} randomised-raster expert episodes ({args.max_steps} steps each)...")
        episodes = collect_expert_demonstrations(env, args.num_demo_episodes, seed=args.seed)
        meta = {"expert": "randomized_raster", "max_steps": args.max_steps, "patient_dir": args.patient_dir,
                "target_vertebra": args.target_vertebra, "seed": args.seed}
        if args.save_demos:
            save_demos(args.save_demos, episodes, meta)
            print(f"[*] Saved demonstrations to {args.save_demos}")
    cov = np.array([e["coverage"] for e in episodes])
    print(f"[expert] {len(episodes)} episodes | final Cov@3 {100 * cov.mean():.1f} +- {100 * cov.std():.1f}% "
          f"(min {100 * cov.min():.1f}, max {100 * cov.max():.1f})")

    train_eps, val_eps = split_episodes(episodes, args.val_frac, args.seed)
    train_ds = UltrasoundDemonstrationDataset(train_eps, horizon=args.horizon)
    val_ds = UltrasoundDemonstrationDataset(val_eps, horizon=args.horizon, deterministic=True)
    print(f"[*] {len(train_eps)} train / {len(val_eps)} val episodes -> {len(train_ds)} / {len(val_ds)} samples")
    point_norm = fit_point_normalization([e["points"][t] for e in train_eps for t in range(len(e["points"]))])
    pose_norm = pose_normalization(env)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    report = {"expert_coverage_mean": float(cov.mean()), "expert_coverage_std": float(cov.std()),
              "num_episodes": len(episodes), "train_samples": len(train_ds), "val_samples": len(val_ds)}
    if not args.no_cfm:
        report["cfm"] = train_flow_matching_policy(train_ds, val_ds, pose_norm, point_norm, epochs=args.cfm_epochs,
                                                   batch_size=args.batch_size, lr=3e-4, horizon=args.horizon,
                                                   save_path=args.cfm_save_path, device=device)
    report["diffusion"] = train_edm_diffusion_policy(train_ds, val_ds, pose_norm, point_norm, epochs=args.epochs,
                                                     batch_size=args.batch_size, lr=args.lr, horizon=args.horizon,
                                                     save_path=args.save_path, device=device)
    if args.report_json:
        with open(args.report_json, "w") as fh:
            json.dump(report, fh, indent=1)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Behaviour cloning of CFM and EDM diffusion policies from a randomised raster expert")
    p.add_argument("--patient-dir", type=str, default="data/sample_totalsegmentator_patient")
    p.add_argument("--target-vertebra", type=str, default="vertebrae_L4")
    p.add_argument("--demo-dataset", type=str, default=None, help=".npz written by --save-demos (must contain logged clouds)")
    p.add_argument("--save-demos", type=str, default=None, help="Write the collected demonstrations to this .npz")
    p.add_argument("--num-demo-episodes", type=int, default=160)
    p.add_argument("--max-steps", type=int, default=72, help="Episode length (benchmark step budget)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=60, help="Diffusion epochs")
    p.add_argument("--cfm-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="checkpoints/diffusion_policy.pt")
    p.add_argument("--cfm-save-path", type=str, default="checkpoints/flow_matching.pt")
    p.add_argument("--no-cfm", action="store_true")
    p.add_argument("--report-json", type=str, default=None, help="Write expert coverage and loss curves here")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
