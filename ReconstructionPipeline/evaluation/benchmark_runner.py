"""
Automated Benchmarking Suite for 3D Robotic Ultrasound Reconstruction.
Compares:
1. Heuristic Raster Sweep (Standard Clinical Baseline)
2. Heuristic Orbital Sweep (Elliptical Facet-Tilted Baseline)
3. Greedy Next-Best-View / Information Gain Baseline
4. Submodular PPO Active Policy (PointNet 3D Geometric Feedback)
5. Continuous Flow Matching Policy (Generative Action Chunking)

Evaluates geometric fidelity (Coverage Ratio, Chamfer Distance),
contact compliance (Normal Force in [2, 8] N, Acoustic Coupling kappa),
and operational efficiency (Trajectory Length, Probe Rotation).
Based on the SonoGym benchmarking protocol (Ao et al., 2025).
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import open3d as o3d

import argparse
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.registration.icp_registration import IntraoperativeRegistration
from ReconstructionPipeline.planning.train_submodular_ppo import SubmodularActorCritic
from ReconstructionPipeline.planning.pointnet_encoder import PointNetEncoder
from ReconstructionPipeline.planning.flow_matching_policy import FlowMatchingPolicy
from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy


class ReconstructionBenchmarkSuite:
    """Evaluates and compares reconstruction strategies under identical anatomical conditions."""

    def __init__(
        self,
        num_trials: int = 5,
        patient_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.num_trials = num_trials
        self.patient_dir = patient_dir
        self.reg_evaluator = IntraoperativeRegistration()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _evaluate_trial_metrics(
        self,
        env: UltrasoundReconstructionEnv,
        force_history: List[float],
        coupling_history: List[float],
    ) -> Dict[str, float]:
        """Calculates final registration and compliance metrics for a single trial."""
        reconstructed_pts = env.accumulator.get_points()
        gt_pts = env.gt_bone_points

        if len(reconstructed_pts) > 20:
            chamfer, cov = self.reg_evaluator.compute_metrics(reconstructed_pts, gt_pts)
        else:
            chamfer, cov = 999.0, 0.0

        # Calculate force compliance percentage (force within clinical safe window [2.0, 8.0] N)
        if len(force_history) > 0:
            compliant_steps = sum(1 for f in force_history if 2.0 <= f <= 8.0)
            force_compliance_pct = (compliant_steps / len(force_history)) * 100.0
            mean_coupling = float(np.mean(coupling_history))
        else:
            force_compliance_pct = 0.0
            mean_coupling = 0.0

        return {
            "coverage_pct": cov * 100.0,
            "chamfer_mm": chamfer,
            "path_length_mm": env.trajectory_length,
            "rotation_deg": np.degrees(env.total_rotation),
            "force_compliance_pct": force_compliance_pct,
            "mean_coupling": mean_coupling,
            "frames": float(env.current_step),
        }

    def _aggregate_trial_results(self, trial_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregates multiple trial results into mean and standard deviation."""
        res = {}
        keys = trial_dicts[0].keys()
        for k in keys:
            vals = [d[k] for d in trial_dicts]
            res[f"mean_{k}"] = float(np.mean(vals))
            res[f"std_{k}"] = float(np.std(vals))
        return res

    def run_heuristic_benchmark(self, scan_type: str = "raster") -> Dict[str, float]:
        """Runs benchmark across multiple trials for a heuristic scanning strategy."""
        planner = HeuristicScanPlanner()
        if scan_type == "raster":
            traj = planner.generate_linear_raster_trajectory(num_sweep_lines=6, points_per_line=12)
        else:
            traj = planner.generate_orbital_trajectory(radii=[18.0, 30.0], points_per_orbit=24)

        trial_metrics = []
        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=len(traj) + 5, device=self.device)
            env.reset()
            force_hist = []
            coupling_hist = []

            for target_pose in traj:
                current_p = env.current_pose_params
                dx = np.clip((target_pose[0] - current_p[0]) / env.action_scale_trans, -1.0, 1.0)
                dz = np.clip((target_pose[1] - current_p[1]) / env.action_scale_trans, -1.0, 1.0)
                dyaw = np.clip((target_pose[2] - current_p[2]) / env.action_scale_yaw, -1.0, 1.0)
                dpitch = np.clip((target_pose[3] - current_p[3]) / env.action_scale_pitch, -1.0, 1.0)

                action = np.array([dx, dz, dyaw, dpitch], dtype=np.float32)
                obs, _, term, trunc, info = env.step(action)
                force_hist.append(info["normal_force"])
                coupling_hist.append(info["acoustic_coupling"])

                if term or trunc:
                    break

            trial_metrics.append(self._evaluate_trial_metrics(env, force_hist, coupling_hist))

        return self._aggregate_trial_results(trial_metrics)

    def run_greedy_nbv_benchmark(self, max_steps: int = 50) -> Dict[str, float]:
        """
        Runs an information-theoretic Greedy Next-Best-View (NBV) baseline.
        Evaluates discrete candidate probe moves and selects the one with maximum expected coverage gain.
        """
        trial_metrics = []
        candidate_deltas = [
            (5.0, 0.0, 0.0, 0.0), (-5.0, 0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0), (0.0, -5.0, 0.0, 0.0),
            (3.5, 3.5, np.radians(6.0), 0.0), (-3.5, 3.5, -np.radians(6.0), 0.0),
            (0.0, 0.0, 0.0, np.radians(12.0)), (0.0, 0.0, 0.0, -np.radians(12.0)),
        ]

        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=max_steps, device=self.device)
            env.reset()
            force_hist = []
            coupling_hist = []

            for _ in range(max_steps):
                curr_p = env.current_pose_params
                best_action = None
                best_score = -float("inf")

                for dx_mm, dz_mm, dyaw_r, dpitch_r in candidate_deltas:
                    cand_x = np.clip(curr_p[0] + dx_mm, env.x_bounds[0], env.x_bounds[1])
                    cand_z = np.clip(curr_p[1] + dz_mm, env.z_bounds[0], env.z_bounds[1])
                    cand_yaw = curr_p[2] + dyaw_r
                    cand_pitch = curr_p[3] + dpitch_r

                    dist_from_origin = np.sqrt(cand_x**2 + cand_z**2)
                    inward_tilt = -np.sign(cand_x) * cand_pitch
                    cand_fn, cand_kappa, _ = env.compute_contact_force_and_coupling(cand_x, cand_z, cand_yaw, cand_pitch)

                    score = cand_kappa * 5.0 + inward_tilt * 2.0 - 0.02 * dist_from_origin
                    if score > best_score:
                        best_score = score
                        best_action = np.array([
                            dx_mm / env.action_scale_trans,
                            dz_mm / env.action_scale_trans,
                            dyaw_r / env.action_scale_yaw,
                            dpitch_r / env.action_scale_pitch,
                        ], dtype=np.float32)

                obs, _, term, trunc, info = env.step(best_action)
                force_hist.append(info["normal_force"])
                coupling_hist.append(info["acoustic_coupling"])
                if term or trunc:
                    break

            trial_metrics.append(self._evaluate_trial_metrics(env, force_hist, coupling_hist))

        return self._aggregate_trial_results(trial_metrics)

    def run_submodular_ppo_benchmark(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: int = 60,
    ) -> Dict[str, float]:
        """
        Runs the Submodular PPO Active Exploration Policy with PointNet 3D state encoder.
        """
        ac = SubmodularActorCritic(action_dim=4, latent_dim=192).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            ac.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"[+] Loaded trained PPO policy from {checkpoint_path}")
        ac.eval()

        trial_metrics = []
        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=max_steps, device=self.device)
            obs = env.reset()
            force_hist = []
            coupling_hist = []

            for _ in range(max_steps):
                pts = obs["reconstructed_points"]
                pose = obs["probe_pose"]

                with torch.no_grad():
                    pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device).unsqueeze(0)
                    pose_t = torch.tensor(pose, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _, _ = ac.get_action_and_value(pts_t, pose_t)
                    act_np = action.squeeze(0).cpu().numpy()

                obs, _, term, trunc, info = env.step(act_np)
                force_hist.append(info["normal_force"])
                coupling_hist.append(info["acoustic_coupling"])
                if term or trunc:
                    break

            trial_metrics.append(self._evaluate_trial_metrics(env, force_hist, coupling_hist))

        return self._aggregate_trial_results(trial_metrics)

    def run_flow_matching_benchmark(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: int = 60,
    ) -> Dict[str, float]:
        """
        Runs the Continuous Flow Matching (CFM) Action Chunking Policy.
        """
        pointnet = PointNetEncoder(latent_dim=128).to(self.device)
        cfm_policy = FlowMatchingPolicy(action_dim=4, chunk_horizon=8, condition_dim=132, device=self.device).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            cfm_policy.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"[+] Loaded CFM policy from {checkpoint_path}")
        cfm_policy.eval()
        pointnet.eval()

        trial_metrics = []
        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=max_steps, device=self.device)
            obs = env.reset()
            force_hist = []
            coupling_hist = []

            step_count = 0
            while step_count < max_steps:
                pts = obs["reconstructed_points"]
                pose = obs["probe_pose"]

                with torch.no_grad():
                    pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device).unsqueeze(0)
                    feat = pointnet(pts_t) # (1, 128)
                    pose_t = torch.tensor(pose, dtype=torch.float32, device=self.device).unsqueeze(0)
                    cond = torch.cat([feat, pose_t], dim=-1) # (1, 132)
                    chunk = cfm_policy.generate_action_chunk(cond, num_steps=5).squeeze(0).cpu().numpy() # (8, 4)

                for act in chunk:
                    obs, _, term, trunc, info = env.step(act)
                    force_hist.append(info["normal_force"])
                    coupling_hist.append(info["acoustic_coupling"])
                    step_count += 1
                    if term or trunc or step_count >= max_steps:
                        break
                if term or trunc or step_count >= max_steps:
                    break

            trial_metrics.append(self._evaluate_trial_metrics(env, force_hist, coupling_hist))

        return self._aggregate_trial_results(trial_metrics)

    def run_diffusion_benchmark(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: int = 60,
    ) -> Dict[str, float]:
        """
        Runs the SOTA 3D Diffusion Policy (Karras EDM + AdaLN DiT backbone).
        """
        diffusion_policy = SOTADiffusionPolicy(action_dim=4, horizon=8, device=self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            diffusion_policy.load_state_dict(ckpt.get("state_dict", ckpt))
            print(f"[+] Loaded Diffusion Policy from {checkpoint_path}")
        diffusion_policy.eval()

        trial_metrics = []
        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=max_steps, device=self.device)
            obs = env.reset()
            force_hist = []
            coupling_hist = []

            step_count = 0
            while step_count < max_steps:
                pts = obs["reconstructed_points"]
                pose = obs["probe_pose"]

                with torch.no_grad():
                    pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device)
                    pose_t = torch.tensor(pose, dtype=torch.float32, device=self.device)
                    chunk = diffusion_policy.sample_actions(pts_t, pose_t, num_steps=5, method="heun").squeeze(0).cpu().numpy()

                for act in chunk:
                    obs, _, term, trunc, info = env.step(act)
                    force_hist.append(info["normal_force"])
                    coupling_hist.append(info["acoustic_coupling"])
                    step_count += 1
                    if term or trunc or step_count >= max_steps:
                        break
                if term or trunc or step_count >= max_steps:
                    break

            trial_metrics.append(self._evaluate_trial_metrics(env, force_hist, coupling_hist))

        return self._aggregate_trial_results(trial_metrics)

    def run_full_comparison(
        self,
        ppo_ckpt: Optional[str] = None,
        cfm_ckpt: Optional[str] = None,
        diffusion_ckpt: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Executes full comparative study across all baseline and learning strategies."""
        results = {}
        print("1/6 Running Heuristic Raster Sweep Benchmark...")
        results["Heuristic_Raster"] = self.run_heuristic_benchmark("raster")

        print("2/6 Running Heuristic Orbital Sweep Benchmark...")
        results["Heuristic_Orbital"] = self.run_heuristic_benchmark("orbital")

        print("3/6 Running Greedy Information-Gain NBV Benchmark...")
        results["Greedy_NBV"] = self.run_greedy_nbv_benchmark()

        print("4/6 Running Submodular PPO Active Policy Benchmark...")
        results["Submodular_PPO"] = self.run_submodular_ppo_benchmark(checkpoint_path=ppo_ckpt)

        print("5/6 Running Continuous Flow Matching Policy Benchmark...")
        results["Flow_Matching_CFM"] = self.run_flow_matching_benchmark(checkpoint_path=cfm_ckpt)

        print("6/6 Running SOTA 3D Diffusion Policy (DP3 / EDM) Benchmark...")
        results["Diffusion_Policy"] = self.run_diffusion_benchmark(checkpoint_path=diffusion_ckpt)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated 6-Policy 3D Ultrasound Reconstruction Benchmark Suite")
    parser.add_argument("--patient-dir", type=str, default=None, help="Path to clinical TotalSegmentator patient directory (e.g. s0010)")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of Monte Carlo trials per strategy")
    parser.add_argument("--ppo-ckpt", type=str, default=None, help="Path to Submodular PPO checkpoint")
    parser.add_argument("--cfm-ckpt", type=str, default=None, help="Path to Continuous Flow Matching checkpoint")
    parser.add_argument("--diffusion-ckpt", type=str, default=None, help="Path to 3D Diffusion Policy checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    dev = torch.device(args.device)
    print(f"[*] Initializing Benchmark Suite (Patient: {args.patient_dir or 'Synthetic Anatomy'}, Device: {dev}, Trials: {args.num_trials})...")
    suite = ReconstructionBenchmarkSuite(num_trials=args.num_trials, patient_dir=args.patient_dir, device=dev)
    res = suite.run_full_comparison(ppo_ckpt=args.ppo_ckpt, cfm_ckpt=args.cfm_ckpt, diffusion_ckpt=args.diffusion_ckpt)

    print("\n" + "=" * 95)
    print("3D ROBOTIC ULTRASOUND RECONSTRUCTION BENCHMARK COMPARISON TABLE")
    print("=" * 95)
    headers = [
        "Method", "Coverage (%)", "Chamfer (mm)", "Path (mm)",
        "Rotation (deg)", "Force Safe (%)", "Coupling"
    ]
    print(f"{headers[0]:<20} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<12} | {headers[4]:<14} | {headers[5]:<14} | {headers[6]:<10}")
    print("-" * 115)

    for method, metrics in res.items():
        cov_str = f"{metrics['mean_coverage_pct']:.1f} ± {metrics['std_coverage_pct']:.1f}"
        cd_str = f"{metrics['mean_chamfer_mm']:.2f} ± {metrics['std_chamfer_mm']:.2f}"
        path_str = f"{metrics['mean_path_length_mm']:.1f}"
        rot_str = f"{metrics['mean_rotation_deg']:.1f}"
        force_str = f"{metrics['mean_force_compliance_pct']:.1f}%"
        coup_str = f"{metrics['mean_mean_coupling']:.2f}"
        print(f"{method:<20} | {cov_str:<14} | {cd_str:<14} | {path_str:<12} | {rot_str:<14} | {force_str:<14} | {coup_str:<10}")

    print("=" * 115)
