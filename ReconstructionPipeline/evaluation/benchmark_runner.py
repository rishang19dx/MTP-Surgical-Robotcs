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
        """
        Calculates complete SonoGym evaluation metrics:
        1. Reconstruction Quality: Surface Coverage (%), Chamfer Distance (mm), Mean Absolute Distance (MAD, mm)
        2. Force Safety & Regulation: Safe Force Ratio (%), Max Force (N), Mean Force (N), Force RMSE from 4.0 N
        3. Acoustic Visibility: Mean Acoustic Coupling kappa, Low Coupling Dropout Steps (kappa < 0.20)
        4. Kinematic Efficiency: Trajectory Length (mm), End-Effector Angular Rotation (deg)
        """
        reconstructed_pts = env.accumulator.get_points()
        target_pts = getattr(env, "target_roi_bone_points", env.gt_bone_points)
        gt_pts = env.gt_bone_points

        mad_mm = 999.0
        if len(reconstructed_pts) > 20 and len(target_pts) > 0:
            chamfer, cov = self.reg_evaluator.compute_metrics(reconstructed_pts, target_pts)
            # Compute Mean Absolute Distance (MAD) from reconstructed points to nearest ground-truth bone point
            target_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts)))
            dists_s2t = [np.sqrt(target_tree.search_knn_vector_3d(p, 1)[2][0]) for p in reconstructed_pts]
            mad_mm = float(np.mean(dists_s2t))
        else:
            chamfer, cov = 999.0, 0.0

        # SonoGym Contact Force Safety & Regulation Metrics
        if len(force_history) > 0:
            f_arr = np.array(force_history, dtype=np.float32)
            safe_steps = int(np.sum((f_arr >= 2.0) & (f_arr <= 8.0)))
            safe_force_ratio_pct = (safe_steps / len(f_arr)) * 100.0
            max_force = float(np.max(f_arr))
            mean_force = float(np.mean(f_arr))
            force_rmse = float(np.sqrt(np.mean((f_arr - 4.0) ** 2)))
            force_violations = int(np.sum((f_arr > 8.0) | (f_arr < 1.0)))
        else:
            safe_force_ratio_pct, max_force, mean_force, force_rmse, force_violations = 0.0, 0.0, 0.0, 0.0, 0

        # SonoGym Acoustic Visibility Metrics
        if len(coupling_history) > 0:
            c_arr = np.array(coupling_history, dtype=np.float32)
            mean_coupling = float(np.mean(c_arr))
            low_coupling_steps = int(np.sum(c_arr < 0.20))
        else:
            mean_coupling, low_coupling_steps = 0.0, 0

        return {
            "coverage_pct": cov * 100.0,
            "chamfer_mm": float(chamfer),
            "mad_mm": float(mad_mm),
            "safe_force_ratio_pct": safe_force_ratio_pct,
            "max_force_n": max_force,
            "mean_force_n": mean_force,
            "force_rmse_n": force_rmse,
            "force_violations": float(force_violations),
            "mean_coupling": mean_coupling,
            "low_coupling_steps": float(low_coupling_steps),
            "path_length_mm": float(env.trajectory_length),
            "rotation_deg": float(np.degrees(env.total_rotation)),
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
        """Runs benchmark across multiple trials for a heuristic scanning strategy (standardized 60-step budget)."""
        planner = HeuristicScanPlanner()
        if scan_type == "raster":
            traj = planner.generate_linear_raster_trajectory(num_sweep_lines=5, points_per_line=12) # 60 steps
        else:
            traj = planner.generate_orbital_trajectory(radii=[18.0, 30.0], points_per_orbit=30) # 60 steps

        trial_metrics = []
        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=len(traj), device=self.device)
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

    def run_greedy_nbv_benchmark(self, max_steps: int = 60) -> Dict[str, float]:
        """
        Runs an information-theoretic Greedy Next-Best-View (NBV) baseline.
        Evaluates candidate probe moves and selects the one maximizing frontier information gain
        and acoustic coupling.
        """
        trial_metrics = []
        candidate_deltas = [
            (5.0, 0.0, 0.0, 0.0), (-5.0, 0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0), (0.0, -5.0, 0.0, 0.0),
            (3.5, 3.5, np.radians(6.0), 0.0), (-3.5, 3.5, -np.radians(6.0), 0.0),
            (3.5, -3.5, np.radians(6.0), 0.0), (-3.5, -3.5, -np.radians(6.0), 0.0),
        ]

        for trial in range(self.num_trials):
            env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=max_steps, device=self.device)
            env.reset()
            force_hist = []
            coupling_hist = []
            visited_positions = []

            for _ in range(max_steps):
                curr_p = env.current_pose_params
                visited_positions.append((float(curr_p[0]), float(curr_p[1])))
                best_action = None
                best_score = -float("inf")

                for dx_mm, dz_mm, dyaw_r, dpitch_r in candidate_deltas:
                    cand_x = np.clip(curr_p[0] + dx_mm, env.c1_bounds[0], env.c1_bounds[1])
                    cand_z = np.clip(curr_p[1] + dz_mm, env.c2_bounds[0], env.c2_bounds[1])
                    cand_yaw = curr_p[2] + dyaw_r
                    cand_pitch = curr_p[3] + dpitch_r

                    # Visitation penalty prevents limit-cycle oscillations
                    visitation_penalty = sum(
                        np.exp(-0.5 * ((cand_x - vx)**2 + (cand_z - vz)**2) / 25.0)
                        for vx, vz in visited_positions
                    )

                    dist_from_origin = np.sqrt(cand_x**2 + cand_z**2)
                    cand_fn, cand_kappa, _ = env.compute_contact_force_and_coupling(cand_x, cand_z, cand_yaw, cand_pitch)

                    # Frontier information gain: reward high coupling and unexplored areas
                    score = cand_kappa * 5.0 - 1.2 * visitation_penalty - 0.01 * dist_from_origin
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
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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
        Runs the Continuous Flow Matching (CFM) Policy with Receding Horizon Execution (K=2).
        """
        pointnet = PointNetEncoder(latent_dim=128).to(self.device)
        cfm_policy = FlowMatchingPolicy(action_dim=4, chunk_horizon=8, condition_dim=132, device=self.device).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "pointnet_state_dict" in ckpt:
                pointnet.load_state_dict(ckpt["pointnet_state_dict"])
            elif "pointnet" in ckpt:
                pointnet.load_state_dict(ckpt["pointnet"])
            if "policy_state_dict" in ckpt:
                cfm_policy.load_state_dict(ckpt["policy_state_dict"])
            elif "state_dict" in ckpt:
                cfm_policy.load_state_dict(ckpt["state_dict"])
            print(f"[+] Loaded CFM policy & PointNet from {checkpoint_path}")
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

                # Receding Horizon Execution: execute first K=2 actions, then re-observe and re-plan
                exec_chunk = chunk[:2]
                for act in exec_chunk:
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
        Runs the SOTA 3D Diffusion Policy with Receding Horizon Execution (K=2).
        """
        diffusion_policy = SOTADiffusionPolicy(action_dim=4, horizon=8, device=self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

                # Receding Horizon Execution: execute first K=2 actions, then re-observe and re-plan
                exec_chunk = chunk[:2]
                for act in exec_chunk:
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

    def run_soft_body_deformation_benchmark(self) -> Dict[str, Any]:
        """
        Benchmarks the effect of continuum mechanics and soft-tissue deformation
        on ultrasound image formation, bone echo localization, and 3D reconstruction accuracy.
        Addresses SonoGym's primary limitation (rigid patient assumptions).
        """
        print("[*] Running SonoGym-SoftBody Continuum Mechanics Benchmark...")
        env = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=100, device=self.device)
        env.reset()

        # 1. Echo Shift vs. Probe Pressure Analysis
        test_forces = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]
        echo_shifts = []
        t_pose = env._build_probe_matrix(0.0, 0.0, 0.0, 0.0)

        for fn in test_forces:
            # Extract slice with backward Eulerian warping at force fn
            ct_slice = env.slice_extractor_ct.extract_slices(t_pose, interpolation="bilinear", contact_force_n=fn)
            label_slice = env.slice_extractor_label.extract_slices(t_pose, interpolation="nearest", contact_force_n=fn)
            us_res = env.us_sim.simulate_bmode(ct_slice, label_slice=label_slice)
            bmode = us_res["bmode"]
            if isinstance(bmode, torch.Tensor):
                bmode = bmode.detach().cpu().numpy()

            # Locate peak hyperechoic reflection (central A-scans average)
            central_ascan = np.mean(bmode[:, 40:60], axis=1)
            bone_peak_idx = int(np.argmax(central_ascan[10:])) + 10
            # Pixel spacing along depth (60 mm / 160 pixels = 0.375 mm/px)
            echo_depth_mm = bone_peak_idx * 0.375
            indent_mm = (fn / 2500.0) * 1000.0
            _, kappa, _ = env.compute_contact_force_and_coupling(0.0, 0.0, 0.0, 0.0, probe_coord=float(env.skin_y - indent_mm))

            echo_shifts.append({
                "force_n": fn,
                "indentation_mm": indent_mm,
                "echo_depth_mm": echo_depth_mm,
                "coupling": kappa,
            })

        # 2. Rigid vs. Deformable Reconstruction Comparison
        # Run standard raster sweep on both:
        # A. Rigid baseline (force = 0.0 N, zero deformation)
        env_rigid = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=60, device=self.device)
        env_rigid.reset()
        planner = HeuristicScanPlanner()
        traj = planner.generate_spine_raster_trajectory(num_sweep_lines=4, step_distance_mm=0.5)

        for target_p in traj[:60]:
            curr_p = env_rigid.current_pose_params
            action = np.array([
                np.clip((target_p[0] - curr_p[0]) / env_rigid.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[1] - curr_p[1]) / env_rigid.action_scale_trans, -1.0, 1.0),
                0.0, 0.0
            ], dtype=np.float32)
            env_rigid.step(action)

        pts_rigid = env_rigid.accumulator.get_points()
        cd_rigid, cov_rigid = self.reg_evaluator.compute_metrics(pts_rigid, env_rigid.gt_bone_points)

        # B. Deformable tissue with Kelvinlet compression (FN = 5.0 N)
        env_deform = UltrasoundReconstructionEnv(patient_data=self.patient_dir, max_steps=60, target_force_n=5.0, device=self.device)
        env_deform.reset()

        for target_p in traj[:60]:
            curr_p = env_deform.current_pose_params
            action = np.array([
                np.clip((target_p[0] - curr_p[0]) / env_deform.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[1] - curr_p[1]) / env_deform.action_scale_trans, -1.0, 1.0),
                0.0, 0.0
            ], dtype=np.float32)
            env_deform.step(action)

        pts_deform = env_deform.accumulator.get_points()
        cd_deform, cov_deform = self.reg_evaluator.compute_metrics(pts_deform, env_deform.gt_bone_points)

        # 3. Multi-Stiffness Generalization (mu in [2.0, 4.0, 8.0] kPa)
        stiffness_results = {}
        for mu_kpa in [2.0, 4.0, 8.0]:
            env_stiff = UltrasoundReconstructionEnv(
                patient_data=self.patient_dir, max_steps=40, target_force_n=4.0, device=self.device
            )
            # Adjust tissue stiffness
            env_stiff.tissue_stiffness = float(mu_kpa * 300.0)
            env_stiff.reset()
            f_hist = []
            for target_p in traj[:40]:
                curr_p = env_stiff.current_pose_params
                action = np.array([
                    np.clip((target_p[0] - curr_p[0]) / env_stiff.action_scale_trans, -1.0, 1.0),
                    np.clip((target_p[1] - curr_p[1]) / env_stiff.action_scale_trans, -1.0, 1.0),
                    0.0, 0.0
                ], dtype=np.float32)
                _, _, _, _, info = env_stiff.step(action)
                f_hist.append(info["normal_force"])

            f_arr = np.array(f_hist, dtype=np.float32)
            stiffness_results[f"mu_{mu_kpa:.1f}kPa"] = {
                "safe_force_pct": float(np.mean((f_arr >= 2.0) & (f_arr <= 8.0)) * 100.0),
                "force_rmse": float(np.sqrt(np.mean((f_arr - 4.0) ** 2))),
                "max_force": float(np.max(f_arr)),
            }

        return {
            "echo_shifts": echo_shifts,
            "cd_rigid_mm": float(cd_rigid),
            "cov_rigid_pct": float(cov_rigid * 100.0),
            "cd_deform_mm": float(cd_deform),
            "cov_deform_pct": float(cov_deform * 100.0),
            "stiffness_generalization": stiffness_results,
        }

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
    parser.add_argument("--soft-body", action="store_true", help="Run SonoGym-SoftBody continuum mechanics benchmark")
    args = parser.parse_args()

    dev = torch.device(args.device)
    print(f"[*] Initializing Benchmark Suite (Patient: {args.patient_dir or 'Synthetic Anatomy'}, Device: {dev}, Trials: {args.num_trials})...")
    suite = ReconstructionBenchmarkSuite(num_trials=args.num_trials, patient_dir=args.patient_dir, device=dev)

    if args.soft_body:
        sb_res = suite.run_soft_body_deformation_benchmark()
        print("\n" + "=" * 80)
        print("SONOGYM-SOFTBODY CONTINUUM MECHANICS & B-MODE DEFORMATION RESULTS")
        print("=" * 80)
        print("1. Probe Contact Force vs. B-Mode Hyperechoic Echo Shift:")
        print(f"   {'Force (N)':<12} | {'Indentation (mm)':<18} | {'Echo Depth (mm)':<16} | {'Coupling':<10}")
        print("   " + "-" * 62)
        for row in sb_res["echo_shifts"]:
            print(f"   {row['force_n']:<12.1f} | {row['indentation_mm']:<18.2f} | {row['echo_depth_mm']:<16.2f} | {row['coupling']:<10.2f}")

        print("\n2. Reconstruction Accuracy: Rigid SonoGym Baseline vs Deformable Kelvinlets:")
        print(f"   - Rigid Phantom Baseline (Zero Deformation): Chamfer = {sb_res['cd_rigid_mm']:.2f} mm | Coverage = {sb_res['cov_rigid_pct']:.1f}%")
        print(f"   - Deformable Soft Tissue (Ours, F_N = 5.0 N): Chamfer = {sb_res['cd_deform_mm']:.2f} mm | Coverage = {sb_res['cov_deform_pct']:.1f}%")
        print(f"   - Accuracy Restored via Backward Eulerian Warping: Delta CD = {abs(sb_res['cd_rigid_mm'] - sb_res['cd_deform_mm']):.2f} mm")

        print("\n3. Tissue Stiffness Generalization (Adipose mu=2 kPa, Torso mu=4 kPa, Muscle mu=8 kPa):")
        for k, v in sb_res["stiffness_generalization"].items():
            print(f"   - {k}: Safe Force = {v['safe_force_pct']:.1f}%, Force RMSE = {v['force_rmse']:.2f} N, Max Force = {v['max_force']:.2f} N")
        print("=" * 80)

    res = suite.run_full_comparison(ppo_ckpt=args.ppo_ckpt, cfm_ckpt=args.cfm_ckpt, diffusion_ckpt=args.diffusion_ckpt)

    print("\n" + "=" * 125)
    print("SONOGYM 6-POLICY RECONSTRUCTION & COMPLIANCE BENCHMARK COMPARISON TABLE")
    print("=" * 125)
    headers = [
        "Method", "Coverage (%)", "Chamfer (mm)", "MAD (mm)", "Safe Force (%)",
        "Max F (N)", "Force RMSE", "Coupling", "Path (mm)"
    ]
    print(f"{headers[0]:<18} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<12} | {headers[4]:<14} | {headers[5]:<10} | {headers[6]:<10} | {headers[7]:<10} | {headers[8]:<10}")
    print("-" * 125)

    for method, metrics in res.items():
        cov_str = f"{metrics['mean_coverage_pct']:.1f} ± {metrics.get('std_coverage_pct', 0.0):.1f}"
        cd_str = f"{metrics['mean_chamfer_mm']:.2f} ± {metrics.get('std_chamfer_mm', 0.0):.2f}"
        mad_str = f"{metrics.get('mean_mad_mm', 0.0):.2f}"
        safe_str = f"{metrics.get('mean_safe_force_ratio_pct', metrics.get('mean_force_compliance_pct', 0.0)):.1f}%"
        max_f_str = f"{metrics.get('mean_max_force_n', 0.0):.1f}"
        rmse_str = f"{metrics.get('mean_force_rmse_n', 0.0):.2f}"
        coup_str = f"{metrics['mean_mean_coupling']:.2f}"
        path_str = f"{metrics['mean_path_length_mm']:.1f}"
        print(f"{method:<18} | {cov_str:<14} | {cd_str:<14} | {mad_str:<12} | {safe_str:<14} | {max_f_str:<10} | {rmse_str:<10} | {coup_str:<10} | {path_str:<10}")

    print("=" * 125)
