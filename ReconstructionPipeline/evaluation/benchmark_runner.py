"""
Benchmark suite for 3D robotic ultrasound bone-surface reconstruction.

Policies: heuristic elevational raster, heuristic orbital, greedy NBV, Submodular PPO, Flow Matching,
Diffusion Policy.  Every trial uses its own seed (initial pose, speckle, segmentation RNG); results are
reported as mean +- std over trials.

Metrics (contract C6, `reconstruction.metrics.compute_reconstruction_metrics`): coverage on the US-visible
dorsal target ROI; precision against the bone SURFACE of the whole volume (env.all_bone_points = boundary
voxels of CT > 300 HU or label 1, so echoes from neighbouring vertebrae are not false positives) and, stricter,
against the dorsal-visible bone surface (env.visible_bone_points = outermost bone voxel per 1 mm (x, z)
column); coverage@2/3 mm, precision@2/3 mm, F-score, chamfer, MAD; plus the coverage-vs-step curve
(mean = normalised AUC), force tracking (mean, std, RMSE to target, max), safety violations (F > 20 N),
coupling, path length and rotation.

The tissue backend is the patient-specific GPU FEM (`--tissue-backend fem`, default) or the analytic
Kelvinlet model (`--tissue-backend kelvinlet`).  One tissue model per backend is built per suite and shared
by all trials (it is stateless between queries apart from its parameters).

`run_soft_body_deformation_benchmark` quantifies what soft-tissue deformation does, using only quantities
computed by the TissueModel / simulator (no hardcoded physics constants), for the FEM with a Kelvinlet
comparison:
  1. force sweep at one pose: indentation (from the undeformed skin under the probe centre; negative when
     the face first touches the paraspinal ridges) and force from the tissue model, bone echo depth detected
     from the B-mode image (NaN when no bone echo passes the sanity check) vs the geometric expectation;
  2. stiffness sweep: the tissue model is re-calibrated for k0 in {0.3, 0.5, 0.8} N/mm;
  3. ablation on the SAME anatomy, SAME trajectory, SAME seed: rigid (no deformation) vs deformation-aware vs
     deformation-UNAWARE back-projection (deformed images, nominal undeformed-skin pose), FEM and Kelvinlet.

Results are written to results/benchmark_<timestamp>.json and a markdown summary results/benchmark_latest.md.
"""

import argparse
import datetime
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.tissue_model import KelvinletTissueModel


def _metrics_fn() -> Callable:
    try:
        from ReconstructionPipeline.reconstruction.metrics import compute_reconstruction_metrics
        return compute_reconstruction_metrics
    except ImportError:  # pragma: no cover - fallback if PERCEPTION's module is absent
        from scipy.spatial import cKDTree

        def _fallback(recon, target, bone=None, tol_mm=3.0, extra_tols_mm=(2.0, 3.0)):
            bone = target if bone is None else bone
            out = {"num_points": int(len(recon))}
            if len(recon) == 0:
                out.update(coverage=0.0, precision=0.0, f_score=0.0, chamfer_mm=float("inf"), mad_mm=float("inf"))
                for t in extra_tols_mm:
                    out[f"coverage@{t:g}"] = out[f"precision@{t:g}"] = out[f"f_score@{t:g}"] = 0.0
                return out
            d_rb = cKDTree(bone).query(recon)[0]
            d_tr = cKDTree(recon).query(target)[0]
            for t in sorted(set(extra_tols_mm) | {tol_mm}):
                c, p = float(np.mean(d_tr <= t)), float(np.mean(d_rb <= t))
                out[f"coverage@{t:g}"], out[f"precision@{t:g}"] = c, p
                out[f"f_score@{t:g}"] = 2 * p * c / (p + c) if p + c > 0 else 0.0
            out.update(coverage=out[f"coverage@{tol_mm:g}"], precision=out[f"precision@{tol_mm:g}"],
                       f_score=out[f"f_score@{tol_mm:g}"], chamfer_mm=float(0.5 * (d_rb.mean() + d_tr.mean())),
                       mad_mm=float(d_rb.mean()))
            return out
        return _fallback


def detect_bone_echo_depth(
    bmode: np.ndarray,
    axial_res_mm: float,
    columns: slice,
    min_depth_mm: float = 3.0,
    shadow_window_mm=(2.0, 10.0),
    min_peak: float = 0.35,
    peak_to_median: float = 2.0,
    shadow_ratio: float = 0.5,
) -> float:
    """
    Depth (mm from the probe face) of the bone echo on the central A-lines, or NaN if no echo passes the
    sanity checks: peak not at the window edges, peak >= min_peak, peak >= peak_to_median x median, and a
    distal shadow (mean in shadow_window below the peak <= shadow_ratio x peak).
    """
    prof = np.asarray(bmode, dtype=np.float64)[:, columns].mean(axis=1)
    n = len(prof)
    r0 = int(np.ceil(min_depth_mm / axial_res_mm))
    s0, s1 = (int(round(v / axial_res_mm)) for v in shadow_window_mm)
    if r0 >= n - s1:
        return float("nan")
    seg = prof[r0:n - s1]
    i = int(np.argmax(seg)) + r0
    peak = prof[i]
    if i <= r0 or i >= n - s1 - 1:
        return float("nan")
    if peak < min_peak or peak < peak_to_median * max(np.median(prof), 1e-6):
        return float("nan")
    if prof[i + s0:i + s1].mean() > shadow_ratio * peak:
        return float("nan")
    return float(i * axial_res_mm)


def ray_bone_depth(env: UltrasoundReconstructionEnv, t_pose: np.ndarray, max_mm: float = 90.0, step: float = 0.25) -> float:
    """Ground-truth distance along the beam from the pose origin to the first bone-label voxel (NaN if none)."""
    lbl = env.label_vol
    org, spc = np.asarray(env.origin, float), np.asarray(env.spacing, float)
    s = np.arange(0.0, max_mm, step)
    pts = t_pose[:3, 3][None] + s[:, None] * t_pose[:3, 1][None]
    idx = np.rint((pts - org) / spc).astype(int)
    D, H, W = lbl.shape
    ok = (idx[:, 0] >= 0) & (idx[:, 0] < W) & (idx[:, 1] >= 0) & (idx[:, 1] < H) & (idx[:, 2] >= 0) & (idx[:, 2] < D)
    hit = np.zeros(len(s), bool)
    hit[ok] = lbl[idx[ok, 2], idx[ok, 1], idx[ok, 0]] == 1
    return float(s[np.argmax(hit)]) if hit.any() else float("nan")


def first_contact(model, t_skin: np.ndarray) -> float:
    fn = getattr(model, "first_contact_indentation", None)
    d0 = float(fn(t_skin)) if fn is not None else 0.0
    return d0 if np.isfinite(d0) else 0.0


def indentation_for_force(model, t_skin: np.ndarray, force_n: float, d_max: float, iters: int = 40) -> float:
    """Bisection on TissueModel.force_at from first contact (works for any backend with monotone force)."""
    lo, hi = first_contact(model, t_skin), float(d_max)
    if model.force_at(t_skin, hi) < force_n:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if model.force_at(t_skin, mid) < force_n:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class ReconstructionBenchmarkSuite:
    """Evaluates and compares reconstruction strategies under identical anatomical conditions."""

    def __init__(
        self,
        num_trials: int = 5,
        patient_dir: Optional[str] = None,
        target_vertebra: Optional[str] = "vertebrae_L4",
        device: Optional[torch.device] = None,
        base_seed: int = 0,
        target_force_n: float = 4.0,
        tissue_backend: str = "fem",
    ):
        self.num_trials = num_trials
        self.patient_dir = patient_dir
        self.target_vertebra = target_vertebra
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_seed = int(base_seed)
        self.target_force_n = float(target_force_n)
        self.metrics_fn = _metrics_fn()
        self.tissue_backend = tissue_backend
        self._patient_data: Optional[Dict[str, Any]] = None
        self._tissue_models: Dict[str, Any] = {}

    # ------------------------------------------------------------- helpers
    @property
    def is_l4(self) -> bool:
        return self.target_vertebra in ["vertebrae_L4", "L4", "l4"]

    @property
    def center_z(self) -> float:
        return -31.0 if self.is_l4 else 0.0

    def patient_data(self) -> Dict[str, Any]:
        """Load the anatomy once and share it between trials (identical anatomy across methods)."""
        if self._patient_data is None:
            if self.patient_dir is None:
                self._patient_data = PatientAssetGenerator.generate_lumbar_phantom()
            elif self.target_vertebra:
                self._patient_data = PatientAssetGenerator.load_totalsegmentator_subject(
                    self.patient_dir, target_vertebrae=[self.target_vertebra], crop_to_spine=False)
            else:
                self._patient_data = PatientAssetGenerator.load_totalsegmentator_subject(self.patient_dir)
        return self._patient_data

    def tissue_model(self, backend: Optional[str] = None):
        """One tissue model per backend, shared by every env of this suite (built by the first env)."""
        backend = backend or self.tissue_backend
        if backend not in self._tissue_models:
            env = UltrasoundReconstructionEnv(patient_data=self.patient_data(), target_vertebra=self.target_vertebra,
                                              max_steps=1, device=self.device, tissue_backend=backend)
            self._tissue_models[backend] = env.tissue_model
            self._window = (env.c1_bounds, env.c2_bounds)
        return self._tissue_models[backend]

    def make_env(self, max_steps: int, seed: int, tissue_backend: Optional[str] = None,
                 **kwargs) -> UltrasoundReconstructionEnv:
        kwargs.setdefault("target_force_n", self.target_force_n)
        kwargs.setdefault("tissue_model", self.tissue_model(tissue_backend))
        return UltrasoundReconstructionEnv(
            patient_data=self.patient_data(), target_vertebra=self.target_vertebra,
            max_steps=max_steps, device=self.device, seed=seed, **kwargs)

    def trial_seed(self, trial: int, offset: int = 0) -> int:
        return self.base_seed + 1000 * offset + trial

    def heuristic_trajectory(self, scan_type: str) -> np.ndarray:
        planner = HeuristicScanPlanner()
        if scan_type == "raster":
            if self.is_l4:
                return planner.generate_linear_raster_trajectory(x_range=(-15.0, 15.0), z_range=(-52.0, -12.0),
                                                                 num_sweep_lines=3, points_per_line=24)
            return planner.generate_linear_raster_trajectory(x_range=(-20.0, 20.0), z_range=(-28.0, 28.0),
                                                             num_sweep_lines=3, points_per_line=20)
        if self.is_l4:
            return planner.generate_orbital_trajectory(center=(0.0, -31.0), radii=[14.0, 24.0], points_per_orbit=36)
        return planner.generate_orbital_trajectory(radii=[18.0, 30.0], points_per_orbit=30)

    @staticmethod
    def follow(env: UltrasoundReconstructionEnv, traj: np.ndarray, hist: Dict[str, List[float]]) -> None:
        for target in traj:
            p = env.current_pose_params
            action = np.array([
                (target[0] - p[0]) / env.action_scale_trans,
                (target[1] - p[1]) / env.action_scale_trans,
                (target[2] - p[2]) / env.action_scale_yaw,
                (target[3] - p[3]) / env.action_scale_pitch,
            ], dtype=np.float32)
            _, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
            ReconstructionBenchmarkSuite._record(hist, info)
            if term or trunc:
                break

    @staticmethod
    def _record(hist: Dict[str, List[float]], info: Dict[str, Any]) -> None:
        for k in ("normal_force", "acoustic_coupling", "indentation_mm", "coverage_ratio"):
            hist.setdefault(k, []).append(float(info[k]))
        hist.setdefault("safety_violation", []).append(float(info["safety_violation"]))

    def _evaluate_trial_metrics(
        self,
        env: UltrasoundReconstructionEnv,
        force_history: List[float],
        coupling_history: List[float],
        hist: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, float]:
        all_bone = getattr(env, "all_bone_points", env.gt_bone_points)
        recon = env.accumulator.get_points()
        m = self.metrics_fn(recon, env.target_roi_bone_points, all_bone, tol_mm=3.0, extra_tols_mm=(2.0, 3.0))
        visible = getattr(env, "visible_bone_points", None)
        mv = self.metrics_fn(recon, env.target_roi_bone_points, visible, tol_mm=3.0, extra_tols_mm=(2.0, 3.0)) \
            if visible is not None else {"precision@2": np.nan, "precision@3": np.nan}
        # Voxels observed in >= 2 frames (suppresses single-frame false detections)
        try:
            recon2 = env.accumulator.get_points(min_observations=2)
        except TypeError:
            recon2 = recon
        m2 = self.metrics_fn(recon2, env.target_roi_bone_points, all_bone, tol_mm=3.0, extra_tols_mm=(2.0, 3.0))
        out = {
            "coverage3_obs2_pct": 100.0 * m2["coverage@3"],
            "precision3_obs2_pct": 100.0 * m2["precision@3"],
            "coverage2_pct": 100.0 * m["coverage@2"],
            "coverage3_pct": 100.0 * m["coverage@3"],
            "precision2_pct": 100.0 * m["precision@2"],
            "precision3_pct": 100.0 * m["precision@3"],
            "precision2_visible_pct": 100.0 * mv["precision@2"],
            "precision3_visible_pct": 100.0 * mv["precision@3"],
            "f_score2": m["f_score@2"],
            "f_score3": m["f_score@3"],
            "chamfer_mm": m["chamfer_mm"],
            "mad_mm": m["mad_mm"],
            "num_points": float(m["num_points"]),
        }
        f = np.asarray(force_history, dtype=np.float64)
        tgt = env.target_force_n
        if len(f):
            out.update(
                mean_force_n=float(f.mean()), std_force_n=float(f.std()), max_force_n=float(f.max()),
                force_rmse_n=float(np.sqrt(np.mean((f - tgt) ** 2))),
                force_in_band_pct=float(100.0 * np.mean(np.abs(f - tgt) <= 1.0)),
                safety_violations=float(np.sum(f > env.safety_force_limit_n)),
            )
        else:
            out.update(mean_force_n=np.nan, std_force_n=np.nan, max_force_n=np.nan, force_rmse_n=np.nan,
                       force_in_band_pct=np.nan, safety_violations=np.nan)
        c = np.asarray(coupling_history, dtype=np.float64)
        out["mean_coupling"] = float(c.mean()) if len(c) else np.nan
        out["low_coupling_steps"] = float(np.sum(c < 0.20)) if len(c) else np.nan
        if hist is not None and hist.get("indentation_mm"):
            out["mean_indentation_mm"] = float(np.mean(hist["indentation_mm"]))
            out["coverage_auc"] = float(np.mean(hist["coverage_ratio"]))
        out["path_length_mm"] = float(env.trajectory_length)
        out["rotation_deg"] = float(np.degrees(env.total_rotation))
        out["frames"] = float(env.current_step)
        return out

    @staticmethod
    def _aggregate_trial_results(trial_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        res = {}
        for k in trial_dicts[0].keys():
            vals = np.array([d.get(k, np.nan) for d in trial_dicts], dtype=np.float64)
            res[f"mean_{k}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
            res[f"std_{k}"] = float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else float("nan")
        res["num_trials"] = len(trial_dicts)
        return res

    def _run_trials(self, max_steps: int, policy_loop: Callable, seed_offset: int, **env_kwargs) -> Dict[str, float]:
        trials = []
        for trial in range(self.num_trials):
            seed = self.trial_seed(trial, seed_offset)
            env = self.make_env(max_steps, seed, **env_kwargs)
            obs = env.reset(seed=seed)
            hist: Dict[str, List[float]] = {}
            policy_loop(env, obs, hist)
            trials.append(self._evaluate_trial_metrics(env, hist.get("normal_force", []),
                                                       hist.get("acoustic_coupling", []), hist))
        return self._aggregate_trial_results(trials)

    # ------------------------------------------------------------- policies
    def run_heuristic_benchmark(self, scan_type: str = "raster") -> Dict[str, float]:
        traj = self.heuristic_trajectory(scan_type)
        return self._run_trials(len(traj), lambda env, obs, hist: self.follow(env, traj, hist),
                                seed_offset=1 if scan_type == "raster" else 2)

    def run_greedy_nbv_benchmark(self, max_steps: int = 60) -> Dict[str, float]:
        """Greedy NBV: picks the move maximising predicted coupling and novelty (visitation penalty)."""
        deltas = [(5.0, 0.0, 0.0, 0.0), (-5.0, 0.0, 0.0, 0.0), (0.0, 5.0, 0.0, 0.0), (0.0, -5.0, 0.0, 0.0),
                  (0.0, 3.0, 0.0, np.radians(8.0)), (0.0, -3.0, 0.0, -np.radians(8.0)),
                  (3.5, 3.5, np.radians(6.0), 0.0), (-3.5, -3.5, -np.radians(6.0), 0.0)]

        def loop(env, obs, hist):
            visited = []
            for _ in range(max_steps):
                p = env.current_pose_params
                visited.append((float(p[0]), float(p[1])))
                best, best_score = None, -np.inf
                for dx, dz, dyaw, dpitch in deltas:
                    cx = float(np.clip(p[0] + dx, *env.c1_bounds))
                    cz = float(np.clip(p[1] + dz, *env.c2_bounds))
                    pen = sum(np.exp(-0.5 * ((cx - vx) ** 2 + (cz - vz) ** 2) / 25.0) for vx, vz in visited)
                    _, kappa, _ = env.compute_contact_force_and_coupling(cx, cz, p[2] + dyaw, p[3] + dpitch)
                    score = 5.0 * kappa - 1.2 * pen - 0.01 * np.hypot(cx, cz - self.center_z)
                    if score > best_score:
                        best_score = score
                        best = np.array([dx / env.action_scale_trans, dz / env.action_scale_trans,
                                         dyaw / env.action_scale_yaw, dpitch / env.action_scale_pitch], np.float32)
                _, _, term, trunc, info = env.step(best)
                self._record(hist, info)
                if term or trunc:
                    break

        return self._run_trials(max_steps, loop, seed_offset=3)

    def run_submodular_ppo_benchmark(self, checkpoint_path: Optional[str] = None, max_steps: int = 60) -> Dict[str, float]:
        from ReconstructionPipeline.planning.train_submodular_ppo import SubmodularActorCritic
        ac = SubmodularActorCritic(action_dim=4, latent_dim=192).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            ac.load_state_dict(ckpt.get("state_dict", ckpt))
        ac.eval()

        def loop(env, obs, hist):
            for _ in range(max_steps):
                with torch.no_grad():
                    pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _, _ = ac.get_action_and_value(pts_t, pose_t)
                obs, _, term, trunc, info = env.step(action.squeeze(0).cpu().numpy())
                self._record(hist, info)
                if term or trunc:
                    break

        return self._run_trials(max_steps, loop, seed_offset=4)

    def _run_chunked(self, max_steps: int, chunk_fn: Callable, seed_offset: int) -> Dict[str, float]:
        def loop(env, obs, hist):
            steps = 0
            while steps < max_steps:
                chunk = chunk_fn(obs)
                for act in chunk[:2]:  # receding horizon K = 2
                    obs, _, term, trunc, info = env.step(act)
                    self._record(hist, info)
                    steps += 1
                    if term or trunc or steps >= max_steps:
                        return
        return self._run_trials(max_steps, loop, seed_offset)

    def run_flow_matching_benchmark(self, checkpoint_path: Optional[str] = None, max_steps: int = 60) -> Dict[str, float]:
        from ReconstructionPipeline.planning.pointnet_encoder import PointNetEncoder
        from ReconstructionPipeline.planning.flow_matching_policy import FlowMatchingPolicy
        pointnet = PointNetEncoder(latent_dim=128).to(self.device)
        cfm = FlowMatchingPolicy(action_dim=4, chunk_horizon=8, condition_dim=132, device=self.device).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            pointnet.load_state_dict(ckpt.get("pointnet_state_dict", ckpt.get("pointnet")))
            cfm.load_state_dict(ckpt.get("policy_state_dict", ckpt.get("state_dict")))
        cfm.eval()
        pointnet.eval()

        def chunk_fn(obs):
            with torch.no_grad():
                pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=self.device).unsqueeze(0)
                pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=self.device).unsqueeze(0)
                cond = torch.cat([pointnet(pts_t), pose_t], dim=-1)
                return cfm.generate_action_chunk(cond, num_steps=5).squeeze(0).cpu().numpy()

        return self._run_chunked(max_steps, chunk_fn, seed_offset=5)

    def run_diffusion_benchmark(self, checkpoint_path: Optional[str] = None, max_steps: int = 60) -> Dict[str, float]:
        from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy
        policy = SOTADiffusionPolicy(action_dim=4, horizon=8, device=self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            policy.load_state_dict(ckpt.get("state_dict", ckpt))
        policy.eval()

        def chunk_fn(obs):
            with torch.no_grad():
                pts_t = torch.tensor(obs["reconstructed_points"], dtype=torch.float32, device=self.device)
                pose_t = torch.tensor(obs["probe_pose"], dtype=torch.float32, device=self.device)
                return policy.sample_actions(pts_t, pose_t, num_steps=5, method="heun").squeeze(0).cpu().numpy()

        return self._run_chunked(max_steps, chunk_fn, seed_offset=6)

    # ------------------------------------------------------------ soft body
    def run_force_sweep(self, forces=(1.0, 2.0, 4.0, 6.0, 8.0, 10.0), seed: int = 0,
                        tissue_backend: Optional[str] = None) -> List[Dict[str, float]]:
        env = self.make_env(10, seed, tissue_backend=tissue_backend)
        env.reset(seed=seed)
        t_skin = env.skin_pose(0.0, self.center_z, 0.0, 0.0)
        model = env.tissue_model
        d_max = env._max_indentation(t_skin)
        d_fc = first_contact(model, t_skin)
        h_center = ray_bone_depth(env, t_skin)
        cols = slice(env.IMAGE_SIZE[1] // 2 - 8, env.IMAGE_SIZE[1] // 2 + 8)
        rows = []
        for fn in forces:
            d = indentation_for_force(model, t_skin, fn, d_max)
            state = model.solve(t_skin, d)
            t_face = state.probe_face_pose(t_skin)
            torch.manual_seed(seed)
            ct = env.slice_extractor_ct.extract_slices(t_face, interpolation="bilinear", tissue_state=state)
            bmode = env.us_sim.simulate_bmode(ct)["bmode"].detach().cpu().numpy()
            rows.append({
                "target_force_n": float(fn),
                "model_force_n": float(state.reaction_force_n),
                "indentation_mm": float(state.indentation_mm),
                "indentation_from_first_contact_mm": float(state.indentation_mm - d_fc),
                "indentation_limited": bool(d >= d_max - 1e-6),
                "echo_depth_mm": detect_bone_echo_depth(bmode, env.axial_res_mm, cols),
                "expected_echo_depth_mm": float(h_center - state.indentation_mm) if np.isfinite(h_center) else float("nan"),
                "tangent_stiffness_n_per_mm": float(state.stiffness_n_per_mm),
            })
        return rows

    def _model_for_k0(self, backend: str, k0: float):
        if backend == "kelvinlet":
            return KelvinletTissueModel.from_patient_data(self.patient_data(), reference_k0_n_per_mm=k0, device=self.device)
        from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
        base = self.tissue_model("fem")
        return FEMTissueModel.from_patient_data(
            self.patient_data(), c1_bounds=self._window[0], c2_bounds=self._window[1], k0_n_per_mm=k0,
            roi_min_mm=base.roi_min, roi_max_mm=base.roi_max, device=self.device, verbose=False)

    def run_stiffness_sweep(self, k0_values=(0.3, 0.5, 0.8), steps: int = 40, seed: int = 0,
                            tissue_backend: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        backend = tissue_backend or self.tissue_backend
        traj = self.heuristic_trajectory("raster")[:steps]
        out = {}
        for k0 in k0_values:
            model = self._model_for_k0(backend, k0)
            env = self.make_env(len(traj), seed, tissue_model=model)
            env.reset(seed=seed)
            hist: Dict[str, List[float]] = {}
            self.follow(env, traj, hist)
            f = np.asarray(hist["normal_force"])
            out[f"k0_{k0:.1f}"] = {
                "modulus_kpa": float(model.shear_modulus_kpa) if backend == "kelvinlet"
                else float(model.effective_moduli_kpa[1]),   # Kelvinlet: shear modulus; FEM: muscle Young's modulus
                "mean_indentation_mm": float(np.mean(hist["indentation_mm"])),
                "mean_force_n": float(f.mean()),
                "force_rmse_n": float(np.sqrt(np.mean((f - env.target_force_n) ** 2))),
                "max_force_n": float(f.max()),
                "coverage3_pct": 100.0 * float(hist["coverage_ratio"][-1]),
            }
        return out

    def run_deformation_ablation(self, seeds: Optional[List[int]] = None,
                                 tissue_backend: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        traj = self.heuristic_trajectory("raster")
        seeds = seeds if seeds is not None else [self.trial_seed(t, 7) for t in range(self.num_trials)]
        out = {}
        for mode in ("rigid", "aware", "unaware"):
            trials = []
            for seed in seeds:
                env = self.make_env(len(traj), seed, tissue_backend=tissue_backend, deformation_mode=mode)
                env.reset(seed=seed)
                hist: Dict[str, List[float]] = {}
                self.follow(env, traj, hist)
                trials.append(self._evaluate_trial_metrics(env, hist["normal_force"], hist["acoustic_coupling"], hist))
            out[mode] = self._aggregate_trial_results(trials)
        return out

    def run_soft_body_deformation_benchmark(self, compare_backend: Optional[str] = "kelvinlet") -> Dict[str, Any]:
        """Soft-body section on the suite's backend (FEM by default) plus the same section on
        `compare_backend` (Kelvinlet by default; None to skip)."""
        out: Dict[str, Any] = {}
        backends = [self.tissue_backend] + ([compare_backend] if compare_backend and compare_backend != self.tissue_backend else [])
        for b in backends:
            out[b] = {
                "force_sweep": self.run_force_sweep(seed=self.base_seed, tissue_backend=b),
                "stiffness_sweep": self.run_stiffness_sweep(seed=self.base_seed, tissue_backend=b),
                "ablation": self.run_deformation_ablation(tissue_backend=b),
            }
        return out

    def run_full_comparison(self, ppo_ckpt=None, cfm_ckpt=None, diffusion_ckpt=None) -> Dict[str, Dict[str, float]]:
        results = {}
        print("1/6 Heuristic elevational raster...")
        results["Heuristic_Raster"] = self.run_heuristic_benchmark("raster")
        print("2/6 Heuristic orbital...")
        results["Heuristic_Orbital"] = self.run_heuristic_benchmark("orbital")
        print("3/6 Greedy NBV...")
        results["Greedy_NBV"] = self.run_greedy_nbv_benchmark()
        print("4/6 Submodular PPO...")
        results["Submodular_PPO"] = self.run_submodular_ppo_benchmark(checkpoint_path=ppo_ckpt)
        print("5/6 Flow Matching...")
        results["Flow_Matching_CFM"] = self.run_flow_matching_benchmark(checkpoint_path=cfm_ckpt)
        print("6/6 Diffusion Policy...")
        results["Diffusion_Policy"] = self.run_diffusion_benchmark(checkpoint_path=diffusion_ckpt)
        return results


def _pm(m: Dict[str, float], key: str, fmt: str = "{:.1f}") -> str:
    return (fmt + " ± " + fmt).format(m.get(f"mean_{key}", float("nan")), m.get(f"std_{key}", float("nan")))


TABLE_COLS = [("Cov@2 %", "coverage2_pct", "{:.1f}"), ("Cov@3 %", "coverage3_pct", "{:.1f}"),
              ("Prec@2 %", "precision2_pct", "{:.1f}"), ("Prec@3 %", "precision3_pct", "{:.1f}"),
              ("Prec@2 vis %", "precision2_visible_pct", "{:.1f}"), ("F@2", "f_score2", "{:.2f}"),
              ("Chamfer mm", "chamfer_mm", "{:.2f}"),
              ("F mean N", "mean_force_n", "{:.2f}"), ("F RMSE N", "force_rmse_n", "{:.2f}"),
              ("F max N", "max_force_n", "{:.1f}"), ("Unsafe", "safety_violations", "{:.0f}"),
              ("Indent mm", "mean_indentation_mm", "{:.1f}"), ("Path mm", "path_length_mm", "{:.0f}")]


def print_results_table(res: Dict[str, Dict[str, float]]) -> None:
    print(f"{'Method':<18} | " + " | ".join(f"{c[0]:<15}" for c in TABLE_COLS))
    for method, m in res.items():
        print(f"{method:<18} | " + " | ".join(f"{_pm(m, k, f):<15}" for _, k, f in TABLE_COLS))


def markdown_table(res: Dict[str, Dict[str, float]], first_col: str = "Method") -> str:
    lines = ["| " + first_col + " | " + " | ".join(c[0] for c in TABLE_COLS) + " |",
             "|---" * (len(TABLE_COLS) + 1) + "|"]
    for method, m in res.items():
        lines.append(f"| {method} | " + " | ".join(_pm(m, k, f) for _, k, f in TABLE_COLS) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "NaN" if not np.isfinite(v) else f"{v:.2f}"
    return str(v)


def soft_body_markdown(sb: Dict[str, Any]) -> str:
    out = []
    backends = list(sb.keys())
    out.append("### Force sweep (midline L4, untilted; echo depth detected from the B-mode, NaN = no valid echo)\n")
    keys = ["target_force_n", "model_force_n", "indentation_mm", "indentation_from_first_contact_mm",
            "echo_depth_mm", "expected_echo_depth_mm", "tangent_stiffness_n_per_mm", "indentation_limited"]
    out.append("| Backend | " + " | ".join(keys) + " |")
    out.append("|---" * (len(keys) + 1) + "|")
    for b in backends:
        for r in sb[b]["force_sweep"]:
            out.append(f"| {b} | " + " | ".join(_fmt(r.get(k, float("nan"))) for k in keys) + " |")
    out.append("\n### Stiffness sweep (raster, first 40 poses; model re-calibrated to k0)\n")
    skeys = None
    for b in backends:
        for k0, v in sb[b]["stiffness_sweep"].items():
            if skeys is None:
                skeys = list(v.keys())
                out.append("| Backend | k0 | " + " | ".join(skeys) + " |")
                out.append("|---" * (len(skeys) + 2) + "|")
            out.append(f"| {b} | {k0} | " + " | ".join(_fmt(v[k]) for k in skeys) + " |")
    out.append("\n### Rigid vs deformation-aware vs deformation-unaware (same anatomy, raster trajectory, seeds)\n")
    abl = {f"{b} / {mode}": m for b in backends for mode, m in sb[b]["ablation"].items()}
    out.append(markdown_table(abl, "Backend / mode"))
    return "\n".join(out)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_results(results: Dict[str, Any], meta: Dict[str, Any], out_dir: str = "results") -> Dict[str, str]:
    """results/benchmark_<timestamp>.json (all numbers) + results/benchmark_latest.md (tables)."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(out_dir, f"benchmark_{stamp}.json")
    with open(jpath, "w") as fh:
        json.dump(_jsonable({"meta": meta, "results": results}), fh, indent=1)
    md = [f"# Benchmark results ({stamp})", "",
          "Generated by `ReconstructionPipeline/evaluation/benchmark_runner.py`; do not edit by hand.", "",
          f"Command: `{meta.get('command', '')}`", "",
          f"Patient: `{meta.get('patient_dir')}` | target `{meta.get('target_vertebra')}` | trials {meta.get('num_trials')} "
          f"| tissue backend `{meta.get('tissue_backend')}` | target force {meta.get('target_force_n')} N | "
          f"wall time {meta.get('wall_time_s', float('nan')):.0f} s | raw: `{os.path.basename(jpath)}`", "",
          "Mean ± std over trials. Cov = coverage of the US-visible dorsal L4 surface; Prec = fraction of reconstructed "
          "points within tol of the bone surface of the whole volume; Prec vis = same against the dorsal-visible bone "
          "surface only (stricter). Force = tissue-model reaction force under the admittance controller.", ""]
    if results.get("policies"):
        md += ["## Policies", "", markdown_table(results["policies"]), ""]
    if results.get("soft_body"):
        md += ["## Soft-body deformation", "", soft_body_markdown(results["soft_body"]), ""]
    mpath = os.path.join(out_dir, "benchmark_latest.md")
    with open(mpath, "w") as fh:
        fh.write("\n".join(md) + "\n")
    return {"json": jpath, "markdown": mpath}


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="3D ultrasound reconstruction benchmark suite")
    parser.add_argument("--patient-dir", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-force", type=float, default=4.0)
    parser.add_argument("--ppo-ckpt", type=str, default=None)
    parser.add_argument("--cfm-ckpt", type=str, default=None)
    parser.add_argument("--diffusion-ckpt", type=str, default=None)
    parser.add_argument("--target-vertebra", type=str, default="vertebrae_L4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tissue-backend", choices=["fem", "kelvinlet"], default="fem")
    parser.add_argument("--soft-body", action="store_true", help="Also run the soft-tissue deformation benchmark")
    parser.add_argument("--soft-body-only", action="store_true", help="Run only the soft-tissue deformation benchmark")
    parser.add_argument("--no-kelvinlet-comparison", action="store_true",
                        help="Skip the Kelvinlet comparison in the soft-body section")
    parser.add_argument("--heuristics-only", action="store_true", help="Heuristic raster + orbital only")
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated subset of raster,orbital,nbv,ppo,cfm,diffusion")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    suite = ReconstructionBenchmarkSuite(num_trials=args.num_trials, patient_dir=args.patient_dir,
                                         target_vertebra=args.target_vertebra, device=torch.device(args.device),
                                         base_seed=args.seed, target_force_n=args.target_force,
                                         tissue_backend=args.tissue_backend)
    t_start = time.time()
    results: Dict[str, Any] = {}
    if not args.soft_body_only:
        if args.policies:
            wanted = [w.strip() for w in args.policies.split(",") if w.strip()]
        elif args.heuristics_only:
            wanted = ["raster", "orbital"]
        else:
            wanted = ["raster", "orbital", "nbv", "ppo", "cfm", "diffusion"]

        def _ckpt(given, default):
            return given or (default if os.path.exists(default) else None)
        runners = {
            "raster": ("Heuristic_Raster", lambda: suite.run_heuristic_benchmark("raster")),
            "orbital": ("Heuristic_Orbital", lambda: suite.run_heuristic_benchmark("orbital")),
            "nbv": ("Greedy_NBV", suite.run_greedy_nbv_benchmark),
            "ppo": ("Submodular_PPO", lambda: suite.run_submodular_ppo_benchmark(
                checkpoint_path=_ckpt(args.ppo_ckpt, "checkpoints/submodular_ppo.pt"))),
            "cfm": ("Flow_Matching_CFM", lambda: suite.run_flow_matching_benchmark(
                checkpoint_path=_ckpt(args.cfm_ckpt, "checkpoints/flow_matching.pt"))),
            "diffusion": ("Diffusion_Policy", lambda: suite.run_diffusion_benchmark(
                checkpoint_path=_ckpt(args.diffusion_ckpt, "checkpoints/diffusion_policy.pt"))),
        }
        res = {}
        for i, key in enumerate(wanted):
            name, fn = runners[key]
            print(f"{i + 1}/{len(wanted)} {name}...")
            res[name] = fn()
        results["policies"] = res
        print_results_table(res)
    if args.soft_body or args.soft_body_only:
        sb = suite.run_soft_body_deformation_benchmark(
            compare_backend=None if args.no_kelvinlet_comparison else "kelvinlet")
        results["soft_body"] = sb
        print(soft_body_markdown(sb))
    meta = {"command": "./.venv/bin/python " + " ".join(sys.argv), "patient_dir": args.patient_dir,
            "target_vertebra": args.target_vertebra, "num_trials": args.num_trials, "seed": args.seed,
            "tissue_backend": args.tissue_backend, "target_force_n": args.target_force, "device": args.device,
            "wall_time_s": time.time() - t_start}
    paths = write_results(results, meta, args.out_dir)
    print(f"[benchmark] wrote {paths['json']} and {paths['markdown']}")
