"""
Gymnasium-compatible Ultrasound Reconstruction & Active View Selection Environment.
Models physical contact dynamics, closed-loop force-acoustic coupling,
GPU-accelerated trilinear slicing, and direct 2D B-mode ultrasound surface reconstruction.
Supports both clinical lateral flank scanning (patient lying down) and dorsal scanning.
"""

from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter
from ReconstructionPipeline.reconstruction.point_cloud_accumulator import PointCloudAccumulator


class UltrasoundReconstructionEnv:
    """
    Simulates autonomous robotic 3D ultrasound scanning and intraoperative bone surface reconstruction.
    """
    def __init__(
        self,
        patient_data: Optional[Dict[str, Any]] = None,
        max_steps: int = 100,
        target_coverage: float = 0.90,
        weight_path: float = 0.01,
        weight_rot: float = 0.05,
        weight_force: float = 0.02,
        weight_decoupling: float = 0.05,
        tissue_stiffness_n_per_mm: float = 2.5,
        target_force_n: float = 4.0,
        skin_curvature: float = 0.002,
        scan_approach: str = "lateral",
        reconstruction_source: str = "bmode",
        pybullet_backend: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps
        self.target_coverage = target_coverage
        self.w_path = weight_path
        self.w_rot = weight_rot
        self.w_force = weight_force
        self.w_decoupling = weight_decoupling
        self.tissue_stiffness = tissue_stiffness_n_per_mm
        self.target_force_n = target_force_n
        self.skin_curvature = skin_curvature
        self.scan_approach = scan_approach.lower()
        self.reconstruction_source = reconstruction_source.lower()
        self.pybullet_backend = pybullet_backend

        # 1. Load or generate patient anatomy
        if patient_data is None:
            patient_data = PatientAssetGenerator.generate_lumbar_phantom()
        elif isinstance(patient_data, str):
            if os.path.isdir(patient_data):
                patient_data = PatientAssetGenerator.load_totalsegmentator_subject(patient_data)
            else:
                raise FileNotFoundError(f"Patient directory not found: {patient_data}")
        self.patient_data = patient_data

        self.ct_vol = patient_data["ct_volume"]
        self.label_vol = patient_data["label_volume"]
        self.spacing = patient_data["spacing_mm"]
        self.origin = patient_data["origin_mm"]
        self.gt_bone_points = patient_data["bone_points"]
        self.skin_y = patient_data.get("skin_y_mm", 55.0)

        # 2. Initialize Subsystems
        self.slice_extractor_ct = VolumeSliceExtractor(
            self.ct_vol, spacing=self.spacing, origin=self.origin, device=self.device
        )
        self.slice_extractor_label = VolumeSliceExtractor(
            self.label_vol, spacing=self.spacing, origin=self.origin, device=self.device
        )
        self.us_sim = ModelBasedUltrasoundSimulator(device=self.device)
        self.conf_estimator = UltrasoundConfidenceEstimator(device=self.device)
        self.segmenter = BoneSurfaceSegmenter()
        self.accumulator = PointCloudAccumulator(voxel_size_mm=1.5)

        # 3. Action Space: 4-DoF [dc1, dc2, dyaw, dpitch]
        self.action_scale_trans = 5.0 # mm
        self.action_scale_yaw = np.radians(10.0) # rad
        self.action_scale_pitch = np.radians(15.0) # rad

        # Scanning bounding box on patient skin
        if self.scan_approach == "lateral":
            # Lateral flank scan: c1 is anterior-posterior Y, c2 is cranial-caudal Z
            self.c1_bounds = (-30.0, 30.0)
            self.c2_bounds = (-30.0, 30.0)
        else:
            # Dorsal scan: c1 is lateral X, c2 is cranial-caudal Z
            self.c1_bounds = (-45.0, 45.0)
            self.c2_bounds = (-30.0, 30.0)
        self.x_bounds = self.c1_bounds
        self.z_bounds = self.c2_bounds

        # State tracking
        self.current_step = 0
        self.current_pose_params = np.zeros(4) # [c1, c2, yaw, pitch]
        self.previous_area = 0.0
        self.trajectory_length = 0.0
        self.total_rotation = 0.0
        self.current_normal_force = target_force_n
        self.current_coupling = 1.0
        self.current_contact_status = "optimal"

    def get_skin_surface(self, c1: float, c2: float) -> float:
        """
        Computes skin surface coordinate accounting for anatomical curvature.
        In lateral mode: returns flank skin x(y, z) where probe contacts the flank.
        In dorsal mode: returns dorsal skin y(x, z).
        """
        if self.scan_approach == "lateral":
            delta_x = self.skin_curvature * (0.6 * (c1 ** 2) + 0.4 * (c2 ** 2))
            return float(48.0 - delta_x)
        else:
            delta_y = self.skin_curvature * (0.6 * (c1 ** 2) + 0.4 * (c2 ** 2))
            return float(self.skin_y - delta_y)

    def get_skin_surface_y(self, x: float, z: float) -> float:
        """Maintains backward compatibility for skin surface coordinate queries."""
        return self.get_skin_surface(x, z)

    def compute_contact_force_and_coupling(
        self,
        c1: float,
        c2: float,
        yaw: float,
        pitch: float,
        probe_coord: Optional[float] = None,
        probe_y: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        """
        Computes normal contact force F_N (N) and acoustic coupling factor kappa in [0.05, 1.0].
        """
        if probe_y is not None and probe_coord is None:
            probe_coord = probe_y
        if self.pybullet_backend is not None and getattr(self.pybullet_backend, "anatomy_body_id", None) is not None:
            contacts = self.pybullet_backend.get_contacts()
            normal_force = float(sum(c["normal_force"] for c in contacts)) if contacts else 0.0
        else:
            surface_coord = self.get_skin_surface(c1, c2)
            if probe_coord is None and probe_y is None:
                penetration = self.target_force_n / self.tissue_stiffness
            elif probe_y is not None:
                penetration = max(0.0, probe_y - self.skin_y)
            else:
                penetration = max(0.0, surface_coord - probe_coord)
            normal_force = float(self.tissue_stiffness * penetration)

        # Beam tilt angle relative to surface normal
        tilt_angle = np.sqrt(yaw ** 2 + pitch ** 2)
        tilt_factor = float(np.exp(-0.5 * (tilt_angle / np.radians(22.0)) ** 2))

        if normal_force < 1.0:
            kappa = 0.05
            status = "decoupled"
        elif normal_force < 2.0:
            kappa = (0.05 + 0.95 * ((normal_force - 1.0) / 1.0)) * tilt_factor
            status = "partial"
        elif normal_force <= 8.0:
            kappa = 1.0 * tilt_factor
            status = "optimal"
        else:
            kappa = max(0.2, (1.0 - 0.05 * (normal_force - 8.0)) * tilt_factor)
            status = "excessive"

        return normal_force, float(np.clip(kappa, 0.05, 1.0)), status

    def _build_probe_matrix(self, c1: float, c2: float, yaw: float, pitch: float, offset: float = 0.0) -> np.ndarray:
        """
        Constructs the 4x4 matrix T_patient_probe mapping probe frame {U} to patient volume frame.
        """
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        if self.scan_approach == "lateral":
            # Lateral flank scan: probe on lateral flank pointing laterally inward (-X) toward spine
            col1 = np.array([0.0, 0.0, 1.0])   # probe +X along cranial-caudal +Z
            col2 = np.array([-1.0, 0.0, 0.0])  # probe +Y (acoustic beam) along -X into spine
            col3 = np.cross(col1, col2)        # probe +Z along anterior-posterior -Y
            r_base = np.column_stack([col1, col2, col3])

            r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            rot = r_base @ r_yaw @ r_pitch

            flank_x = self.get_skin_surface(c1, c2)
            t_matrix = np.eye(4)
            t_matrix[:3, :3] = rot
            t_matrix[0, 3] = flank_x + offset
            t_matrix[1, 3] = c1
            t_matrix[2, 3] = c2
            return t_matrix
        else:
            # Dorsal scan: probe on dorsal back pointing down (-Y) into patient
            r_base = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0]
            ])
            r_yaw = np.array([
                [cy,  0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy]
            ])
            r_pitch = np.array([
                [1.0, 0.0,  0.0],
                [0.0, cp, -sp],
                [0.0, sp,  cp]
            ])
            rot = r_yaw @ r_pitch @ r_base

            surface_y = self.get_skin_surface(c1, c2)
            t_matrix = np.eye(4)
            t_matrix[:3, :3] = rot
            t_matrix[0, 3] = c1
            t_matrix[1, 3] = surface_y + offset
            t_matrix[2, 3] = c2
            return t_matrix

    def reset(self) -> Dict[str, Any]:
        """Resets the environment for a new scanning episode."""
        self.current_step = 0
        self.accumulator.clear()
        self.previous_area = 0.0
        self.trajectory_length = 0.0
        self.total_rotation = 0.0

        # Randomize initial probe pose on patient skin
        init_c1 = np.random.uniform(-10.0, 10.0)
        init_c2 = np.random.uniform(-10.0, 10.0)
        init_yaw = np.random.uniform(-np.radians(5.0), np.radians(5.0))
        init_pitch = np.random.uniform(-np.radians(5.0), np.radians(5.0))
        self.current_pose_params = np.array([init_c1, init_c2, init_yaw, init_pitch], dtype=np.float32)

        f_n, kappa, status = self.compute_contact_force_and_coupling(init_c1, init_c2, init_yaw, init_pitch)
        self.current_normal_force = f_n
        self.current_coupling = kappa
        self.current_contact_status = status

        obs = self._observe_current_state()
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one action step, acquiring an ultrasound frame and updating reconstruction.
        """
        self.current_step += 1

        action = np.clip(action, -1.0, 1.0)
        dc1 = action[0] * self.action_scale_trans
        dc2 = action[1] * self.action_scale_trans
        dyaw = action[2] * self.action_scale_yaw
        dpitch = action[3] * self.action_scale_pitch

        new_c1 = np.clip(self.current_pose_params[0] + dc1, self.c1_bounds[0], self.c1_bounds[1])
        new_c2 = np.clip(self.current_pose_params[1] + dc2, self.c2_bounds[0], self.c2_bounds[1])
        new_yaw = self.current_pose_params[2] + dyaw
        new_pitch = np.clip(self.current_pose_params[3] + dpitch, -np.radians(35.0), np.radians(35.0))

        actual_dc1 = new_c1 - self.current_pose_params[0]
        actual_dc2 = new_c2 - self.current_pose_params[1]
        step_path = np.sqrt(actual_dc1**2 + actual_dc2**2)
        step_rot = np.abs(dyaw) + np.abs(dpitch)

        self.trajectory_length += step_path
        self.total_rotation += step_rot
        self.current_pose_params = np.array([new_c1, new_c2, new_yaw, new_pitch], dtype=np.float32)

        # Contact mechanics & acoustic coupling
        f_n, kappa, status = self.compute_contact_force_and_coupling(new_c1, new_c2, new_yaw, new_pitch)
        self.current_normal_force = f_n
        self.current_coupling = kappa
        self.current_contact_status = status

        # 1. Extract 2D CT & Label slices at current probe pose
        t_pose = self._build_probe_matrix(new_c1, new_c2, new_yaw, new_pitch)
        ct_slice = self.slice_extractor_ct.extract_slices(t_pose, interpolation="bilinear")
        label_slice = self.slice_extractor_label.extract_slices(t_pose, interpolation="nearest")

        # 2. Simulate B-mode ultrasound frame
        us_res = self.us_sim.simulate_bmode(ct_slice, label_slice=label_slice, bone_label_id=1)
        bmode_raw = us_res["bmode"]
        energy_map = us_res["energy"]

        # 3. Apply Acoustic Coupling degradation
        if kappa < 0.99:
            noise_floor = torch.rand_like(bmode_raw) * 0.08
            bmode_img = bmode_raw * kappa + noise_floor * (1.0 - kappa)
        else:
            bmode_img = bmode_raw

        # 4. Compute Ultrasound Confidence Map
        conf_map = self.conf_estimator.compute_confidence_map(bmode_img, energy_map=energy_map) * kappa

        # 5. Extract bone contour and update 3D reconstruction directly from US image
        if kappa >= 0.20:
            if self.reconstruction_source == "bmode":
                # Real ultrasound reconstruction directly from B-mode frames
                pts_2d = self.segmenter.segment_from_bmode(
                    bmode_img.detach().cpu().numpy(),
                    confidence_map=conf_map.detach().cpu().numpy()
                )
            else:
                pts_2d = self.segmenter.segment_from_label_map(label_slice.cpu().numpy(), bone_label_id=1)
            self.accumulator.add_frame(pts_2d, t_pose)
        else:
            pts_2d = np.empty((0, 2), dtype=np.float32)

        # 6. Calculate Submodular Marginal Gain Reward + Compliance Penalties
        current_area = self.accumulator.get_reconstruction_area()
        marginal_area_gain = current_area - self.previous_area
        self.previous_area = current_area

        force_penalty = 0.0
        if f_n > 8.0:
            force_penalty = self.w_force * float((f_n - 8.0) ** 2)
        elif f_n < 2.0:
            force_penalty = self.w_decoupling * float(2.0 - f_n)

        reward = float(marginal_area_gain * 0.05) - (
            self.w_path * step_path + self.w_rot * step_rot + force_penalty
        )

        reconstructed_pts = self.accumulator.get_points()
        coverage_ratio = min(1.0, float(len(reconstructed_pts)) / float(max(1, len(self.gt_bone_points))))

        terminated = bool(coverage_ratio >= self.target_coverage)
        truncated = bool(self.current_step >= self.max_steps)

        obs = {
            "probe_pose": self.current_pose_params.copy(),
            "bmode": bmode_img.detach().cpu().numpy(),
            "confidence_map": conf_map.detach().cpu().numpy(),
            "reconstructed_points": reconstructed_pts,
            "coverage_ratio": coverage_ratio,
            "normal_force": self.current_normal_force,
            "acoustic_coupling": self.current_coupling,
            "contact_status": self.current_contact_status,
        }

        info = {
            "marginal_gain": marginal_area_gain,
            "total_area": current_area,
            "total_points": len(reconstructed_pts),
            "coverage_ratio": coverage_ratio,
            "trajectory_length": self.trajectory_length,
            "total_rotation": self.total_rotation,
            "normal_force": self.current_normal_force,
            "acoustic_coupling": self.current_coupling,
            "contact_status": self.current_contact_status,
            "step": self.current_step,
            "scan_approach": self.scan_approach,
            "reconstruction_source": self.reconstruction_source,
        }

        return obs, reward, terminated, truncated, info

    def _observe_current_state(self) -> Dict[str, Any]:
        """Generates observation for the initial state."""
        t_pose = self._build_probe_matrix(*self.current_pose_params)
        ct_slice = self.slice_extractor_ct.extract_slices(t_pose)
        label_slice = self.slice_extractor_label.extract_slices(t_pose, interpolation="nearest")
        us_res = self.us_sim.simulate_bmode(ct_slice, label_slice=label_slice)
        bmode_img = us_res["bmode"] * self.current_coupling
        conf_map = self.conf_estimator.compute_confidence_map(bmode_img, energy_map=us_res["energy"]) * self.current_coupling

        if self.current_coupling >= 0.20:
            if self.reconstruction_source == "bmode":
                pts_2d = self.segmenter.segment_from_bmode(
                    bmode_img.detach().cpu().numpy(),
                    confidence_map=conf_map.detach().cpu().numpy()
                )
            else:
                pts_2d = self.segmenter.segment_from_label_map(label_slice.cpu().numpy())
            self.accumulator.add_frame(pts_2d, t_pose)
        self.previous_area = self.accumulator.get_reconstruction_area()

        return {
            "probe_pose": self.current_pose_params.copy(),
            "bmode": bmode_img.detach().cpu().numpy(),
            "confidence_map": conf_map.detach().cpu().numpy(),
            "reconstructed_points": self.accumulator.get_points(),
            "coverage_ratio": 0.0,
            "normal_force": self.current_normal_force,
            "acoustic_coupling": self.current_coupling,
            "contact_status": self.current_contact_status,
        }
