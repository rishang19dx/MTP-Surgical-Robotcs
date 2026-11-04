"""
Gymnasium-style environment for autonomous robotic ultrasound bone-surface reconstruction
(dorsal lumbar scanning).

Per step:
  1. 4-DoF action [dc1, dc2, dyaw, dpitch] moves the probe over the dorsal skin
     (c1 = lateral X, c2 = cranio-caudal Z, yaw = rotation ABOUT THE BEAM, pitch = cranio-caudal tilt).
  2. The probe origin is placed on the REAL skin (height map from the CT body mask, frame P,
     +Y toward the dorsal skin).
  3. An inner-loop admittance controller (contract C2) adjusts the indentation along the beam to
     track `target_force_n` using the TissueModel's force (contract C1).  Force is an OUTPUT of the
     tissue model: it varies with anatomy (tissue thickness over bone), tilt and transients.
     The indentation is measured from the undeformed skin under the probe CENTRE and may be negative:
     a flat transverse probe over the midline groove touches the paraspinal ridges first (FEM
     `first_contact_indentation`, about -9 mm on the sample patient).  The controller works in
     [first_contact, max_indentation] and is warm-started across steps relative to first contact.
  4. The tissue model is solved at the final indentation; CT/label slices are sampled with backward
     Eulerian warping at the indented probe-face pose.
  5. B-mode simulation (explicit pixel spacing, contract C3), image-only confidence (C4),
     bone segmentation from B-mode, back-projection and fusion.
  6. Coverage = fraction of the US-visible target ROI within `coverage_tol_mm` of the reconstruction (C6).

Tissue backends (`tissue_backend`): "fem" (default) = patient-specific precomputed GPU linear FEM
(`simulation.fem_tissue.FEMTissueModel`, calibrated over the scan window); "kelvinlet" = the analytic
regularised-Kelvinlet model (`simulation.tissue_model.KelvinletTissueModel`), kept for comparison.

Deformation modes (for ablations):
  "aware"   (default) deformed images, back-projection with the tracked (indented) face pose.
  "unaware" deformed images, back-projection with the nominal pose at the undeformed skin
            (ignores the probe indentation, i.e. assumes rigid tissue).
  "rigid"   SonoGym-style rigid patient: images are not deformed and the probe sits on the skin.
"""

from contextlib import nullcontext
from typing import Optional, Tuple, Dict, Any
import os
import warnings

import numpy as np
import torch
from scipy.spatial import cKDTree

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator
from ReconstructionPipeline.simulation.tissue_model import (
    KelvinletTissueModel,
    SkinSurface,
    TissueState,
    tilt_cosine,
)
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter
from ReconstructionPipeline.reconstruction.point_cloud_accumulator import PointCloudAccumulator


def _try_construct(cls, kwargs_full: Dict[str, Any], kwargs_min: Dict[str, Any]):
    """Construct with the new keyword args (contract C3); fall back if an older signature is present."""
    try:
        return cls(**kwargs_full)
    except TypeError:
        warnings.warn(f"{cls.__name__} does not accept {sorted(set(kwargs_full) - set(kwargs_min))}; using defaults")
        return cls(**kwargs_min)


class UltrasoundReconstructionEnv:
    """Simulated robotic 3D ultrasound scanning with soft-tissue contact and force control."""

    IMAGE_SIZE = (200, 150)          # (H, W) px
    PHYSICAL_SIZE_MM = (75.0, 50.0)  # (depth, width) mm

    def __init__(
        self,
        patient_data: Optional[Any] = None,
        max_steps: int = 100,
        target_coverage: float = 0.90,
        weight_path: float = 0.01,
        weight_rot: float = 0.05,
        weight_force: float = 0.02,
        weight_decoupling: float = 0.05,
        tissue_stiffness_n_per_mm: Optional[float] = None,  # deprecated, ignored (force comes from tissue_model)
        target_force_n: float = 4.0,
        skin_curvature: Optional[float] = None,             # deprecated, ignored (skin from CT)
        scan_approach: str = "dorsal",
        reconstruction_source: str = "bmode",
        target_vertebra: Optional[str] = None,
        pybullet_backend: Optional[Any] = None,             # visualisation only; force no longer read from it
        device: Optional[torch.device] = None,
        tissue_model: Optional[Any] = None,
        tissue_backend: str = "fem",
        deformation_mode: str = "aware",
        admittance_gain_mm_per_n: float = 0.5,
        admittance_max_step_mm: float = 1.5,
        admittance_iterations: int = 2,
        safety_force_limit_n: float = 20.0,
        coverage_tol_mm: float = 3.0,
        max_yaw_deg: float = 45.0,
        max_pitch_deg: float = 35.0,
        seed: Optional[int] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if scan_approach.lower() != "dorsal":
            raise ValueError("Only the dorsal scan approach is supported (lateral flank mode was removed).")
        if deformation_mode not in ("aware", "unaware", "rigid"):
            raise ValueError(f"deformation_mode must be aware|unaware|rigid, got {deformation_mode}")
        if tissue_backend not in ("fem", "kelvinlet"):
            raise ValueError(f"tissue_backend must be fem|kelvinlet, got {tissue_backend}")
        self.scan_approach = "dorsal"
        self.deformation_mode = deformation_mode
        self.max_steps = max_steps
        self.target_coverage = target_coverage
        self.w_path = weight_path
        self.w_rot = weight_rot
        self.w_force = weight_force
        self.w_decoupling = weight_decoupling
        if not (3.0 <= target_force_n <= 12.0):
            warnings.warn(f"target_force_n={target_force_n} outside the supported 3-12 N range; clipping")
        self.target_force_n = float(np.clip(target_force_n, 3.0, 12.0))
        self.safety_force_limit_n = float(safety_force_limit_n)
        self.k_adm = float(admittance_gain_mm_per_n)
        self.adm_max_step = float(admittance_max_step_mm)
        self.adm_iters = int(admittance_iterations)
        self.coverage_tol_mm = float(coverage_tol_mm)
        self.max_yaw = np.radians(max_yaw_deg)
        self.max_pitch = np.radians(max_pitch_deg)
        self.reconstruction_source = reconstruction_source.lower()
        self.target_vertebra = target_vertebra
        self.pybullet_backend = pybullet_backend
        self.rng = np.random.default_rng(seed)

        # 1. Patient anatomy
        if patient_data is None:
            patient_data = PatientAssetGenerator.generate_lumbar_phantom()
        elif isinstance(patient_data, str):
            if not os.path.isdir(patient_data):
                raise FileNotFoundError(f"Patient directory not found: {patient_data}")
            if target_vertebra:
                patient_data = PatientAssetGenerator.load_totalsegmentator_subject(
                    patient_data, target_vertebrae=[target_vertebra], crop_to_spine=False
                )
            else:
                patient_data = PatientAssetGenerator.load_totalsegmentator_subject(patient_data)
        if "skin_height_map" not in patient_data:
            patient_data.update(PatientAssetGenerator.compute_skin_height_map(
                patient_data["ct_volume"], patient_data["origin_mm"], patient_data["spacing_mm"]))
        self.patient_data = patient_data

        loaded_structs = self.patient_data.get("loaded_structures", [])
        self.is_l4_target = (target_vertebra in ["vertebrae_L4", "L4", "l4"]) or (loaded_structs == ["vertebrae_L4"])

        self.ct_vol = patient_data["ct_volume"]
        self.label_vol = patient_data["label_volume"]
        self.spacing = patient_data["spacing_mm"]
        self.origin = patient_data["origin_mm"]
        self.gt_bone_points = patient_data["bone_points"]
        self.skin = SkinSurface.from_patient_data(patient_data)
        self.skin_y = float(patient_data.get("skin_y_mm", np.median(patient_data["skin_height_map"])))

        # Scan window (probe-centre bounds, frame P mm)
        if self.is_l4_target:
            self.c1_bounds = (-32.0, 32.0)
            self.c2_bounds = (-55.0, -10.0)
        else:
            self.c1_bounds = (-45.0, 45.0)
            self.c2_bounds = (-30.0, 30.0)

        # Tissue model (contract C1): FEM by default; an explicit `tissue_model` overrides the backend.
        if tissue_model is not None:
            self.tissue_model = tissue_model
        elif tissue_backend == "fem":
            from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
            self.tissue_model = FEMTissueModel.from_patient_data(
                patient_data, c1_bounds=self.c1_bounds, c2_bounds=self.c2_bounds, device=self.device, verbose=False)
        else:
            self.tissue_model = KelvinletTissueModel.from_patient_data(
                patient_data, skin_surface=self.skin, device=self.device)
        self.tissue_backend = "fem" if hasattr(self.tissue_model, "first_contact_indentation") else "kelvinlet"
        self._fit_window_to_tissue_region()

        # Target ROI: US-visible (outermost-along-beam) dorsal bone surface
        pts = self.gt_bone_points
        if len(pts) > 5000:
            if self.is_l4_target:
                roi_mask = (np.abs(pts[:, 0]) <= 32.0) & (pts[:, 2] >= -56.0) & (pts[:, 2] <= -10.0) & (pts[:, 1] >= 10.0)
            else:
                roi_mask = (np.abs(pts[:, 0]) <= 25.0) & (np.abs(pts[:, 2]) <= 35.0) & (pts[:, 1] >= 15.0)
            roi_pts = pts[roi_mask] if np.any(roi_mask) else pts[:5000]
            self.target_roi_bone_points = self._dorsal_visible_surface(roi_pts)
        else:
            self.target_roi_bone_points = np.asarray(pts, dtype=np.float32)
        self.target_voxel_count = max(1, len(self.target_roi_bone_points))
        # All bone surface in the volume (any vertebra / pelvis, CT > 300 HU or label 1): used for precision so
        # that echoes from neighbouring vertebrae are not counted as false positives.
        self.all_bone_points = self._all_bone_surface(self.ct_vol, self.label_vol, self.origin, self.spacing)
        # Stricter precision reference: the dorsal-visible bone surface (outermost bone voxel per (x, z) column).
        self.visible_bone_points = self._dorsal_visible_bone(self.ct_vol, self.label_vol, self.origin, self.spacing)
        self._roi_tree = cKDTree(self.target_roi_bone_points) if len(self.target_roi_bone_points) else None

        # 2. Subsystems
        h_px, w_px = self.IMAGE_SIZE
        depth_mm, width_mm = self.PHYSICAL_SIZE_MM
        self.axial_res_mm = depth_mm / (h_px - 1)
        self.lateral_res_mm = width_mm / (w_px - 1)
        self.slice_extractor_ct = VolumeSliceExtractor(
            self.ct_vol, spacing=self.spacing, origin=self.origin, image_size=self.IMAGE_SIZE,
            physical_size_mm=self.PHYSICAL_SIZE_MM, device=self.device)
        self.slice_extractor_label = VolumeSliceExtractor(
            self.label_vol, spacing=self.spacing, origin=self.origin, image_size=self.IMAGE_SIZE,
            physical_size_mm=self.PHYSICAL_SIZE_MM, device=self.device)
        self.us_sim = _try_construct(
            ModelBasedUltrasoundSimulator,
            dict(device=self.device, axial_resolution_mm=self.axial_res_mm, lateral_resolution_mm=self.lateral_res_mm,
                 seed=seed),
            dict(device=self.device))
        self.conf_estimator = _try_construct(
            UltrasoundConfidenceEstimator,
            dict(device=self.device, axial_resolution_mm=self.axial_res_mm), dict(device=self.device))
        self.segmenter = _try_construct(
            BoneSurfaceSegmenter, dict(axial_resolution_mm=self.axial_res_mm, rng=np.random.default_rng(seed)), {})
        self.accumulator = PointCloudAccumulator(
            voxel_size_mm=1.5, image_size=self.IMAGE_SIZE, physical_size_mm=self.PHYSICAL_SIZE_MM)

        # 3. Action space: 4-DoF [dc1, dc2, dyaw, dpitch]
        self.action_scale_trans = 5.0
        self.action_scale_yaw = np.radians(10.0)
        self.action_scale_pitch = np.radians(15.0)
        self.x_bounds = self.c1_bounds
        self.z_bounds = self.c2_bounds

        # State
        self.current_step = 0
        self.current_pose_params = np.zeros(4, dtype=np.float32)
        self.current_indentation_mm = 0.0
        self._indent_rel_mm = 0.0           # indentation beyond first contact (controller warm start)
        self.current_state: Optional[TissueState] = None
        self.previous_area = 0.0
        self.trajectory_length = 0.0
        self.total_rotation = 0.0
        self.current_normal_force = 0.0
        self.current_coupling = 0.0
        self.current_contact_status = "decoupled"
        self.coverage_history = []

    # ------------------------------------------------------------------ geometry
    WINDOW_MARGIN_MM = 5.0

    def _fit_window_to_tissue_region(self) -> None:
        """Shrink the scan window so the whole probe face (any yaw) stays `WINDOW_MARGIN_MM` inside the
        tissue model's region (FEM ROI: its side faces are Dirichlet, so contact near them is over-stiff)."""
        lo = getattr(self.tissue_model, "roi_min", None)
        hi = getattr(self.tissue_model, "roi_max", None)
        if lo is None or hi is None:
            return
        half = 0.5 * float(getattr(self.tissue_model, "face_dims", (45.0, 10.0))[0]) + self.WINDOW_MARGIN_MM
        new = []
        for (a, b), i in ((self.c1_bounds, 0), (self.c2_bounds, 2)):
            a2, b2 = max(a, float(lo[i]) + half), min(b, float(hi[i]) - half)
            if a2 >= b2:
                raise ValueError(f"tissue model region {lo[i]:.0f}..{hi[i]:.0f} mm cannot hold the scan window")
            new.append((a2, b2))
        if (tuple(new[0]), tuple(new[1])) != (tuple(self.c1_bounds), tuple(self.c2_bounds)):
            warnings.warn(f"scan window {self.c1_bounds} x {self.c2_bounds} clipped to {new[0]} x {new[1]} "
                          "to stay inside the tissue model region")
        self.c1_bounds, self.c2_bounds = new[0], new[1]

    @property
    def tissue_state(self) -> Optional[TissueState]:
        """Latest TissueState (None in rigid mode or before the first acquisition)."""
        return self.current_state

    @staticmethod
    def _dorsal_visible_surface(pts: np.ndarray) -> np.ndarray:
        """Outermost (max +Y) bone point per 1 mm (x, z) column = surface reachable by a dorsal beam."""
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)
        keys = np.round(pts[:, [0, 2]]).astype(np.int64)
        order = np.lexsort((-pts[:, 1], keys[:, 1], keys[:, 0]))
        k_sorted = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = np.any(k_sorted[1:] != k_sorted[:-1], axis=1)
        return pts[order[first]].astype(np.float32)

    @staticmethod
    def _all_bone_surface(ct, lbl, origin, spacing) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        mask = np.asarray(ct) > 300.0
        if lbl is not None:
            mask |= np.asarray(lbl) == 1
        surf = mask & ~binary_erosion(mask)
        idx = np.argwhere(surf)[:, ::-1].astype(np.float64)  # (x, y, z) voxel indices
        return (np.asarray(origin, float) + idx * np.asarray(spacing, float)).astype(np.float32)

    @staticmethod
    def _dorsal_visible_bone(ct, lbl, origin, spacing) -> np.ndarray:
        """Outermost (max +Y) voxel of CT > 300 HU or label 1 per (z, x) voxel column, in P mm."""
        mask = np.asarray(ct) > 300.0
        if lbl is not None:
            mask |= np.asarray(lbl) == 1
        ny = mask.shape[1]
        top = ny - 1 - np.argmax(mask[:, ::-1, :], axis=1)          # (nz, nx)
        iz, ix = np.nonzero(mask.any(axis=1))
        idx = np.stack([ix, top[iz, ix], iz], axis=1).astype(np.float64)
        return (np.asarray(origin, float) + idx * np.asarray(spacing, float)).astype(np.float32)

    def get_skin_surface(self, c1: float, c2: float) -> float:
        """Undeformed dorsal skin height y_skin(x=c1, z=c2) in mm (frame P) from the CT body mask."""
        return float(self.skin.height(c1, c2))

    def get_skin_surface_y(self, x: float, z: float) -> float:
        return self.get_skin_surface(x, z)

    @staticmethod
    def _probe_rotation(yaw: float, pitch: float) -> np.ndarray:
        """
        R = Rx(pitch) Ry(-yaw) R_base.  R_base: probe x = +X (transverse image plane), beam (+Y_probe) = -Y.
        yaw is a body rotation about the beam (R_base Ry_probe(yaw) = Ry(-yaw) R_base), so it never changes
        the beam direction; pitch tilts the beam cranio-caudally (about world X).
        """
        r_base = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        r_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return r_x @ r_y @ r_base

    def skin_pose(self, c1: float, c2: float, yaw: float, pitch: float) -> np.ndarray:
        """T_P<-U with origin at the UNDEFORMED skin point under the probe centre (contract C1)."""
        t = np.eye(4)
        t[:3, :3] = self._probe_rotation(yaw, pitch)
        t[:3, 3] = [c1, self.get_skin_surface(c1, c2), c2]
        return t

    def _build_probe_matrix(self, c1: float, c2: float, yaw: float, pitch: float, offset: float = 0.0) -> np.ndarray:
        """Image-plane (probe face) pose: skin pose pushed along the beam by the current indentation + offset."""
        t = self.skin_pose(c1, c2, yaw, pitch)
        d = 0.0 if self.deformation_mode == "rigid" else self.current_indentation_mm
        t[:3, 3] += (d + offset) * t[:3, 1]
        return t

    # ---------------------------------------------------------- contact / force
    def _max_indentation(self, t_skin: np.ndarray) -> float:
        fn = getattr(self.tissue_model, "max_indentation", None)
        return float(fn(t_skin)) if fn is not None else 30.0

    def _first_contact(self, t_skin: np.ndarray) -> float:
        """Indentation (mm, may be < 0) at which the probe face first touches the undeformed skin."""
        fn = getattr(self.tissue_model, "first_contact_indentation", None)
        d0 = float(fn(t_skin)) if fn is not None else 0.0
        return d0 if np.isfinite(d0) else 0.0

    def _blas_scope(self):
        fn = getattr(self.tissue_model, "blas_scope", None)
        return fn() if fn is not None else nullcontext()

    def regulate_force(self, t_skin: np.ndarray, d_rel0: float, iterations: Optional[int] = None) -> Tuple[float, float]:
        """
        Admittance controller along the beam (contract C2), from first contact d_fc (may be < 0):
            d_{k+1} = clip(d_k + clip(K_a (F_target - F(d_k)), +-max_step), d_fc, d_max)
        `d_rel0` is the warm start relative to first contact (d = d_fc + d_rel0).
        Returns (indentation_mm, force_n) with force = TissueModel.force_at(final indentation).
        """
        n_it = self.adm_iters if iterations is None else int(iterations)
        with self._blas_scope():
            d_fc = self._first_contact(t_skin)
            d_max = max(self._max_indentation(t_skin), d_fc)
            d = float(np.clip(d_fc + d_rel0, d_fc, d_max))
            for _ in range(n_it):
                f = self.tissue_model.force_at(t_skin, d)
                step = np.clip(self.k_adm * (self.target_force_n - f), -self.adm_max_step, self.adm_max_step)
                d = float(np.clip(d + step, d_fc, d_max))
            return d, float(self.tissue_model.force_at(t_skin, d))

    def _coupling(self, force_n: float, beam: np.ndarray, c1: float, c2: float) -> Tuple[float, str, float]:
        """kappa from contact force and the angle between the beam and the local skin normal."""
        cos_t = tilt_cosine(beam, self.skin.normal(c1, c2))
        tilt = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
        tilt_factor = float(np.exp(-0.5 * (tilt / np.radians(22.0)) ** 2))
        if force_n < 1.0:
            kappa, status = 0.05, "decoupled"
        elif force_n < 2.0:
            kappa, status = (0.05 + 0.95 * (force_n - 1.0)) * tilt_factor, "partial"
        elif force_n <= 12.0:
            kappa, status = tilt_factor, "optimal"
        else:
            kappa, status = max(0.2, 1.0 - 0.05 * (force_n - 12.0)) * tilt_factor, "excessive"
        if force_n > self.safety_force_limit_n:
            status = "unsafe"
        return float(np.clip(kappa, 0.05, 1.0)), status, tilt

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
        Stateless query of (normal force N, kappa, status) at a pose.
        If `probe_y` (probe-face Y in frame P) is given, the indentation is the depth of the face below the
        local skin, measured along the beam (no controller).  Otherwise the admittance controller is run from
        the current indentation (warm start), as env.step would.
        """
        if probe_coord is not None and probe_y is None:
            probe_y = probe_coord
        t_skin = self.skin_pose(c1, c2, yaw, pitch)
        beam = t_skin[:3, 1]
        if probe_y is not None:
            d = (t_skin[1, 3] - float(probe_y)) / max(-beam[1], 1e-3)
            d = min(d, self._max_indentation(t_skin))
            f = float(self.tissue_model.force_at(t_skin, d))
        else:
            _, f = self.regulate_force(t_skin, self._indent_rel_mm)
        kappa, status, _ = self._coupling(f, beam, c1, c2)
        return f, kappa, status

    # --------------------------------------------------------------- episode
    def reset(self, seed: Optional[int] = None, clear_accumulator: bool = True) -> Dict[str, Any]:
        """Resets the episode. `seed` re-seeds the env RNG (initial pose, segmentation) and torch (speckle)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if hasattr(self.segmenter, "rng"):
                self.segmenter.rng = np.random.default_rng(seed)
            if getattr(self.us_sim, "generator", None) is not None:
                self.us_sim.generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))
        self.current_step = 0
        self.coverage_history = []
        if clear_accumulator:
            self.accumulator.clear()
            self.previous_area = 0.0
            self.trajectory_length = 0.0
            self.total_rotation = 0.0

        if self.is_l4_target:
            init_c1 = self.rng.uniform(-6.0, 6.0)
            init_c2 = self.rng.uniform(-36.0, -26.0)
        else:
            init_c1 = self.rng.uniform(-10.0, 10.0)
            init_c2 = self.rng.uniform(-10.0, 10.0)
        init_yaw = self.rng.uniform(-np.radians(5.0), np.radians(5.0))
        init_pitch = self.rng.uniform(-np.radians(5.0), np.radians(5.0))
        self.current_pose_params = np.array([init_c1, init_c2, init_yaw, init_pitch], dtype=np.float32)

        # Approach phase: let the controller converge from first touch.
        t_skin = self.skin_pose(*self.current_pose_params)
        self.current_indentation_mm, _ = self.regulate_force(t_skin, 0.0, iterations=15)
        self._indent_rel_mm = self.current_indentation_mm - self._first_contact(t_skin)
        obs, _ = self._acquire(self.current_pose_params)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        dc1 = action[0] * self.action_scale_trans
        dc2 = action[1] * self.action_scale_trans
        dyaw = action[2] * self.action_scale_yaw
        dpitch = action[3] * self.action_scale_pitch

        p = self.current_pose_params
        new_c1 = float(np.clip(p[0] + dc1, *self.c1_bounds))
        new_c2 = float(np.clip(p[1] + dc2, *self.c2_bounds))
        new_yaw = float(np.clip(p[2] + dyaw, -self.max_yaw, self.max_yaw))
        new_pitch = float(np.clip(p[3] + dpitch, -self.max_pitch, self.max_pitch))

        step_path = float(np.hypot(new_c1 - p[0], new_c2 - p[1]))
        step_rot = float(abs(new_yaw - p[2]) + abs(new_pitch - p[3]))
        self.trajectory_length += step_path
        self.total_rotation += step_rot
        self.current_pose_params = np.array([new_c1, new_c2, new_yaw, new_pitch], dtype=np.float32)

        obs, step_info = self._acquire(self.current_pose_params)
        f_n = self.current_normal_force

        current_area = self.accumulator.get_reconstruction_area()
        marginal_area_gain = current_area - self.previous_area
        self.previous_area = current_area

        safety_violation = f_n > self.safety_force_limit_n
        force_penalty = 0.0
        if f_n > 12.0:
            force_penalty = self.w_force * float((f_n - 12.0) ** 2)
        elif f_n < 2.0:
            force_penalty = self.w_decoupling * float(2.0 - f_n)
        reward = float(marginal_area_gain * 0.05) - (self.w_path * step_path + self.w_rot * step_rot + force_penalty)
        if safety_violation:
            reward -= 10.0

        coverage_ratio = obs["coverage_ratio"]
        terminated = bool(coverage_ratio >= self.target_coverage or safety_violation)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "marginal_gain": marginal_area_gain,
            "total_area": current_area,
            "total_points": len(obs["reconstructed_points"]),
            "coverage_ratio": coverage_ratio,
            "trajectory_length": self.trajectory_length,
            "total_rotation": self.total_rotation,
            "normal_force": f_n,
            "force_error_n": f_n - self.target_force_n,
            "indentation_mm": self.current_indentation_mm,
            "safety_violation": bool(safety_violation),
            "acoustic_coupling": self.current_coupling,
            "contact_status": self.current_contact_status,
            "step": self.current_step,
            "scan_approach": self.scan_approach,
            "reconstruction_source": self.reconstruction_source,
            "deformation_mode": self.deformation_mode,
            **step_info,
        }
        return obs, reward, terminated, truncated, info

    def _acquire(self, pose_params: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Force control -> tissue solve -> warped slices -> B-mode -> segmentation -> fusion."""
        c1, c2, yaw, pitch = (float(v) for v in pose_params)
        t_skin = self.skin_pose(c1, c2, yaw, pitch)
        with self._blas_scope():
            d, f_n = self.regulate_force(t_skin, self._indent_rel_mm)
            self.current_indentation_mm = d
            self._indent_rel_mm = d - self._first_contact(t_skin)
            if self.deformation_mode == "rigid":
                state = None
                t_img = t_skin
            else:
                state = self.tissue_model.solve(t_skin, d)
                f_n = float(state.reaction_force_n)
                t_img = state.probe_face_pose(t_skin)
        self.current_state = state
        kappa, status, tilt = self._coupling(f_n, t_skin[:3, 1], c1, c2)
        self.current_normal_force, self.current_coupling, self.current_contact_status = f_n, kappa, status

        ct_slice = self.slice_extractor_ct.extract_slices(t_img, interpolation="bilinear", tissue_state=state)
        label_slice = None
        if self.reconstruction_source != "bmode":
            label_slice = self.slice_extractor_label.extract_slices(t_img, interpolation="nearest", tissue_state=state)
        us_res = self.us_sim.simulate_bmode(ct_slice)
        # Poor coupling = amplitude transmission loss of 20 log10(kappa) dB applied before display:
        # with log compression over `dynamic_range_db` this is a constant downward shift of the B-mode.
        # (Approximation: it also attenuates the simulator's electronic-noise term.)
        bmode_raw = us_res["bmode"]
        loss_db = -20.0 * float(np.log10(max(kappa, 1e-3)))
        dr_db = float(getattr(self.us_sim, "dynamic_range_db", 60.0))
        bmode_img = torch.clamp(bmode_raw - loss_db / dr_db, 0.0, 1.0) if loss_db > 1e-3 else bmode_raw
        conf_map = self.conf_estimator.compute_confidence_map(bmode_img)  # image-only (C4)

        bmode_np = bmode_img.detach().cpu().numpy()
        conf_np = conf_map.detach().cpu().numpy()
        # Frames are only used when the force sensor reports contact (no privileged kappa gating).
        if f_n >= 1.0:
            if self.reconstruction_source == "bmode":
                try:
                    pts_2d = self.segmenter.segment_from_bmode(bmode_np, confidence_map=conf_np, rng=self.rng)
                except TypeError:
                    pts_2d = self.segmenter.segment_from_bmode(bmode_np, confidence_map=conf_np)
            else:
                pts_2d = self.segmenter.segment_from_label_map(label_slice.cpu().numpy(), bone_label_id=1)
            t_bp = t_skin if self.deformation_mode == "unaware" else t_img
            self.accumulator.add_frame(pts_2d, t_bp)
        else:
            pts_2d = np.empty((0, 2), dtype=np.float32)

        recon = self.accumulator.get_points()
        coverage = self._coverage(recon)
        self.coverage_history.append(coverage)
        obs = {
            "probe_pose": self.current_pose_params.copy(),
            "bmode": bmode_np,
            "confidence_map": conf_np,
            "reconstructed_points": recon,
            "coverage_ratio": coverage,
            "normal_force": f_n,
            "indentation_mm": d,
            "acoustic_coupling": kappa,
            "contact_status": status,
        }
        info = {"tilt_deg": float(np.degrees(tilt)), "frame_points": int(len(pts_2d)),
                "tissue_thickness_mm": float(state.bone_depth_mm) if state is not None else float("nan")}
        return obs, info

    def _coverage(self, recon: np.ndarray) -> float:
        """C6 coverage: fraction of target ROI points with a reconstructed point within coverage_tol_mm."""
        if self._roi_tree is None or len(recon) == 0:
            return 0.0
        d, _ = cKDTree(recon).query(self.target_roi_bone_points, k=1, distance_upper_bound=self.coverage_tol_mm)
        return float(np.mean(np.isfinite(d)))

    def _observe_current_state(self) -> Dict[str, Any]:
        """Backward-compatible alias: acquire a frame at the current pose."""
        return self._acquire(self.current_pose_params)[0]
