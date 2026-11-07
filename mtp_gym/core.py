"""
GPU-batched robotic-ultrasound vector environment (DEC-041, task T3).

`UltrasoundVectorEnv` steps N probes over dorsal skin at once.  Per step it reproduces the single-env
pipeline of `ReconstructionPipeline/planning/env_reconstruction.py` (`step` / `_acquire`), but every stage
is a batched tensor operation over the envs of one patient (and a Python loop over patients only):

  1. action -> pose deltas [dc1, dc2, dyaw, dpitch] (task-defined mapping), clipped to the per-patient scan
     window and angle limits;
  2. skin pose on the undeformed skin (height map), rotation R = Rx(pitch) Ry(-yaw) R_base;
  3. admittance force control along the beam from first contact (`regulate_force`), warm-started per env;
     the target force may come from the action (force-scan task) or a constant;
  4. tissue solve at the final indentation (`BatchedFEMSolver`), indented face pose, backward-warped CT
     slice; "rigid" mode slices the undeformed CT at the skin pose (SonoGym-style rigid patient);
  5. model-based B-mode, acoustic coupling kappa from force and beam tilt (dB loss before display),
     image-only confidence map.

Tasks (`mtp_gym.tasks`) turn the resulting `Frame` into observations, rewards and terminations.
No PyBullet anywhere: this is the training path; the 1 kHz PyBullet + Slicer path is unchanged.

Autoreset is same-step (as GPU RL frameworks such as Isaac Lab / skrl expect): when env i terminates or
truncates, the returned observation for i is already the first observation of its next episode, and the
final observation is in `info["final_obs"]` (masked by `info["_final_obs"]`).

Observations and rewards are torch tensors on the env device.  `mtp_gym.wrappers` converts to numpy for
CPU libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
from ReconstructionPipeline.simulation.tissue_model import SkinSurface
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator

IMAGE_SIZE = (200, 150)          # (H, W) px, as env_reconstruction
PHYSICAL_SIZE_MM = (75.0, 50.0)  # (depth, width) mm
WINDOW_MARGIN_MM = 5.0
L4_WINDOW = ((-32.0, 32.0), (-55.0, -10.0))
DEFAULT_WINDOW = ((-45.0, 45.0), (-30.0, 30.0))

# contact status codes (env_reconstruction._coupling strings)
DECOUPLED, PARTIAL, OPTIMAL, EXCESSIVE, UNSAFE = 0, 1, 2, 3, 4
STATUS_NAMES = ("decoupled", "partial", "optimal", "excessive", "unsafe")


def load_patient(patient: Union[str, Dict[str, Any]], target_vertebra: str = "vertebrae_L4") -> Dict[str, Any]:
    """TotalSegmentator subject directory (or a ready patient dict) -> patient dict with a skin height map."""
    if isinstance(patient, str):
        patient = PatientAssetGenerator.load_totalsegmentator_subject(
            patient, target_vertebrae=[target_vertebra], crop_to_spine=False)
    if "skin_height_map" not in patient:
        patient.update(PatientAssetGenerator.compute_skin_height_map(
            patient["ct_volume"], patient["origin_mm"], patient["spacing_mm"]))
    return patient


def _bilinear(arr: torch.Tensor, o, s, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup in a (nz, nx) map with the clamping of `SkinSurface._interp`."""
    nz, nx = arr.shape
    fx = ((x - o[0]) / s[0]).clamp(0, nx - 1)
    fz = ((z - o[1]) / s[1]).clamp(0, nz - 1)
    x0 = fx.floor().long().clamp(0, max(nx - 2, 0))
    z0 = fz.floor().long().clamp(0, max(nz - 2, 0))
    x1 = (x0 + 1).clamp_max(nx - 1)
    z1 = (z0 + 1).clamp_max(nz - 1)
    tx, tz = fx - x0, fz - z0
    return (arr[z0, x0] * (1 - tx) * (1 - tz) + arr[z0, x1] * tx * (1 - tz)
            + arr[z1, x0] * (1 - tx) * tz + arr[z1, x1] * tx * tz)


def probe_rotation(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Batched `UltrasoundReconstructionEnv._probe_rotation`: R = Rx(pitch) Ry(-yaw) R_base -> (B,3,3)."""
    B = yaw.shape[0]
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    cy, sy = torch.cos(-yaw), torch.sin(-yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    r_y = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    r_x = torch.stack([o, z, z, z, cp, -sp, z, sp, cp], 1).view(B, 3, 3)
    r_base = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], dtype=yaw.dtype, device=yaw.device)
    return r_x @ r_y @ r_base


# Shared tissue stiffness (DEC-041, user decision 2026-11-05): every patient uses the same fat / muscle moduli,
# so the force at a given indentation depends on anatomy only.  The value is the modulus scale calibrated on the
# sample patient (case-0004) to the literature curve F = (k0/beta)(exp(beta d) - 1), k0 = 0.5 N/mm, beta = 0.15/mm:
# e_scale 1.0958 -> E_fat / E_muscle = 10.96 / 21.92 kPa (results/patients_fem_20261105_225601.json, first row).
SHARED_E_SCALE = 1.0958


class PatientSim:
    """Everything that is per patient and shared by all envs scanning that patient.

    stiffness="shared" (default): moduli E_fat, E_muscle = SHARED_E_SCALE x (10, 20) kPa for every patient.
    stiffness="calibrated": per-patient calibration of the modulus scale to the literature force curve over the
    scan window (the single-env default, `FEMTissueModel.from_patient_data(calibrate=True)`); it equalises the
    force response across patients and is kept as an ablation."""

    def __init__(self, patient: Union[str, Dict[str, Any]], device: torch.device, name: Optional[str] = None,
                 tissue_model: Optional[FEMTissueModel] = None, stiffness: str = "shared",
                 shared_e_scale: float = SHARED_E_SCALE):
        if stiffness not in ("shared", "calibrated"):
            raise ValueError(f"stiffness must be shared|calibrated, got {stiffness}")
        data = load_patient(patient) if not isinstance(patient, dict) or "skin_height_map" not in patient \
            else patient
        self.name = name or (patient if isinstance(patient, str) else data.get("name", "patient"))
        self.data = data
        self.device = device
        self.is_l4 = data.get("loaded_structures", []) == ["vertebrae_L4"]
        (c1b, c2b) = L4_WINDOW if self.is_l4 else DEFAULT_WINDOW
        self.stiffness = stiffness
        if tissue_model is None:
            tissue_model = FEMTissueModel.from_patient_data(
                data, calibrate=(stiffness == "calibrated"), c1_bounds=c1b, c2_bounds=c2b, device=device,
                verbose=False)
            if stiffness == "shared":
                tissue_model.set_parameters(e_scale=shared_e_scale)
        self.model = tissue_model
        self.solver = BatchedFEMSolver(self.model)
        self.ct = VolumeSliceExtractor(data["ct_volume"], spacing=data["spacing_mm"], origin=data["origin_mm"],
                                       image_size=IMAGE_SIZE, physical_size_mm=PHYSICAL_SIZE_MM, device=device)
        skin = SkinSurface.from_patient_data(data)
        f32 = dict(dtype=torch.float32, device=device)
        self.skin_o, self.skin_s = (skin.ox, skin.oz), (skin.sx, skin.sz)
        self.skin_h = torch.as_tensor(skin.h, **f32)
        self.skin_dhdx = torch.as_tensor(skin._dhdx, **f32)
        self.skin_dhdz = torch.as_tensor(skin._dhdz, **f32)
        self.c1_bounds, self.c2_bounds = self._fit_window(c1b, c2b)

    def _fit_window(self, c1b, c2b):
        """`UltrasoundReconstructionEnv._fit_window_to_tissue_region`: keep the face inside the FEM ROI."""
        lo, hi = self.model.roi_min, self.model.roi_max
        half = 0.5 * float(self.model.face_dims[0]) + WINDOW_MARGIN_MM
        out = []
        for (a, b), i in ((c1b, 0), (c2b, 2)):
            a2, b2 = max(a, float(lo[i]) + half), min(b, float(hi[i]) - half)
            if a2 >= b2:
                raise ValueError(f"{self.name}: FEM region cannot hold the scan window")
            out.append((a2, b2))
        return out[0], out[1]

    def skin_height(self, x, z):
        return _bilinear(self.skin_h, self.skin_o, self.skin_s, x, z)

    def skin_normal(self, x, z):
        gx = _bilinear(self.skin_dhdx, self.skin_o, self.skin_s, x, z)
        gz = _bilinear(self.skin_dhdz, self.skin_o, self.skin_s, x, z)
        n = torch.stack([-gx, torch.ones_like(gx), -gz], -1)
        return n / n.norm(dim=-1, keepdim=True)

    def skin_pose(self, params: torch.Tensor) -> torch.Tensor:
        """(B,4) [c1, c2, yaw, pitch] -> (B,4,4) float64 pose on the undeformed skin (`skin_pose`)."""
        p = params.double()
        T = torch.eye(4, dtype=torch.float64, device=self.device).repeat(p.shape[0], 1, 1)
        T[:, :3, :3] = probe_rotation(p[:, 2], p[:, 3])
        T[:, 0, 3] = p[:, 0]
        T[:, 1, 3] = self.skin_height(p[:, 0].float(), p[:, 1].float()).double()
        T[:, 2, 3] = p[:, 1]
        return T


@dataclass
class Frame:
    """One acquisition for a set of envs (`ids`).  Frame P, mm / N / rad."""
    ids: torch.Tensor                 # (n,) env indices
    pose_params: torch.Tensor         # (n,4) [c1, c2, yaw, pitch]
    t_skin: torch.Tensor              # (n,4,4) probe pose on the undeformed skin
    t_img: torch.Tensor               # (n,4,4) image-plane pose (indented face; = t_skin in rigid mode)
    t_backproject: torch.Tensor       # (n,4,4) pose a reconstruction should back-project with (mode-aware)
    bmode: torch.Tensor               # (n,H,W) in [0,1], after the coupling loss
    confidence: torch.Tensor          # (n,H,W) image-only confidence
    force_n: torch.Tensor             # (n,) normal force from the tissue model
    target_force_n: torch.Tensor      # (n,)
    indentation_mm: torch.Tensor      # (n,)
    kappa: torch.Tensor               # (n,) acoustic coupling
    status: torch.Tensor              # (n,) int, STATUS_NAMES
    tilt_rad: torch.Tensor            # (n,)
    tissue_thickness_mm: torch.Tensor  # (n,) along the beam under the probe centre, NaN if no bone
    patient: torch.Tensor             # (n,) patient index


class UltrasoundVectorEnv(VectorEnv):
    """N-env batched robotic-ultrasound scanning with FEM soft tissue.  See module docstring."""

    metadata = {"autoreset_mode": "same-step", "render_modes": []}

    def __init__(
        self,
        num_envs: int,
        task,
        patients: Sequence[Union[str, Dict[str, Any], PatientSim]],
        env_patient: Optional[Sequence[int]] = None,
        deformation_mode: str = "aware",
        max_steps: int = 100,
        target_force_n: float = 4.0,
        admittance_gain_mm_per_n: float = 0.5,
        admittance_max_step_mm: float = 1.5,
        admittance_iterations: int = 2,
        approach_iterations: int = 15,
        safety_force_limit_n: float = 20.0,
        max_yaw_deg: float = 45.0,
        max_pitch_deg: float = 35.0,
        action_scale_trans_mm: float = 5.0,
        action_scale_yaw_deg: float = 10.0,
        action_scale_pitch_deg: float = 15.0,
        e_scale_range: Optional[Sequence[float]] = None,
        bmode_physics: str = "depth",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        if deformation_mode not in ("aware", "unaware", "rigid"):
            raise ValueError(f"deformation_mode must be aware|unaware|rigid, got {deformation_mode}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = int(num_envs)
        self.deformation_mode = deformation_mode
        self.max_steps = int(max_steps)
        self.default_target_force = float(target_force_n)
        self.k_adm, self.adm_max_step = float(admittance_gain_mm_per_n), float(admittance_max_step_mm)
        self.adm_iters, self.approach_iters = int(admittance_iterations), int(approach_iterations)
        self.safety_limit = float(safety_force_limit_n)
        self.max_yaw, self.max_pitch = math.radians(max_yaw_deg), math.radians(max_pitch_deg)
        self.action_scale = torch.tensor([action_scale_trans_mm, action_scale_trans_mm,
                                          math.radians(action_scale_yaw_deg), math.radians(action_scale_pitch_deg)],
                                         device=self.device)
        self.e_scale_range = tuple(e_scale_range) if e_scale_range is not None else None

        self.patients: List[PatientSim] = [p if isinstance(p, PatientSim) else PatientSim(p, self.device)
                                           for p in patients]
        if env_patient is None:   # contiguous blocks, so each patient's envs are one batch
            env_patient = (np.arange(self.num_envs) * len(self.patients)) // self.num_envs
        self.env_patient = torch.as_tensor(np.asarray(env_patient), dtype=torch.long, device=self.device)
        if self.env_patient.shape != (self.num_envs,) or int(self.env_patient.max()) >= len(self.patients):
            raise ValueError("env_patient must assign each env to a patient index")
        bounds = torch.tensor([[*p.c1_bounds, *p.c2_bounds] for p in self.patients], device=self.device)
        self.bounds = bounds[self.env_patient]                                   # (N,4) c1lo c1hi c2lo c2hi

        h, w = IMAGE_SIZE
        if bmode_physics not in ("depth", "legacy"):
            raise ValueError(f"bmode_physics must be depth|legacy, got {bmode_physics}")
        self.bmode_physics = bmode_physics
        sim_kw = dict(device=self.device, axial_resolution_mm=PHYSICAL_SIZE_MM[0] / (h - 1),
                      lateral_resolution_mm=PHYSICAL_SIZE_MM[1] / (w - 1), seed=seed)
        # "depth" (default, DEC-042): per-tissue attenuation + receiver noise before the TGC; "legacy" = the
        # single-env simulator (env_reconstruction), used for parity tests
        self.us_sim = (ModelBasedUltrasoundSimulator.depth_physics(**sim_kw) if bmode_physics == "depth"
                       else ModelBasedUltrasoundSimulator(**sim_kw))
        self.conf = UltrasoundConfidenceEstimator(device=self.device,
                                                  axial_resolution_mm=PHYSICAL_SIZE_MM[0] / (h - 1))
        self._np_rng = np.random.default_rng(seed)

        N, dev = self.num_envs, self.device
        self.pose = torch.zeros(N, 4, device=dev)
        self.indent_rel = torch.zeros(N, dtype=torch.float64, device=dev)
        self.e_scale = torch.tensor([self.patients[i].model.e_scale for i in self.env_patient.tolist()],
                                    dtype=torch.float64, device=dev)
        self.step_count = torch.zeros(N, dtype=torch.long, device=dev)
        self.target_force = torch.full((N,), self.default_target_force, device=dev)
        self.last_frame: Optional[Frame] = None

        self.task = task
        task.bind(self)
        self.single_action_space = spaces.Box(-1.0, 1.0, (task.action_dim,), np.float32)
        base = {"bmode": spaces.Box(0.0, 1.0, (1, h, w), np.float32),
                "probe_pose": spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "force": spaces.Box(0.0, np.inf, (1,), np.float32)}
        base.update(task.observation_spaces())
        self.single_observation_space = spaces.Dict(base)
        self.action_space = spaces.Box(-1.0, 1.0, (N, task.action_dim), np.float32)
        self.observation_space = spaces.Dict({k: spaces.Box(np.broadcast_to(v.low, (N, *v.shape)),
                                                            np.broadcast_to(v.high, (N, *v.shape)),
                                                            (N, *v.shape), v.dtype)
                                              for k, v in base.items()})

    # ------------------------------------------------------------- episode
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
            if self.us_sim.generator is not None:
                self.us_sim.generator.manual_seed(int(seed))
            torch.manual_seed(int(seed))
        ids = torch.arange(self.num_envs, device=self.device)
        init = None if options is None else options.get("init_pose")
        obs, info = self._reset_ids(ids, init)
        return obs, info

    def _sample_init_pose(self, ids: torch.Tensor) -> torch.Tensor:
        n = len(ids)
        rng = self._np_rng
        l4 = np.array([self.patients[p].is_l4 for p in self.env_patient[ids].tolist()])
        c1 = np.where(l4, rng.uniform(-6.0, 6.0, n), rng.uniform(-10.0, 10.0, n))
        c2 = np.where(l4, rng.uniform(-36.0, -26.0, n), rng.uniform(-10.0, 10.0, n))
        ang = rng.uniform(-math.radians(5.0), math.radians(5.0), (n, 2))
        return torch.as_tensor(np.column_stack([c1, c2, ang]), dtype=torch.float32, device=self.device)

    def _clip_pose(self, p: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        b = self.bounds[ids]
        return torch.stack([torch.maximum(torch.minimum(p[:, 0], b[:, 1]), b[:, 0]),
                            torch.maximum(torch.minimum(p[:, 1], b[:, 3]), b[:, 2]),
                            p[:, 2].clamp(-self.max_yaw, self.max_yaw),
                            p[:, 3].clamp(-self.max_pitch, self.max_pitch)], 1)

    def _reset_ids(self, ids: torch.Tensor, init_pose=None):
        task_init = self.task.initial_pose(ids)
        if init_pose is not None:
            p = torch.as_tensor(np.asarray(init_pose), dtype=torch.float32, device=self.device).reshape(-1, 4)
            p = p.expand(len(ids), 4) if p.shape[0] == 1 else p[ids]
        elif task_init is not None:
            p = task_init
        else:
            p = self._sample_init_pose(ids)
        self.pose[ids] = self._clip_pose(p, ids)
        self.step_count[ids] = 0
        self.indent_rel[ids] = 0.0
        self.target_force[ids] = self.default_target_force
        if self.e_scale_range is not None:
            lo, hi = self.e_scale_range
            base = torch.tensor([self.patients[i].model.e_scale for i in self.env_patient[ids].tolist()],
                                dtype=torch.float64, device=self.device)
            mult = torch.as_tensor(np.exp(self._np_rng.uniform(np.log(lo), np.log(hi), len(ids))),
                                   dtype=torch.float64, device=self.device)
            self.e_scale[ids] = base * mult
        self.task.reset(ids)
        frame = self._acquire(ids, approach=True)
        extra, info = self.task.observe(frame, first=True)
        return self._obs(frame, extra), info

    def step(self, actions):
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device).clamp(-1.0, 1.0)
        ids = torch.arange(self.num_envs, device=self.device)
        dpose, f_target = self.task.map_action(a)
        if f_target is not None:
            self.target_force = f_target.float()
        prev = self.pose.clone()
        self.pose = self._clip_pose(self.pose + dpose * self.action_scale, ids)
        self.step_count += 1
        frame = self._acquire(ids, approach=False)
        step_path = torch.hypot(self.pose[:, 0] - prev[:, 0], self.pose[:, 1] - prev[:, 1])
        step_rot = (self.pose[:, 2] - prev[:, 2]).abs() + (self.pose[:, 3] - prev[:, 3]).abs()
        reward, terminated, extra, info = self.task.evaluate(frame, step_path, step_rot)
        safety = frame.force_n > self.safety_limit
        terminated = terminated | safety
        truncated = self.step_count >= self.max_steps
        info.update(normal_force=frame.force_n, indentation_mm=frame.indentation_mm, kappa=frame.kappa,
                    contact_status=frame.status, safety_violation=safety, step=self.step_count.clone(),
                    tissue_thickness_mm=frame.tissue_thickness_mm, patient=frame.patient)
        obs = self._obs(frame, extra)
        done = terminated | truncated
        if bool(done.any()):
            d_ids = done.nonzero()[:, 0]
            info["final_obs"] = {k: v.clone() for k, v in obs.items()}
            info["_final_obs"] = done.clone()
            new_obs, _ = self._reset_ids(d_ids)
            for k in obs:
                obs[k][d_ids] = new_obs[k]
        return obs, reward, terminated, truncated, info

    def _obs(self, frame: Frame, extra: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full-batch observation dict; `frame` may cover a subset of envs (reset), then only rows `ids`."""
        n = len(frame.ids)
        obs = {"bmode": frame.bmode.unsqueeze(1).float(), "probe_pose": frame.pose_params.float(),
               "force": frame.force_n.view(n, 1).float()}
        obs.update(extra)
        return obs

    # ---------------------------------------------------------- simulation
    def _regulate(self, ps: PatientSim, t_skin, rel, target, es, iters):
        """Batched `regulate_force`: admittance along the beam within [first contact, max indentation]."""
        s = ps.solver
        d_fc = s.first_contact(t_skin)
        d_fc = torch.where(torch.isfinite(d_fc), d_fc, torch.zeros_like(d_fc))
        d_max = torch.maximum(s.max_indentation(t_skin, first_contact=d_fc), d_fc)
        d = torch.minimum(torch.maximum(d_fc + rel, d_fc), d_max)
        for _ in range(iters):
            f = s.force(t_skin, d, e_scale=es)
            step = (self.k_adm * (target.double() - f)).clamp(-self.adm_max_step, self.adm_max_step)
            d = torch.minimum(torch.maximum(d + step, d_fc), d_max)
        return d, d_fc

    @torch.no_grad()
    def _acquire(self, ids: torch.Tensor, approach: bool) -> Frame:
        n = len(ids)
        h, w = IMAGE_SIZE
        dev = self.device
        slices = torch.empty(n, h, w, device=dev)
        force = torch.empty(n, dtype=torch.float64, device=dev)
        indent = torch.empty(n, dtype=torch.float64, device=dev)
        thick = torch.empty(n, dtype=torch.float64, device=dev)
        t_skin_all = torch.empty(n, 4, 4, dtype=torch.float64, device=dev)
        t_img_all = torch.empty_like(t_skin_all)
        normals = torch.empty(n, 3, device=dev)
        pat = self.env_patient[ids]
        for p_idx in pat.unique().tolist():
            loc = (pat == p_idx).nonzero()[:, 0]
            g = ids[loc]
            ps = self.patients[p_idx]
            t_skin = ps.skin_pose(self.pose[g])
            rel = torch.zeros(len(g), dtype=torch.float64, device=dev) if approach else self.indent_rel[g]
            it = self.approach_iters if approach else self.adm_iters
            es = self.e_scale[g]
            d, d_fc = self._regulate(ps, t_skin, rel, self.target_force[g], es, it)
            self.indent_rel[g] = (d - d_fc).double()
            if self.deformation_mode == "rigid":
                f = ps.solver.force(t_skin, d, e_scale=es)
                t_img = t_skin
                sl = ps.ct.extract_slices(t_img, interpolation="bilinear")
            else:
                st = ps.solver.solve(t_skin, d, e_scale=es, stiffness=False)
                f = st.reaction_force_n
                t_img = st.probe_face_poses(t_skin)
                sl = ps.ct.extract_slices(t_img, interpolation="bilinear", batched_warp_fn=st.warp_to_reference)
            slices[loc], force[loc], indent[loc] = sl.float(), f.double(), d.double()
            thick[loc] = ps.solver.tissue_thickness(t_skin).double()
            t_skin_all[loc], t_img_all[loc] = t_skin, t_img
            normals[loc] = ps.skin_normal(self.pose[g, 0], self.pose[g, 1])

        f32 = force.float()
        beam = t_skin_all[:, :3, 1].float()
        cos_t = (-(beam / beam.norm(dim=1, keepdim=True)) * normals).sum(1).clamp(0.0, 1.0)
        tilt = torch.arccos(cos_t.clamp(-1.0, 1.0))
        kappa, status = self._coupling(f32, tilt)
        bmode = self.us_sim.simulate_bmode(slices)["bmode"]
        loss_db = -20.0 * torch.log10(kappa.clamp_min(1e-3))
        shifted = (bmode - (loss_db / self.us_sim.dynamic_range_db)[:, None, None]).clamp(0.0, 1.0)
        bmode = torch.where((loss_db > 1e-3)[:, None, None], shifted, bmode)
        conf = self.conf.compute_confidence_map(bmode)
        t_bp = t_skin_all if self.deformation_mode in ("unaware", "rigid") else t_img_all
        frame = Frame(ids=ids, pose_params=self.pose[ids].clone(), t_skin=t_skin_all, t_img=t_img_all,
                      t_backproject=t_bp, bmode=bmode, confidence=conf, force_n=f32,
                      target_force_n=self.target_force[ids].clone(), indentation_mm=indent.float(), kappa=kappa,
                      status=status, tilt_rad=tilt, tissue_thickness_mm=thick.float(), patient=pat)
        self.last_frame = frame
        return frame

    def _coupling(self, f: torch.Tensor, tilt: torch.Tensor):
        """Batched `UltrasoundReconstructionEnv._coupling` -> (kappa, status code)."""
        tf = torch.exp(-0.5 * (tilt / math.radians(22.0)) ** 2)
        kappa = torch.where(f < 1.0, torch.full_like(f, 0.05),
                            torch.where(f < 2.0, (0.05 + 0.95 * (f - 1.0)) * tf,
                                        torch.where(f <= 12.0, tf, (1.0 - 0.05 * (f - 12.0)).clamp_min(0.2) * tf)))
        status = torch.where(f < 1.0, DECOUPLED, torch.where(f < 2.0, PARTIAL, torch.where(f <= 12.0, OPTIMAL,
                                                                                            EXCESSIVE)))
        status = torch.where(f > self.safety_limit, torch.full_like(status, UNSAFE), status)
        return kappa.clamp(0.05, 1.0), status

    def close_extras(self, **kwargs):
        self.patients = []
