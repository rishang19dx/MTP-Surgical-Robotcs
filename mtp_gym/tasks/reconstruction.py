"""
Bone-surface reconstruction task (SonoGym task 2 analogue; DEC-041, task T5).

Observation (image-only, AGENTS.md invariant 5): the core B-mode / pose / force, plus
  * "recon_grid" (1, G, G, G): occupancy of this episode's reconstruction (voxels hit by B-mode-segmented,
    back-projected bone points) max-pooled over a box around the target, like SonoGym's 40^3 voxel observation;
  * "coverage" (1,): the coverage ratio below.

Reward (as env_reconstruction.step): 0.05 x marginal reconstructed area (occupied 1.5 mm voxels x 1.5^2 mm^2)
- w_path x path - w_rot x rotation - force penalty (w_force (F-12)^2 above 12 N, w_decoupling (2-F) below 2 N),
-10 on a safety violation.  Terminates at `target_coverage`.

Coverage (reward/termination only): fraction of the patient's US-visible target surface (outermost bone point
per 1 mm column inside the L4 ROI, `UltrasoundReconstructionEnv._dorsal_visible_surface`) with a
back-projected bone point within `coverage_tol_mm`.  Ground truth is used here for the reward only.

Differences from the single env, stated so they are not hidden:
  * coverage is measured against the raw back-projected points of every frame, not against the voxel-mean
    points of the accumulator (voxel 1.5 mm), so it can differ by a fraction of a voxel at the tolerance edge;
  * points outside the occupancy box (target box + `box_margin_mm`) do not count towards the area;
  * no statistical outlier removal (the single env also applies it only in `finalize()`, not per step).
Benchmarks should report the exact point-cloud metrics (`ReconstructionPipeline.reconstruction.metrics`).

Frames are fused only when the sensed force is >= 1 N (as the single env).  In `unaware` mode points are
back-projected with the skin pose (the frame's `t_backproject`), in `aware` mode with the indented face pose.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter
from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM
from mtp_gym.perception import backproject, segment_bmode_batch
from mtp_gym.tasks.base import Task


def target_surface(patient_data: Dict, is_l4: bool) -> np.ndarray:
    """US-visible target bone surface, exactly as `UltrasoundReconstructionEnv.__init__`."""
    pts = np.asarray(patient_data["bone_points"])
    if len(pts) > 5000:
        if is_l4:
            m = (np.abs(pts[:, 0]) <= 32.0) & (pts[:, 2] >= -56.0) & (pts[:, 2] <= -10.0) & (pts[:, 1] >= 10.0)
        else:
            m = (np.abs(pts[:, 0]) <= 25.0) & (np.abs(pts[:, 2]) <= 35.0) & (pts[:, 1] >= 15.0)
        roi = pts[m] if np.any(m) else pts[:5000]
        return UltrasoundReconstructionEnv._dorsal_visible_surface(roi)
    return pts.astype(np.float32)


class ReconstructionTask(Task):
    action_dim = 4

    def __init__(self, target_coverage: float = 0.90, coverage_tol_mm: float = 3.0, voxel_mm: float = 1.5,
                 box_margin_mm: float = 20.0, grid_obs: int = 32, weight_path: float = 0.01,
                 weight_rot: float = 0.05, weight_force: float = 0.02, weight_decoupling: float = 0.05,
                 min_fusion_force_n: float = 1.0):
        self.target_coverage = target_coverage
        self.tol = coverage_tol_mm
        self.voxel = voxel_mm
        self.margin = box_margin_mm
        self.G = grid_obs
        self.w_path, self.w_rot, self.w_force, self.w_dec = weight_path, weight_rot, weight_force, weight_decoupling
        self.min_force = min_fusion_force_n
        self.seg = BoneSurfaceSegmenter(axial_resolution_mm=PHYSICAL_SIZE_MM[0] / (IMAGE_SIZE[0] - 1))

    # ------------------------------------------------------------ setup
    def bind(self, env) -> None:
        super().bind(env)
        dev = env.device
        self.targets: List[torch.Tensor] = []
        lo_all, hi_all = [], []
        for ps in env.patients:
            t = target_surface(ps.data, ps.is_l4)
            if len(t) == 0:
                raise ValueError(f"{ps.name}: empty target surface")
            self.targets.append(torch.as_tensor(t, dtype=torch.float32, device=dev))
            lo_all.append(t.min(0) - self.margin)
            hi_all.append(t.max(0) + self.margin)
        ext = np.max(np.stack(hi_all) - np.stack(lo_all), 0)
        self.dims = tuple(int(v) for v in np.ceil(ext / self.voxel).astype(int) + 1)
        box_lo = torch.as_tensor(np.stack(lo_all), dtype=torch.float32, device=dev)
        self.box_lo = box_lo[env.env_patient]                                          # (N,3)
        N = env.num_envs
        self.t_max = max(len(t) for t in self.targets)
        self.occ = torch.zeros(N, *self.dims, dtype=torch.bool, device=dev)
        self.covered = torch.zeros(N, self.t_max, dtype=torch.bool, device=dev)
        self.n_targets = torch.tensor([len(self.targets[p]) for p in env.env_patient.tolist()],
                                      dtype=torch.float32, device=dev)
        self.prev_area = torch.zeros(N, device=dev)

    def observation_spaces(self):
        return {"recon_grid": spaces.Box(0.0, 1.0, (1, self.G, self.G, self.G), np.float32),
                "coverage": spaces.Box(0.0, 1.0, (1,), np.float32)}

    def reset(self, ids):
        self.occ[ids] = False
        self.covered[ids] = False
        self.prev_area[ids] = 0.0

    # ------------------------------------------------------------ fusion
    def _ingest(self, frame) -> None:
        ids = frame.ids
        has, y = segment_bmode_batch(frame.bmode, frame.confidence, self.seg)
        has &= (frame.force_n >= self.min_force)[:, None]
        pts = backproject(has, y, frame.t_backproject, IMAGE_SIZE, PHYSICAL_SIZE_MM).float()   # (n, W, 3)
        # occupancy
        vi = torch.floor((pts - self.box_lo[ids][:, None, :]) / self.voxel)
        dims = torch.tensor(self.dims, device=pts.device)
        inb = has & (vi >= 0).all(-1) & (vi < dims).all(-1)
        e, c = inb.nonzero(as_tuple=True)
        v = vi[e, c].long()
        self.occ[ids[e], v[:, 0], v[:, 1], v[:, 2]] = True
        # coverage: target points within tol of any new point, per patient group
        pat = frame.patient
        for p in pat.unique().tolist():
            loc = (pat == p).nonzero()[:, 0]
            tg = self.targets[p]
            q = torch.where(has[loc][..., None], pts[loc], torch.full_like(pts[loc], 1e6))     # (m, W, 3)
            hit = torch.zeros(len(loc), len(tg), dtype=torch.bool, device=pts.device)
            step = max(1, int(2e7 // max(1, q.shape[1] * len(tg))))
            for s in range(0, len(loc), step):
                d = torch.cdist(q[s:s + step], tg[None].expand(min(step, len(loc) - s), -1, -1))
                hit[s:s + step] = (d <= self.tol).any(1)
            self.covered[ids[loc], :len(tg)] |= hit

    def _coverage(self, ids) -> torch.Tensor:
        return self.covered[ids].sum(1).float() / self.n_targets[ids]

    def _area(self, ids) -> torch.Tensor:
        return self.occ[ids].flatten(1).sum(1).float() * self.voxel ** 2

    def _extra(self, ids) -> Dict[str, torch.Tensor]:
        g = F.adaptive_max_pool3d(self.occ[ids].unsqueeze(1).float(), self.G)
        return {"recon_grid": g, "coverage": self._coverage(ids)[:, None]}

    # ------------------------------------------------------------ steps
    def observe(self, frame, first):
        self._ingest(frame)          # the reset acquisition is fused too; prev_area stays 0 (as the single env)
        return self._extra(frame.ids), {}

    def evaluate(self, frame, step_path, step_rot):
        ids = frame.ids
        self._ingest(frame)
        area = self._area(ids)
        gain = area - self.prev_area[ids]
        self.prev_area[ids] = area
        f = frame.force_n
        pen = torch.where(f > 12.0, self.w_force * (f - 12.0) ** 2,
                          torch.where(f < 2.0, self.w_dec * (2.0 - f), torch.zeros_like(f)))
        safety = f > self.env.safety_limit
        reward = 0.05 * gain - (self.w_path * step_path + self.w_rot * step_rot + pen) - 10.0 * safety.float()
        cov = self._coverage(ids)
        terminated = cov >= self.target_coverage
        info = {"coverage_ratio": cov, "marginal_gain": gain, "total_area": area}
        return reward, terminated, self._extra(ids), info
