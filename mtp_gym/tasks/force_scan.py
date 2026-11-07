"""
Force-regulated scanning (new task; DEC-041, task T7).  Only meaningful with soft tissue.

The bone-reconstruction task (`ReconstructionTask`) with the contact force as part of the action:

  action (5,) in [-1, 1]: [dc1, dc2, dyaw, dpitch, f];  target force F* = f_min + (f + 1)/2 (f_max - f_min)
  (default 1-15 N), tracked by the env's admittance controller (2 iterations of at most 1.5 mm per step, so the
  force follows the command over a few steps, as on a real robot).

  reward = reconstruction reward (area gain, path/rotation cost, force penalty outside 2-12 N, safety)
           + w_vis x visibility          (fraction of scanlines with a B-mode bone surface: IMAGE ONLY)
           - w_force_cost x F / 1 N       (contact cost; pressing harder is not free)

Why the task exists: the compression sweep (results/compression_sweep_20261105_235841.md) shows that in the
deformable simulator force moves the bone echo 6-10 mm toward the probe and, in thick backs, decides whether bone
is visible at all.  So the best force depends on the patient (hypothesis H2: it rises with soft-tissue thickness).
In `rigid` mode the image does not depend on force, which makes it the control condition.

Extra observation: "target_force" (1,), the commanded force in N (the sensed force is the core "force").
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from mtp_gym.tasks.reconstruction import ReconstructionTask


class ForceScanTask(ReconstructionTask):
    action_dim = 5

    def __init__(self, f_min_n: float = 1.0, f_max_n: float = 15.0, weight_visibility: float = 2.0,
                 weight_force_cost: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.f_min, self.f_max = float(f_min_n), float(f_max_n)
        self.w_vis, self.w_fc = float(weight_visibility), float(weight_force_cost)

    def observation_spaces(self):
        sp = super().observation_spaces()
        sp["target_force"] = spaces.Box(0.0, np.inf, (1,), np.float32)
        return sp

    def force_command(self, f_action: torch.Tensor) -> torch.Tensor:
        return self.f_min + 0.5 * (f_action.clamp(-1, 1) + 1.0) * (self.f_max - self.f_min)

    def force_action(self, force_n: float) -> float:
        """Inverse of `force_command` (for scripted policies)."""
        return float(np.clip(2.0 * (force_n - self.f_min) / (self.f_max - self.f_min) - 1.0, -1.0, 1.0))

    def map_action(self, a):
        return a[:, :4], self.force_command(a[:, 4])

    def _extra(self, ids):
        out = super()._extra(ids)
        out["target_force"] = self.env.target_force[ids][:, None].float()
        return out

    def evaluate(self, frame, step_path, step_rot):
        reward, terminated, extra, info = super().evaluate(frame, step_path, step_rot)
        vis = self.last_visibility
        reward = reward + self.w_vis * vis - self.w_fc * frame.force_n
        info.update(bone_visibility=vis, target_force_n=frame.target_force_n)
        return reward, terminated, extra, info
