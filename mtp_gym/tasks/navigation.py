"""
Navigation to a standard plane (SonoGym task 1 analogue; DEC-041, task T6).

Goal (user decision 2026-11-07): the transverse plane through L4.  The probe centre sits over the centroid (x, z)
of the patient's US-visible dorsal L4 surface (the reconstruction task's target, `target_surface`), with yaw 0
and pitch 0.  The goal comes from the CT labels and is used for the reward and success test ONLY; it is never
part of the observation (AGENTS.md invariant 5), so a policy has to find the plane from the B-mode image.

Action: 3-DoF [dc1, dc2, dyaw] in [-1, 1] (x the env action scale: 5 mm, 5 mm, 10 deg); pitch stays at 0.
Observation: the core B-mode, probe pose and force (no task extras).
Reward: progress toward the goal, r = (dist_prev - dist) / 1 mm + (ang_prev - ang) / 1 deg x `angle_weight`,
+ `success_bonus` on success, minus the same force penalty as the reconstruction task.
Success: dist < `success_mm` and |yaw error| < `success_deg`; the episode terminates on success.
Start pose: uniform over the per-patient scan window, yaw uniform in +-`start_yaw_deg`, pitch 0.
Hidden patient placement (DEC-045): by default the env draws a per-episode offset of +-15 mm (c1, c2) and +-10 deg
(yaw) and observes the pose relative to it.  Without it every patient's goal is nearly the same absolute pose (all
patients are anchored at L4), and a blind policy driving to a fixed pose succeeds on held-out patients.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from mtp_gym.tasks.base import Task
from mtp_gym.tasks.reconstruction import target_surface


class NavigationTask(Task):
    action_dim = 3
    default_pose_offset = (15.0, 15.0, 10.0)   # DEC-045: the goal must not be readable from the pose

    def __init__(self, success_mm: float = 3.0, success_deg: float = 5.0, angle_weight: float = 0.5,
                 success_bonus: float = 10.0, start_yaw_deg: float = 30.0, weight_force: float = 0.02,
                 weight_decoupling: float = 0.05):
        self.success_mm, self.success_deg = success_mm, success_deg
        self.angle_weight, self.bonus = angle_weight, success_bonus
        self.start_yaw = math.radians(start_yaw_deg)
        self.w_force, self.w_dec = weight_force, weight_decoupling

    def bind(self, env) -> None:
        super().bind(env)
        goals = []
        for ps in env.patients:
            t = target_surface(ps.data, ps.is_l4)
            goals.append([float(t[:, 0].mean()), float(t[:, 2].mean()), 0.0, 0.0])
        goals = torch.tensor(goals, dtype=torch.float32, device=env.device)
        self.goal = env._clip_pose(goals[env.env_patient], torch.arange(env.num_envs, device=env.device))
        self.prev_dist = torch.zeros(env.num_envs, device=env.device)
        self.prev_ang = torch.zeros(env.num_envs, device=env.device)

    # ------------------------------------------------------------ helpers
    def errors(self, ids: Optional[torch.Tensor] = None):
        """(distance mm, |yaw error| deg) of envs `ids` (all if None) from the current env pose."""
        env = self.env
        ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
        p, g = env.pose[ids], self.goal[ids]
        dist = torch.hypot(p[:, 0] - g[:, 0], p[:, 1] - g[:, 1])
        ang = torch.rad2deg((p[:, 2] - g[:, 2]).abs())
        return dist, ang

    def initial_pose(self, ids):
        env, rng, n = self.env, self.env._np_rng, len(ids)
        b = env.bounds[ids].cpu().numpy()
        c1 = rng.uniform(b[:, 0], b[:, 1])
        c2 = rng.uniform(b[:, 2], b[:, 3])
        yaw = rng.uniform(-self.start_yaw, self.start_yaw, n)
        return torch.as_tensor(np.column_stack([c1, c2, yaw, np.zeros(n)]), dtype=torch.float32, device=env.device)

    def map_action(self, a):
        return torch.cat([a[:, :3], torch.zeros_like(a[:, :1])], 1), None

    def oracle_action(self) -> torch.Tensor:
        """Privileged scripted expert (uses the goal; for tests and demonstrations only)."""
        env = self.env
        d = (self.goal - env.pose) / env.action_scale
        return d[:, :3].clamp(-1.0, 1.0)

    # ------------------------------------------------------------ steps
    def observe(self, frame, first):
        dist, ang = self.errors(frame.ids)
        self.prev_dist[frame.ids], self.prev_ang[frame.ids] = dist, ang
        return {}, {}

    def evaluate(self, frame, step_path, step_rot):
        ids = frame.ids
        dist, ang = self.errors(ids)
        success = (dist < self.success_mm) & (ang < self.success_deg)
        f = frame.force_n
        pen = torch.where(f > 12.0, self.w_force * (f - 12.0) ** 2,
                          torch.where(f < 2.0, self.w_dec * (2.0 - f), torch.zeros_like(f)))
        reward = (self.prev_dist[ids] - dist) + self.angle_weight * (self.prev_ang[ids] - ang) \
            + self.bonus * success.float() - pen
        self.prev_dist[ids], self.prev_ang[ids] = dist, ang
        info = {"goal_distance_mm": dist, "goal_yaw_error_deg": ang, "success": success}
        return reward, success, {}, info
