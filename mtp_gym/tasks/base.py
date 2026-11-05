"""Task interface for `UltrasoundVectorEnv` (DEC-041).

A task owns everything that differs between benchmark tasks: the action mapping, extra observations,
reward, termination and per-episode state.  The core env owns the physics, imaging and force control.

Rules every task must follow (AGENTS.md invariant 5):
  * observations may only use the B-mode image, the image-only confidence map, the probe pose, the
    sensed force and quantities computed from them;
  * labels / CT / ground-truth bone may be used for REWARDS, GOALS and METRICS only, never observations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from gymnasium import spaces


class Task:
    action_dim: int = 4

    def bind(self, env) -> None:
        self.env = env

    def observation_spaces(self) -> Dict[str, spaces.Space]:
        return {}

    def initial_pose(self, ids: torch.Tensor) -> Optional[torch.Tensor]:
        """(n,4) start poses for envs `ids`, or None for the env default."""
        return None

    def reset(self, ids: torch.Tensor) -> None:
        """Clear per-episode state of envs `ids` (called before their first acquisition)."""

    def map_action(self, a: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(N, action_dim) in [-1,1] -> pose deltas (N,4) in units of the env action scale, and an optional
        (N,) target force [N] for the admittance controller (None = keep the current target)."""
        return a[:, :4], None

    def observe(self, frame, first: bool) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Extra observations (rows = frame.ids) after a reset acquisition."""
        return {}, {}

    def evaluate(self, frame, step_path: torch.Tensor, step_rot: torch.Tensor):
        """Full-batch step -> (reward (N,), terminated (N,) bool, extra obs, info dict)."""
        raise NotImplementedError


class FreeScanTask(Task):
    """No objective: zero reward, never terminates (core env tests, throughput, teleoperation)."""

    def evaluate(self, frame, step_path, step_rot):
        n = len(frame.ids)
        z = torch.zeros(n, device=frame.force_n.device)
        return z, torch.zeros(n, dtype=torch.bool, device=z.device), {}, {}
