"""
Wide start-pose randomisation shared by policy training and the benchmark.

The env's default reset draws the start pose from a small box near the L4 centre (c1 +-6 mm, c2 +-5 mm,
+-5 deg), so trials of an open-loop heuristic differ only by speckle. Training and evaluation instead draw
the start uniformly over the whole scan window (inset by `margin_mm`) with up to +-15 deg yaw and
+-10 deg pitch, so a policy has to recover from arbitrary starts and benchmark trials differ meaningfully.
"""

from typing import Any, Dict

import numpy as np

WIDE_START = {"margin_mm": 4.0, "max_yaw_deg": 15.0, "max_pitch_deg": 10.0}


def sample_start_pose(env, rng: np.random.Generator, margin_mm: float = WIDE_START["margin_mm"],
                      max_yaw_deg: float = WIDE_START["max_yaw_deg"],
                      max_pitch_deg: float = WIDE_START["max_pitch_deg"]) -> np.ndarray:
    """[c1, c2, yaw, pitch] uniform over the env scan window (inset by margin_mm) and the given angle range."""
    (a0, a1), (b0, b1) = env.c1_bounds, env.c2_bounds
    return np.array([
        rng.uniform(a0 + margin_mm, a1 - margin_mm),
        rng.uniform(b0 + margin_mm, b1 - margin_mm),
        rng.uniform(-np.radians(max_yaw_deg), np.radians(max_yaw_deg)),
        rng.uniform(-np.radians(max_pitch_deg), np.radians(max_pitch_deg)),
    ], dtype=np.float32)


def reset_wide(env, seed: int) -> Dict[str, Any]:
    """env.reset(seed) with a start pose drawn by `sample_start_pose` from a generator seeded by `seed`."""
    pose = sample_start_pose(env, np.random.default_rng([int(seed), 7919]))
    return env.reset(seed=int(seed), options={"init_pose": pose})
