"""
Stable-Baselines3 adapter for `UltrasoundVectorEnv` (DEC-041, task T10).

    from mtp_gym.wrappers import SB3VecEnv
    venv = SB3VecEnv(UltrasoundVectorEnv(32, ReconstructionTask(), patients))
    model = stable_baselines3.PPO("MultiInputPolicy", venv)

The core env keeps everything on the GPU; SB3 works on numpy.  The adapter converts once per step and makes the
observations SB3-friendly without changing what they contain:
  * "bmode": uint8 (1, H/2, W/2), 2x2 average-pooled B-mode (SB3's NatureCNN needs uint8 image boxes);
  * "recon_grid" (if present): max-pooled to (1, G/2, G/2, G/2) floats, flattened by SB3's combined extractor;
  * "probe_pose": [c1, c2] / 50 mm and [yaw, pitch] in rad;  "force", "target_force": / 10 N;
  * everything else unchanged.
Episode ends follow SB3's convention: `dones = terminated | truncated`, `infos[i]["terminal_observation"]` holds the
final observation (the core env already reset env i in the same step), `infos[i]["TimeLimit.truncated"]` is set
for time-limit ends.  Scalar per-env diagnostics from the core info dict are copied into `infos[i]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from mtp_gym.core import UltrasoundVectorEnv

POSE_SCALE = np.array([50.0, 50.0, 1.0, 1.0], dtype=np.float32)


class SB3VecEnv(VecEnv):
    def __init__(self, env: UltrasoundVectorEnv, image_pool: int = 2, grid_pool: int = 2):
        self.env = env
        self.image_pool, self.grid_pool = int(image_pool), int(grid_pool)
        src = env.single_observation_space.spaces
        h, w = src["bmode"].shape[1:]
        obs_spaces = {"bmode": spaces.Box(0, 255, (1, h // self.image_pool, w // self.image_pool), np.uint8)}
        for k, sp in src.items():
            if k == "bmode":
                continue
            if k == "recon_grid":
                g = sp.shape[1] // self.grid_pool
                obs_spaces[k] = spaces.Box(0.0, 1.0, (1, g, g, g), np.float32)
            else:
                obs_spaces[k] = spaces.Box(-np.inf, np.inf, sp.shape, np.float32)
        super().__init__(env.num_envs, spaces.Dict(obs_spaces), env.single_action_space)
        self._actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------ conversion
    def _convert(self, obs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        out = {}
        b = obs["bmode"].float()
        if self.image_pool > 1:
            b = F.avg_pool2d(b, self.image_pool)
        out["bmode"] = (b.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        for k, v in obs.items():
            if k == "bmode":
                continue
            v = v.float()
            if k == "recon_grid" and self.grid_pool > 1:
                v = F.max_pool3d(v, self.grid_pool)
            elif k == "probe_pose":
                v = v / torch.as_tensor(POSE_SCALE, device=v.device)
            elif k in ("force", "target_force"):
                v = v / 10.0
            out[k] = v.cpu().numpy().astype(np.float32)
        return out

    def _infos(self, info: Dict[str, Any], terminated, truncated) -> List[Dict[str, Any]]:
        n = self.num_envs
        infos: List[Dict[str, Any]] = [{} for _ in range(n)]
        for k, v in info.items():
            if k in ("final_obs", "_final_obs"):
                continue
            if torch.is_tensor(v) and v.shape[:1] == (n,) and v.dim() == 1:
                arr = v.detach().cpu().numpy()
                for i in range(n):
                    infos[i][k] = arr[i].item()
        if "final_obs" in info:
            fin = self._convert(info["final_obs"])
            mask = info["_final_obs"].cpu().numpy()
            for i in np.nonzero(mask)[0]:
                infos[i]["terminal_observation"] = {k: v[i] for k, v in fin.items()}
        term, trunc = terminated.cpu().numpy(), truncated.cpu().numpy()
        for i in range(n):
            infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
        return infos

    # ------------------------------------------------------------ VecEnv API
    def reset(self):
        seed = self._seeds[0] if getattr(self, "_seeds", None) and self._seeds[0] is not None else None
        obs, _ = self.env.reset(seed=seed)
        self._seeds = [None] * self.num_envs
        return self._convert(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        obs, rew, term, trunc, info = self.env.step(self._actions)
        dones = (term | trunc).cpu().numpy()
        return self._convert(obs), rew.float().cpu().numpy(), dones, self._infos(info, term, trunc)

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * len(self._idx(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def _idx(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)
