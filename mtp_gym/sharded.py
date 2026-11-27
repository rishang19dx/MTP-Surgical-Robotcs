"""
Sharded SB3 vector env over several GPUs (DEC-072).

    venv = ShardedSB3VecEnv(specs, gpus)      # specs: one dict per shard (patients, num_envs, task, env kwargs, seed)

One worker process per shard, pinned to one GPU (the parent sets CUDA_VISIBLE_DEVICES just before starting it),
holds its own `UltrasoundVectorEnv` over its block of patients, wrapped in `SB3VecEnv`. The parent concatenates
observations, rewards, dones and infos in shard order, so to SB3 it is one VecEnv of sum(num_envs) envs. Used when the
training patients do not fit one GPU (about 0.45 GB of FEM field per patient). Each shard's env is seeded with its
spec seed (train_vec_ppo uses seed + 1000 k), so the random streams differ from an unsharded run of the same seed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def _worker(conn, spec: Dict[str, Any], gpu: str) -> None:
    # CUDA_VISIBLE_DEVICES was set by the parent before this process started: the spawned child re-imports the
    # training module, whose imports may query CUDA, after which changing the variable has no effect.
    import torch

    from mtp_gym.core import PatientSim, UltrasoundVectorEnv
    from mtp_gym.wrappers import SB3VecEnv
    from ReconstructionPipeline.planning.train_vec_ppo import make_task

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patients = [PatientSim(p, dev) for p in spec["patients"]]
    env = UltrasoundVectorEnv(spec["num_envs"], make_task(spec["task"]), patients, seed=spec["seed"],
                              **spec.get("env_kwargs", {}))
    venv = SB3VecEnv(env)
    conn.send(("ready", venv.observation_space, venv.action_space))
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            venv.step_async(data)
            conn.send(venv.step_wait())
        elif cmd == "reset":
            if data is not None:
                venv.seed(data)
            conn.send(venv.reset())
        elif cmd == "close":
            venv.close()
            conn.send(None)
            return


class ShardedSB3VecEnv(VecEnv):
    def __init__(self, specs: Sequence[Dict[str, Any]], gpus: Sequence[str]):
        ctx = mp.get_context("spawn")
        self._conns, self._procs, self._sizes = [], [], [int(s["num_envs"]) for s in specs]
        for k, spec in enumerate(specs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, spec, gpus[k % len(gpus)]), daemon=True)
            old = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k % len(gpus)])     # inherited by the spawned worker
            try:
                p.start()
            finally:
                if old is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old
            child.close()
            self._conns.append(parent)
            self._procs.append(p)
        spaces = [c.recv() for c in self._conns]
        for s in spaces:
            if s[0] != "ready":
                raise RuntimeError(f"shard failed to start: {s}")
        self._seed = None
        self._actions = None
        super().__init__(sum(self._sizes), spaces[0][1], spaces[0][2])

    @staticmethod
    def _cat_obs(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}

    def reset(self):
        for k, c in enumerate(self._conns):
            c.send(("reset", None if self._seed is None else self._seed + 1000 * k))
        self._seed = None
        return self._cat_obs([c.recv() for c in self._conns])

    def seed(self, seed=None):
        self._seed = seed
        return [seed] * self.num_envs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self):
        o = 0
        for c, n in zip(self._conns, self._sizes):
            c.send(("step", self._actions[o:o + n]))
            o += n
        res = [c.recv() for c in self._conns]
        obs = self._cat_obs([r[0] for r in res])
        rew = np.concatenate([r[1] for r in res])
        done = np.concatenate([r[2] for r in res])
        infos = [i for r in res for i in r[3]]
        return obs, rew, done, infos

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(("close", None))
                c.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self._procs:
            p.join(timeout=30)

    def get_attr(self, attr_name, indices=None):
        return [None] * len(self._idx(indices))

    def set_attr(self, attr_name, value, indices=None):
        return None

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * len(self._idx(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._idx(indices))

    def _idx(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)
