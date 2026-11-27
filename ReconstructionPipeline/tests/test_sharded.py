"""Sharded SB3 vector env (DEC-072): shapes, stepping and episode ends across worker processes.

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_sharded.py -q
"""
import os

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
P = [os.path.join(ROOT, "data", "sample_totalsegmentator_patient"),
     os.path.join(ROOT, "data", "patients", "sonogym_s0015")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="workers use the GPU")
def test_two_shards_step_like_one_env():
    if not all(os.path.isdir(p) for p in P):
        pytest.skip("patients not available")
    from mtp_gym.sharded import ShardedSB3VecEnv
    kw = dict(deformation_mode="aware", max_steps=5)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    venv = ShardedSB3VecEnv([{"patients": [P[0]], "num_envs": 4, "task": "nav", "seed": 0, "env_kwargs": kw},
                             {"patients": [P[1]], "num_envs": 4, "task": "nav", "seed": 1000, "env_kwargs": kw}], gpus)
    try:
        assert venv.num_envs == 8
        obs = venv.reset()
        assert obs["bmode"].shape[0] == 8 and obs["bmode"].dtype == np.uint8
        rng = np.random.default_rng(0)
        ended = np.zeros(8, bool)
        for _ in range(6):
            venv.step_async(rng.uniform(-1, 1, (8, venv.action_space.shape[0])).astype(np.float32))
            obs, rew, done, infos = venv.step_wait()
            assert rew.shape == (8,) and done.shape == (8,) and len(infos) == 8
            for i in np.nonzero(done)[0]:
                assert "terminal_observation" in infos[i]
            ended |= done
        assert ended.all(), "every env must end within max_steps + 1 steps"
        if len(gpus) > 1:   # each shard must sit on its own GPU
            import subprocess
            out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                          text=True)
            uu = {l.split(",")[1].strip() for l in out.strip().splitlines()
                  if int(l.split(",")[0]) in {p.pid for p in venv._procs}}
            assert len(uu) == 2, uu
    finally:
        venv.close()
