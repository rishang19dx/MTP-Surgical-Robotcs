"""Stable-Baselines3 adapter (DEC-041, task T10).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_sb3_wrapper.py -q -s
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import ForceScanTask, NavigationTask, ReconstructionTask
from mtp_gym.wrappers import SB3VecEnv

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")


@pytest.fixture(scope="module")
def patient():
    return PatientSim(SUBJECT, DEV)


def test_spaces_and_terminal_observation(patient):
    venv = SB3VecEnv(UltrasoundVectorEnv(3, ReconstructionTask(), [patient], max_steps=2, seed=0))
    obs = venv.reset()
    for k, sp in venv.observation_space.spaces.items():
        assert obs[k].shape == (3, *sp.shape) and obs[k].dtype == sp.dtype, k
    assert obs["bmode"].shape == (3, 1, 100, 75) and obs["recon_grid"].shape == (3, 1, 16, 16, 16)
    venv.step(np.zeros((3, 4), np.float32))
    obs, rew, dones, infos = venv.step(np.zeros((3, 4), np.float32))
    assert dones.all() and all("terminal_observation" in i and i["TimeLimit.truncated"] for i in infos)
    assert set(infos[0]["terminal_observation"]) == set(obs)


@pytest.mark.parametrize("task_cls", [ReconstructionTask, NavigationTask, ForceScanTask])
def test_sb3_ppo_learns_a_few_steps(patient, task_cls):
    from stable_baselines3 import PPO
    venv = SB3VecEnv(UltrasoundVectorEnv(4, task_cls(), [patient], max_steps=20, seed=0))
    model = PPO("MultiInputPolicy", venv, n_steps=16, batch_size=32, n_epochs=1, device=DEV, verbose=0, seed=0)
    model.learn(total_timesteps=128)
    a, _ = model.predict(venv.reset(), deterministic=True)
    assert a.shape == (4, task_cls.action_dim)
