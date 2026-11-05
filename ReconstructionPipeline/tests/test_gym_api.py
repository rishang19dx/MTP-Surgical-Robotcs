"""mtp_gym vector env (DEC-041, task T3).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_gym_api.py -q -s
"""
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

import mtp_gym
from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import FreeScanTask

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")


@pytest.fixture(scope="module")
def patient():
    return PatientSim(SUBJECT, DEV)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def test_make_vec_and_spaces(patient):
    envs = gym.make_vec("MTP/FreeScan-v0", num_envs=4, vectorization_mode="vector_entry_point",
                        patients=[patient], seed=0)
    obs, info = envs.reset(seed=0)
    assert set(obs) == {"bmode", "probe_pose", "force"}
    for k, v in obs.items():
        assert envs.single_observation_space[k].contains(_np(v[0]).astype(np.float32)), k
        assert v.shape == envs.observation_space[k].shape
    obs, r, term, trunc, info = envs.step(envs.action_space.sample())
    assert r.shape == (4,) and term.dtype == torch.bool and trunc.shape == (4,)
    assert envs.single_action_space.shape == (4,)


def test_parity_with_single_env(patient):
    """Same start pose -> same admittance-controlled force and indentation as env_reconstruction."""
    from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
    single = UltrasoundReconstructionEnv(patient_data=patient.data, tissue_model=patient.model, device=DEV, seed=0)
    vec = UltrasoundVectorEnv(1, FreeScanTask(), [patient], seed=0)
    poses = [[0.0, -30.0, 0.0, 0.0], [-15.0, -40.0, 0.3, -0.1], [18.0, -25.0, -0.2, 0.2]]
    for p in poses:
        o_s = single.reset(options={"init_pose": p})
        o_v, _ = vec.reset(options={"init_pose": p})
        assert abs(float(o_v["force"][0, 0]) - o_s["normal_force"]) < 1e-3 * max(1.0, o_s["normal_force"]), p
        assert abs(float(vec.last_frame.indentation_mm[0]) - o_s["indentation_mm"]) < 1e-3, p
        assert abs(float(vec.last_frame.kappa[0]) - o_s["acoustic_coupling"]) < 1e-3, p
        a = np.array([[0.5, -0.4, 0.2, 0.1]], dtype=np.float32)
        o_s, *_ = single.step(a[0])
        o_v, *_ = vec.step(a)
        np.testing.assert_allclose(_np(o_v["probe_pose"][0]), o_s["probe_pose"], atol=1e-5)
        assert abs(float(o_v["force"][0, 0]) - o_s["normal_force"]) < 1e-3 * max(1.0, o_s["normal_force"]), p


def test_autoreset_same_step(patient):
    env = UltrasoundVectorEnv(3, FreeScanTask(), [patient], max_steps=2, seed=1)
    env.reset(seed=1)
    a = np.zeros((3, 4), dtype=np.float32)
    _, _, _, trunc, info = env.step(a)
    assert not bool(trunc.any()) and "final_obs" not in info
    obs, _, _, trunc, info = env.step(a)
    assert bool(trunc.all()) and bool(info["_final_obs"].all())
    assert int(env.step_count.max()) == 0                 # already reset
    assert info["final_obs"]["bmode"].shape == obs["bmode"].shape


def test_rigid_mode_images_at_skin(patient):
    env = UltrasoundVectorEnv(2, FreeScanTask(), [patient], deformation_mode="rigid", seed=2)
    env.reset(seed=2)
    fr = env.last_frame
    assert torch.equal(fr.t_img, fr.t_skin)
    assert bool((fr.force_n > 1.0).all())                 # force is still simulated (controller in contact)


def test_stiffness_randomisation(patient):
    """Wider e_scale -> different forces at the same commanded indentation."""
    env = UltrasoundVectorEnv(8, FreeScanTask(), [patient], e_scale_range=(0.5, 2.0), seed=3)
    env.reset(seed=3, options={"init_pose": [-20.0, -45.0, 0.0, 0.0]})
    es = env.e_scale.cpu().numpy()
    assert es.std() > 0.1 * es.mean()
    T = env.last_frame.t_skin
    F = env.patients[0].solver.force(T, torch.full((8,), 6.0, dtype=torch.float64, device=DEV), e_scale=env.e_scale)
    order = np.argsort(es)
    assert np.all(np.diff(_np(F)[order]) > 0)             # stiffer -> larger force


def test_mtp_gym_never_imports_pybullet():
    code = "import sys, mtp_gym, mtp_gym.core; assert 'pybullet' not in sys.modules, 'pybullet imported'"
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)
