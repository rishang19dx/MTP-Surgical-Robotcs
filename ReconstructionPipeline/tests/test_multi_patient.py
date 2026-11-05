"""Multi-patient vector env with shared tissue moduli (DEC-041, task T4).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_multi_patient.py -q -s
Needs the imported SonoGym subjects (ReconstructionPipeline/data/import_sonogym_subjects.py).
"""
import json
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import SHARED_E_SCALE, PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import ReconstructionTask

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SAMPLE = os.path.join(ROOT, "data", "sample_totalsegmentator_patient")
THIN, THICK = os.path.join(ROOT, "data", "patients", "sonogym_s0015"), os.path.join(ROOT, "data", "patients",
                                                                                     "sonogym_s0014")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not (os.path.isdir(THIN) and os.path.isdir(THICK)),
                                reason="imported SonoGym subjects not available")


@pytest.fixture(scope="module")
def patients():
    return [PatientSim(THIN, DEV), PatientSim(THICK, DEV)]


def test_splits_file_is_consistent():
    s = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    assert not set(s["train"]) & set(s["test"])
    for n in s["train"] + s["test"]:
        assert os.path.isfile(os.path.join(ROOT, "data", "patients", n, "ct.nii.gz")), n


def test_shared_moduli(patients):
    for p in patients:
        assert p.model.e_scale == pytest.approx(SHARED_E_SCALE)
        assert p.model.effective_moduli_kpa == pytest.approx((10.0 * SHARED_E_SCALE, 20.0 * SHARED_E_SCALE))


def test_anatomy_changes_force_under_shared_moduli(patients):
    """Same moduli, same probe pose and the same indentation past first contact: the thin back (s0015, 15 mm
    midline soft tissue) must push back harder than the thick back (s0014, 78 mm), because bone is closer."""
    F = []
    for p in patients:
        T = p.model.dorsal_pose(0.0, -32.5)
        d0 = p.model.first_contact_indentation(T)
        F.append(p.model.force_at(T, d0 + 6.0))
    print(f"\n[multi] force at first contact + 6 mm: thin {F[0]:.2f} N, thick {F[1]:.2f} N")
    assert F[0] > 1.2 * F[1]


def test_two_patient_recon_env(patients):
    env = UltrasoundVectorEnv(4, ReconstructionTask(target_coverage=1.01), patients, seed=0)
    assert env.env_patient.tolist() == [0, 0, 1, 1]
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.random.default_rng(0).uniform(-1, 1, (4, 4)).astype(np.float32))
    assert obs["recon_grid"].shape[0] == 4 and torch.isfinite(r).all()
    assert info["patient"].tolist() == [0, 0, 1, 1]
    th = info["tissue_thickness_mm"].cpu().numpy()
    assert np.nanmean(th[2:]) > np.nanmean(th[:2]) + 10.0      # thick patient's envs see thicker tissue
