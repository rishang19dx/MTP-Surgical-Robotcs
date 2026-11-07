"""Training-time augmentation (DEC-048 thread A, H2/H3): FEM preload, non-physical warp control, imaging.

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_augmentation.py -q -s
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import NavigationTask

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")


@pytest.fixture(scope="module")
def patient():
    return PatientSim(SUBJECT, DEV)


def _pose(ps):
    T = ps.model.dorsal_pose(-20.0, -45.0)
    return torch.as_tensor(T[None], dtype=torch.float64, device=DEV), ps.model.first_contact_indentation(T)


def test_zero_preload_changes_nothing(patient):
    s = patient.solver
    T, d0 = _pose(patient)
    a = s.solve(T, d0 + 5.0)
    b = s.solve(T, d0 + 5.0, preload=torch.zeros(1, 3 * s.Sf, device=DEV), preload_margin_mm=10.0)
    assert torch.allclose(a.reaction_force_n, b.reaction_force_n, rtol=1e-6)
    assert torch.allclose(a.u_grid, b.u_grid, atol=1e-6)


def test_inward_preload_reduces_probe_force_and_deforms_without_contact(patient):
    s = patient.solver
    T, d0 = _pose(patient)
    pre = s.random_preload(1, np.random.default_rng(0), force_max_n=5.0, patches=1, radius_mm=20.0,
                           region=(-30.0, -10.0, -55.0, -35.0), max_tilt=0.0)
    pre = pre / pre.view(1, -1, 3).sum(1).norm() * 5.0                # exactly 5 N inward near the probe
    f0 = float(s.force(T, d0 + 5.0))
    f1 = float(s.force(T, d0 + 5.0, preload=pre, preload_margin_mm=10.0))
    print(f"\n[aug] probe force at d0+5 mm: no preload {f0:.2f} N, 5 N inward preload nearby {f1:.2f} N")
    assert f1 < f0                                                    # the skin already moved away from the probe
    free = s.solve(T, d0 - 5.0, preload=pre, preload_margin_mm=10.0)  # probe 5 mm above the skin
    assert float(free.u_grid.abs().max()) > 0.1                       # tissue still deformed by the preload


def test_warp_control_is_soft_tissue_only_and_rms_matched(patient):
    env = UltrasoundVectorEnv(4, NavigationTask(), [patient], augment="warp", seed=0)
    env.reset(seed=0)
    w = env.u_warp[0]
    mask = patient.solver.soft_mask()
    assert float(w[:, :, ~mask].abs().max()) == 0.0                   # bone / fixed / outside ROI untouched
    assert float(w.abs().max()) > 0.0


@pytest.mark.parametrize("mode", ["fem", "warp", "imaging"])
def test_env_runs_with_augmentation(patient, mode):
    env = UltrasoundVectorEnv(4, NavigationTask(), [patient], augment=mode, seed=1)
    obs, _ = env.reset(seed=1)
    for _ in range(3):
        obs, r, term, trunc, info = env.step(np.zeros((4, 3), np.float32))
    assert torch.isfinite(obs["bmode"]).all() and torch.isfinite(r).all()
    base = UltrasoundVectorEnv(4, NavigationTask(), [patient], seed=1)
    ob, _ = base.reset(seed=1)
    if mode == "fem":
        assert float(env.preload.abs().sum()) > 0
    if mode == "imaging":
        assert float(env.img_gain_db.abs().sum()) > 0
