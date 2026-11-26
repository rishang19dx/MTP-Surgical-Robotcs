"""Parity of the cross-patient batched FEM path (DEC-066) with the per-patient path.

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_multi.py -q -s
Needs the sample patient and two SonoGym patients with a different lattice depth (NY) and free-skin count.
"""
import os
import time

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
NAMES = [os.path.join(ROOT, "data", "sample_totalsegmentator_patient"),
         os.path.join(ROOT, "data", "patients", "sonogym_s0015"),
         os.path.join(ROOT, "data", "patients", "sonogym_s0004")]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def patients():
    if not all(os.path.isdir(p) for p in NAMES):
        pytest.skip("patient assets not available")
    from mtp_gym.core import PatientSim
    return [PatientSim(p, DEV) for p in NAMES]


def _batch(patients, n_per, seed=0):
    """Interleaved envs of all patients: random poses in each window, indentations around first contact."""
    rng = np.random.default_rng(seed)
    pid, T, d = [], [], []
    for k in range(n_per):
        for i, ps in enumerate(patients):
            (a, b), (c, e) = ps.c1_bounds, ps.c2_bounds
            prm = torch.tensor([[rng.uniform(a, b), rng.uniform(c, e), rng.uniform(-0.5, 0.5),
                                 rng.uniform(-0.4, 0.4)]], dtype=torch.float32, device=DEV)
            Ti = ps.skin_pose(prm)
            d0 = ps.solver.first_contact(Ti)
            d0 = torch.where(torch.isfinite(d0), d0, torch.zeros_like(d0))
            pid.append(i)
            T.append(Ti[0])
            d.append(float(d0[0]) + (rng.uniform(-2.0, -0.1) if k % 7 == 0 else rng.uniform(0.5, 12.0)))
    return (torch.tensor(pid, device=DEV), torch.stack(T),
            torch.tensor(d, dtype=torch.float64, device=DEV))


def test_patients_differ_in_shape(patients):
    ny = {p.model.NY for p in patients}
    sf = {len(p.model.free_skin) for p in patients}
    assert len(ny) > 1 or len(sf) > 1, "the parity test needs patients with different lattice/free-skin sizes"


def test_queries_match_per_patient(patients):
    from ReconstructionPipeline.simulation.fem_multi import MultiPatientFEMSolver
    mp = MultiPatientFEMSolver([p.solver for p in patients])
    pid, T, d = _batch(patients, 24)
    es = torch.rand(len(pid), dtype=torch.float64, device=DEV) * 1.5 + 0.5
    fc = mp.first_contact(pid, T)
    th = mp.tissue_thickness(pid, T)
    mx = mp.max_indentation(pid, T)
    res = mp.contact(pid, T, d, es)
    for i, ps in enumerate(patients):
        r = (pid == i).nonzero()[:, 0]
        s = ps.solver
        torch.testing.assert_close(fc[r], s.first_contact(T[r]), equal_nan=True, rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(th[r], s.tissue_thickness(T[r]), equal_nan=True, rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(mx[r], s.max_indentation(T[r]), equal_nan=True, rtol=1e-6, atol=1e-5)
        ref = s.solve(T[r], d[r], e_scale=es[r], field=True, stiffness=False)
        torch.testing.assert_close(res["F"][r].double(), ref.reaction_force_n.double(), rtol=1e-4, atol=1e-5)
        u, _ = s._field(res["f"][r], res["fr"][r], res["b"][r], res["es"][r])
        err = (u - ref.u_grid).abs().max().item()
        assert err < 1e-3 * max(ref.u_grid.abs().max().item(), 1.0), err
    assert bool((res["F"] > 0).any()) and bool((res["F"] == 0).any()), "batch must mix contact and no contact"


@pytest.mark.parametrize("mode,augment", [("aware", None), ("rigid", None), ("aware", "fem"), ("aware", "warp")])
def test_env_matches_per_patient_path(patients, mode, augment):
    from mtp_gym.core import UltrasoundVectorEnv
    from mtp_gym.tasks import FreeScanTask
    out = {}
    for batched in (True, False):
        env = UltrasoundVectorEnv(12, FreeScanTask(), patients, deformation_mode=mode, max_steps=10 ** 9, seed=3,
                                  augment=augment)
        env.batch_patients = batched
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        rows = []
        t0 = time.perf_counter()
        for _ in range(4):
            obs, _, _, _, info = env.step(rng.uniform(-1, 1, (12, 4)).astype(np.float32))
            rows.append((info["normal_force"].double(), info["indentation_mm"].double(), obs["bmode"].float()))
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        print(f"[*] {mode} {augment} batched={batched}: {(time.perf_counter() - t0) * 250:.1f} ms/step (12 envs, 3 patients)")
        out[batched] = rows
    for (fa, ia, ba), (fb, ib, bb) in zip(out[True], out[False]):
        torch.testing.assert_close(fa, fb, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ia, ib, rtol=1e-5, atol=1e-5)
        assert (ba - bb).abs().mean().item() < 1e-3, (ba - bb).abs().mean().item()
