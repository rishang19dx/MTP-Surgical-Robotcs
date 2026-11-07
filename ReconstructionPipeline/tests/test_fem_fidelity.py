"""Neo-Hookean reference FEM (DEC-041 task T14): the reference must linearise to the precomputed linear model.

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_fidelity.py -q -s   (about 1 min on CPU)
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import PatientSim
from ReconstructionPipeline.simulation.fem_nonlinear import NeoHookeanReference, contact_nodal_forces

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")


@pytest.fixture(scope="module")
def setup():
    ps = PatientSim(SUBJECT, DEV)
    m = ps.model
    T = m.dorsal_pose(-20.0, -45.0)
    f, loaded, b = contact_nodal_forces(m, T, m.first_contact_indentation(T) + 3.0)
    return m, NeoHookeanReference(m), f, loaded, b


def test_linearisation_equals_precomputed_compliance(setup):
    """K(u=0) of the Neo-Hookean energy reproduces the precomputed skin compliance C / e_scale."""
    m, ref, f, loaded, b = setup
    sk = m.mesh.skin_nodes
    u = ref.linear_solve(f)[sk[loaded]].ravel()
    fr = m.skin_to_free[loaded]
    rows = (3 * fr[:, None] + np.arange(3)).ravel()
    u_C = m.C_base[np.ix_(rows, rows)] @ f[sk[loaded]].ravel() / m.e_scale
    rel = np.abs(u - u_C).max() / np.abs(u_C).max()
    print(f"\n[fidelity] linearisation vs precomputed compliance: max rel diff {rel:.2e}")
    assert rel < 1e-4          # the compliance comes from a float32 PCG with rtol 1e-5


def test_small_load_nonlinear_equals_linear(setup):
    m, ref, f, loaded, b = setup
    eps = 1e-3
    r = ref.solve(f * eps, load_steps=1)
    u_lin = ref.linear_solve(f) * eps
    rel = np.abs(r["u"] - u_lin).max() / np.abs(u_lin).max()
    assert r["history"][-1]["rel_residual"] < 1e-8
    assert rel < 1e-3          # nonlinear terms are O(load): 7e-5 measured at 1e-3 x 3 N
