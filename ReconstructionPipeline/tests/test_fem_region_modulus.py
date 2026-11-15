"""Opt-in region modulus of the FEM (DEC-059: a liver modulus by label).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_region_modulus.py -q -s
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import load_patient
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")
DEV = torch.device("cpu")
ROI = (np.array([-30.0, 20.0, -55.0]), np.array([30.0, 80.0, -10.0]))   # small box under the L4 window


@pytest.fixture(scope="module")
def data():
    return load_patient(SUBJECT)


def _model(data, region=None):
    return FEMTissueModel(data, roi_min_mm=ROI[0], roi_max_mm=ROI[1], cache_dir=None, device=DEV, verbose=False,
                          region_modulus=region)


def _deep_mask(data, y_below):
    ct = np.asarray(data["ct_volume"])
    oy, sy = float(data["origin_mm"][1]), float(data["spacing_mm"][1])
    y = oy + np.arange(ct.shape[1]) * sy
    return np.broadcast_to((y < y_below)[None, :, None], ct.shape).copy()


def _u_at(model, X):
    T = model.dorsal_pose(0.0, -32.0)
    st = model.solve(T, model.first_contact_indentation(T) + 4.0)
    return np.asarray(st._displacement_fn(torch.as_tensor(X, dtype=torch.float32)).cpu()), st.reaction_force_n


def test_empty_region_changes_nothing(data):
    base, empty = _model(data), _model(data, (np.zeros_like(np.asarray(data["ct_volume"]), dtype=bool), 1.0))
    assert empty.cells_in_region().sum() == 0
    assert empty.cache_key != base.cache_key
    X = np.array([[0.0, 45.0, -32.0], [10.0, 35.0, -30.0]])
    np.testing.assert_allclose(_u_at(base, X)[0], _u_at(empty, X)[0], atol=1e-5)


def test_soft_region_moves_more(data):
    """Softening the deep tissue increases the displacement there, at the same indentation."""
    base = _model(data)
    soft = _model(data, (_deep_mask(data, 45.0), 2.0))
    n = soft.cells_in_region()
    assert 0 < n.sum() < len(n)
    X = np.array([[0.0, 40.0, -32.0], [5.0, 38.0, -32.0]])
    ub, fb = _u_at(base, X)
    us, fs = _u_at(soft, X)
    print(f"\n[region] |u| deep: base {np.linalg.norm(ub, axis=1)}, soft {np.linalg.norm(us, axis=1)}; F {fb:.2f} vs {fs:.2f} N")
    assert np.all(np.linalg.norm(us, axis=1) > np.linalg.norm(ub, axis=1))
    assert fs < fb                                   # a softer body pushes back less at the same indentation
