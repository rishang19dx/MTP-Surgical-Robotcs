"""Frictionless sliding interface of the FEM (DEC-060: the liver slides against the abdominal wall).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_sliding.py -q -s
The sample patient stands in for the anatomy: the "region" is the tissue below y = 45 mm (a slab under the probe),
the "wall" is everything above it.
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import load_patient
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUBJECT), reason="sample patient not available")
DEV = torch.device("cpu")
ROI = (np.array([-30.0, 20.0, -55.0]), np.array([30.0, 80.0, -10.0]))


@pytest.fixture(scope="module")
def data():
    return load_patient(SUBJECT)


@pytest.fixture(scope="module")
def region(data):
    ct = np.asarray(data["ct_volume"])
    y = float(data["origin_mm"][1]) + np.arange(ct.shape[1]) * float(data["spacing_mm"][1])
    return np.broadcast_to((y < 45.0)[None, :, None], ct.shape).copy()


def _model(data, **kw):
    return FEMTissueModel(data, roi_min_mm=ROI[0], roi_max_mm=ROI[1], cache_dir=None, device=DEV, verbose=False, **kw)


def _solve(model, force_free_pose=None):
    s = BatchedFEMSolver(model)
    T = torch.as_tensor(np.stack([model.dorsal_pose(0.0, -32.0), model.dorsal_pose(5.0, -32.0, pitch=0.25)]),
                        dtype=torch.float64)
    d0 = s.first_contact(T)
    return s.solve(T, d0 + 4.0, stiffness=False), T


@pytest.fixture(scope="module")
def models(data, region):
    return {"base": _model(data), "tied": _model(data, sliding_region=region, sliding_mode="tied"),
            "slide": _model(data, sliding_region=region)}


def test_split_finds_an_interface(models):
    m = models["slide"]
    assert len(m.if_pairs) > 20
    assert np.allclose(m.mesh.nodes[m.if_pairs[:, 0]], m.mesh.nodes[m.if_pairs[:, 1]])
    assert np.all(np.abs(m.if_normals[:, 1]) > 0.5)        # the slab's top face: normals along y


def test_tied_split_reproduces_the_unsplit_model(models):
    """Control: tying the duplicated nodes in all directions must give back the original mesh's answer."""
    sb, _ = _solve(models["base"])
    st, _ = _solve(models["tied"])
    assert torch.allclose(st.reaction_force_n, sb.reaction_force_n, rtol=2e-2)
    X = torch.tensor([[[0.0, 40.0, -32.0], [6.0, 30.0, -30.0], [-8.0, 55.0, -35.0]]] * 2)
    np.testing.assert_allclose(st.displacement(X, inner=True).numpy(), sb.displacement(X).numpy(), atol=0.03)


def test_normal_continuity_and_tangential_slip(models):
    m = models["slide"]
    st, _ = _solve(m)
    idx = torch.as_tensor(m.if_lat_zyx)
    uw = st.u_grid.view(2, 3, -1)[:, :, idx].transpose(1, 2).numpy()          # wall side (B, P, 3)
    ur = st.u_grid_inner.view(2, 3, -1)[:, :, idx].transpose(1, 2).numpy()    # region side
    jump = ur - uw
    n = m.if_normals[None]
    jn = np.abs((jump * n).sum(-1))
    jt = np.linalg.norm(jump - (jump * n).sum(-1, keepdims=True) * n, axis=-1)
    scale = np.linalg.norm(uw, axis=-1).max()
    print(f"\n[sliding] max |u| {scale:.3f} mm, normal gap max {jn.max():.4f} mm, tangential slip max {jt.max():.3f} mm")
    assert jn.max() < 0.02 * scale                                            # tied along the normal
    assert jt.max() > 0.1 * jn.max() and jt.max() > 1e-3                      # free to slip tangentially


def test_sliding_report_region_motion(models):
    """Reported, not asserted: how sliding changes the motion of points inside the region."""
    sb, _ = _solve(models["base"])
    ss, _ = _solve(models["slide"])
    X = torch.tensor([[[0.0, 40.0, -32.0], [0.0, 30.0, -32.0], [10.0, 35.0, -30.0]]] * 2)
    ub = np.linalg.norm(sb.displacement(X).numpy(), axis=-1)
    us = np.linalg.norm(ss.displacement(X, inner=True).numpy(), axis=-1)
    print(f"\n[sliding] region |u| tied {np.round(ub, 3).tolist()} vs sliding {np.round(us, 3).tolist()}; "
          f"F tied {sb.reaction_force_n.numpy().round(2)} sliding {ss.reaction_force_n.numpy().round(2)}")
    assert np.all(np.isfinite(us))
