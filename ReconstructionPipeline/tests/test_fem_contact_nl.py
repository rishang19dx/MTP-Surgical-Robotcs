"""Large-strain FEM with contact on the deformed skin (DEC-071): gradient, small-strain agreement, vein collapse.

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_contact_nl.py -q -s
Uses the Sato & Karino popliteal phantom of vein_calibration.py (fat block, 5 mm lumen 11.5 mm deep) at h = 2 mm.
"""
import numpy as np
import pytest
import torch

from mtp_gym.core import SHARED_E_SCALE
from ReconstructionPipeline.evaluation import vein_calibration as vc
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
from ReconstructionPipeline.simulation.fem_contact_nl import NeoHookeanContact
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel

H = 2.0


def _model(lumen_e_eff=None):
    data, lumen = vc.phantom()
    lo = np.array([-vc.HALF_X, 0.0, -vc.HALF_Z])
    hi = np.array([vc.HALF_X, vc.SKIN_Y + 4.0, vc.HALF_Z])
    reg = {} if lumen_e_eff is None else {"region_modulus": (lumen, lumen_e_eff / SHARED_E_SCALE, vc.NU_LUMEN)}
    m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=H, cache_dir=None,
                       device=torch.device("cpu"), verbose=False, **reg)
    m.set_parameters(e_scale=SHARED_E_SCALE)
    return m


@pytest.fixture(scope="module")
def slab():
    return _model()


def test_gradient_matches_finite_differences(slab):
    nh = NeoHookeanContact(slab)
    T = slab.dorsal_pose(0.0, 0.0)
    d = float(slab.first_contact_indentation(T)) + 1.0
    rng = np.random.default_rng(0)
    u = np.zeros(nh.n)
    u[nh.fidx] = rng.normal(0, 0.05, len(nh.fidx))
    _, w, _ = nh._contact(u, T, d)
    g, _, _, _ = nh.grad_hess(u, T, d, w)
    top = nh.skin[np.argsort(-w)[:3]]                       # skin nodes under the face: contact term active
    for k in [3 * int(top[0]) + 1, 3 * int(top[1]) + 1, int(nh.fidx[len(nh.fidx) // 2])]:
        e = np.zeros(nh.n)
        e[k] = 1e-6
        fd = (nh.energy(u + e, T, d, w) - nh.energy(u - e, T, d, w)) / 2e-6
        assert abs(fd - g[k]) <= 1e-4 * max(1.0, abs(g[k])), (k, fd, g[k])


def test_small_indentation_force_close_to_linear_model(slab):
    nh = NeoHookeanContact(slab)
    T = slab.dorsal_pose(0.0, 0.0)
    res = nh.sweep(T, d_max_mm=0.5, step_mm=0.25)
    assert res["converged"]
    f_nl = res["curve"][-1]["force_n"]
    s = BatchedFEMSolver(slab, contact_dtype=torch.float64)
    Tt = torch.as_tensor(T[None], dtype=torch.float64)
    st = s.solve(Tt, s.first_contact(Tt) + 0.5, field=False, stiffness=False)
    f_lin = float(st.linear_force_n[0])
    print(f"[*] 0.5 mm: Neo-Hookean + deformed-skin contact {f_nl:.3f} N, linear (before stiffening) {f_lin:.3f} N")
    assert abs(f_nl - f_lin) <= 0.25 * f_lin


def test_vein_collapses_monotonically():
    m = _model(lumen_e_eff=vc.E_LUMEN_VOID)
    nh = NeoHookeanContact(m)
    T = m.dorsal_pose(0.0, 0.0)
    yc = vc.SKIN_Y - vc.DEPTH_MM
    walls = np.array([[0.0, yc + vc.DIAM_MM / 2, 0.0], [0.0, yc - vc.DIAM_MM / 2, 0.0]])
    rows = []

    def rec(dd, F, u):
        du = nh.displacement_at(u, walls)
        rows.append((F, (vc.DIAM_MM + du[0, 1] - du[1, 1]) / vc.DIAM_MM))

    res = nh.sweep(T, d_max_mm=8.0, step_mm=0.5, f_stop_n=3.0, on_step=rec)
    print(f"[*] force / AP ratio: {[(round(f, 2), round(r, 3)) for f, r in rows]}; converged {res['converged']}")
    assert res["converged"] and len(rows) >= 4
    F = np.array([r[0] for r in rows])
    R = np.array([r[1] for r in rows])
    assert np.all(np.diff(F) > 0) and np.all(np.diff(R) <= 1e-6) and R[-1] < 0.9
