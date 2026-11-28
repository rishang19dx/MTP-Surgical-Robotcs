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


def _model(lumen_e_eff=None, precompute=True):
    data, lumen = vc.phantom()
    lo = np.array([-vc.HALF_X, 0.0, -vc.HALF_Z])
    hi = np.array([vc.HALF_X, vc.SKIN_Y + 4.0, vc.HALF_Z])
    reg = {} if lumen_e_eff is None else {"region_modulus": (lumen, lumen_e_eff / SHARED_E_SCALE, vc.NU_LUMEN)}
    m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=H, cache_dir=None,
                       device=torch.device("cpu"), verbose=False, precompute=precompute, **reg)
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


def test_mesh_only_model_gives_the_same_force(slab):
    """precompute=False (DEC-071) builds the same mesh and materials: identical large-strain force."""
    lite = _model(precompute=False)
    assert lite.G_lat is None
    f = [NeoHookeanContact(mm).sweep(mm.dorsal_pose(0.0, 0.0), d_max_mm=0.5, step_mm=0.25)["curve"][-1]["force_n"]
         for mm in (slab, lite)]
    assert abs(f[0] - f[1]) <= 1e-9 * max(f[0], 1e-9), f


def test_vein_collapses_monotonically():
    m = _model(lumen_e_eff=vc.E_LUMEN_VOID, precompute=False)
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


def _cavity_model():
    nh = NeoHookeanContact(_model(lumen_e_eff=vc.E_LUMEN_VOID, precompute=False), cavity=True)
    return nh, nh.model.dorsal_pose(0.0, 0.0)


def test_cavity_wall_contact_gradient_and_hessian():
    """DEC-075: the wall-to-wall term's gradient and Hessian match finite differences (pairs made active by flipping
    their sign, which keeps the same energy form at small u)."""
    nh, T = _cavity_model()
    pr = nh.cavity_pairs(T)
    assert len(pr["node"]) > 0 and np.all(pr["g0"] > 0)
    pr["sign"] = -pr["sign"]
    nh._pairs = pr
    u = np.zeros(nh.n)
    u[nh.fidx] = np.random.default_rng(1).normal(0, 0.01, len(nh.fidx))
    d = float(nh.model.first_contact_indentation(T)) - 1.0          # probe off the skin: wall term only
    _, w, _ = nh._contact(u, T, d)
    g, K, _, _ = nh.grad_hess(u, T, d, w)
    _, nd, _, q = nh._cavity_terms(u)
    assert len(q) == len(pr["node"])
    for k in [3 * int(nd[0, 0]) + 1, 3 * int(nd[0, 1]) + 1, 3 * int(nd[1, 0]) + 1]:
        e = np.zeros(nh.n)
        e[k] = 1e-6
        fd = (nh.energy(u + e, T, d, w) - nh.energy(u - e, T, d, w)) / 2e-6
        assert abs(fd - g[k]) <= 1e-4 * max(1.0, abs(g[k])), (k, fd, g[k])
        e[k] = 1e-5
        col = (nh.grad_hess(u + e, T, d, w)[0] - nh.grad_hess(u - e, T, d, w)[0]) / 2e-5
        assert np.abs(col - K[:, k].toarray().ravel()).max() <= 1e-6 * max(1.0, np.abs(col).max())


def test_cavity_lumen_collapses_fully():
    """DEC-075: with the lumen as a cavity the phantom vein reaches full collapse (AP ratio <= 0.1), which the 0.1 kPa
    solid lumen cannot (DEC-073: stall at 0.36), with monotone force and ratio."""
    nh, T = _cavity_model()
    centre = np.array([0.0, vc.SKIN_Y - vc.DEPTH_MM, 0.0])
    rows = []

    def rec(dd, F, u):
        rows.append((F, nh.cavity_ratio(u.ravel(), centre)))
        return rows[-1][1] <= 0.1

    res = nh.sweep(T, d_max_mm=20.0, step_mm=0.5, f_stop_n=20.0, on_step=rec)
    print(f"[*] cavity force / AP ratio: {[(round(f, 2), round(r, 3)) for f, r in rows]}")
    F = np.array([r[0] for r in rows])
    R = np.array([r[1] for r in rows])
    assert res["converged"] and R[-1] <= 0.1
    assert np.all(np.diff(F) > 0) and np.all(np.diff(R) <= 1e-6)


def test_rounded_edge_contact_gradient_and_profile(slab):
    """DEC-077: with a rounded probe edge the contact gradient matches finite differences, and columns under the
    rounded edge are set back by r - sqrt(r^2 - q^2) (0 on the flat part, r at the footprint boundary)."""
    nh = NeoHookeanContact(slab, edge_radius_mm=3.0)
    T = slab.dorsal_pose(0.0, 0.0)
    w, h = nh._edge(T)
    assert set(np.unique(w)) <= {0.0, 1.0} and h.min() >= 0 and h[w > 0].max() <= 3.0 + 1e-9
    o, ex, _, ez = __import__("ReconstructionPipeline.simulation.fem_contact_nl", fromlist=["_frame"])._frame(T)
    r = nh.skin_ref - o
    flat = (np.abs(r @ ex) <= 0.5 * nh.L - 3.0) & (np.abs(r @ ez) <= 0.5 * nh.W - 3.0)
    assert np.all(h[flat] == 0) and np.all(w[flat] == 1)
    d = float(slab.first_contact_indentation(T)) + 2.0
    u = np.zeros(nh.n)
    u[nh.fidx] = np.random.default_rng(2).normal(0, 0.05, len(nh.fidx))
    p, w, _ = nh._contact(u, T, d)
    g, _, _, _ = nh.grad_hess(u, T, d, w)
    edge_cols = np.nonzero((p > 0) & (h > 0.2))[0]
    assert len(edge_cols) > 0
    for k in [3 * int(nh.skin[edge_cols[0]]) + 1, 3 * int(nh.skin[edge_cols[-1]]) + 1]:
        e = np.zeros(nh.n)
        e[k] = 1e-6
        fd = (nh.energy(u + e, T, d, w) - nh.energy(u - e, T, d, w)) / 2e-6
        assert abs(fd - g[k]) <= 1e-4 * max(1.0, abs(g[k])), (k, fd, g[k])


def test_projected_hessian_is_psd_and_fallback_keeps_the_answer(slab):
    """DEC-077: the projected element Hessians are PSD, and enabling the fallback leaves a converged sweep unchanged."""
    nh = NeoHookeanContact(slab, psd_fallback=True)
    T = slab.dorsal_pose(0.0, 0.0)
    d = float(slab.first_contact_indentation(T)) + 1.0
    u = np.zeros(nh.n)
    u[nh.fidx] = np.random.default_rng(3).normal(0, 0.2, len(nh.fidx))
    _, K, _, _ = nh.grad_hess(u, T, d, None, psd=True)
    Kf = K[nh.fidx][:, nh.fidx]
    v = np.random.default_rng(4).normal(size=(Kf.shape[0], 8))
    assert np.all(np.einsum("ij,ij->j", v, Kf @ v) >= -1e-9)
    f = [NeoHookeanContact(slab, psd_fallback=fb).sweep(T, d_max_mm=1.0, step_mm=0.5)["curve"][-1]["force_n"]
         for fb in (False, True)]
    assert abs(f[0] - f[1]) <= 1e-6 * f[0], f
