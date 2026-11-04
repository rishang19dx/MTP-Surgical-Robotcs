"""Falsifiable validation of the precomputed linear FEM tissue model (contract C1 backend, DEC-037).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_fem_tissue.py -q -s
(first run precomputes ~6 s on GPU and caches to ReconstructionPipeline/.fem_cache/).
"""
import os
import time

import numpy as np
import pytest
import torch

from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
from ReconstructionPipeline.simulation.tissue_model import TissueModel

SUBJECT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_totalsegmentator_patient")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (name, x, z) in frame P (mm); env L4 window x in [-32, 32], z in [-55, -10]
SITES = [("L4 centre (midline)", 0.0, -32.5), ("left paraspinal", -20.0, -45.0),
         ("right paraspinal", 20.0, -32.5), ("L3/L4 midline", 0.0, -15.0)]
FLAT = (-20.0, -45.0)   # skin is flat under a transverse probe here (first contact at d = 0)


@pytest.fixture(scope="module")
def patient():
    if not os.path.isdir(SUBJECT):
        pytest.skip("sample patient not available")
    return PatientAssetGenerator.load_totalsegmentator_subject(SUBJECT, target_vertebrae=["vertebrae_L4"],
                                                               crop_to_spine=False)


@pytest.fixture(scope="module")
def model(patient):
    base = 0.0
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 1e6   # other test modules may still hold tensors
    t = time.perf_counter()
    m = FEMTissueModel.from_patient_data(patient, device=DEV)
    m._test_vram_base_mb = base
    print(f"\n[FEM] build+calibrate {time.perf_counter() - t:.2f} s (cache={m.loaded_from_cache}); "
          f"timings {({k: round(v, 3) for k, v in m.timings.items()})}")
    print(f"[FEM] e_scale={m.e_scale:.3f} -> E fat/muscle = {m.effective_moduli_kpa[0]:.1f}/"
          f"{m.effective_moduli_kpa[1]:.1f} kPa, rms log err per calib pose "
          f"{np.round(m.calibration['rms_log_error_per_pose'], 2).tolist()}")
    return m


def _force_for(m, T, target, lo, hi=None):
    hi = lo + 25.0 if hi is None else hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if m.force_at(T, mid) < target else (lo, mid)
    return hi


def _tilted(m, x, z, roll_rad):
    """Probe rotated about its own elevation axis (lateral tilt of the array), origin on the skin."""
    T = m.dorsal_pose(x, z)
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T[:3, :3] = T[:3, :3] @ Rz
    return T


def test_protocol(model):
    assert isinstance(model, TissueModel)


def test_zero_indentation_zero_force_identity_warp(model):
    T = model.dorsal_pose(*FLAT)
    d0 = model.first_contact_indentation(T)
    assert abs(d0) < 0.5
    for d in (d0 - 1.0, d0 - 1e-3):
        st = model.solve(T, d)
        assert st.reaction_force_n == 0.0 and not st.in_contact
        X = torch.randn(1000, 3, device=DEV) * 20 + torch.tensor(T[:3, 3], dtype=torch.float32, device=DEV)
        assert torch.equal(st.warp_to_reference(X), X) and torch.equal(st.displace(X), X)
    assert model.force_at(T, d0) < 1e-6
    # lifted 2 mm above first contact anywhere -> no force
    for _, x, z in SITES:
        T = model.dorsal_pose(x, z)
        assert model.force_at(T, model.first_contact_indentation(T) - 2.0) == 0.0


def test_force_monotonic_and_table(model):
    ds = [0, 1, 2, 3, 4, 6, 8, 10]
    lit = [0.5 / 0.15 * np.expm1(0.15 * d) for d in ds]
    print("\n[FEM] force (N) vs indentation beyond first contact d (mm), untilted transverse probe 45x10 mm")
    print("      d:                    " + " ".join(f"{d:6d}" for d in ds))
    print("      literature (k0=.5,b=.15)" + " ".join(f"{f:6.2f}" for f in lit))
    f6 = []
    for name, x, z in SITES:
        T = model.dorsal_pose(x, z)
        d0 = model.first_contact_indentation(T)
        F = np.array([model.force_at(T, d0 + d) for d in ds])
        dd = np.linspace(d0, d0 + 12, 49)
        Fd = np.array([model.force_at(T, d) for d in dd])
        assert np.all(np.diff(Fd) > 0), name
        st = model.solve(T, d0 + 6)
        f6.append(F[ds.index(6)])
        print(f"      {name:22s} d0={d0:+5.1f} bone {st.bone_depth_mm:4.0f} mm " + " ".join(f"{f:6.2f}" for f in F)
              + f" | F at d_centre=0: {model.force_at(T, 0.0):5.2f} N")
    # anatomy must matter: >= 1.5x spread in force at 6 mm across sites
    assert max(f6) / min(f6) > 1.5


def test_contact_stiffness_matches_finite_difference(model):
    for _, x, z in SITES:
        T = model.dorsal_pose(x, z)
        d = _force_for(model, T, 4.0, model.first_contact_indentation(T))
        K = model.solve(T, d).contact_stiffness_n_per_mm
        fd = (model.force_at(T, d + 0.05) - model.force_at(T, d - 0.05)) / 0.1
        assert K > 0 and abs(K - fd) / fd < 0.2, (K, fd)


def test_bone_does_not_move(model, patient):
    ct, lab = patient["ct_volume"], patient["label_volume"]
    o, sp = np.array(patient["origin_mm"]), np.array(patient["spacing_mm"])
    zyx = np.argwhere(((lab == 1) | (ct > 250))[::2, ::2, ::2]) * 2
    P = o + zyx[:, ::-1] * sp
    for _, x, z in SITES:
        T = model.dorsal_pose(x, z)
        st = model.solve(T, _force_for(model, T, 8.0, model.first_contact_indentation(T)))
        near = np.linalg.norm((P - T[:3, 3])[:, [0, 2]], axis=1) < 40
        X = torch.tensor(P[near], dtype=torch.float32, device=DEV)
        ub = (st.displace(X) - X).norm(dim=1)
        ref = st.u_grid.norm(dim=1).max()
        print(f"\n[FEM] bone |u| at 8 N ({x},{z}): median {ub.median():.4f} p99 {ub.quantile(0.99):.3f} "
              f"max {ub.max():.3f} mm; max tissue |u| {ref:.2f} mm")
        assert float(ub.quantile(0.99)) < 0.05 * float(ref)
        assert float(ub.max()) < 0.15 * float(ref)


def test_displacement_decays_away_from_probe(model):
    T = model.dorsal_pose(*FLAT)
    st = model.solve(T, _force_for(model, T, 5.0, 0.0))
    V, _ = model.skin_surface_mesh()
    Vd, _ = model.skin_surface_mesh(st)
    u = np.linalg.norm(Vd - V, axis=1)
    r = np.linalg.norm((V - T[:3, 3])[:, [0, 2]], axis=1)
    bins = [(0, 10), (25, 35), (45, 60)]
    means = [u[(r >= a) & (r < b)].mean() for a, b in bins]
    print(f"\n[FEM] skin |u| by distance from probe centre {bins}: {np.round(means, 3).tolist()} mm")
    assert means[0] > means[1] > means[2] and means[2] < 0.25 * means[0]
    # along the beam: displacement decreases with depth under the probe centre
    ys = torch.tensor([[T[0, 3], T[1, 3] - s, T[2, 3]] for s in (2, 8, 16, 24)], dtype=torch.float32, device=DEV)
    uy = (st.displace(ys) - ys).norm(dim=1).cpu().numpy()
    assert np.all(np.diff(uy) < 0), uy


def test_warp_round_trip(model, patient):
    ct = patient["ct_volume"]
    o, sp = np.array(patient["origin_mm"]), np.array(patient["spacing_mm"])
    for _, x, z in SITES:
        T = model.dorsal_pose(x, z)
        st = model.solve(T, _force_for(model, T, 8.0, model.first_contact_indentation(T)))
        rng = np.random.default_rng(0)
        X = T[:3, 3] + rng.uniform([-40, -45, -40], [40, -1, 40], size=(20000, 3))
        idx = np.rint((X - o) / sp).astype(int)[:, ::-1]
        idx = np.clip(idx, 0, np.array(ct.shape) - 1)
        hu = ct[idx[:, 0], idx[:, 1], idx[:, 2]]
        lab = patient["label_volume"][idx[:, 0], idx[:, 1], idx[:, 2]]
        soft = (hu > -200) & (hu < 250) & (lab != 1) & (X[:, 1] < model.skin_height(X[:, 0], X[:, 2]) - 0.5)
        Xt = torch.tensor(X[soft], dtype=torch.float32, device=DEV)
        err = (st.warp_to_reference(st.displace(Xt)) - Xt).norm(dim=1)
        print(f"\n[FEM] round-trip ({x},{z}) 8 N, {st.warp_iterations} it: mean {err.mean():.5f} "
              f"max {err.max():.4f} mm (n={len(err)}, max|u|={st.u_grid.norm(dim=1).max():.2f})")
        assert float(err.max()) < 0.2


def test_tilt_breaks_symmetry(model):
    x, z = FLAT
    out = []
    for roll in (0.0, np.radians(8.0)):
        T = _tilted(model, x, z, roll)
        st = model.solve(T, _force_for(model, T, 5.0, model.first_contact_indentation(T)))
        ez = T[:3, 2]
        lever = abs(float(st.torque_nmm @ ez)) / st.reaction_force_n
        out.append((lever, st.num_active_nodes))
    print(f"\n[FEM] lateral tilt 0 vs 8 deg at 5 N: lever arm {out[0][0]:.1f} vs {out[1][0]:.1f} mm, "
          f"active nodes {out[0][1]} vs {out[1][1]}")
    assert out[1][0] > out[0][0] + 3.0
    assert out[1][1] < out[0][1]


def test_latency_and_memory(model):
    T = model.dorsal_pose(0.0, -32.5)
    d = _force_for(model, T, 4.0, model.first_contact_indentation(T))
    for i in range(10):
        model.solve(T, d + 1e-3 * i)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(100):
        st = model.solve(T, d + 1e-3 * (i + 11))
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t_solve = (time.perf_counter() - t) * 10
    t = time.perf_counter()
    for i in range(300):
        model.force_at(T, d + 1e-4 * i)
    t_force = (time.perf_counter() - t) / 300 * 1e3
    pts = torch.rand(200 * 150, 3, device=DEV) * 40 + torch.tensor(T[:3, 3] - [20, 40, 20], dtype=torch.float32,
                                                                  device=DEV)
    st.warp_to_reference(pts)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        st.warp_to_reference(pts)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t_warp = (time.perf_counter() - t) * 20
    vram = (torch.cuda.max_memory_allocated() / 1e6 - model._test_vram_base_mb) if DEV.type == "cuda" else 0.0
    print(f"\n[FEM] solve {t_solve:.2f} ms, force_at {t_force:.2f} ms, warp 200x150 ({st.warp_iterations} it) "
          f"{t_warp:.2f} ms, peak VRAM {vram:.0f} MB")
    assert t_solve < 10.0 and t_force < 5.0
    if DEV.type == "cuda":
        assert vram < 300.0


def test_cache_reload_identical(model, patient):
    if not model.cache_path:
        pytest.skip("no cache")
    m2 = FEMTissueModel(patient, e_scale=model.e_scale, beta_per_mm=model.beta, device=DEV, verbose=False)
    assert m2.loaded_from_cache
    T = model.dorsal_pose(*FLAT)
    a, b = model.force_at(T, 4.0), m2.force_at(T, 4.0)
    assert abs(a - b) < 1e-9 * max(1.0, a)


def test_max_indentation(model):
    for _, x, z in SITES:
        T = model.dorsal_pose(x, z)
        dmax = model.max_indentation(T)
        assert np.isfinite(dmax) and dmax > model.first_contact_indentation(T)
        assert model.force_at(T, dmax) > 0.0


def test_compare_with_kelvinlet(model, patient):
    from ReconstructionPipeline.simulation.tissue_model import KelvinletTissueModel
    try:
        kel = KelvinletTissueModel.from_patient_data(patient, device=DEV)
    except Exception as e:  # ENV agent's backend may be mid-edit
        pytest.skip(f"Kelvinlet backend unavailable: {e}")
    print("\n[FEM vs Kelvinlet] force (N) at indentation d from the skin under the probe centre (contract semantics)")
    for name, x, z in SITES:
        T = model.dorsal_pose(x, z)
        Tk = T.copy()
        Tk[1, 3] = kel.skin.height(x, z) if kel.skin is not None else T[1, 3]
        row = []
        for d in (2.0, 4.0, 6.0):
            sf, sk = model.solve(T, d), kel.solve(Tk, d)
            p = torch.tensor([[x, T[1, 3] - 10.0, z]], dtype=torch.float32, device=DEV)
            uf = float((sf.displace(p) - p).norm()) if sf.in_contact else 0.0
            uk = float((sk.displace(p) - p).norm()) if sk.in_contact else 0.0
            row.append(f"d={d:.0f}: F {sf.reaction_force_n:5.2f}/{sk.reaction_force_n:5.2f}  |u|@10mm {uf:4.2f}/{uk:4.2f}")
            assert np.isfinite(sf.reaction_force_n) and np.isfinite(sk.reaction_force_n)
        print(f"   {name:22s} " + " | ".join(row))
        d0 = model.first_contact_indentation(T)
        print(f"   {'':22s} FEM from first contact (d0={d0:+.1f}): " + " ".join(
            f"F(d0+{d:.0f})={model.force_at(T, d0 + d):5.2f}" for d in (2.0, 4.0, 6.0)) +
            f" | Kelvinlet max_indentation {kel.max_indentation(Tk):.1f} mm, FEM {model.max_indentation(T):.1f} mm")
