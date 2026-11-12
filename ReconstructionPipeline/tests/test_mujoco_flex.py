"""MuJoCo flex tissue worker (DEC-053). The worker runs in `.venv-mj4` (MuJoCo 4); these tests drive it through
that interpreter on a synthetic slab and are skipped when `.venv-mj4` is absent."""

import json
import os
import secrets
import subprocess
import sys
import textwrap
import time
from multiprocessing.connection import Listener

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJ4 = os.path.join(ROOT, ".venv-mj4", "bin", "python")
WORKER_DIR = os.path.join(ROOT, "surgical_physics_bridge", "src")
WORKER = os.path.join(WORKER_DIR, "mujoco_flex_worker.py")

pytestmark = pytest.mark.skipif(not os.path.exists(MJ4), reason=".venv-mj4 (MuJoCo 4) not installed")

KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
CORNERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def _slab(nx=9, ny=5, nz=9, h=4.0):
    """Kuhn-tet slab (mm), +Y up, bottom face pinned; skin = top face."""
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    nodes = np.stack([I, J, K], -1).reshape(-1, 3) * h
    idx = lambda i, j, k: (i * ny + j) * nz + k  # noqa: E731
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [idx(i + a, j + b, k + cc) for a, b, cc in CORNERS]
                tets += [[c[t] for t in kt] for kt in KUHN]
    tets = np.array(tets)
    fixed = nodes[:, 1] == 0
    top = np.nonzero(nodes[:, 1] == h * (ny - 1))[0]
    return nodes.astype(np.float64), tets, fixed, top


def _mj4(script: str, tmp_path, **arrays):
    """Run `script` in .venv-mj4 with the worker importable and `arrays` loaded as `z`; it prints one JSON line."""
    np.savez(tmp_path / "in.npz", **arrays)
    code = textwrap.dedent(f"""
        import sys, json, numpy as np, mujoco, warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, {WORKER_DIR!r})
        import mujoco_flex_worker as W
        z = np.load({str(tmp_path / 'in.npz')!r})
    """) + textwrap.dedent(script)
    out = subprocess.run([MJ4, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


STATIC = """
def static_u(young, material="svk", e_ref=None):
    m, d, info = W.build_model(z["nodes"], z["tets"], z["fixed"], young, 0.45, material, e_ref=e_ref)
    f = np.zeros((len(z["nodes"]), 3)); f[z["load"], 1] = -float(z["force"]) / len(z["load"])
    W.set_nodal_forces(d, info, f)
    ok, steps, _ = W.settle(m, d, info, z["load"], np.ones(len(z["load"])), np.array([0.0, -1.0, 0.0]))
    u = W.displacement_m(d, info) * 1e3
    return u, ok, W.inverted_tets(d, info)
"""


def _slab_arrays(force=0.2):
    nodes, tets, fixed, top = _slab()
    centre = nodes[top][:, [0, 2]].mean(0)
    load = top[np.linalg.norm(nodes[top][:, [0, 2]] - centre, axis=1) <= 6.0]
    return dict(nodes=nodes, tets=tets, fixed=fixed, load=load, force=force)


@pytest.mark.parametrize("material", ["svk", "snh"])
def test_per_tet_scaling_equals_changing_young(material, tmp_path):
    """G5 of DEC-053: a flex compiled at E and scaled x2 per tet behaves as one compiled at 2E; and a heterogeneous
    field (stiff lower half) gives a different answer than the uniform one (so the scaling is not a no-op)."""
    a = _slab_arrays()
    nt = len(a["tets"])
    res = _mj4(STATIC + f"""
nt = {nt}
E = 15000.0
u_compiled, ok1, _ = static_u(np.full(nt, 2 * E), {material!r})
u_scaled, ok2, _ = static_u(np.full(nt, 2 * E), {material!r}, e_ref=E)
cy = W.build_model(z["nodes"], z["tets"], z["fixed"], np.full(nt, E), 0.45)[2]["nodes_m"][z["tets"]].mean(1)[:, 1]
het = np.where(cy < np.median(cy), 4 * E, E)
u_het, ok3, _ = static_u(het, {material!r})
u_uni, ok4, _ = static_u(np.full(nt, E), {material!r})
L = z["load"]
print(json.dumps(dict(ok=bool(ok1 and ok2 and ok3 and ok4),
                      diff=float(np.abs(u_compiled - u_scaled).max()), umax=float(np.abs(u_compiled).max()),
                      het=float(-u_het[L, 1].mean()), uni=float(-u_uni[L, 1].mean()))))
""", tmp_path, **a)
    assert res["ok"]
    assert res["umax"] > 1e-3
    assert res["diff"] <= 1e-3 * res["umax"]
    assert res["het"] < 0.9 * res["uni"]


def test_both_materials_agree_at_small_load_and_settle_without_inversion(tmp_path):
    """SVK and Stable Neo-Hookean share the same linearisation: at a small load their compressions agree."""
    a = _slab_arrays(force=0.05)
    res = _mj4(STATIC + """
nt = len(z["tets"])
u_s, ok_s, inv_s = static_u(np.full(nt, 15000.0), "svk")
u_n, ok_n, inv_n = static_u(np.full(nt, 15000.0), "snh")
L = z["load"]
print(json.dumps(dict(ok=bool(ok_s and ok_n), inv=int(inv_s + inv_n), cs=float(-u_s[L, 1].mean()),
                      cn=float(-u_n[L, 1].mean()))))
""", tmp_path, **a)
    assert res["ok"] and res["inv"] == 0
    assert res["cs"] > 0.01
    assert res["cn"] == pytest.approx(res["cs"], rel=0.03)


def test_live_worker_reports_a_restoring_contact_force(tmp_path):
    """Live mode: a probe pressed 3 mm into the slab along the beam (-Y) is pushed back (+Y), the skin moves."""
    nodes, tets, fixed, top = _slab(nx=13, ny=6, nz=9)
    np.savez(tmp_path / "mesh.npz", nodes=nodes, tets=tets, fixed=fixed, young_pa=np.full(len(tets), 15000.0),
             nu=0.45, material="snh")
    key = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=key)
    proc = subprocess.Popen([MJ4, WORKER, "live", str(tmp_path / "mesh.npz"), "127.0.0.1",
                             str(listener.address[1]), key.hex()], stderr=subprocess.DEVNULL)
    try:
        conn = listener.accept()
        hello = conn.recv()
        assert hello["type"] == "ready" and hello["nodes"] == len(nodes)
        T = np.eye(4)
        T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # x lateral, beam -Y, z elevation
        T[:3, 3] = [nodes[:, 0].max() / 2, nodes[:, 1].max(), nodes[:, 2].max() / 2]
        conn.send({"type": "target", "pose": T.tobytes(), "indentation_mm": 3.0})
        last, end = None, time.time() + 20.0
        while time.time() < end:
            if conn.poll(0.1):
                msg = conn.recv()
                assert msg["type"] == "state", msg
                last = msg
                if last["sim_time_s"] > 1.5:
                    break
        conn.send({"type": "stop"})
    finally:
        proc.wait(timeout=10)
        listener.close()
    assert last is not None and last["sim_time_s"] > 1.5
    f = np.asarray(last["force_vec_n"])
    u = np.frombuffer(last["u_mm"], dtype=np.float32).reshape(-1, 3)
    assert f[1] > 0.1                                       # pushes the probe back out of the tissue
    assert abs(f[1]) > 3 * max(abs(f[0]), abs(f[2]))
    assert u[top, 1].min() < -1.0                           # the skin under the probe is pressed in
    assert last["probe_face_mm"][1] == pytest.approx(nodes[:, 1].max() - 3.0, abs=0.05)


# ------------------------------------------------------------------ .venv side (MujocoFlexTissueModel)
def test_lattice_field_puts_nodes_on_the_lattice_and_copies_skin_values_into_the_air_above():
    from types import SimpleNamespace
    sys.path.insert(0, ROOT)
    from ReconstructionPipeline.simulation.mujoco_flex_tissue import lattice_field
    # 2 x 4 x 1 lattice; column (0,0) has nodes j=0,1 (skin at j=1), column (1,0) has j=0 only (skin at j=0)
    node_ijk = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    mesh = SimpleNamespace(lattice_shape=(2, 4, 1), node_ijk=node_ijk, skin_nodes=np.array([1, 2]),
                           skin_column_ik=np.array([[0, 0], [1, 0]]))
    u = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=np.float32)
    g = lattice_field(mesh, u)                       # (3, NZ, NY, NX)
    assert g.shape == (3, 1, 4, 2)
    assert g[0, 0, :, 0].tolist() == [1.0, 2.0, 2.0, 2.0]   # node, skin, air, air
    assert g[0, 0, :, 1].tolist() == [3.0, 3.0, 3.0, 3.0]


def test_mujoco_flex_is_not_accepted_by_the_env_or_the_benchmark():
    """DEC-053: demo only. Only the server builds it; the env and the benchmark reject the name."""
    sys.path.insert(0, ROOT)
    from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv
    with pytest.raises(ValueError):
        UltrasoundReconstructionEnv(patient_data=None, tissue_backend="mujoco-flex")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "ReconstructionPipeline", "evaluation",
                                                       "benchmark_runner.py"), "--tissue-backend", "mujoco-flex"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode != 0 and "invalid choice" in out.stderr


def test_live_flex_model_presses_the_patient_skin(tmp_path):
    """Sample patient, 8 mm flex: a probe 4 mm past first contact pushes back, the tissue under it moves along the
    beam, and the reported indentation is the worker's actual probe depth."""
    import torch
    sys.path.insert(0, ROOT)
    from mtp_gym.core import load_patient
    from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
    from ReconstructionPipeline.simulation.mujoco_flex_tissue import MujocoFlexTissueModel
    data = load_patient(os.path.join(ROOT, "data", "sample_totalsegmentator_patient"))
    geom = FEMTissueModel.from_patient_data(data, c1_bounds=(-32.0, 32.0), c2_bounds=(-55.0, -10.0),
                                            device=torch.device("cpu"), verbose=False)
    flex = MujocoFlexTissueModel(data, geometry=geom, element_size_mm=8.0, verbose=False)
    try:
        T = geom.dorsal_pose(-20.0, -45.0)
        d = geom.first_contact_indentation(T) + 4.0
        st = flex.solve(T, d)
        s0 = flex.wait_for_state(0.0)
        flex.wait_for_state(s0["sim_time_s"] + 3.0, timeout_s=60.0)
        st = flex.solve(T, d)
    finally:
        flex.close()
    assert st.in_contact and st.reaction_force_n > 0.2
    assert st.indentation_mm == pytest.approx(d, abs=0.05)
    b = T[:3, 1]
    p = torch.tensor((T[:3, 3] + d * b)[None], dtype=torch.float32)
    moved = (st.displace(p) - p).numpy()[0]
    assert moved @ b > 0.5                                   # tissue under the face is pushed along the beam
