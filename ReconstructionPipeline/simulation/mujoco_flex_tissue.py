"""
Live MuJoCo flex tissue (DEC-053), DEMO ONLY unless the DEC-053 gate passes: it backs no metric, benchmark, gym
run or study (the env, benchmark and gym accept only "fem" and "kelvinlet"; the server builds this model itself).

The tissue runs as a dynamic MuJoCo 4 flex simulation in its own process (`.venv-mj4`,
`surgical_physics_bridge/src/mujoco_flex_worker.py live`): the patient tet mesh (same CT, ROI and fat/muscle moduli
as the FEM, coarser elements), and a probe body that MuJoCo contact presses into it. The process runs at its own
rate (close to real time on the 8 mm mesh) and is never waited on: a background thread here keeps the latest
published state, and `solve` / `force_at` answer from it. The model therefore lags the commanded pose by the
worker's rate limit and settling time (it is a dynamic tissue, not a quasi-static solve).

What it reports, per TissueState:
  * reaction force = contact force on the probe along -beam (MuJoCo contact, frame P);
  * indentation = the worker's actual probe-face depth along the beam below the undeformed skin point, so the
    force and indentation sent to the 1 kHz loop belong together;
  * contact stiffness = secant dF/dd over recent states (no tangent is available from a dynamic simulation);
    held at its last value while the depth changes by < 0.1 mm;
  * displacement field = the nodal displacements on the flex lattice (air above the skin takes the value of
    its column's skin node, bone and empty cells 0, as the FEM's `G_lat`), sampled with `grid_sample`.
Geometry queries (first contact, maximum indentation, thickness, skin patch) are the FEM model's, built from the
same CT, so the Slicer skin patch and the env's window are unchanged.
"""

from __future__ import annotations

import atexit
import os
import secrets
import subprocess
import tempfile
import threading
import time
from multiprocessing.connection import Listener
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel, FEMTissueState
from ReconstructionPipeline.simulation.tet_mesh import build_tet_mesh
from ReconstructionPipeline.simulation.tissue_model import tilt_cosine

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJ4_PYTHON = os.path.join(ROOT, ".venv-mj4", "bin", "python")
WORKER = os.path.join(ROOT, "surgical_physics_bridge", "src", "mujoco_flex_worker.py")
CONTACT_FORCE_MIN_N = 0.05
SECANT_MIN_STEP_MM = 0.1
STIFFNESS_MAX_N_PER_MM = 10.0
DEMO_ONLY_BANNER = ("[!] --tissue-backend mujoco-flex is DEMO ONLY (DEC-053): a live MuJoCo flex simulation; "
                    "it backs no metric, benchmark or study.")


def lattice_field(mesh, u_nodes_mm: np.ndarray) -> np.ndarray:
    """(3, NZ, NY, NX) displacement on the mesh lattice from nodal displacements (Nn, 3): nodes at their lattice
    index, air above the skin = the column's skin-node value, everything else 0 (the FEM's G_lat convention)."""
    NX, NY, NZ = mesh.lattice_shape
    u = np.zeros((NX, NY, NZ, 3), dtype=np.float32)
    ijk = mesh.node_ijk
    u[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = u_nodes_mm
    sk = mesh.skin_nodes
    ci, ck = mesh.skin_column_ik[:, 0], mesh.skin_column_ik[:, 1]
    js = ijk[sk, 1]
    for j in range(NY):
        above = js < j
        if above.any():
            u[ci[above], j, ck[above]] = u_nodes_mm[sk[above]]
    return np.ascontiguousarray(u.transpose(3, 2, 1, 0))


class MujocoFlexTissueModel:
    """TissueModel backed by the live flex worker (see module docstring). Call `close()` when done."""

    def __init__(self, patient_data: Dict[str, Any], geometry: FEMTissueModel, element_size_mm: float = 8.0,
                 material: str = "snh", device: Optional[torch.device] = None, connect_timeout_s: float = 120.0,
                 verbose: bool = True):
        if not os.path.exists(MJ4_PYTHON):
            raise RuntimeError(f"{MJ4_PYTHON} not found: create .venv-mj4 (AGENTS.md, invariant 1)")
        self.geom = geometry
        self.device = device or geometry.device
        self.material = material
        g = geometry
        self.mesh = build_tet_mesh(patient_data, g.roi_min, g.roi_max, h=element_size_mm)
        m = self.mesh
        young = np.where(m.cell_hu < g.fat_hu, g.e_fat, g.e_muscle)[m.tet_cell] * 1e3 * g.e_scale
        self._tmp = tempfile.mkdtemp(prefix="mj_flex_live_")
        mesh_path = os.path.join(self._tmp, "mesh.npz")
        np.savez(mesh_path, nodes=m.nodes, tets=m.tets, fixed=m.fixed, young_pa=young, nu=g.nu, material=material)
        NX, NY, NZ = m.lattice_shape
        self._org = torch.tensor(m.lattice_origin, dtype=torch.float32, device=self.device)
        self._ext = torch.tensor([(NX - 1) * m.h, (NY - 1) * m.h, (NZ - 1) * m.h], dtype=torch.float32,
                                 device=self.device)
        key = secrets.token_bytes(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=key)
        env = dict(os.environ)
        if env.get("MTP_FLEX_SNAPSHOT_DIR"):
            env.setdefault("MUJOCO_GL", "egl")     # offscreen renders in the worker only; the viewer keeps its GL
        self._proc = subprocess.Popen([MJ4_PYTHON, WORKER, "live", mesh_path, "127.0.0.1",
                                       str(self._listener.address[1]), key.hex()], env=env)
        accepted = {}
        t = threading.Thread(target=lambda: accepted.setdefault("conn", self._listener.accept()), daemon=True)
        t.start()
        t.join(connect_timeout_s)
        if "conn" not in accepted:
            self._proc.kill()
            raise RuntimeError("MuJoCo flex worker did not connect")
        self._conn = accepted["conn"]
        self.info = self._conn.recv()
        if self.info.get("type") != "ready":
            raise RuntimeError(f"MuJoCo flex worker failed: {self.info}")
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._sent_key: Optional[bytes] = None
        self._k = 0.0
        self._hist = []                 # (depth_mm, force_n) of recent states
        self._closed = False
        self._rx = threading.Thread(target=self._receive, daemon=True)
        self._rx.start()
        atexit.register(self.close)
        self.timings = {"mesh": m.summary()}
        if verbose:
            print(DEMO_ONLY_BANNER)
            print(f"[MujocoFlexTissue] {m.summary()}; material {material}; worker dt {self.info['dt']} s")

    # ------------------------------------------------------------------ worker I/O
    def _receive(self) -> None:
        while not self._closed:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                break
            if msg.get("type") == "state":
                with self._lock:
                    self._latest = msg
            elif msg.get("type") == "error":
                self._error = msg.get("message")
                print(f"[MujocoFlexTissue] worker error: {self._error}")
                break

    def _command(self, pose: np.ndarray, indentation_mm: float) -> None:
        key = np.asarray(pose, dtype=np.float64).tobytes() + np.float64(indentation_mm).tobytes()
        if key == self._sent_key or self._closed:
            return
        self._sent_key = key
        with self._send_lock:
            self._conn.send({"type": "target", "pose": np.asarray(pose, dtype=np.float64).tobytes(),
                             "indentation_mm": float(indentation_mm)})

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest

    def wait_for_state(self, min_sim_time_s: float = 0.0, timeout_s: float = 30.0) -> Optional[Dict[str, Any]]:
        """Block until the worker has published a state at or after ``min_sim_time_s`` (scripts and tests only)."""
        end = time.time() + timeout_s
        while time.time() < end:
            s = self.latest()
            if s is not None and s["sim_time_s"] >= min_sim_time_s:
                return s
            time.sleep(0.01)
        return self.latest()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            with self._send_lock:
                self._conn.send({"type": "stop"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        try:
            self._listener.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ TissueModel
    def _displacement_fn(self, u_nodes_mm: np.ndarray):
        u_grid = torch.from_numpy(lattice_field(self.mesh, u_nodes_mm)).to(self.device)[None]
        org, ext = self._org, self._ext

        def disp(X: torch.Tensor) -> torch.Tensor:
            X = torch.as_tensor(X).to(self.device, torch.float32)
            shp = X.shape
            g = (2.0 * (X.reshape(-1, 3) - org) / ext - 1.0).view(1, -1, 1, 1, 3)
            u = F.grid_sample(u_grid, g, mode="bilinear", padding_mode="border", align_corners=True)
            return u.view(3, -1).t().reshape(shp)

        return disp

    def solve(self, probe_pose_P: np.ndarray, indentation_mm: float) -> FEMTissueState:
        T = np.asarray(probe_pose_P, dtype=np.float64)
        self._command(T, indentation_mm)
        o, b = T[:3, 3].copy(), T[:3, 1] / np.linalg.norm(T[:3, 1])
        cos_t = tilt_cosine(b, self.geom.skin_surface.normal(o[0], o[2]))
        s = self.latest()
        empty = FEMTissueState(indentation_mm=float(indentation_mm), reaction_force_n=0.0, in_contact=False,
                               contact_point_mm=o, beam_dir=b.copy(), stiffness_n_per_mm=0.0,
                               bone_depth_mm=self.geom.tissue_thickness(T), cos_tilt=cos_t)
        if s is None or s.get("probe_face_mm") is None:
            return empty
        fvec = np.asarray(s["force_vec_n"], dtype=np.float64)
        force = float(max(0.0, -fvec @ b))
        depth = float((np.asarray(s["probe_face_mm"]) - o) @ b)
        self._update_stiffness(depth, force)
        if force < CONTACT_FORCE_MIN_N:
            empty.indentation_mm = depth
            return empty
        u = np.frombuffer(s["u_mm"], dtype=np.float32).reshape(-1, 3)
        return FEMTissueState(
            indentation_mm=depth, reaction_force_n=force, in_contact=True, contact_point_mm=o, beam_dir=b.copy(),
            stiffness_n_per_mm=self._k, bone_depth_mm=self.geom.tissue_thickness(T), cos_tilt=cos_t,
            _displacement_fn=self._displacement_fn(u), contact_stiffness_n_per_mm=self._k, force_vec_n=fvec,
            num_active_nodes=int(s.get("ncon", 0)), grid_origin_mm=self.mesh.lattice_origin.copy(),
            grid_spacing_mm=self.mesh.h)

    def _update_stiffness(self, depth: float, force: float) -> None:
        if force < CONTACT_FORCE_MIN_N:
            self._hist.clear()
            return
        self._hist = [h for h in self._hist[-20:]]
        for d0, f0 in reversed(self._hist):
            if abs(depth - d0) >= SECANT_MIN_STEP_MM:
                self._k = float(np.clip((force - f0) / (depth - d0), 0.0, STIFFNESS_MAX_N_PER_MM))
                break
        self._hist.append((depth, force))

    def force_at(self, probe_pose_P: np.ndarray, indentation_mm: float) -> float:
        """The live force (stale by the worker's lag; the env's admittance loop must run one iteration per step)."""
        return self.solve(probe_pose_P, indentation_mm).reaction_force_n

    # ------------------------------------------------------------------ geometry (FEM model of the same CT)
    def first_contact_indentation(self, probe_pose_P: np.ndarray, iters: int = 4) -> float:
        return self.geom.first_contact_indentation(probe_pose_P, iters)

    def max_indentation(self, probe_pose_P: np.ndarray, default_mm: float = 30.0) -> float:
        return self.geom.max_indentation(probe_pose_P, default_mm)

    def tissue_thickness(self, probe_pose_P: np.ndarray) -> float:
        return self.geom.tissue_thickness(probe_pose_P)

    def skin_surface_mesh(self, state=None):
        return self.geom.skin_surface_mesh(state)

    def dorsal_pose(self, *a, **k):
        return self.geom.dorsal_pose(*a, **k)

    @property
    def skin_surface(self):
        return self.geom.skin_surface

    @staticmethod
    def blas_scope():
        return FEMTissueModel.blas_scope()
