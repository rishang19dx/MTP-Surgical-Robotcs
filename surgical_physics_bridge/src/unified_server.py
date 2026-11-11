"""
Unified Surgical Physics & Autonomous 3D Ultrasound Reconstruction Server.

Runs two decoupled loops (AGENTS invariant 2):

* **Physics loop (1 kHz)** -- ``physics_common.run_physics_loop`` with the backend chosen by
  ``--physics-backend`` (PyBullet by default, or MuJoCo). It owns *every* engine call
  (IK, motor commands, stepping, contact queries, virtual fixtures). By default
  it runs in its own process (``--physics-mode process``) so the Python-heavy ultrasound /
  OpenIGTLink work cannot steal its GIL. ``thread`` mode runs it in a thread of this process
  (kept for comparison), ``inline`` is used by ``step_simulation_and_stream`` in tests.
* **Streaming loop (``--fps``, default 30 Hz)** -- steps ``UltrasoundReconstructionEnv`` (tissue
  model, B-mode simulation, reconstruction), sends the resulting end-effector target to the
  physics loop, and publishes to 3D Slicer. It never touches an engine; it only reads the physics
  loop's contact snapshot and rate statistics through ``PhysicsChannels``.

OpenIGTLink messages published (all patient-frame data is in frame P, mm; the Slicer side applies
the documented P->RAS matrix, see ``coordinates.P_TO_RAS``):

    UltrasoundProbe    TransformMessage  4x4 T_P<-probe of the (indented) probe face / image plane, +Y = beam
    US_Image           ImageMessage      uint8 (H, W, 1) B-mode frame
    ReconstructedBone  ImageMessage      float32 (1, N, 3) accumulated bone points in P (array transport;
                                         pyigtl 0.3.4 has no NDARRAY and its PointMessage packer is O(N^2))
    DeformedSkin       ImageMessage      float32 (1, N, 3) all N deformed skin-patch vertices (mm, P), in the
                                         vertex order of ``meshes/patient_fem_skin.obj``, which this server
                                         writes at startup from the tissue model's skin surface
                                         (``FEMTissueModel.skin_surface_mesh``). Sent when the deformation
                                         changes (and once undeformed after contact is lost).
    Telemetry          StringMessage     JSON: force, indentation, coupling, physics-rate stats, rigid contacts

Incoming (StringMessage):
    CMD             JSON {"reset": true} | {"mode": "<scan mode>"} | {"roi_target": [R, A, S]}
    VirtualFixture  JSON {"vertices": [[R,A,S],...], "indices": [...], "policy": "keep_out"}

Physics coupling (DEC-037): every streaming step sends the env's latest tissue solution (FEM force F0,
contact stiffness K_c, indentation delta0, undeformed contact point and beam) with the end-effector
target; the 1 kHz loop applies F = F0 + K_c (delta - delta0) to the probe without waiting on the FEM.
With ``--gui`` a separate viewer process renders the scene and the deformed skin patch.

Usage:
    ./.venv/bin/python surgical_physics_bridge/src/unified_server.py --mode orbital_l4 --fps 30 [--gui]
    ./.venv/bin/python surgical_physics_bridge/src/unified_server.py --benchmark-seconds 20
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import pyigtl
import pyigtl.comm
import torch


class RobustTCPRequestHandler(pyigtl.comm.TCPRequestHandler):
    """pyigtl request handler that tolerates momentary send-buffer saturation by a busy Slicer UI."""

    def handle(self):
        self.server._connected = True
        while not self.server.communication_thread_stop_requested:
            self.request.settimeout(0.005)
            try:
                while self.server._receive_message_from_socket(self.request):
                    if self.server.communication_thread_stop_requested:
                        break
            except Exception:
                pass
            self.request.settimeout(2.0)
            try:
                while self.server._send_queued_message_from_socket(self.request):
                    pass
            except (socket.timeout, TimeoutError):
                continue
            except Exception:
                self.server._communication_error_occurred()
                break


pyigtl.comm.TCPRequestHandler = RobustTCPRequestHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surgical_physics_bridge.src.coordinates import (  # noqa: E402
    P_TO_PB_ROT,
    patient_to_pybullet_points,
    probe_axes_to_pybullet_quaternion,
    ras_to_patient_points,
)
from surgical_physics_bridge.src.obj_io import write_obj  # noqa: E402
from surgical_physics_bridge.src.physics_rate import PhysicsRateMonitor  # noqa: E402
from surgical_physics_bridge.src.physics_common import (  # noqa: E402
    BACKENDS,
    PROBE_LENGTH_M,
    PhysicsBackendBase,
    PhysicsChannels,
    create_backend,
    dedicated_physics_cpu,
    exclude_dedicated_cpu,
    physics_process_main,
    run_physics_loop,
    viewer_process_main,
)
from ReconstructionPipeline.planning.env_reconstruction import UltrasoundReconstructionEnv  # noqa: E402
from ReconstructionPipeline.planning.heuristic_planner import HeuristicScanPlanner  # noqa: E402
from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator  # noqa: E402

BENCH_CLIENT_SRC = (
    "import sys, time, pyigtl\n"
    "c = pyigtl.OpenIGTLinkClient(host='127.0.0.1', port=int(sys.argv[1]))\n"
    "while True:\n"
    "    c.get_latest_messages(); time.sleep(0.01)\n"
)
SCAN_MODES = ["orbital_l4", "raster_l4", "orbital", "raster", "nbv", "diffusion", "cfm", "interactive"]
FEM_SKIN_OBJ = "patient_fem_skin.obj"
SKIN_CHANGE_MIN_MM = 0.01
MAX_OUTGOING_QUEUE = 64


class UnifiedSurgicalBridgeServer:
    def __init__(
        self,
        port: int = 18944,
        urdf_path: Optional[str] = None,
        patient_dir: Optional[str] = None,
        gui: bool = False,
        publish_fps: float = 30.0,
        scan_mode: str = "orbital_l4",
        device: Optional[torch.device] = None,
        physics_mode: str = "process",
        physics_hz: float = 1000.0,
        rate_log_interval_s: float = 5.0,
        rate_window: int = 60000,
        tissue_backend: str = "fem",
        physics_backend: str = "pybullet",
    ):
        if physics_mode not in ("process", "thread", "inline"):
            raise ValueError(f"physics_mode must be process|thread|inline, got {physics_mode}")
        if physics_backend not in BACKENDS:
            raise ValueError(f"physics_backend must be one of {BACKENDS}, got {physics_backend}")
        self.physics_backend = physics_backend
        self.port = port
        self.gui = gui
        self.publish_fps = publish_fps
        self.publish_interval = 1.0 / publish_fps
        self.scan_mode = scan_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.physics_mode = physics_mode
        self.physics_hz = physics_hz
        self.dt_physics = 1.0 / physics_hz
        self.rate_log_interval_s = rate_log_interval_s

        print(f"[*] Initializing OpenIGTLink Server on port {self.port}...")
        self.server = pyigtl.OpenIGTLinkServer(port=self.port)

        self.patient_dir = self._resolve_patient_data(patient_dir)
        self._ensure_meshes()

        # Physics: never constructed in this process unless thread/inline mode.
        self.channels = PhysicsChannels(ctx=mp.get_context("spawn") if physics_mode == "process" else None,
                                        rate_window=rate_window)
        self._cached_stats = self.channels.rate_reader().snapshot()
        self._last_stats_refresh = 0.0
        self.backend: Optional[PhysicsBackendBase] = None
        self.physics_proc = None
        self.physics_thread = None
        self.viewer_proc = None
        # The physics instance is always headless; the GUI is a separate viewer process (process mode only).
        self._physics_config = dict(backend=physics_backend, gui=False, patient_dir=self.patient_dir,
                                    urdf_path=urdf_path, dt=self.dt_physics)
        self._start_physics()
        if gui:
            self._start_viewer()

        print(f"[*] Initializing Ultrasound Reconstruction Environment (dorsal, {tissue_backend} tissue)...")
        self.env = UltrasoundReconstructionEnv(
            patient_data=self.patient_dir,
            target_vertebra="vertebrae_L4",
            max_steps=10000,
            device=self.device,
            tissue_backend=tissue_backend,
        )
        self.skin_vertices, self.skin_tris = self._skin_patch()
        self.skin_obj_path = self._export_skin_obj()
        self._last_skin_sent: Optional[np.ndarray] = None
        if self.viewer_proc is not None:
            self.channels.viewer.put(("skin_mesh", {"vertices_pb": patient_to_pybullet_points(self.skin_vertices),
                                                    "indices": self.skin_tris.ravel().tolist()}))
        self.obs = self.env.reset()

        self.planner = HeuristicScanPlanner()
        self.trajectory = self._build_trajectory(self.scan_mode)
        self.traj_idx = 0
        self.candidate_deltas = [
            (5.0, 0.0, 0.0, 0.0), (-5.0, 0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0), (0.0, -5.0, 0.0, 0.0),
            (3.5, 3.5, np.radians(6.0), 0.0), (-3.5, 3.5, -np.radians(6.0), 0.0),
            (3.5, -3.5, np.radians(6.0), 0.0), (-3.5, -3.5, -np.radians(6.0), 0.0),
        ]
        self.visited_positions = []
        self.action_queue = []
        self.diffusion_policy = None
        self.cfm_policy = None
        self.pointnet = None
        self._load_policies()

        self.is_running = True
        self.last_publish_time = 0.0
        self.total_frames_published = 0
        self.last_points_sent = 0
        self.last_points_publish_time = 0.0
        self.last_rate_log = time.perf_counter()
        self.last_target: Optional[Dict[str, Any]] = None
        self._send_target(teleport=True)

    # ----------------------------------------------------------------- setup
    def _start_physics(self) -> None:
        if self.physics_mode == "process":
            self._dedicated_cpu = dedicated_physics_cpu()
            ctx = mp.get_context("spawn")
            self.physics_proc = ctx.Process(
                target=physics_process_main,
                args=(self._physics_config, self.channels, self.physics_hz),
                name="PhysicsProcess",
                daemon=True,
            )
            self.physics_proc.start()
            if not self.channels.ready.wait(timeout=60.0) or not self.physics_proc.is_alive():
                raise RuntimeError("Physics process failed to start")
            # The child pinned itself to the dedicated CPU; keep torch/open3d worker threads of this process off it.
            exclude_dedicated_cpu(self._dedicated_cpu)
            print(f"[+] Physics process started (pid {self.physics_proc.pid}, {self.physics_hz:.0f} Hz target)")
        else:
            cfg = dict(self._physics_config)
            self.backend = create_backend(cfg.pop("backend"), **cfg)
            if self.physics_mode == "thread":
                # Default GIL switch interval is 5 ms, which alone caps a GIL-sharing 1 kHz loop near 200 Hz.
                sys.setswitchinterval(2e-4)
                self.physics_thread = threading.Thread(
                    target=run_physics_loop, args=(self.backend, self.channels, self.physics_hz),
                    daemon=True, name="PhysicsThread")

    def _start_viewer(self) -> None:
        if self.physics_mode != "process":
            print("[!] --gui needs --physics-mode process (separate viewer process); running headless.")
            return
        ctx = mp.get_context("spawn")
        cfg = dict(self._physics_config, gui=True)
        self.viewer_proc = ctx.Process(target=viewer_process_main, args=(cfg, self.channels),
                                       name="ViewerProcess", daemon=True)
        self.viewer_proc.start()
        print(f"[+] {self.physics_backend} GUI viewer process started (pid {self.viewer_proc.pid})")

    def _resolve_patient_data(self, patient_dir: Optional[str]) -> str:
        if patient_dir and os.path.isdir(patient_dir):
            return patient_dir
        default = os.path.join(PROJECT_ROOT, "data", "sample_totalsegmentator_patient")
        if not os.path.isdir(default):
            print("[*] Sample patient dataset not found. Exporting it...")
            PatientAssetGenerator.export_totalsegmentator_dataset(default)
        return default

    def _ensure_meshes(self) -> None:
        mesh_dir = os.path.join(self.patient_dir, "meshes")
        needed = ["patient_lumbar_spine.obj", "patient_torso_skin.obj", "patient_torso_skin_ras.obj"]
        if not all(os.path.exists(os.path.join(mesh_dir, n)) for n in needed):
            PatientAssetGenerator.export_patient_meshes(self.patient_dir)

    def _skin_patch(self):
        """Undeformed skin patch (vertices (S,3) P mm, triangles (T,3)) whose deformation is streamed:
        the FEM skin surface (`FEMTissueModel.skin_surface_mesh`, deformed = state.displace(vertices))."""
        model = self.env.tissue_model
        if hasattr(model, "skin_surface_mesh"):
            v, f = model.skin_surface_mesh(None)
        else:  # Kelvinlet backend: 4 mm grid on the CT skin height map over the scan window +- 30 mm
            xs = np.arange(self.env.c1_bounds[0] - 30.0, self.env.c1_bounds[1] + 30.01, 4.0)
            zs = np.arange(self.env.c2_bounds[0] - 30.0, self.env.c2_bounds[1] + 30.01, 4.0)
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            v = np.stack([X.ravel(), self.env.skin.height(X.ravel(), Z.ravel()), Z.ravel()], 1)
            nx, nz = len(xs), len(zs)
            a = (np.arange(nx - 1)[:, None] * nz + np.arange(nz - 1)[None, :]).ravel()
            f = np.concatenate([np.stack([a, a + nz, a + 1], 1), np.stack([a + 1, a + nz, a + nz + 1], 1)])
        return np.ascontiguousarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64)

    def _export_skin_obj(self) -> str:
        """Write the undeformed skin patch (frame P mm, streamed vertex order) for the Slicer bridge."""
        path = os.path.join(self.patient_dir, "meshes", FEM_SKIN_OBJ)
        write_obj(path, self.skin_vertices, self.skin_tris)
        return path

    def _load_policies(self) -> None:
        if self.scan_mode == "diffusion" and os.path.exists("checkpoints/diffusion_policy.pt"):
            try:
                from ReconstructionPipeline.planning.diffusion_policy import SOTADiffusionPolicy
                self.diffusion_policy = SOTADiffusionPolicy(action_dim=4, horizon=8, device=self.device)
                ckpt = torch.load("checkpoints/diffusion_policy.pt", map_location=self.device, weights_only=False)
                self.diffusion_policy.load_state_dict(ckpt.get("state_dict", ckpt))
                self.diffusion_policy.eval()
            except Exception as e:
                print(f"[!] Failed to load diffusion policy: {e}")
        elif self.scan_mode == "cfm" and os.path.exists("checkpoints/flow_matching.pt"):
            try:
                from ReconstructionPipeline.planning.flow_matching_policy import FlowMatchingPolicy
                from ReconstructionPipeline.planning.pointnet_encoder import PointNetEncoder
                self.pointnet = PointNetEncoder(latent_dim=128).to(self.device)
                self.cfm_policy = FlowMatchingPolicy(action_dim=4, chunk_horizon=8, condition_dim=132,
                                                     device=self.device).to(self.device)
                ckpt = torch.load("checkpoints/flow_matching.pt", map_location=self.device, weights_only=False)
                if "pointnet_state_dict" in ckpt:
                    self.pointnet.load_state_dict(ckpt["pointnet_state_dict"])
                if "policy_state_dict" in ckpt:
                    self.cfm_policy.load_state_dict(ckpt["policy_state_dict"])
                self.pointnet.eval()
                self.cfm_policy.eval()
            except Exception as e:
                print(f"[!] Failed to load CFM policy: {e}")

    def _build_trajectory(self, mode: str) -> Optional[np.ndarray]:
        if mode == "raster_l4":
            return self.planner.generate_spine_raster_trajectory(
                x_range=(-20.0, 20.0), z_range=(-52.0, -10.0), step_distance_mm=0.33, num_sweep_lines=6)
        if mode == "orbital_l4":
            return self.planner.generate_orbital_trajectory(center=(0.0, -31.0), radii=[16.0, 26.0], step_distance_mm=0.33)
        if mode == "raster":
            return self.planner.generate_spine_raster_trajectory(
                x_range=(-18.0, 18.0), z_range=(-55.0, 55.0), step_distance_mm=0.33, num_sweep_lines=5)
        if mode == "orbital":
            return self.planner.generate_orbital_trajectory(center=(0.0, -31.0), radii=[16.0, 26.0, 34.0],
                                                            step_distance_mm=0.33)
        return None

    # ------------------------------------------------------------ networking
    def _safe_send(self, msg) -> None:
        """Queue a message only while a client is connected; drop the oldest frames if Slicer lags."""
        srv = self.server
        if not srv.is_connected():
            return
        with srv.lock_outgoing_messages:
            while len(srv.outgoing_messages) >= MAX_OUTGOING_QUEUE:
                srv.outgoing_messages.popleft()
        try:
            srv.send_message(msg, wait=False)
        except Exception as e:
            print(f"[!] OpenIGTLink send failed: {e}")

    def handle_incoming_messages(self) -> None:
        for message in self.server.get_latest_messages():
            if not isinstance(message, pyigtl.StringMessage):
                continue
            try:
                data = json.loads(message.string)
            except json.JSONDecodeError as err:
                print(f"[!] Malformed {message.device_name} JSON: {err}")
                continue
            if message.device_name == "CMD":
                if data.get("reset"):
                    self.obs = self.env.reset()
                    self.traj_idx = 0
                    self._send_target(teleport=True)
                if "mode" in data:
                    self.scan_mode = str(data["mode"]).lower()
                    self.trajectory = self._build_trajectory(self.scan_mode)
                    self.traj_idx = 0
                if "roi_target" in data:
                    target_p = ras_to_patient_points(np.asarray(data["roi_target"], dtype=np.float64))
                    self.env.current_pose_params[0] = np.clip(target_p[0], *self.env.c1_bounds)
                    self.env.current_pose_params[1] = np.clip(target_p[2], *self.env.c2_bounds)
                    self.obs = self.env._acquire(self.env.current_pose_params)[0]
                    self._send_target()
            elif message.device_name == "VirtualFixture":
                if "vertices" in data and "indices" in data:
                    self.channels.commands.put(("fixture", data))

    # -------------------------------------------------------- env -> physics
    def _tissue_state(self):
        return self.env.tissue_state

    def _indentation_mm(self) -> float:
        ts = self._tissue_state()
        if ts is not None and hasattr(ts, "indentation_mm"):
            return float(ts.indentation_mm)
        return float(getattr(self.env, "current_indentation_mm", 0.0))

    def _skin_pose(self) -> np.ndarray:
        """T_P<-probe with origin on the UNDEFORMED skin (``env.skin_pose``, contract C1)."""
        return np.asarray(self.env.skin_pose(*self.env.current_pose_params), dtype=np.float64)

    def _send_target(self, teleport: bool = False) -> None:
        """Probe pose from the env (frame P) -> end-effector target in PyBullet, sent to the physics loop."""
        t_p = self._skin_pose()
        beam_p, lateral_p = t_p[:3, 1], t_p[:3, 0]
        indent = self._indentation_mm()
        face_p = t_p[:3, 3] + indent * beam_p
        contact_pb = patient_to_pybullet_points(face_p)
        beam_pb = P_TO_PB_ROT @ beam_p
        beam_pb /= np.linalg.norm(beam_pb)
        ee_pos = contact_pb - beam_pb * PROBE_LENGTH_M
        fn = float(self.env.current_normal_force)
        coupling = float(self.env.current_coupling)
        stats = self._cached_stats
        hud = (
            f"ROBOTIC ULTRASOUND | mode {self.scan_mode}\n"
            f"Normal force {fn:.2f} N | indentation {indent:.2f} mm | coupling {coupling * 100:.0f}%\n"
            f"Reconstructed {len(self.env.accumulator.get_points())} pts | "
            f"physics {stats['achieved_hz']:.0f} Hz (p99 {stats['period_p99_ms']:.2f} ms)"
        )
        ts = self._tissue_state()
        in_contact = bool(ts is not None and ts.in_contact)
        tissue = {
            "contact_pb": patient_to_pybullet_points(t_p[:3, 3]).tolist(),
            "beam_pb": beam_pb.tolist(),
            "force_n": float(ts.reaction_force_n) if in_contact else 0.0,
            "stiffness_n_per_mm": float(getattr(ts, "contact_stiffness_n_per_mm", ts.stiffness_n_per_mm))
            if in_contact else 0.0,
            "indentation_mm": indent,
            "in_contact": in_contact,
        }
        target = {
            "tissue": tissue,
            "ee_pos": ee_pos.tolist(),
            "ee_orn": list(probe_axes_to_pybullet_quaternion(beam_p, lateral_p)),
            "contact_pb": contact_pb.tolist(),
            "beam_pb": beam_pb.tolist(),
            "lateral_pb": (P_TO_PB_ROT @ lateral_p).tolist(),
            "teleport": teleport,
            "force_n": fn,
            "hud": hud,
        }
        self.last_target = target
        self.channels.commands.put(("target", target))
        if self.viewer_proc is not None:
            self.channels.viewer.put(("target", target))

    # --------------------------------------------------------------- scanning
    def _policy_action(self) -> Optional[np.ndarray]:
        env = self.env
        curr = env.current_pose_params
        if self.trajectory is not None:
            target_p = self.trajectory[self.traj_idx % len(self.trajectory)]
            self.traj_idx += 1
            return np.array([
                np.clip((target_p[0] - curr[0]) / env.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[1] - curr[1]) / env.action_scale_trans, -1.0, 1.0),
                np.clip((target_p[2] - curr[2]) / env.action_scale_yaw, -1.0, 1.0),
                np.clip((target_p[3] - curr[3]) / env.action_scale_pitch, -1.0, 1.0),
            ], dtype=np.float32)
        if self.scan_mode == "nbv":
            self.visited_positions.append((float(curr[0]), float(curr[1])))
            coupling_fn = getattr(env, "compute_contact_force_and_coupling", None)
            best, best_score = None, -float("inf")
            for dx, dz, dyaw, dpitch in self.candidate_deltas:
                cx = np.clip(curr[0] + dx, *env.c1_bounds)
                cz = np.clip(curr[1] + dz, *env.c2_bounds)
                penalty = sum(np.exp(-0.5 * ((cx - vx) ** 2 + (cz - vz) ** 2) / 25.0)
                              for vx, vz in self.visited_positions[-40:])
                kappa = coupling_fn(cx, cz, curr[2] + dyaw, curr[3] + dpitch)[1] if coupling_fn else 1.0
                score = 5.0 * kappa - 1.2 * penalty - 0.01 * np.hypot(cx, cz + 31.0)
                if score > best_score:
                    best_score = score
                    best = np.array([dx / env.action_scale_trans, dz / env.action_scale_trans,
                                     dyaw / env.action_scale_yaw, dpitch / env.action_scale_pitch], dtype=np.float32)
            return best
        if self.scan_mode in ("diffusion", "cfm"):
            if not self.action_queue:
                pts_t = torch.tensor(self.obs["reconstructed_points"], dtype=torch.float32, device=self.device).unsqueeze(0)
                pose_t = torch.tensor(self.obs["probe_pose"], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    if self.scan_mode == "diffusion" and self.diffusion_policy is not None:
                        chunk = self.diffusion_policy.sample_actions(pts_t, pose_t, num_steps=5).squeeze(0).cpu().numpy()
                    elif self.scan_mode == "cfm" and self.cfm_policy is not None:
                        cond = torch.cat([self.pointnet(pts_t), pose_t], dim=-1)
                        chunk = self.cfm_policy.generate_action_chunk(cond, num_steps=5).squeeze(0).cpu().numpy()
                    else:
                        return None
                self.action_queue.extend(list(chunk[:2]))
            return self.action_queue.pop(0)
        return None  # interactive: hold pose

    def _execute_scan_step_and_publish(self, current_time: float) -> None:
        action = self._policy_action()
        if action is not None:
            self.obs, _, _, _, info = self.env.step(action)
            self._send_target()
            if self.trajectory is not None and self.traj_idx % len(self.trajectory) == 0:
                print(f"[+] Scan pass complete: {len(self.env.accumulator.get_points())} points, "
                      f"coverage {info.get('coverage_ratio', float('nan')) * 100:.1f}%")
        self.publish(current_time)

    def deformed_skin_vertices(self) -> np.ndarray:
        """(S, 3) float32 deformed skin-patch vertices (P mm, same order as ``self.skin_vertices``)."""
        ts = self._tissue_state()
        if ts is None or not ts.in_contact:
            return self.skin_vertices
        model = self.env.tissue_model
        with torch.no_grad():
            if hasattr(model, "skin_surface_mesh"):
                v = model.skin_surface_mesh(ts)[0]
            else:
                x = torch.as_tensor(self.skin_vertices, dtype=torch.float32, device=self.device)
                v = ts.displace(x).detach().cpu().numpy()
        return np.ascontiguousarray(v, dtype=np.float32)

    def _skin_message(self) -> Optional[np.ndarray]:
        """(1, S, 3) DeformedSkin payload, or None if the skin did not change since the last one sent."""
        v = self.deformed_skin_vertices()
        if self._last_skin_sent is not None and np.max(np.abs(v - self._last_skin_sent)) < SKIN_CHANGE_MIN_MM:
            return None
        self._last_skin_sent = v
        return v[np.newaxis]

    def publish(self, current_time: float) -> None:
        env = self.env
        t_probe = np.asarray(env._build_probe_matrix(*env.current_pose_params), dtype=np.float64)  # indented face
        self._safe_send(pyigtl.TransformMessage(t_probe, device_name="UltrasoundProbe"))

        bmode = (np.clip(self.obs["bmode"], 0.0, 1.0) * 255.0).astype(np.uint8)
        self._safe_send(pyigtl.ImageMessage(bmode[:, :, np.newaxis], device_name="US_Image"))

        pts = env.accumulator.get_points()
        n_pts = len(pts)
        if n_pts > 0 and (n_pts >= self.last_points_sent + 5 or current_time - self.last_points_publish_time >= 0.15):
            arr = np.ascontiguousarray(np.asarray(pts, dtype=np.float32).reshape(1, -1, 3))
            self._safe_send(pyigtl.ImageMessage(arr, device_name="ReconstructedBone"))
            self.last_points_sent = n_pts
            self.last_points_publish_time = current_time

        skin = self._skin_message()
        if skin is not None:
            self._safe_send(pyigtl.ImageMessage(skin, device_name="DeformedSkin"))
            if self.viewer_proc is not None:
                self.channels.viewer.put(("skin", patient_to_pybullet_points(skin[0]).astype(np.float32)))

        ts = self._tissue_state()
        telemetry = {
            "normal_force_n": float(env.current_normal_force),
            "indentation_mm": self._indentation_mm(),
            "in_contact": bool(ts.in_contact) if ts is not None else None,
            "contact_stiffness_n_per_mm": float(getattr(ts, "contact_stiffness_n_per_mm", 0.0)) if ts is not None else None,
            "coupling": float(env.current_coupling),
            "num_points": int(n_pts),
            "physics": self._cached_stats,
            "rigid_contacts": self.channels.read_contact(),
        }
        self._safe_send(pyigtl.StringMessage(json.dumps(telemetry), device_name="Telemetry"))
        self.last_publish_time = current_time
        self.total_frames_published += 1

    # ------------------------------------------------------------------ loops
    def physics_stats(self) -> Dict[str, float]:
        return self.channels.rate_reader().snapshot()

    def _maybe_log_rate(self) -> None:
        now = time.perf_counter()
        if now - self._last_stats_refresh >= 1.0:  # percentiles over the window are not free; refresh at 1 Hz
            self._last_stats_refresh = now
            self._cached_stats = self.physics_stats()
        if now - self.last_rate_log >= self.rate_log_interval_s:
            self.last_rate_log = now
            print(f"[rate] {PhysicsRateMonitor.format(self._cached_stats)} | "
                  f"stream frames {self.total_frames_published}")

    def _network_streaming_worker(self, duration_s: Optional[float] = None) -> None:
        t_begin = time.perf_counter()
        while self.is_running:
            t0 = time.perf_counter()
            self.handle_incoming_messages()
            self._execute_scan_step_and_publish(time.time())
            self._maybe_log_rate()
            if self.physics_proc is not None and not self.physics_proc.is_alive():
                print("[!] Physics process exited; stopping.")
                break
            if duration_s is not None and t0 - t_begin >= duration_s:
                break
            sleep_time = self.publish_interval - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def step_simulation_and_stream(self, physics_ticks: int = 1) -> None:
        """Deterministic synchronous step (inline mode only): one scan/publish step + N physics ticks."""
        if self.backend is None or self.physics_mode != "inline":
            raise RuntimeError("step_simulation_and_stream requires physics_mode='inline'")
        self.handle_incoming_messages()
        self._execute_scan_step_and_publish(time.time())
        for _ in range(physics_ticks):
            t0 = time.perf_counter()
            self.backend.tick(self.channels)
            self._inline_monitor().tick(t0, time.perf_counter())

    def _inline_monitor(self) -> PhysicsRateMonitor:
        if not hasattr(self, "_inline_mon"):
            self._inline_mon = PhysicsRateMonitor(target_hz=self.physics_hz, window=self.channels.rate_window,
                                                  shared_buffer=self.channels.rate)
        return self._inline_mon

    def run(self, duration_s: Optional[float] = None) -> Dict[str, float]:
        print("=" * 70)
        print(f"[*] Unified server | port {self.port} | stream {self.publish_fps} Hz | mode {self.scan_mode} | "
              f"physics {self.physics_mode} @ {self.physics_hz:.0f} Hz | GUI {self.gui}")
        print("=" * 70)
        if self.physics_thread is not None:
            self.physics_thread.start()
        # Count only the steady state (both loops running), not scene/env construction.
        self.channels.rate_reader().reset()
        self.total_frames_published = 0
        try:
            self._network_streaming_worker(duration_s)
        except KeyboardInterrupt:
            print("\n[*] Stopping unified server...")
        stats = self.physics_stats()
        self.shutdown()
        return stats

    def shutdown(self) -> None:
        self.is_running = False
        self.channels.stop.set()
        if self.physics_thread is not None and self.physics_thread.is_alive():
            self.physics_thread.join(timeout=2.0)
        for proc in (self.physics_proc, self.viewer_proc):
            if proc is not None:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
        # A child that died early leaves queued messages unread; without this the queue feeder threads
        # block interpreter exit (seen when the viewer process crashed).
        for q in (self.channels.commands, self.channels.viewer):
            if hasattr(q, "cancel_join_thread"):
                q.cancel_join_thread()
        if self.backend is not None:
            self.backend.disconnect()
            self.backend = None
        try:
            self.server.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Unified Surgical Physics & Ultrasound Reconstruction Server")
    parser.add_argument("--port", type=int, default=18944, help="OpenIGTLink port (default 18944)")
    parser.add_argument("--urdf", type=str, default="", help="Path to robot URDF (default: Franka Panda)")
    parser.add_argument("--patient-dir", type=str, default="", help="TotalSegmentator-style patient directory")
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI")
    parser.add_argument("--fps", type=float, default=30.0, help="Streaming / scan-step rate to Slicer (Hz)")
    parser.add_argument("--mode", type=str, choices=SCAN_MODES, default="orbital_l4", help="Scan execution mode")
    parser.add_argument("--physics-mode", choices=["process", "thread"], default="process",
                        help="Run the 1 kHz physics loop in its own process (default) or in a thread")
    parser.add_argument("--physics-hz", type=float, default=1000.0, help="Physics loop rate (Hz)")
    parser.add_argument("--tissue-backend", choices=["fem", "kelvinlet"], default="fem",
                        help="Soft-tissue model: precomputed GPU FEM (default) or analytic Kelvinlet")
    parser.add_argument("--physics-backend", choices=list(BACKENDS), default="pybullet",
                        help="Rigid-body engine of the 1 kHz loop (robot, probe, fixtures, viewer)")
    parser.add_argument("--benchmark-seconds", type=float, default=0.0,
                        help="Headless benchmark: run both loops (PyBullet DIRECT) for N s and print physics-rate stats")
    parser.add_argument("--no-benchmark-client", action="store_true",
                        help="Benchmark without a draining OpenIGTLink client (messages are then never packed)")
    args = parser.parse_args()

    bench = args.benchmark_seconds > 0
    client = None
    if bench and not args.no_benchmark_client:
        # A separate-process OpenIGTLink client that drains everything, so the benchmark includes the real
        # server-side message packing / CRC / socket cost (nothing is packed while no client is connected).
        import subprocess
        client = subprocess.Popen([sys.executable, "-c", BENCH_CLIENT_SRC, str(args.port)])
    server = UnifiedSurgicalBridgeServer(
        port=args.port,
        urdf_path=args.urdf or None,
        patient_dir=args.patient_dir or None,
        gui=False if bench else args.gui,
        publish_fps=args.fps,
        scan_mode=args.mode,
        physics_mode=args.physics_mode,
        physics_hz=args.physics_hz,
        tissue_backend=args.tissue_backend,
        physics_backend=args.physics_backend,
    )
    if client is not None:
        t_wait = time.time()
        while not server.server.is_connected() and time.time() - t_wait < 15.0:
            time.sleep(0.1)
        print(f"[benchmark] OpenIGTLink drain client connected: {server.server.is_connected()}")
    stats = server.run(duration_s=args.benchmark_seconds if bench else None)
    if client is not None:
        client.terminate()
    if bench:
        frames = server.total_frames_published
        print("[benchmark] " + PhysicsRateMonitor.format(stats))
        print(f"[benchmark] streaming loop: {frames} scan steps in {args.benchmark_seconds:.1f} s "
              f"({frames / args.benchmark_seconds:.1f} Hz, target {args.fps:.0f} Hz)")
        print("[benchmark] json " + json.dumps({"physics_mode": args.physics_mode,
                                                 "physics_backend": args.physics_backend, **stats,
                                                 "stream_hz": frames / args.benchmark_seconds}))


if __name__ == "__main__":
    main()
