"""
PyBullet physics backend for the unified server.

Threading / process model (invariant 2, clock decoupling):
    * Every PyBullet call of the simulation is made by the *physics loop* only (``run_physics_loop``),
      which runs either in its own process (default, ``physics_process_main``; no GIL shared with the
      ultrasound / streaming work) or, for tests and comparison, in-process. The physics client is
      always DIRECT.
    * The GUI is a separate *viewer process* (``viewer_process_main``) with its own PyBullet GUI client:
      it mirrors the robot joints / probe pose from the physics loop at ~60 Hz, draws the force / beam
      overlays and renders the FEM-deformed skin patch (visual mesh rebuilt at <= ``SKIN_VIEW_HZ``).
      Rendering cost therefore never lands on the 1 kHz loop.
    * The streaming side never touches PyBullet. It talks to both through ``PhysicsChannels``:
        - ``commands``: queue of ``("target", dict)`` / ``("fixture", dict)`` messages
          (streaming -> physics; the loop drains it each tick, latest target wins).
        - ``contact``: lock-protected latest-value snapshot written by the physics loop every tick
          (physics -> streaming), see ``CONTACT_FIELDS``.
        - ``view``: latest joint / probe pose snapshot (physics -> viewer), see ``VIEW_FIELDS``.
        - ``viewer``: queue of ``("target", dict)`` / ``("skin_mesh", dict)`` / ``("skin", array)``
          (streaming -> viewer).
        - ``rate``: lock-free ring buffers of ``PhysicsRateMonitor`` (physics -> streaming).

Probe-tissue contact (DEC-037): the torso is visual-only (no rigid probe-torso collision). The target
carries the latest FEM tissue solution ``tissue = {contact_pb, beam_pb, force_n, stiffness_n_per_mm,
indentation_mm, in_contact}`` from the streaming side (~30 Hz). Every 1 kHz tick the loop measures the
probe face indentation delta along the beam from the PyBullet probe pose and applies
``F = F0 + K_c (delta - delta0)`` (clipped to [0, ``TISSUE_FORCE_MAX_N``]) to the probe against the beam
(first-order extrapolation of the FEM force with its contact stiffness; the loop never waits on the FEM).
PyBullet provides robot kinematics/dynamics, probe collisions with rigid structures (table, virtual
keep-out fixtures) and the GUI.
"""

import os
import queue
import threading
import time
from typing import Dict, Optional

import numpy as np
import pybullet as p
import pybullet_data

try:
    from coordinates import ras_to_pybullet_points
    from physics_rate import PhysicsRateMonitor
except ImportError:
    from surgical_physics_bridge.src.coordinates import ras_to_pybullet_points
    from surgical_physics_bridge.src.physics_rate import PhysicsRateMonitor

#: Fields of the contact snapshot published by the physics loop every tick.
CONTACT_FIELDS = (
    "seq",               # physics tick counter at the time of the snapshot
    "sim_time_s",        # simulated time
    "n_contacts",        # probe contacts with rigid bodies (table, virtual keep-out fixtures)
    "normal_force_n",    # summed PyBullet normal force of those contacts (rigid collision, NOT tissue force)
    "min_distance_m",    # most negative contact distance (penetration), 0 if no contact
    "ee_error_m",        # distance between commanded and achieved end-effector position
    "tissue_force_n",    # FEM-extrapolated tissue reaction force applied to the probe this tick
    "tissue_indentation_mm",  # probe face indentation along the beam measured from the PyBullet probe pose
)
#: Latest robot / probe pose for the viewer process: 7 arm joints, probe position (3), quaternion (4).
VIEW_FIELDS = 14
TISSUE_FORCE_MAX_N = 40.0
SKIN_VIEW_HZ = 10.0
SKIN_RGBA = [0.93, 0.62, 0.52, 0.95]


class LatestSnapshot:
    """Lock-protected latest-value buffer of floats (process-shared if built from a mp context)."""

    def __init__(self, n: int, ctx=None):
        if ctx is not None:
            self._arr = ctx.Array("d", n, lock=True)
            self._lock = self._arr.get_lock()
            self._view = None
        else:
            self._arr = None
            self._lock = threading.Lock()
            self._view = np.zeros(n, dtype=np.float64)

    def write(self, values) -> None:
        with self._lock:
            if self._arr is not None:
                self._arr[:] = values
            else:
                self._view[:] = values

    def read(self) -> np.ndarray:
        with self._lock:
            if self._arr is not None:
                return np.array(self._arr[:], dtype=np.float64)
            return self._view.copy()


class PhysicsChannels:
    """IPC between the streaming side and the physics loop (``ctx=None`` -> in-process primitives)."""

    def __init__(self, ctx=None, rate_window: int = 5000):
        self.ctx = ctx
        if ctx is not None:
            self.commands = ctx.Queue()
            self.stop = ctx.Event()
            self.ready = ctx.Event()
            self.rate = ctx.RawArray("d", PhysicsRateMonitor(window=rate_window)._buf.size)
        else:
            self.commands = queue.SimpleQueue()
            self.stop = threading.Event()
            self.ready = threading.Event()
            self.rate = PhysicsRateMonitor(window=rate_window)._buf  # plain numpy buffer
        self.contact = LatestSnapshot(len(CONTACT_FIELDS), ctx)
        self.view = LatestSnapshot(VIEW_FIELDS, ctx)
        self.viewer = ctx.Queue() if ctx is not None else queue.SimpleQueue()
        self.rate_window = rate_window

    def read_contact(self) -> Dict[str, float]:
        return dict(zip(CONTACT_FIELDS, self.contact.read().tolist()))

    def rate_reader(self) -> PhysicsRateMonitor:
        return PhysicsRateMonitor.reader(self.rate)


class PyBulletBackend:
    """Owns the PyBullet client and operating-room scene. Use only from the physics loop."""

    EE_LINK = 11            # Franka Panda grasptarget link
    ARM_JOINTS = list(range(7))
    PROBE_LENGTH_M = 0.09   # end-effector mount -> transducer face along the beam
    ROBOT_BASE_Z = 0.30

    def __init__(
        self,
        gui: bool = False,
        patient_dir: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt: float = 0.001,
    ):
        self.gui = gui
        self.dt = dt
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(dt, physicsClientId=self.cid)
        self.sim_time = 0.0
        self.step_count = 0
        self.robot_id: Optional[int] = None
        self.probe_body_id: Optional[int] = None
        self.torso_id = -1
        self.spine_id = -1
        self.fixture_ids = []
        self._ee_target = None
        self._q_seed = None
        self._overlay_ids: Dict[str, int] = {}
        self._tissue: Optional[dict] = None
        self.tissue_force_n = 0.0
        self.tissue_indentation_mm = 0.0
        self.skin_body_id = -1
        self._skin_indices: Optional[list] = None
        if patient_dir is not None:
            self.build_scene(patient_dir, urdf_path)

    # ------------------------------------------------------------------ scene
    def build_scene(self, patient_dir: str, urdf_path: Optional[str] = None) -> None:
        c = self.cid
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=c)
            p.resetDebugVisualizerCamera(1.10, 45, -22, [0.55, 0.0, 0.38], physicsClientId=c)

        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=c)
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15], physicsClientId=c)
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.42, 0.30, 0.15],
                                        rgbaColor=[0.68, 0.72, 0.75, 1.0], physicsClientId=c)
        self.table_id = p.createMultiBody(0, table_col, table_vis, [0.55, 0.0, 0.15], physicsClientId=c)

        mesh_dir = os.path.join(patient_dir, "meshes")
        spine = os.path.join(mesh_dir, "patient_lumbar_spine.obj")
        l4 = os.path.join(mesh_dir, "patient_l4_vertebra.obj")
        torso = os.path.join(mesh_dir, "patient_torso_skin.obj")

        if os.path.exists(spine):
            # Visual only. A concave-trimesh collider for the spine was measured to cost 0.7-2 ms per
            # stepSimulation (BVH queries against the ~20k-triangle mesh), breaking the 1 ms budget, and
            # its convex hull is anatomically meaningless. Probe-bone interaction is the tissue model's job.
            vis = p.createVisualShape(p.GEOM_MESH, fileName=spine, rgbaColor=[0.92, 0.89, 0.78, 1.0], physicsClientId=c)
            self.spine_id = p.createMultiBody(0, -1, vis, [0, 0, 0], physicsClientId=c)
        else:
            print(f"[!] Spine mesh missing: {spine}")
        if os.path.exists(l4):
            vis = p.createVisualShape(p.GEOM_MESH, fileName=l4, rgbaColor=[1.0, 0.85, 0.40, 0.95], physicsClientId=c)
            p.createMultiBody(0, -1, vis, [0, 0, 0], physicsClientId=c)
        if os.path.exists(torso):
            # Visual only: soft-tissue response comes from the tissue model, not rigid PyBullet contact.
            vis = p.createVisualShape(p.GEOM_MESH, fileName=torso, rgbaColor=[0.86, 0.68, 0.58, 0.55], physicsClientId=c)
            self.torso_id = p.createMultiBody(0, -1, vis, [0, 0, 0], physicsClientId=c)
        else:
            print(f"[!] Torso mesh missing: {torso}")

        robot_urdf = urdf_path if (urdf_path and os.path.exists(urdf_path)) else "franka_panda/panda.urdf"
        # Base on a 0.30 m pedestal (table-top height): from z = 0.10 m the dorsal-skin targets with the
        # probe pointing down were measured to be outside the Panda workspace (best IK error ~0.15 m).
        self.robot_id = p.loadURDF(robot_urdf, [0.0, 0.0, self.ROBOT_BASE_Z], useFixedBase=True, physicsClientId=c)
        ped_half = self.ROBOT_BASE_Z / 2.0
        ped_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.10, 0.10, ped_half],
                                      rgbaColor=[0.3, 0.3, 0.32, 1.0], physicsClientId=c)
        p.createMultiBody(0, -1, ped_vis, [0.0, 0.0, ped_half], physicsClientId=c)

        probe_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.014, length=self.PROBE_LENGTH_M,
                                        rgbaColor=[0.2, 0.5, 0.85, 1.0], physicsClientId=c)
        probe_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.014, height=self.PROBE_LENGTH_M, physicsClientId=c)
        self.probe_body_id = p.createMultiBody(0.2, probe_col, probe_vis, [0.55, 0.0, 0.45], physicsClientId=c)
        # Probe centre sits half a probe length beyond the EE along EE +Z (= beam), so its face is at the skin.
        self._probe_offset = [0.0, 0.0, -self.PROBE_LENGTH_M / 2.0]  # EE origin expressed in the probe frame
        p.createConstraint(self.robot_id, self.EE_LINK, self.probe_body_id, -1, p.JOINT_FIXED,
                           [0, 0, 0], [0, 0, 0], self._probe_offset, physicsClientId=c)
        # IK is seeded from the Panda "ready" posture: seeded from zeros it converges to an elbow-inverted
        # solution (q4 > 0, outside the joint limit) that the limit solver then fights (~10 cm EE error).
        n_movable = sum(p.getJointInfo(self.robot_id, j, physicsClientId=c)[2] != p.JOINT_FIXED
                        for j in range(p.getNumJoints(self.robot_id, physicsClientId=c)))
        rest = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]
        self._ik_rest = rest + [0.0] * (n_movable - len(rest))
        for j, qv in zip(self.ARM_JOINTS, rest):
            p.resetJointState(self.robot_id, j, qv, physicsClientId=c)
        self._ik_offset = np.zeros(3)
        self._ik_offset = self._calibrate_ik_offset()
        # The probe overlaps the hand it is mounted on; never report that as a collision.
        for link in range(-1, p.getNumJoints(self.robot_id, physicsClientId=c)):
            p.setCollisionFilterPair(self.robot_id, self.probe_body_id, link, -1, 0, physicsClientId=c)

    def _calibrate_ik_offset(self) -> np.ndarray:
        """
        PyBullet 3.2.6 ``calculateInverseKinematics(..., currentPositions=...)`` was measured to solve in a
        frame translated from the world by the base offset (EE lands exactly +0.35 m in z for a base at
        z = 0.30 m). Measure that translation once with a solve+FK round trip and subtract it from targets.
        """
        c, r = self.cid, self.robot_id
        saved = [p.getJointState(r, j, physicsClientId=c)[0] for j in self.ARM_JOINTS]
        target = np.array([0.55, 0.0, 0.50])
        orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        q = self._ik_rest
        for _ in range(3):
            q = p.calculateInverseKinematics(r, self.EE_LINK, target.tolist(), orn, currentPositions=list(q),
                                             maxNumIterations=200, residualThreshold=1e-6, physicsClientId=c)
        for j, qv in zip(self.ARM_JOINTS, q[:7]):
            p.resetJointState(r, j, qv, physicsClientId=c)
        ee = np.asarray(p.getLinkState(r, self.EE_LINK, computeForwardKinematics=1, physicsClientId=c)[4])
        for j, qv in zip(self.ARM_JOINTS, saved):
            p.resetJointState(r, j, qv, physicsClientId=c)
        return ee - target

    # ---------------------------------------------------------------- commands
    def apply_target(self, target: dict) -> None:
        """IK to the commanded end-effector pose (PyBullet world); optional teleport and GUI overlays."""
        if self.robot_id is None:
            return
        c = self.cid
        pos = target["ee_pos"]
        orn = target["ee_orn"]
        ik_pos = (np.asarray(pos, dtype=np.float64) - self._ik_offset).tolist()
        teleport = target.get("teleport", False)
        # Warm-start from the previous IK solution (not the lagging measured joints): ~10 iterations per
        # 30 Hz target keep the tick under 1 ms and still converge; a teleport gets a full cold solve.
        seed = self._ik_rest if (teleport or self._q_seed is None) else self._q_seed
        kwargs = {"currentPositions": list(seed)}
        q = p.calculateInverseKinematics(self.robot_id, self.EE_LINK, ik_pos, orn,
                                         maxNumIterations=100 if teleport else 10,
                                         residualThreshold=1e-5, physicsClientId=c, **kwargs)
        self._q_seed = list(q)
        if teleport:
            for j, qv in zip(self.ARM_JOINTS, q[:7]):
                p.resetJointState(self.robot_id, j, qv, 0.0, physicsClientId=c)
            # Move the constrained probe body with the arm, otherwise the fixed constraint yanks the arm.
            ee_pos, ee_orn = p.getLinkState(self.robot_id, self.EE_LINK, computeForwardKinematics=1,
                                            physicsClientId=c)[4:6]
            probe_pos, _ = p.multiplyTransforms(ee_pos, ee_orn, [0.0, 0.0, self.PROBE_LENGTH_M / 2.0],
                                                [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self.probe_body_id, probe_pos, ee_orn, physicsClientId=c)
            p.resetBaseVelocity(self.probe_body_id, [0, 0, 0], [0, 0, 0], physicsClientId=c)
        p.setJointMotorControlArray(self.robot_id, self.ARM_JOINTS, p.POSITION_CONTROL,
                                    targetPositions=q[:7], forces=[150.0] * 7, physicsClientId=c)
        self._ee_target = np.asarray(pos, dtype=np.float64)
        tissue = target.get("tissue")
        if tissue is not None:
            self._tissue = {
                "contact": np.asarray(tissue["contact_pb"], dtype=np.float64),
                "beam": np.asarray(tissue["beam_pb"], dtype=np.float64),
                "F0": float(tissue["force_n"]),
                "K": float(tissue["stiffness_n_per_mm"]),
                "d0": float(tissue["indentation_mm"]),
                "in_contact": bool(tissue["in_contact"]),
            }

    def apply_tissue_force(self) -> None:
        """F = F0 + K_c (delta - delta0) on the probe, against the beam, at the probe face (every tick)."""
        tis = self._tissue
        if tis is None or self.probe_body_id is None:
            self.tissue_force_n = 0.0
            return
        c = self.cid
        pos = np.asarray(p.getBasePositionAndOrientation(self.probe_body_id, physicsClientId=c)[0])
        beam = tis["beam"]
        face = pos + beam * (self.PROBE_LENGTH_M / 2.0)
        delta = 1000.0 * float(np.dot(face - tis["contact"], beam))
        f = tis["F0"] + tis["K"] * (delta - tis["d0"]) if tis["in_contact"] else 0.0
        f = float(np.clip(f, 0.0, TISSUE_FORCE_MAX_N))
        if f > 0.0:
            p.applyExternalForce(self.probe_body_id, -1, (-beam * f).tolist(), face.tolist(), p.WORLD_FRAME,
                                 physicsClientId=c)
        self.tissue_force_n = f
        self.tissue_indentation_mm = delta

    def view_snapshot(self) -> list:
        c, r = self.cid, self.robot_id
        q = [s[0] for s in p.getJointStates(r, self.ARM_JOINTS, physicsClientId=c)]
        pos, orn = p.getBasePositionAndOrientation(self.probe_body_id, physicsClientId=c)
        return list(q) + list(pos) + list(orn)

    # ------------------------------------------------------------- viewer side
    def set_view(self, snap: np.ndarray) -> None:
        """Viewer process: mirror the physics loop's joints and probe pose (no simulation stepping)."""
        c = self.cid
        for j, qv in zip(self.ARM_JOINTS, snap[:7]):
            p.resetJointState(self.robot_id, j, float(qv), physicsClientId=c)
        p.resetBasePositionAndOrientation(self.probe_body_id, snap[7:10].tolist(), snap[10:14].tolist(),
                                          physicsClientId=c)

    def set_skin_mesh(self, indices) -> None:
        """Triangle indices of the FEM skin patch (vertex order of the streamed skin vertices)."""
        tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        # P -> PB is a reflection (det -1): reverse the winding so the outward faces stay front-facing.
        self._skin_indices = tris[:, ::-1].ravel().tolist()

    def update_skin(self, vertices_pb: np.ndarray) -> None:
        """Rebuild the visual-only skin body (PyBullet cannot update rigid visual mesh vertices in place:
        ``resetMeshData`` only supports soft bodies)."""
        if self._skin_indices is None:
            return
        c = self.cid
        # With rendering enabled the GUI client syncs a frame per body change (measured 68 ms per rebuild of
        # the 864-vertex patch); toggling rendering off around the rebuild brings it to ~1.4 ms.
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=c)
        if self.skin_body_id >= 0:
            p.removeBody(self.skin_body_id, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_MESH, vertices=np.asarray(vertices_pb, dtype=np.float64).tolist(),
                                  indices=self._skin_indices, rgbaColor=SKIN_RGBA, physicsClientId=c)
        self.skin_body_id = p.createMultiBody(0, -1, vis, [0, 0, 0], physicsClientId=c)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=c)

    def _debug_line(self, key, a, b, color, width):
        new_id = p.addUserDebugLine(a, b, color, lineWidth=width,
                                    replaceItemUniqueId=self._overlay_ids.get(key, -1), physicsClientId=self.cid)
        self._overlay_ids[key] = new_id

    def _draw_overlays(self, target: dict) -> None:
        contact = np.asarray(target["contact_pb"], dtype=np.float64)
        beam = np.asarray(target["beam_pb"], dtype=np.float64)
        lateral = np.asarray(target.get("lateral_pb", [1.0, 0.0, 0.0]), dtype=np.float64)
        fn = float(target.get("force_n", 0.0))
        color = [0.1, 0.95, 0.3] if fn <= 6.0 else ([1.0, 0.75, 0.1] if fn <= 8.0 else [1.0, 0.15, 0.1])
        # Reaction force on the probe acts against the beam direction.
        self._debug_line("force", contact.tolist(), (contact - beam * fn * 0.015).tolist(), color, 4)
        half_w, depth = 0.018, 0.06
        tl, tr = contact - lateral * half_w, contact + lateral * half_w
        bl, br = tl + beam * depth, tr + beam * depth
        cyan = [0.0, 0.85, 1.0]
        self._debug_line("beam_l", tl.tolist(), bl.tolist(), cyan, 2)
        self._debug_line("beam_r", tr.tolist(), br.tolist(), cyan, 2)
        self._debug_line("beam_b", bl.tolist(), br.tolist(), cyan, 2)
        hud = target.get("hud")
        if hud:
            self._overlay_ids["hud"] = p.addUserDebugText(
                hud, textPosition=[0.42, -0.30, 0.62], textColorRGB=[0.1, 0.95, 0.4], textSize=1.1,
                replaceItemUniqueId=self._overlay_ids.get("hud", -1), physicsClientId=self.cid)

    def set_virtual_fixture(self, vf_data: dict) -> None:
        """Creates a static keep-out collision body from a Slicer mesh (vertices in RAS mm)."""
        vertices = vf_data.get("vertices", [])
        indices = vf_data.get("indices", [])
        if not vertices or not indices:
            print("[!] Invalid VirtualFixture: needs 'vertices' and 'indices'.")
            return
        if vf_data.get("policy", "keep_out") != "keep_out":
            print(f"[!] Unsupported VirtualFixture policy: {vf_data.get('policy')}")
            return
        pb_vertices = ras_to_pybullet_points(np.asarray(vertices, dtype=np.float64).reshape(-1, 3)).tolist()
        col = p.createCollisionShape(p.GEOM_MESH, vertices=pb_vertices, indices=list(indices),
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_MESH, vertices=pb_vertices, indices=list(indices),
                                  rgbaColor=[1, 0, 0, 0.3], physicsClientId=self.cid)
        self.fixture_ids.append(p.createMultiBody(0, col, vis, physicsClientId=self.cid))
        print(f"[+] Virtual keep-out fixture created ({len(pb_vertices)} vertices, {len(indices) // 3} faces).")

    # -------------------------------------------------------------------- loop
    def step(self) -> None:
        self.apply_tissue_force()   # external forces are cleared by every stepSimulation
        p.stepSimulation(physicsClientId=self.cid)
        self.sim_time += self.dt
        self.step_count += 1

    def contact_snapshot(self) -> list:
        n, force, min_d, ee_err = 0, 0.0, 0.0, 0.0
        if self.probe_body_id is not None:
            for cp in p.getContactPoints(bodyA=self.probe_body_id, physicsClientId=self.cid):
                n += 1
                force += float(cp[9])
                min_d = min(min_d, float(cp[8]))
        if self._ee_target is not None:
            ee = p.getLinkState(self.robot_id, self.EE_LINK, physicsClientId=self.cid)[4]
            ee_err = float(np.linalg.norm(np.asarray(ee) - self._ee_target))
        return [float(self.step_count), self.sim_time, float(n), force, min_d, ee_err,
                self.tissue_force_n, self.tissue_indentation_mm]

    def tick(self, channels: PhysicsChannels) -> None:
        """One physics tick: drain commands (latest target wins), step, publish contact snapshot."""
        latest_target = None
        while True:
            try:
                kind, payload = channels.commands.get_nowait()
            except queue.Empty:
                break
            if kind == "target":
                if latest_target is not None and latest_target.get("teleport"):
                    payload = dict(payload, teleport=True)
                latest_target = payload
            elif kind == "fixture":
                self.set_virtual_fixture(payload)
        if latest_target is not None:
            self.apply_target(latest_target)
        self.step()
        channels.contact.write(self.contact_snapshot())
        if self.step_count % 10 == 0:   # the viewer renders at <= 60 Hz; 100 Hz snapshots are plenty
            channels.view.write(self.view_snapshot())

    def disconnect(self) -> None:
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass


def run_physics_loop(
    backend: PyBulletBackend,
    channels: PhysicsChannels,
    target_hz: float = 1000.0,
    spin_s: float = 2e-4,
    max_lag_ticks: int = 5,
    duration_s: Optional[float] = None,
) -> PhysicsRateMonitor:
    """
    Deadline-scheduled fixed-rate loop. Sleeps until ``spin_s`` before each deadline and busy-waits
    the remainder (Linux sleep overshoots by ~50-100 us). If the loop falls more than
    ``max_lag_ticks`` behind it re-synchronises instead of bursting catch-up steps (counted as a resync).
    """
    monitor = PhysicsRateMonitor(target_hz=target_hz, window=channels.rate_window, shared_buffer=channels.rate)
    monitor.reset()
    dt = 1.0 / target_hz
    t_begin = time.perf_counter()
    next_t = t_begin
    stop = channels.stop
    while not stop.is_set():
        remaining = next_t - time.perf_counter()
        if remaining > spin_s:
            time.sleep(remaining - spin_s)
        while time.perf_counter() < next_t:
            pass
        t0 = time.perf_counter()
        backend.tick(channels)
        t1 = time.perf_counter()
        monitor.tick(t0, t1)
        next_t += dt
        if t1 - next_t > max_lag_ticks * dt:
            next_t = t1 + dt
            monitor.note_resync()
        if duration_s is not None and t1 - t_begin >= duration_s:
            break
    return monitor


def dedicated_physics_cpu() -> Optional[int]:
    """The CPU reserved for the physics process (highest-numbered allowed CPU), or None if unsupported."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    cpus = sorted(os.sched_getaffinity(0))
    return cpus[-1] if len(cpus) >= 2 else None


def pin_to_dedicated_cpu() -> Optional[int]:
    """Pin the calling process to the dedicated physics CPU (Linux only)."""
    cpu = dedicated_physics_cpu()
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    return cpu


def exclude_dedicated_cpu(cpu: Optional[int]) -> None:
    """Keep the calling (streaming) process, ALL its existing threads and its future threads off the physics
    CPU. ``sched_setaffinity(0)`` only affects the calling thread on Linux, so thread pools created earlier
    (numpy's OpenBLAS pool spins between calls; torch / CUDA workers) are re-pinned through /proc/self/task:
    measured period p99 3.5 ms with only the calling thread excluded."""
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return
    others = set(os.sched_getaffinity(0)) - {cpu}
    if not others:
        return
    os.sched_setaffinity(0, others)
    try:
        tids = [int(t) for t in os.listdir("/proc/self/task")]
    except OSError:
        tids = []
    for tid in tids:
        try:
            os.sched_setaffinity(tid, others)
        except OSError:
            pass


def run_viewer(backend: PyBulletBackend, channels: PhysicsChannels, fps: float = 60.0,
               skin_hz: float = SKIN_VIEW_HZ, log_interval_s: float = 10.0) -> None:
    """Viewer loop (GUI client): mirror the physics state, draw overlays, rebuild the skin mesh."""
    period = 1.0 / fps
    last_skin = 0.0
    pending_skin = None
    skin_ms = []
    last_log = time.perf_counter()
    while not channels.stop.is_set():
        t0 = time.perf_counter()
        target = None
        while True:
            try:
                kind, payload = channels.viewer.get_nowait()
            except queue.Empty:
                break
            if kind == "target":
                target = payload
            elif kind == "skin_mesh":
                backend.set_skin_mesh(payload["indices"])
                pending_skin = np.asarray(payload["vertices_pb"])
            elif kind == "skin":
                pending_skin = payload
        snap = channels.view.read()
        if np.any(snap[10:14]):
            backend.set_view(snap)
        if target is not None:
            backend._draw_overlays(target)
        if pending_skin is not None and t0 - last_skin >= 1.0 / skin_hz:
            ts = time.perf_counter()
            backend.update_skin(pending_skin)
            skin_ms.append(1e3 * (time.perf_counter() - ts))
            pending_skin = None
            last_skin = t0
        if skin_ms and t0 - last_log >= log_interval_s:
            last_log = t0
            print(f"[viewer] skin mesh rebuild: median {np.median(skin_ms):.2f} ms, max {np.max(skin_ms):.2f} ms "
                  f"over {len(skin_ms)} updates (viewer process, not the physics loop)")
            skin_ms = skin_ms[-500:]
        sleep = period - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def viewer_process_main(config: dict, channels: PhysicsChannels, fps: float = 60.0) -> None:
    """Entry point of the GUI viewer process (spawned by the unified server with ``--gui``)."""
    backend = None
    try:
        backend = PyBulletBackend(**dict(config, gui=True))
        run_viewer(backend, channels, fps=fps)
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            backend.disconnect()


def physics_process_main(config: dict, channels: PhysicsChannels, target_hz: float = 1000.0) -> None:
    """Entry point of the dedicated physics process (spawned by the unified server)."""
    backend = None
    try:
        dedicated = pin_to_dedicated_cpu()
        if dedicated is not None:
            print(f"[+] Physics process pinned to CPU {dedicated}")
        backend = PyBulletBackend(**config)
        channels.ready.set()
        run_physics_loop(backend, channels, target_hz=target_hz)
    except KeyboardInterrupt:
        pass
    finally:
        channels.ready.set()  # never leave the parent waiting if scene construction failed
        if backend is not None:
            backend.disconnect()
