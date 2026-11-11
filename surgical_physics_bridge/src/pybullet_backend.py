"""
PyBullet rigid-body backend for the unified server (the default ``--physics-backend``).

The engine-independent parts (IPC channels, the tick, the tissue force law, the 1 kHz loop, the viewer loop
and the process entry points) live in ``physics_common.py``; see its docstring for the process model. This
module only holds the PyBullet calls: scene, IK, position motors, external force, contacts, fixtures, GUI.

PyBullet provides robot kinematics/dynamics, probe collisions with rigid structures (table, virtual
keep-out fixtures) and the GUI. The torso is visual-only (DEC-037): the probe-tissue force comes from the
tissue model through ``PhysicsBackendBase.apply_tissue_force``.
"""

import os
import time
from typing import Dict, Optional

import numpy as np
import pybullet as p
import pybullet_data

# The shared names are re-exported: tests and scripts import them from this module.
if __name__.rpartition(".")[0]:   # same copy of physics_common as the importer (see create_backend)
    from surgical_physics_bridge.src.physics_common import (  # noqa: F401
        CONTACT_FIELDS, PROBE_LENGTH_M, ROBOT_BASE_Z, SKIN_RGBA, SKIN_VIEW_HZ, TISSUE_FORCE_MAX_N, VIEW_CLOSEUP_DISTANCE_M,
        VIEW_CLOSEUP_PITCH_DEG, VIEW_CLOSEUP_YAW_DEG, VIEW_FIELDS, ARM_REST_POSTURE, LatestSnapshot,
        PhysicsBackendBase, PhysicsChannels, dedicated_physics_cpu, exclude_dedicated_cpu, physics_process_main,
        pin_to_dedicated_cpu, run_physics_loop, run_viewer, viewer_process_main,
    )
else:
    from physics_common import (  # noqa: F401
        CONTACT_FIELDS, PROBE_LENGTH_M, ROBOT_BASE_Z, SKIN_RGBA, SKIN_VIEW_HZ, TISSUE_FORCE_MAX_N, VIEW_CLOSEUP_DISTANCE_M,
        VIEW_CLOSEUP_PITCH_DEG, VIEW_CLOSEUP_YAW_DEG, VIEW_FIELDS, ARM_REST_POSTURE, LatestSnapshot,
        PhysicsBackendBase, PhysicsChannels, dedicated_physics_cpu, exclude_dedicated_cpu, physics_process_main,
        pin_to_dedicated_cpu, run_physics_loop, run_viewer, viewer_process_main,
    )


class PyBulletBackend(PhysicsBackendBase):
    """Owns the PyBullet client and operating-room scene. Use only from the physics loop."""

    EE_LINK = 11            # Franka Panda grasptarget link
    ARM_JOINTS = list(range(7))

    def __init__(
        self,
        gui: bool = False,
        patient_dir: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt: float = 0.001,
    ):
        self._init_common(gui, dt)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(dt, physicsClientId=self.cid)
        self.robot_id: Optional[int] = None
        self.probe_body_id: Optional[int] = None
        self.torso_id = -1
        self.spine_id = -1
        self._q_seed = None
        self._overlay_ids: Dict[str, int] = {}
        self.skin_body_id = -1
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
        rest = list(ARM_REST_POSTURE)
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
        self._store_target(target)

    # ------------------------------------------------------------ engine hooks
    def _probe_centre(self) -> Optional[np.ndarray]:
        if self.probe_body_id is None:
            return None
        return np.asarray(p.getBasePositionAndOrientation(self.probe_body_id, physicsClientId=self.cid)[0])

    def _apply_probe_force(self, force: np.ndarray, point: np.ndarray) -> None:
        # External forces are cleared by every stepSimulation, so the base class re-applies it each tick.
        p.applyExternalForce(self.probe_body_id, -1, force.tolist(), point.tolist(), p.WORLD_FRAME,
                             physicsClientId=self.cid)

    def _step_engine(self) -> None:
        p.stepSimulation(physicsClientId=self.cid)

    def _probe_contacts(self):
        n, force, min_d = 0, 0.0, 0.0
        if self.probe_body_id is not None:
            for cp in p.getContactPoints(bodyA=self.probe_body_id, physicsClientId=self.cid):
                n += 1
                force += float(cp[9])
                min_d = min(min_d, float(cp[8]))
        return n, force, min_d

    def _ee_position(self) -> np.ndarray:
        return np.asarray(p.getLinkState(self.robot_id, self.EE_LINK, physicsClientId=self.cid)[4])

    def _ee_orientation_xyzw(self):
        return p.getLinkState(self.robot_id, self.EE_LINK, physicsClientId=self.cid)[5]

    def _create_fixture(self, pb_vertices: np.ndarray, indices: list) -> int:
        """Static keep-out body colliding as the concave triangle mesh itself."""
        verts = pb_vertices.tolist()
        col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=indices,
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=indices,
                                  rgbaColor=[1, 0, 0, 0.3], physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, physicsClientId=self.cid)

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

    def focus_camera(self, target_pb, distance_m: float = VIEW_CLOSEUP_DISTANCE_M) -> None:
        """Close-up of the scan site, so the skin indentation under the probe is visible.

        Only the initial view: the user can still orbit and zoom with the mouse."""
        if self.gui:
            p.resetDebugVisualizerCamera(distance_m, VIEW_CLOSEUP_YAW_DEG, VIEW_CLOSEUP_PITCH_DEG,
                                         np.asarray(target_pb, dtype=np.float64).tolist(), physicsClientId=self.cid)

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

    def draw_overlays(self, target: dict) -> None:
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

    def save_snapshot(self, out_dir: str, width: int = 1280, height: int = 800) -> None:
        """Render the GUI client's current debug camera to ``out_dir/viewer_<unix ms>.png``."""
        cam = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        _, _, rgba, _, _ = p.getCameraImage(width, height, viewMatrix=cam[2], projectionMatrix=cam[3],
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.cid)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        from PIL import Image  # local import: only needed when snapshots are enabled
        Image.fromarray(rgb).save(os.path.join(out_dir, f"viewer_{int(time.time() * 1e3)}.png"))

    def disconnect(self) -> None:
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
