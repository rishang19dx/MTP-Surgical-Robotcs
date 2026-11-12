"""
MuJoCo rigid-body backend for the unified server (``--physics-backend mujoco``, DEC-052).

It does the job of ``pybullet_backend.py`` with MuJoCo: robot dynamics, the probe, rigid contacts with the
table and virtual keep-out fixtures, and the GUI viewer. The soft tissue is NOT simulated here: the
probe-tissue force still comes from the tissue model through ``PhysicsBackendBase.apply_tissue_force``.

Design:
    * **World frame**: the PB frame of ``coordinates.py`` (Z-up, metres), identical to the PyBullet scene, so
      the coordinate helpers and the exported ``meshes/*.obj`` are shared. Quaternions cross the boundary as
      (x, y, z, w); MuJoCo is (w, x, y, z) inside.
    * **Robot**: the same Franka Panda URDF PyBullet loads (``pybullet_data``; located without importing
      pybullet), compiled by MuJoCo. The gripper fingers are removed (7 dofs). The probe is a child body of
      ``panda_grasptarget``, so there is no constraint to drift.
    * **Control (torque level)**: ``tau = qfrc_bias + M(q) (Kp (q_d - q) - Kd qdot)``, clipped to the URDF
      effort limits. ``q_d`` follows a damped-least-squares IK solution at a bounded joint speed. Unlike
      PyBullet's position motors this arm is compliant: an external force deflects it.
    * **Step**: ``mj_step1`` (kinematics, contacts, bias forces), then tissue force and control torques from
      that fresh state, then ``mj_step2``.
    * **Fixtures**: MuJoCo cannot add a geom to a compiled model, so a fixture triggers one ``MjSpec``
      recompile (state carried over). That tick is long (measured ~6 ms for a small mesh); steady-state
      ticks are unaffected. A mesh geom collides as its convex hull, so each fixture TRIANGLE becomes its own
      thin convex prism (``FIXTURE_SHELL_M`` thick, centred on the triangle): the union follows a concave
      surface and a flat sheet has a volume. PyBullet collides with the triangle mesh itself.
    * **Viewer** (``gui=True``, viewer process only): ``mujoco.viewer.launch_passive``. Nothing in the
      headless path imports or touches OpenGL.

Written against MuJoCo 3.17.0 (the last release for Python 3.10); only APIs that also exist in 4.0 are used.
"""

import importlib.util
import math
import os
import re
import time
from typing import Optional

import mujoco
import numpy as np

if __name__.rpartition(".")[0]:   # same copy of physics_common as the importer (see create_backend)
    from surgical_physics_bridge.src.physics_common import (
        ARM_REST_POSTURE, PROBE_MASS_KG, PROBE_RADIUS_M, SKIN_RGBA, TABLE_CENTRE, TABLE_HALF_EXTENTS,
        VIEW_CLOSEUP_DISTANCE_M, VIEW_CLOSEUP_PITCH_DEG, VIEW_CLOSEUP_YAW_DEG, PhysicsBackendBase,
    )
else:
    from physics_common import (
        ARM_REST_POSTURE, PROBE_MASS_KG, PROBE_RADIUS_M, SKIN_RGBA, TABLE_CENTRE, TABLE_HALF_EXTENTS,
        VIEW_CLOSEUP_DISTANCE_M, VIEW_CLOSEUP_PITCH_DEG, VIEW_CLOSEUP_YAW_DEG, PhysicsBackendBase,
    )

# Collision bits. Robot links never collide with each other or with the probe mounted on the hand; both
# collide with the environment (floor, table, fixtures).
_ROBOT_BIT, _PROBE_BIT = 1, 2
_ENV_AFFINITY = _ROBOT_BIT | _PROBE_BIT

_PATIENT_VISUALS = (
    ("patient_lumbar_spine.obj", (0.92, 0.89, 0.78, 1.0)),
    ("patient_l4_vertebra.obj", (1.0, 0.85, 0.40, 0.95)),
    ("patient_torso_skin.obj", (0.86, 0.68, 0.58, 0.55)),
)


def _default_urdf() -> str:
    """The Panda URDF shipped with pybullet_data, found without importing pybullet."""
    spec = importlib.util.find_spec("pybullet_data")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("pybullet_data (Franka Panda URDF) is not installed; pass urdf_path")
    return os.path.join(list(spec.submodule_search_locations)[0], "franka_panda", "panda.urdf")


def _urdf_to_spec(urdf_path: str) -> "mujoco.MjSpec":
    """Parse a URDF with MuJoCo, keeping every link as a body and resolving ``package://`` mesh paths."""
    mesh_dir = os.path.dirname(os.path.abspath(urdf_path))
    with open(urdf_path) as f:
        urdf = f.read().replace("package://", "")
    compiler = (f'<mujoco><compiler meshdir="{mesh_dir}" strippath="false" discardvisual="true" '
                f'fusestatic="false" balanceinertia="true"/></mujoco>')
    urdf = re.sub(r"(<robot\b[^>]*>)", lambda m: m.group(1) + compiler, urdf, count=1)
    return mujoco.MjSpec.from_string(urdf)


def _xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q) -> list:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


class MujocoBackend(PhysicsBackendBase):
    """Owns the MuJoCo model and operating-room scene. Use only from the physics loop (or the viewer)."""

    EE_BODY = "panda_grasptarget"
    N_ARM = 7
    # Acceleration-level PD (computed torque): natural frequency 10 Hz, critically damped.
    KP = (2.0 * np.pi * 10.0) ** 2
    KD = 2.0 * (2.0 * np.pi * 10.0)
    JOINT_ARMATURE = 0.1          # reflected rotor inertia (kg m^2); the URDF has none
    MAX_JOINT_SPEED = 2.0         # rad/s, slew limit of the commanded joint positions
    MAX_FOLLOWING_ERROR = 0.005   # rad, see step()
    FIXTURE_SHELL_M = 0.004       # thickness of the per-triangle fixture prisms
    MAX_FIXTURE_TRIANGLES = 2000  # one geom per triangle; larger keep-out meshes must be decimated first
    MAX_INTERP_TICKS = 100        # a target is interpolated over at most this many ticks
    IK_MAX_TICKS = 300            # a streamed target gets one IK iteration per tick until converged, at most this many
                                  # (a teleport gets a cold 200-iteration solve)
    IK_DAMPING = 0.05
    MAX_VISUAL_RADIUS_M = 2.0     # anatomy meshes larger than this are not in metres and are not drawn
    IK_MAX_STEP = 0.3             # rad per iteration

    def __init__(
        self,
        gui: bool = False,
        patient_dir: Optional[str] = None,
        urdf_path: Optional[str] = None,
        dt: float = 0.001,
    ):
        self._init_common(gui, dt)
        self.spec = None
        self.model = None
        self.data = None
        self.viewer = None
        self._q_cmd = None
        self._q_goal = None
        self._q_rate = np.zeros(self.N_ARM)
        self._last_target_step = 0
        self._ik_target = None
        self._ik_left = 0
        self._ik_converged = True
        self._interp_ticks = 1
        self._skin_mesh_id = -1
        self._skin_geom_id = -1
        self._skin_nvert = 0
        self._renderer = None
        self._snapshot_failed = False
        self._camera = ([0.55, 0.0, 0.38], 1.6, 135.0, -22.0)
        if patient_dir is not None:
            self.build_scene(patient_dir, urdf_path)

    # ------------------------------------------------------------------ scene
    def build_scene(self, patient_dir: str, urdf_path: Optional[str] = None) -> None:
        robot_urdf = urdf_path if (urdf_path and os.path.exists(urdf_path)) else _default_urdf()
        spec = _urdf_to_spec(robot_urdf)
        # ``discardvisual`` was needed while parsing (the URDF's visual meshes are not loaded); left on, the
        # compiler would also drop every non-colliding geom added below (pedestal, anatomy, skin patch).
        spec.compiler.discardvisual = False
        # Gripper: not used. Remove the finger joints (and the mimic equality that names them).
        for eq in list(spec.equalities):
            spec.delete(eq)
        hinge = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
        for j in list(spec.joints):
            if j not in hinge[:self.N_ARM]:
                spec.delete(j)
        for j in spec.joints:
            j.armature = self.JOINT_ARMATURE
        for g in spec.geoms:
            g.contype, g.conaffinity = _ROBOT_BIT, 0
        base = spec.worldbody.first_body()
        base.pos = [0.0, 0.0, self.ROBOT_BASE_Z]

        ee = spec.body(self.EE_BODY)
        if ee is None:
            raise ValueError(f"URDF has no link named {self.EE_BODY!r}")
        half = self.PROBE_LENGTH_M / 2.0
        # Probe centre sits half a probe length beyond the EE along EE +Z (= beam): its face is EE + L z.
        probe = ee.add_body(name="probe", pos=[0.0, 0.0, half])
        probe.add_geom(name="probe_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PROBE_RADIUS_M, half, 0.0],
                       mass=PROBE_MASS_KG, rgba=[0.2, 0.5, 0.85, 1.0], contype=_PROBE_BIT, conaffinity=0)

        world = spec.worldbody
        world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1],
                       rgba=[0.55, 0.57, 0.6, 1.0], contype=0, conaffinity=_ENV_AFFINITY)
        world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_EXTENTS),
                       pos=list(TABLE_CENTRE), rgba=[0.68, 0.72, 0.75, 1.0], contype=0,
                       conaffinity=_ENV_AFFINITY)
        ped_half = self.ROBOT_BASE_Z / 2.0
        world.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.10, 0.10, ped_half],
                       pos=[0.0, 0.0, ped_half], rgba=[0.3, 0.3, 0.32, 1.0], contype=0, conaffinity=0)
        world.add_light(pos=[0.5, -0.5, 2.0], dir=[0.0, 0.3, -1.0])
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 800   # viewer snapshots
        spec.option.timestep = self.dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER   # mj_step1 / mj_step2 need a one-stage scheme
        self.spec = spec
        self.model = spec.compile()
        if self.gui:
            self._add_patient_visuals(os.path.join(patient_dir, "meshes"))
        self.data = mujoco.MjData(self.model)
        self._bind()
        rest = np.asarray(ARM_REST_POSTURE, dtype=np.float64)
        self.data.qpos[self._qadr] = rest
        mujoco.mj_forward(self.model, self.data)
        self._q_cmd = rest.copy()
        self._q_goal = rest.copy()
        if self.gui:
            self._launch_viewer()

    def _add_patient_visuals(self, mesh_dir: str) -> None:
        """Visual-only anatomy (viewer process). A mesh MuJoCo cannot compile is skipped, never fatal."""
        for fname, rgba in _PATIENT_VISUALS:
            path = os.path.abspath(os.path.join(mesh_dir, fname))   # meshdir points at the robot URDF
            if not os.path.exists(path):
                print(f"[!] Patient mesh missing: {path}")
                continue
            name = os.path.splitext(fname)[0]
            mesh = self.spec.add_mesh(name=name, file=path)
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom = self.spec.worldbody.add_geom(name=name + "_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                                rgba=list(rgba), contype=0, conaffinity=0, group=1)
            try:
                self.model = self.spec.compile()
                radius = float(self.model.geom_rbound[self.model.geom(name + "_vis").id])
                if radius > self.MAX_VISUAL_RADIUS_M:
                    raise ValueError(f"bounding radius {radius:.1f} m: the file is not in the PB frame (metres)")
            except Exception as exc:  # e.g. degenerate geometry, or a mesh exported in the wrong frame
                print(f"[!] MuJoCo viewer skips {fname} ({str(exc).splitlines()[0]}).")
                self.spec.delete(geom)
                self.spec.delete(mesh)
                self.model = self.spec.compile()

    def _bind(self) -> None:
        """Resolve ids and allocate work buffers; call after every (re)compile."""
        m = self.model
        self._ee_id = m.body(self.EE_BODY).id
        self._probe_id = m.body("probe").id
        self._probe_geom = m.geom("probe_geom").id
        arm = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE][:self.N_ARM]
        self._qadr = m.jnt_qposadr[arm].copy()
        self._dofs = m.jnt_dofadr[arm].copy()
        self._q_lo = m.jnt_range[arm, 0].copy()
        self._q_hi = m.jnt_range[arm, 1].copy()
        limited = m.jnt_actfrclimited[arm].astype(bool)
        self._tau_max = np.where(limited, m.jnt_actfrcrange[arm, 1], np.inf)
        self._acc = np.zeros(m.nv)
        self._tau = np.zeros(m.nv)
        self._ext = np.zeros(m.nv)
        # One 6 x nv Jacobian buffer; mj_jacBody writes into its two halves. Plain slices (not index arrays)
        # keep the per-iteration numpy overhead low: IK runs inside the 1 ms tick.
        self._jac = np.zeros((6, m.nv))
        self._jacp, self._jacr = self._jac[:3], self._jac[3:]
        contiguous = np.array_equal(self._dofs, np.arange(self._dofs[0], self._dofs[0] + len(self._dofs)))
        self._dof_slice = slice(int(self._dofs[0]), int(self._dofs[0]) + len(self._dofs)) if contiguous else self._dofs
        self._ik_damping = self.IK_DAMPING ** 2 * np.eye(6)
        self._ik_err = np.zeros(6)
        self._ik_data = mujoco.MjData(m)   # scratch state: IK never disturbs the simulated one
        self._zero3 = np.zeros(3)

    # --------------------------------------------------------------------- IK
    def _solve_ik(self, pos: np.ndarray, quat_wxyz: np.ndarray, seed: np.ndarray, iterations: int,
                  tol: float = 1e-5) -> np.ndarray:
        """Damped least squares on the 6-D end-effector error, within the joint limits."""
        m, d = self.model, self._ik_data
        q = np.array(seed, dtype=np.float64)
        err = self._ik_err
        ee, tol2 = self._ee_id, tol * tol
        tw, tx, ty, tz = (float(v) for v in quat_wxyz)
        for _ in range(iterations):
            d.qpos[self._qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err[:3] = pos - d.xpos[ee]
            # World-frame rotation vector taking the current orientation to the target: log(q_t * q_c^-1),
            # in plain floats (cheaper than wrapped calls on 4-vectors).
            cw, cx, cy, cz = d.xquat[ee].tolist()
            ew = tw * cw + tx * cx + ty * cy + tz * cz
            ex = -tw * cx + tx * cw - ty * cz + tz * cy
            ey = -tw * cy + tx * cz + ty * cw - tz * cx
            ez = -tw * cz - tx * cy + ty * cx + tz * cw
            if ew < 0.0:                 # shortest rotation
                ew, ex, ey, ez = -ew, -ex, -ey, -ez
            sin_half = math.sqrt(ex * ex + ey * ey + ez * ez)
            scale = 2.0 * math.atan2(sin_half, ew) / sin_half if sin_half > 1e-12 else 2.0
            err[3], err[4], err[5] = ex * scale, ey * scale, ez * scale
            if err[:3] @ err[:3] < tol2 and err[3:] @ err[3:] < 100.0 * tol2:
                self._ik_converged = True
                break
            self._ik_converged = False
            mujoco.mj_jacBody(m, d, self._jacp, self._jacr, ee)
            jac = self._jac[:, self._dof_slice]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + self._ik_damping, err)
            step = max(dq.max(), -dq.min())
            if step > self.IK_MAX_STEP:
                dq *= self.IK_MAX_STEP / step
            q += dq
            np.clip(q, self._q_lo, self._q_hi, out=q)
        return q

    # ---------------------------------------------------------------- commands
    def apply_target(self, target: dict) -> None:
        """IK to the commanded end-effector pose (world frame, quaternion xyzw); optional teleport."""
        if self.model is None:
            return
        pos = np.asarray(target["ee_pos"], dtype=np.float64)
        quat = _xyzw_to_wxyz(target["ee_orn"])
        quat /= np.linalg.norm(quat)
        teleport = bool(target.get("teleport", False))
        # Warm start from the previous IK solution (not the lagging measured joints); a teleport gets a
        # full cold solve from the ready posture.
        if teleport:
            self._q_goal = self._solve_ik(pos, quat, np.asarray(ARM_REST_POSTURE), iterations=200)
            self._q_cmd = self._q_goal.copy()
            self._q_rate[:] = 0.0
            self._ik_left = 0
            self.data.qpos[self._qadr] = self._q_goal
            self.data.qvel[:] = 0.0
            self.data.qacc_warmstart[:] = 0.0
        else:
            # The IK for a streamed target is spread over the next ticks (one iteration each, in step(), until
            # it converges), so no single tick carries the whole solve. The command is interpolated to reach the
            # goal when the next target is due (targets arrive at the streaming rate).
            self._ik_target = (pos, quat)
            self._ik_left = self.IK_MAX_TICKS
            self._interp_ticks = int(np.clip(self.step_count - self._last_target_step, 1, self.MAX_INTERP_TICKS))
        self._last_target_step = self.step_count
        self._store_target(target)

    # -------------------------------------------------------------------- loop
    def step(self) -> None:
        m, d = self.model, self.data
        if m is None:
            return
        mujoco.mj_step1(m, d)             # kinematics, collisions, M and bias forces at the current state
        self.apply_tissue_force()         # fills self._ext through _apply_probe_force
        if self._ik_left > 0:
            self._ik_left -= 1
            self._q_goal = self._solve_ik(*self._ik_target, self._q_goal, iterations=1)
            if self._ik_converged:
                self._ik_left = 0
            remaining = max(1, self._interp_ticks - (self.step_count - self._last_target_step))
            self._q_rate = np.clip(np.abs(self._q_goal - self._q_cmd) / (remaining * self.dt),
                                   0.0, self.MAX_JOINT_SPEED)
        q = d.qpos[self._qadr]
        max_dq = self._q_rate * self.dt
        dq = np.clip(self._q_goal - self._q_cmd, -max_dq, max_dq)
        self._q_cmd = self._q_cmd + dq
        qd_cmd = dq / self.dt
        # Following-error limit: the position error fed to the PD law is bounded, so a blocked arm (keep-out
        # fixture) pushes with a bounded force. The bound acts on the error only, never on the planned command:
        # clamping the command to the arm would drag it past the goal whenever the arm overshoots and turn the
        # overshoot into a sustained, torque-saturated limit cycle (seen in review, DEC-052 addendum).
        err = np.clip(self._q_cmd - q, -self.MAX_FOLLOWING_ERROR, self.MAX_FOLLOWING_ERROR)
        self._acc[:] = 0.0
        self._acc[self._dofs] = self.KP * err + self.KD * (qd_cmd - d.qvel[self._dofs])
        mujoco.mj_mulM(m, d, self._tau, self._acc)
        tau = np.clip(self._tau[self._dofs] + d.qfrc_bias[self._dofs], -self._tau_max, self._tau_max)
        d.qfrc_applied[:] = self._ext
        d.qfrc_applied[self._dofs] += tau
        self._ext[:] = 0.0
        mujoco.mj_step2(m, d)
        self.sim_time += self.dt
        self.step_count += 1

    # ------------------------------------------------------------ engine hooks
    def _probe_centre(self) -> Optional[np.ndarray]:
        if self.data is None:
            return None
        return self.data.xpos[self._probe_id].copy()

    def _apply_probe_force(self, force: np.ndarray, point: np.ndarray) -> None:
        mujoco.mj_applyFT(self.model, self.data, np.asarray(force, dtype=np.float64), self._zero3,
                          np.asarray(point, dtype=np.float64), self._probe_id, self._ext)

    def _probe_contacts(self):
        n, force, min_d = 0, 0.0, 0.0
        d = self.data
        if d is None:
            return n, force, min_d
        wrench = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 != self._probe_geom and con.geom2 != self._probe_geom:
                continue
            n += 1
            mujoco.mj_contactForce(self.model, d, i, wrench)
            force += float(wrench[0])
            min_d = min(min_d, float(con.dist))
        return n, force, min_d

    def _ee_position(self) -> np.ndarray:
        return self.data.xpos[self._ee_id]

    def _ee_orientation_xyzw(self):
        return _wxyz_to_xyzw(self.data.xquat[self._ee_id])

    def _create_fixture(self, pb_vertices: np.ndarray, indices: list):
        """Static keep-out shell: one thin convex prism per triangle (see module doc); needs a recompile."""
        name = f"fixture_{len(self.fixture_ids)}"
        v = np.asarray(pb_vertices, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        if len(tris) > self.MAX_FIXTURE_TRIANGLES:
            print(f"[!] VirtualFixture has {len(tris)} triangles (> {self.MAX_FIXTURE_TRIANGLES}); ignored.")
            return None
        added = []
        half = 0.5 * self.FIXTURE_SHELL_M
        for k, (a, b, c) in enumerate(v[tris]):
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-12:            # degenerate triangle
                continue
            n *= half / norm
            prism = np.vstack([a + n, b + n, c + n, a - n, b - n, c - n])
            mesh = self.spec.add_mesh(name=f"{name}_{k}", uservert=prism.ravel().tolist())
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom = self.spec.worldbody.add_geom(name=f"{name}_{k}_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
                                                meshname=f"{name}_{k}", rgba=[1.0, 0.0, 0.0, 0.3], contype=0,
                                                conaffinity=_ENV_AFFINITY)
            added += [geom, mesh]
        if not added:
            print("[!] VirtualFixture has only degenerate triangles; ignored.")
            return None
        try:
            self._recompile()
        except Exception as exc:
            print(f"[!] VirtualFixture rejected by MuJoCo ({str(exc).splitlines()[0]}); ignored.")
            for el in added:
                self.spec.delete(el)
            self._recompile()
            return None
        return name

    def _recompile(self) -> None:
        """Rebuild the model from the spec, carrying the simulation state over."""
        if self._renderer is not None:      # bound to the old compiled model
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self.model, self.data = self.spec.recompile(self.model, self.data)
        self._bind()
        mujoco.mj_forward(self.model, self.data)

    def view_snapshot(self) -> list:
        d = self.data
        return (d.qpos[self._qadr].tolist() + d.xpos[self._probe_id].tolist()
                + _wxyz_to_xyzw(d.xquat[self._probe_id]))

    # ------------------------------------------------------------- viewer side
    def _launch_viewer(self) -> None:
        import mujoco.viewer   # local import: pulls in glfw / OpenGL, viewer process only
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        lookat, distance, azimuth, elevation = self._camera     # kept across a viewer restart
        self.viewer.cam.lookat[:] = lookat
        self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = distance, azimuth, elevation
        self.viewer.opt.geomgroup[1] = 1

    def set_view(self, snap: np.ndarray) -> None:
        """Viewer process: mirror the physics loop's joints (the probe follows by forward kinematics)."""
        self.data.qpos[self._qadr] = snap[:self.N_ARM]
        mujoco.mj_kinematics(self.model, self.data)

    def viewer_sync(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def focus_camera(self, target_pb, distance_m: float = VIEW_CLOSEUP_DISTANCE_M) -> None:
        """Close-up of the scan site (initial view only: the user can still orbit and zoom)."""
        self._camera = (np.asarray(target_pb, dtype=np.float64).tolist(), 2.5 * distance_m,
                        90.0 + VIEW_CLOSEUP_YAW_DEG, VIEW_CLOSEUP_PITCH_DEG)
        if self.viewer is None:
            return
        lookat, distance, azimuth, elevation = self._camera
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = lookat
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = distance, azimuth, elevation

    def update_skin(self, vertices_pb: np.ndarray) -> None:
        """Deformed skin patch. The mesh is allocated once (one recompile and viewer restart, viewer process
        only); afterwards its vertex buffer is overwritten in place and re-uploaded."""
        if self._skin_indices is None or self.viewer is None:
            return
        verts = np.asarray(vertices_pb, dtype=np.float64).reshape(-1, 3)
        if self._skin_mesh_id < 0:
            if self._skin_mesh_id == -2:       # allocation failed before: do not retry every frame
                return
            self._allocate_skin(verts)
            if self._skin_mesh_id < 0:
                return
        m = self.model
        # Compiled mesh vertices live in the geom frame (MuJoCo recentres meshes at compile time).
        local = (verts - self.data.geom_xpos[self._skin_geom_id]) @ self.data.geom_xmat[self._skin_geom_id].reshape(3, 3)
        adr = m.mesh_vertadr[self._skin_mesh_id]
        with self.viewer.lock():
            m.mesh_vert[adr:adr + self._skin_nvert] = local
        self.viewer.update_mesh(self._skin_mesh_id)

    def _allocate_skin(self, verts: np.ndarray) -> None:
        mesh = self.spec.add_mesh(name="skin_patch", uservert=verts.ravel().tolist(), userface=self._skin_indices)
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = self.spec.worldbody.add_geom(name="skin_patch_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
                                            meshname="skin_patch", rgba=list(SKIN_RGBA), contype=0, conaffinity=0)
        self._close_viewer()     # the viewer thread renders from the model that is about to be replaced
        try:
            self._recompile()
            mesh_id = self.model.mesh("skin_patch").id
            if self.model.mesh_vertnum[mesh_id] != len(verts):
                raise ValueError("MuJoCo changed the vertex count of the skin mesh")
            self._skin_mesh_id = mesh_id
            self._skin_geom_id = self.model.geom("skin_patch_geom").id
            self._skin_nvert = len(verts)
        except Exception as exc:
            print(f"[!] Skin patch not shown in the MuJoCo viewer ({str(exc).splitlines()[0]}).")
            self.spec.delete(geom)
            self.spec.delete(mesh)
            self._recompile()
            self._skin_mesh_id = -2
        self._launch_viewer()

    def draw_overlays(self, target: dict) -> None:
        """Force arrow (against the beam) and the image-plane outline, as decorative scene geoms."""
        if self.viewer is None:
            return
        contact = np.asarray(target["contact_pb"], dtype=np.float64)
        beam = np.asarray(target["beam_pb"], dtype=np.float64)
        lateral = np.asarray(target.get("lateral_pb", [1.0, 0.0, 0.0]), dtype=np.float64)
        fn = float(target.get("force_n", 0.0))
        color = (0.1, 0.95, 0.3, 1.0) if fn <= 6.0 else ((1.0, 0.75, 0.1, 1.0) if fn <= 8.0 else (1.0, 0.15, 0.1, 1.0))
        half_w, depth = 0.018, 0.06
        tl, tr = contact - lateral * half_w, contact + lateral * half_w
        bl, br = tl + beam * depth, tr + beam * depth
        cyan = (0.0, 0.85, 1.0, 1.0)
        segments = [(contact, contact - beam * max(fn, 1e-3) * 0.015, color, 0.004),
                    (tl, bl, cyan, 0.0015), (tr, br, cyan, 0.0015), (bl, br, cyan, 0.0015)]
        scn = self.viewer.user_scn
        with self.viewer.lock():
            scn.ngeom = 0
            for a, b, rgba, width in segments:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
                scn.ngeom += 1
        hud = target.get("hud")
        if hud:
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud, ""))

    def save_snapshot(self, out_dir: str, width: int = 1280, height: int = 800) -> None:
        """Offscreen render of the viewer's camera to ``out_dir/viewer_<unix ms>.png`` (best effort)."""
        if self._snapshot_failed or self.viewer is None:
            return
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer.update_scene(self.data, camera=self.viewer.cam)
            rgb = self._renderer.render()
            from PIL import Image  # local import: only needed when snapshots are enabled
            Image.fromarray(rgb).save(os.path.join(out_dir, f"viewer_{int(time.time() * 1e3)}.png"))
        except Exception as exc:
            self._snapshot_failed = True
            print(f"[!] MuJoCo viewer snapshots unavailable ({str(exc).splitlines()[0]}).")

    def _close_viewer(self) -> None:
        """Close the passive viewer and wait for its render thread to stop using the model."""
        if self.viewer is None:
            return
        self.viewer.close()
        deadline = time.perf_counter() + 3.0
        while self.viewer.is_running() and time.perf_counter() < deadline:
            time.sleep(0.02)
        # launch_passive exposes no handle to its render thread, so there is nothing to join: after
        # is_running() goes false the thread may still be inside its last frame. Wait it out.
        time.sleep(0.5)
        self.viewer = None

    def disconnect(self) -> None:
        try:
            if self._renderer is not None:
                self._renderer.close()
            self._close_viewer()
        except Exception:
            pass
