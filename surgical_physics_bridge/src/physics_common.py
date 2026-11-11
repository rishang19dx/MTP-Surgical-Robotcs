"""
Engine-independent part of the 1 kHz physics loop (DEC-052).

This module holds everything the rigid-body backends share and imports no physics engine:

    * ``PhysicsChannels`` / ``LatestSnapshot``: IPC between the streaming side and the physics loop.
    * ``PhysicsBackendBase``: the tick (drain commands, latest target wins, step, publish snapshots), the
      tissue force law and virtual-fixture validation. An engine implements a handful of hooks.
    * ``run_physics_loop``, CPU pinning, the viewer loop and the process entry points.
    * ``create_backend(name, **config)``: lazy import of ``pybullet_backend`` or ``mujoco_backend``.

Threading / process model (invariant 2, clock decoupling):
    * Every engine call of the simulation is made by the *physics loop* only (``run_physics_loop``), which
      runs either in its own process (default, ``physics_process_main``) or, for tests and comparison,
      in-process.
    * The GUI is a separate *viewer process* (``viewer_process_main``) with its own engine instance: it
      mirrors the robot joints / probe pose from the physics loop at ~60 Hz, draws the overlays and renders
      the deformed skin patch. Rendering cost never lands on the 1 kHz loop.
    * The streaming side never touches an engine. It talks to both through ``PhysicsChannels``:
        - ``commands``: queue of ``("target", dict)`` / ``("fixture", dict)`` messages
          (streaming -> physics; the loop drains it each tick, latest target wins).
        - ``contact``: latest-value snapshot written by the physics loop every tick, see ``CONTACT_FIELDS``.
        - ``view``: latest joint / probe pose snapshot (physics -> viewer), see ``VIEW_FIELDS``.
        - ``viewer``: queue of ``("target", dict)`` / ``("skin_mesh", dict)`` / ``("skin", array)``.
        - ``rate``: lock-free ring buffers of ``PhysicsRateMonitor`` (physics -> streaming).

World frame of every backend: the "PB" frame of ``coordinates.py`` (Z-up, metres). Quaternions cross the
backend boundary as (x, y, z, w).

Probe-tissue contact (DEC-037, DEC-038): the torso is visual-only. The target carries the latest tissue
solution ``tissue = {contact_pb, beam_pb, force_n, stiffness_n_per_mm, indentation_mm, in_contact}`` from
the streaming side (~30 Hz). Every tick the loop measures the probe-face indentation delta along the beam
from the engine's own probe pose and applies ``F = F0 + K_c (delta - delta0)`` (clipped to
[0, ``TISSUE_FORCE_MAX_N``]) to the probe against the beam. The loop never waits on the tissue solver.
"""

import importlib
import math
import os
import queue
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from coordinates import ras_to_pybullet_points
    from physics_rate import PhysicsRateMonitor
except ImportError:
    from surgical_physics_bridge.src.coordinates import ras_to_pybullet_points
    from surgical_physics_bridge.src.physics_rate import PhysicsRateMonitor

#: Rigid-body engines selectable with ``--physics-backend``.
BACKENDS = ("pybullet", "mujoco")

#: Fields of the contact snapshot published by the physics loop every tick.
CONTACT_FIELDS = (
    "seq",               # physics tick counter at the time of the snapshot
    "sim_time_s",        # simulated time
    "n_contacts",        # probe contacts with rigid bodies (table, virtual keep-out fixtures)
    "normal_force_n",    # summed engine normal force of those contacts (rigid collision, NOT tissue force)
    "min_distance_m",    # most negative contact distance (penetration), 0 if no contact
    "ee_error_m",        # distance between commanded and achieved end-effector position
    "tissue_force_n",    # extrapolated tissue reaction force applied to the probe this tick
    "tissue_indentation_mm",  # probe face indentation along the beam measured from the engine's probe pose
    "ee_orn_error_rad",  # angle between commanded and achieved end-effector orientation
)
#: Latest robot / probe pose for the viewer process: 7 arm joints, probe position (3), quaternion xyzw (4).
VIEW_FIELDS = 14
TISSUE_FORCE_MAX_N = 40.0
SKIN_VIEW_HZ = 10.0
SKIN_RGBA = [0.93, 0.62, 0.52, 0.95]
# Viewer close-up on the skin patch (set once when the patch mesh arrives).
VIEW_CLOSEUP_DISTANCE_M = 0.24
VIEW_CLOSEUP_YAW_DEG = 20.0
VIEW_CLOSEUP_PITCH_DEG = -32.0

PROBE_LENGTH_M = 0.09       # end-effector mount -> transducer face along the beam
PROBE_RADIUS_M = 0.014
PROBE_MASS_KG = 0.2
ROBOT_BASE_Z = 0.30         # pedestal height (table-top level)
#: Franka Panda "ready" posture: IK seed and initial configuration of both backends.
ARM_REST_POSTURE = (0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79)
TABLE_HALF_EXTENTS = (0.42, 0.30, 0.15)
TABLE_CENTRE = (0.55, 0.0, 0.15)


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


class PhysicsBackendBase:
    """
    Shared behaviour of the rigid-body backends. Use only from the physics loop (or the viewer process).

    Engine hooks a subclass must provide:
        ``apply_target(target)``      track ``ee_pos`` / ``ee_orn`` (xyzw), honour ``teleport``; call
                                      ``_store_target(target)`` for the bookkeeping.
        ``_probe_centre()``           probe body centre in the world (m), or None before the scene exists.
        ``_apply_probe_force(f, p)``  world-frame force ``f`` (N) on the probe at world point ``p`` for the
                                      next step only.
        ``_step_engine()``            advance the simulation by ``dt``.
        ``_probe_contacts()``         (count, summed normal force N, most negative distance m) of the probe's
                                      rigid contacts.
        ``_ee_position()``            end-effector position (m).
        ``_ee_orientation_xyzw()``    end-effector orientation, quaternion (x, y, z, w).
        ``_create_fixture(v, tris)``  static keep-out collision body from PB-frame vertices (m).
        ``view_snapshot()``           7 joint angles + probe position + probe quaternion (xyzw).
        ``disconnect()``
    Viewer-side hooks (only with ``gui=True``): ``set_view``, ``update_skin``, ``focus_camera``,
    ``draw_overlays``, ``save_snapshot``, and optionally ``viewer_sync``.
    """

    PROBE_LENGTH_M = PROBE_LENGTH_M
    ROBOT_BASE_Z = ROBOT_BASE_Z

    def _init_common(self, gui: bool, dt: float) -> None:
        self.gui = gui
        self.dt = dt
        self.sim_time = 0.0
        self.step_count = 0
        self.fixture_ids = []
        self._ee_target: Optional[np.ndarray] = None
        self._ee_target_quat: Optional[tuple] = None   # commanded orientation, xyzw, unit norm
        self._tissue: Optional[dict] = None
        self.tissue_force_n = 0.0
        self.tissue_indentation_mm = 0.0
        self._skin_indices: Optional[list] = None

    # ---------------------------------------------------------------- commands
    def _store_target(self, target: dict) -> None:
        """Record the commanded end-effector position and the tissue solution carried by a target."""
        self._ee_target = np.asarray(target["ee_pos"], dtype=np.float64)
        q = [float(v) for v in target["ee_orn"]]
        norm = math.sqrt(sum(v * v for v in q))
        self._ee_target_quat = tuple(v / norm for v in q)
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
        centre = self._probe_centre() if tis is not None else None
        if centre is None:
            self.tissue_force_n = 0.0
            return
        beam = tis["beam"]
        face = centre + beam * (self.PROBE_LENGTH_M / 2.0)
        delta = 1000.0 * float(np.dot(face - tis["contact"], beam))
        f = tis["F0"] + tis["K"] * (delta - tis["d0"]) if tis["in_contact"] else 0.0
        f = min(max(float(f), 0.0), TISSUE_FORCE_MAX_N)     # plain floats: this runs every 1 kHz tick
        if f > 0.0:
            self._apply_probe_force(-beam * f, face)
        self.tissue_force_n = f
        self.tissue_indentation_mm = delta

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
        try:
            tris = [int(i) for i in indices]
            pb_vertices = ras_to_pybullet_points(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
        except (TypeError, ValueError):
            print("[!] Invalid VirtualFixture: 'vertices' must be N x 3 numbers and 'indices' integers.")
            return
        if len(tris) % 3 != 0 or min(tris) < 0 or max(tris) >= len(pb_vertices):
            print("[!] Invalid VirtualFixture: 'indices' must be whole triangles of valid vertex indices.")
            return
        fixture = self._create_fixture(pb_vertices, tris)
        if fixture is None:          # the engine rejected the mesh
            return
        self.fixture_ids.append(fixture)
        print(f"[+] Virtual keep-out fixture created ({len(pb_vertices)} vertices, {len(tris) // 3} faces).")

    def set_skin_mesh(self, indices) -> None:
        """Triangle indices of the skin patch (vertex order of the streamed skin vertices)."""
        tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        # P -> PB is a reflection (det -1): reverse the winding so the outward faces stay front-facing.
        self._skin_indices = tris[:, ::-1].ravel().tolist()

    # -------------------------------------------------------------------- loop
    def step(self) -> None:
        self.apply_tissue_force()   # per-step external force: re-applied every tick
        self._step_engine()
        self.sim_time += self.dt
        self.step_count += 1

    def contact_snapshot(self) -> list:
        n, force, min_d = self._probe_contacts()
        ee_err = orn_err = 0.0
        if self._ee_target is not None:
            ee_err = float(np.linalg.norm(np.asarray(self._ee_position()) - self._ee_target))
            qc, qt = self._ee_orientation_xyzw(), self._ee_target_quat
            dot = abs(sum(float(a) * b for a, b in zip(qc, qt)))     # plain floats: runs every tick
            orn_err = 2.0 * math.acos(min(1.0, dot))
        return [float(self.step_count), self.sim_time, float(n), float(force), float(min_d), ee_err,
                self.tissue_force_n, self.tissue_indentation_mm, orn_err]

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

    # ------------------------------------------------------------ engine hooks
    def _probe_centre(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _apply_probe_force(self, force: np.ndarray, point: np.ndarray) -> None:
        raise NotImplementedError

    def _step_engine(self) -> None:
        raise NotImplementedError

    def _probe_contacts(self) -> Tuple[int, float, float]:
        raise NotImplementedError

    def _ee_position(self) -> np.ndarray:
        raise NotImplementedError

    def _ee_orientation_xyzw(self):
        raise NotImplementedError

    def _create_fixture(self, pb_vertices: np.ndarray, indices: list):
        """Return an id, or None if the engine rejected the mesh (the fixture is then not recorded)."""
        raise NotImplementedError

    def viewer_sync(self) -> None:
        """Called once per viewer frame after the state was mirrored (no-op unless the engine needs it)."""


def create_backend(name: str, **config) -> PhysicsBackendBase:
    """Build the named rigid-body backend. The engine module is imported here, never at module level."""
    if name not in BACKENDS:
        raise ValueError(f"physics backend must be one of {BACKENDS}, got {name!r}")
    # Import the sibling module under the same package this module was imported as, so that the backend
    # subclasses THIS module's PhysicsBackendBase (the package supports both import styles).
    package = __name__.rpartition(".")[0]
    module = importlib.import_module(f"{package}.{name}_backend" if package else f"{name}_backend")
    cls = {"pybullet": "PyBulletBackend", "mujoco": "MujocoBackend"}[name]
    return getattr(module, cls)(**config)


def run_physics_loop(
    backend,
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


def run_viewer(backend, channels: PhysicsChannels, fps: float = 60.0,
               skin_hz: float = SKIN_VIEW_HZ, log_interval_s: float = 10.0) -> None:
    """Viewer loop (GUI instance): mirror the physics state, draw overlays, update the skin mesh."""
    period = 1.0 / fps
    last_skin = 0.0
    pending_skin = None
    skin_ms = []
    last_log = time.perf_counter()
    # Opt-in debug snapshots (viewer process only, never the physics loop): on Wayland an
    # external tool cannot capture the GUI window, so the viewer renders its own view.
    snapshot_dir = os.environ.get("MTP_VIEWER_SNAPSHOT_DIR")
    snapshot_interval_s = float(os.environ.get("MTP_VIEWER_SNAPSHOT_S", "5"))
    last_snapshot = 0.0
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
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
                backend.focus_camera(pending_skin.mean(axis=0))
            elif kind == "skin":
                pending_skin = payload
        snap = channels.view.read()
        if np.any(snap[10:14]):
            backend.set_view(snap)
        if target is not None:
            backend.draw_overlays(target)
        if pending_skin is not None and t0 - last_skin >= 1.0 / skin_hz:
            ts = time.perf_counter()
            backend.update_skin(pending_skin)
            skin_ms.append(1e3 * (time.perf_counter() - ts))
            pending_skin = None
            last_skin = t0
        backend.viewer_sync()
        if snapshot_dir and t0 - last_snapshot >= snapshot_interval_s:
            last_snapshot = t0
            backend.save_snapshot(snapshot_dir)
        if skin_ms and t0 - last_log >= log_interval_s:
            last_log = t0
            print(f"[viewer] skin mesh update: median {np.median(skin_ms):.2f} ms, max {np.max(skin_ms):.2f} ms "
                  f"over {len(skin_ms)} updates (viewer process, not the physics loop)")
            skin_ms = skin_ms[-500:]
        sleep = period - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def _split_config(config: dict) -> Tuple[str, dict]:
    config = dict(config)
    return config.pop("backend", "pybullet"), config


def viewer_process_main(config: dict, channels: PhysicsChannels, fps: float = 60.0) -> None:
    """Entry point of the GUI viewer process (spawned by the unified server with ``--gui``)."""
    backend = None
    try:
        name, cfg = _split_config(config)
        backend = create_backend(name, **dict(cfg, gui=True))
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
        name, cfg = _split_config(config)
        backend = create_backend(name, **cfg)
        channels.ready.set()
        run_physics_loop(backend, channels, target_hz=target_hz)
    except KeyboardInterrupt:
        pass
    finally:
        channels.ready.set()  # never leave the parent waiting if scene construction failed
        if backend is not None:
            backend.disconnect()
