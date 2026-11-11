"""
Contract tests for the rigid-body physics backends of the unified server (DEC-052).

Every backend named in ``physics_common.BACKENDS`` must satisfy the same contract, because the 1 kHz loop,
the streaming side and the Slicer bridge are shared. The world frame of every backend is the "PB" frame
of ``coordinates.py`` (Z-up, metres); quaternions cross the backend boundary as (x, y, z, w).

The tests are written against behaviour (poses, forces and contacts read back through the public
snapshots), not against engine calls, so they hold for PyBullet and MuJoCo alike.
"""

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from surgical_physics_bridge.src.coordinates import (
    patient_to_ras_points,
    pybullet_to_patient_points,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATIENT_DIR = os.path.join(REPO_ROOT, "data", "sample_totalsegmentator_patient")
BACKEND_NAMES = ("pybullet", "mujoco")

# A reachable scan pose above the dorsal skin, probe pointing straight down (PB frame, metres).
CONTACT_PB = np.array([0.55, -0.03, 0.44])
BEAM_PB = np.array([0.0, 0.0, -1.0])
LATERAL_PB = np.array([1.0, 0.0, 0.0])


def _common():
    import surgical_physics_bridge.src.physics_common as pc
    return pc


def _target(contact_pb=CONTACT_PB, beam_pb=BEAM_PB, lateral_pb=LATERAL_PB, teleport=False, tissue=None):
    """Target message as built by ``unified_server._send_target`` (beam/lateral already in PB)."""
    pc = _common()
    # probe_axes_to_pybullet_quaternion takes frame-P axes; build the same proper rotation directly in PB:
    # EE +Z = beam, EE +X = lateral.
    z = beam_pb / np.linalg.norm(beam_pb)
    x = lateral_pb - z * float(lateral_pb @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat_xyzw = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    target = {
        "ee_pos": (contact_pb - z * pc.PROBE_LENGTH_M).tolist(),
        "ee_orn": quat_xyzw.tolist(),
        "contact_pb": contact_pb.tolist(),
        "beam_pb": z.tolist(),
        "lateral_pb": x.tolist(),
        "teleport": teleport,
        "force_n": 0.0,
    }
    if tissue is not None:
        target["tissue"] = tissue
    return target


def _tissue(force_n, stiffness, indentation_mm, in_contact=True, contact_pb=CONTACT_PB, beam_pb=BEAM_PB):
    return {"contact_pb": contact_pb.tolist(), "beam_pb": beam_pb.tolist(), "force_n": force_n,
            "stiffness_n_per_mm": stiffness, "indentation_mm": indentation_mm, "in_contact": in_contact}


def _run(backend, channels, ticks, target=None):
    if target is not None:
        channels.commands.put(("target", target))
    for _ in range(ticks):
        backend.tick(channels)
    return dict(zip(_common().CONTACT_FIELDS, channels.contact.read().tolist()))


def _probe_face(backend, channels, beam_pb=BEAM_PB):
    """Probe face position (PB, m) from the public view snapshot: probe centre + L/2 along the beam."""
    snap = np.asarray(backend.view_snapshot(), dtype=np.float64)
    return snap[7:10] + beam_pb * (_common().PROBE_LENGTH_M / 2.0)


@pytest.fixture(params=BACKEND_NAMES)
def backend_and_channels(request):
    pc = _common()
    backend = pc.create_backend(request.param, gui=False, patient_dir=PATIENT_DIR, urdf_path=None, dt=0.001)
    channels = pc.PhysicsChannels(rate_window=2000)
    yield backend, channels
    backend.disconnect()


# --------------------------------------------------------------------------- module boundaries
def test_physics_common_imports_no_engine():
    """The shared loop / IPC module must be importable without loading any physics engine."""
    code = ("import sys; import surgical_physics_bridge.src.physics_common; "
            "print(int('pybullet' in sys.modules), int('mujoco' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["0", "0"]


def test_streaming_side_never_imports_an_engine():
    """Invariant 2: only the physics loop calls the engine. The streaming module must not even load one."""
    import surgical_physics_bridge.src.unified_server as us
    src = inspect.getsource(us)
    assert "import pybullet" not in src and "import mujoco" not in src
    code = ("import sys; import surgical_physics_bridge.src.unified_server; "
            "print(int('pybullet' in sys.modules), int('mujoco' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["0", "0"]


def test_backend_registry_names_both_engines_and_rejects_unknown():
    pc = _common()
    assert set(BACKEND_NAMES) <= set(pc.BACKENDS)
    with pytest.raises(ValueError):
        pc.create_backend("bullet3d", gui=False, patient_dir=None, urdf_path=None, dt=0.001)


# --------------------------------------------------------------------------- pose tracking
def test_teleport_then_hold_tracks_the_end_effector_target(backend_and_channels):
    backend, channels = backend_and_channels
    c = _run(backend, channels, 300, _target(teleport=True))
    assert c["seq"] == 300
    assert c["sim_time_s"] == pytest.approx(0.3, abs=1e-9)
    assert c["ee_error_m"] < 0.002
    # the probe face (EE + L along the beam) sits on the commanded contact point
    assert np.linalg.norm(_probe_face(backend, channels) - CONTACT_PB) < 0.003


def test_probe_orientation_matches_the_commanded_quaternion_xyzw(backend_and_channels):
    """Guards the quaternion convention at the boundary (MuJoCo is w-first internally)."""
    backend, channels = backend_and_channels
    beam = np.array([0.17, 0.0, -0.985])
    beam /= np.linalg.norm(beam)
    target = _target(beam_pb=beam, teleport=True)
    _run(backend, channels, 300, target)
    snap = np.asarray(backend.view_snapshot(), dtype=np.float64)
    assert len(snap) == _common().VIEW_FIELDS
    quat = snap[10:14]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)
    err = Rotation.from_quat(quat) * Rotation.from_quat(target["ee_orn"]).inv()
    assert np.degrees(err.magnitude()) < 2.0


def test_moving_target_is_followed_without_teleport(backend_and_channels):
    """A 30 Hz stream of 0.33 mm steps (10 mm/s scan speed) is tracked by the 1 kHz loop."""
    backend, channels = backend_and_channels
    _run(backend, channels, 200, _target(teleport=True))
    contact = CONTACT_PB.copy()
    for _ in range(30):
        contact = contact + np.array([0.00033, 0.0, 0.0])
        c = _run(backend, channels, 33, _target(contact_pb=contact))
    assert c["ee_error_m"] < 0.003
    assert np.linalg.norm(_probe_face(backend, channels) - contact) < 0.004


def test_tilting_sweep_is_tracked(backend_and_channels):
    """A 30 Hz target stream that translates and tilts the probe about the contact point (peak 0.67 rad/s,
    about 6 cm/s at the end effector) is followed within 1 cm throughout, not only at the end. The bound is
    the server's existing "while moving" bound (test_server.py); each new target alone steps the error by
    about 2 mm."""
    backend, channels = backend_and_channels
    _run(backend, channels, 200, _target(teleport=True))
    contact = CONTACT_PB.copy()
    worst = 0.0
    for i in range(90):
        contact = contact + np.array([0.00033, 0.0002, 0.0])
        tilt = 0.2 * np.sin(33 * i / 300.0)
        beam = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
        c = _run(backend, channels, 33, _target(contact_pb=contact, beam_pb=beam))
        if i >= 3:
            worst = max(worst, c["ee_error_m"])
    print(f"[{type(backend).__name__}] worst EE error in the tilting sweep: {1e3 * worst:.2f} mm")
    assert worst < 0.01


def test_backends_agree_on_the_probe_pose_for_the_same_target():
    """Same URDF, same target: the two engines must place the probe at the same world pose."""
    pc = _common()
    poses = {}
    for name in BACKEND_NAMES:
        backend = pc.create_backend(name, gui=False, patient_dir=PATIENT_DIR, urdf_path=None, dt=0.001)
        channels = pc.PhysicsChannels(rate_window=2000)
        try:
            _run(backend, channels, 300, _target(teleport=True))
            poses[name] = np.asarray(backend.view_snapshot(), dtype=np.float64)[7:14]
        finally:
            backend.disconnect()
    a, b = poses["pybullet"], poses["mujoco"]
    assert np.linalg.norm(a[:3] - b[:3]) < 0.003
    rel = Rotation.from_quat(a[3:]) * Rotation.from_quat(b[3:]).inv()
    assert np.degrees(rel.magnitude()) < 2.0


# --------------------------------------------------------------------------- tissue force (DEC-038 item 3)
@pytest.mark.parametrize("f0, k, d0", [(4.0, 0.8, 3.0), (2.0, 0.0, 0.0), (6.0, 2.5, 5.0)])
def test_tissue_force_follows_the_first_order_extrapolation(backend_and_channels, f0, k, d0):
    """F = clip(F0 + K_c (delta - delta0), 0, max), with delta measured from the engine's own probe pose."""
    backend, channels = backend_and_channels
    pc = _common()
    skin = CONTACT_PB - BEAM_PB * (d0 * 1e-3)        # undeformed skin point: face is d0 mm past it
    target = _target(teleport=True, tissue=_tissue(f0, k, d0, contact_pb=skin))
    c = _run(backend, channels, 300, target)
    expect = float(np.clip(f0 + k * (c["tissue_indentation_mm"] - d0), 0.0, pc.TISSUE_FORCE_MAX_N))
    assert c["tissue_force_n"] == pytest.approx(expect, rel=1e-6, abs=1e-6)
    # the reported indentation is the measured one: face position along the beam from the skin point
    measured = 1000.0 * float((_probe_face(backend, channels) - skin) @ BEAM_PB)
    assert c["tissue_indentation_mm"] == pytest.approx(measured, abs=0.05)
    assert abs(c["tissue_indentation_mm"] - d0) < 3.0


def test_tissue_force_is_zero_out_of_contact_and_clipped_at_the_maximum(backend_and_channels):
    backend, channels = backend_and_channels
    pc = _common()
    c = _run(backend, channels, 100, _target(teleport=True, tissue=_tissue(4.0, 1.0, 0.0, in_contact=False)))
    assert c["tissue_force_n"] == 0.0
    c = _run(backend, channels, 5, _target(tissue=_tissue(1e4, 0.0, 0.0)))
    assert c["tissue_force_n"] == pytest.approx(pc.TISSUE_FORCE_MAX_N)


def test_tissue_force_really_pushes_the_robot_back(backend_and_channels, request):
    """The force must act on the dynamics, not only be reported: a constant push against the beam moves
    the probe face backwards along the beam relative to the unloaded hold.

    PyBullet is a strict xfail: its position motors are velocity constraints with a 150 N limit, so the arm
    is effectively kinematic (measured 0.4 um of deflection under 30 N). This is the "kinematic robot"
    caveat of DEC-048; the torque-controlled MuJoCo arm is the backend that removes it."""
    backend, channels = backend_and_channels
    if type(backend).__name__ == "PyBulletBackend":
        request.applymarker(pytest.mark.xfail(strict=True, reason="PyBullet position motors: kinematic arm"))
    _run(backend, channels, 600, _target(teleport=True))
    unloaded = float(_probe_face(backend, channels) @ BEAM_PB)
    _run(backend, channels, 600, _target(tissue=_tissue(30.0, 0.0, 0.0)))
    loaded = float(_probe_face(backend, channels) @ BEAM_PB)
    print(f"[{type(backend).__name__}] deflection under 30 N: {1e6 * (unloaded - loaded):.1f} um")
    assert unloaded - loaded > 1e-5      # > 10 um of deflection against the beam under 30 N


# --------------------------------------------------------------------------- virtual keep-out fixture
def _box_fixture_ras(centre_pb, half_m):
    """Closed box (8 vertices, 12 triangles) as the Slicer VirtualFixture JSON: vertices in RAS mm."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    verts_pb = centre_pb + signs * half_m
    verts_ras = patient_to_ras_points(pybullet_to_patient_points(verts_pb))
    tris = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
            2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return {"vertices": verts_ras.tolist(), "indices": tris, "policy": "keep_out"}


def test_virtual_fixture_blocks_the_probe(backend_and_channels):
    backend, channels = backend_and_channels
    start = CONTACT_PB + np.array([0.0, 0.0, 0.06])
    c = _run(backend, channels, 200, _target(contact_pb=start, teleport=True))
    assert c["n_contacts"] == 0 and c["normal_force_n"] == 0.0
    # a 6 cm keep-out box whose top face is 3 cm below the probe face, then command the face 2 cm into it
    box_centre = start + BEAM_PB * 0.06
    channels.commands.put(("fixture", _box_fixture_ras(box_centre, 0.03)))
    goal = start + BEAM_PB * 0.05
    steps = 50
    for i in range(1, steps + 1):
        c = _run(backend, channels, 20, _target(contact_pb=start + (goal - start) * i / steps))
    c = _run(backend, channels, 300)
    assert c["n_contacts"] >= 1
    assert c["normal_force_n"] > 1.0
    # the fixture holds: the face stays out of the box (at most 5 mm of penetration)
    depth_into_box = float((_probe_face(backend, channels) - (box_centre - BEAM_PB * 0.03)) @ BEAM_PB)
    assert depth_into_box < 0.005
    assert c["ee_error_m"] > 0.01          # the robot did NOT reach the commanded pose inside the box


def test_invalid_fixture_is_ignored(backend_and_channels):
    backend, channels = backend_and_channels
    channels.commands.put(("fixture", {"vertices": [], "indices": []}))
    channels.commands.put(("fixture", dict(_box_fixture_ras(CONTACT_PB, 0.01), policy="keep_in")))
    c = _run(backend, channels, 50, _target(contact_pb=CONTACT_PB + np.array([0.0, 0.0, 0.06]), teleport=True))
    assert c["n_contacts"] == 0


# --------------------------------------------------------------------------- 1 kHz budget (invariant 2)
@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_median_tick_fits_the_one_millisecond_budget(name):
    """Median in-process tick time with a 30 Hz target stream. The full-rate figure (p99, overruns) comes
    from ``unified_server.py --benchmark-seconds`` into ``results/``; this guards against gross regressions."""
    import time
    pc = _common()
    backend = pc.create_backend(name, gui=False, patient_dir=PATIENT_DIR, urdf_path=None, dt=0.001)
    channels = pc.PhysicsChannels(rate_window=2000)
    try:
        _run(backend, channels, 200, _target(teleport=True, tissue=_tissue(4.0, 0.8, 0.0)))
        times = []
        for i in range(1000):
            if i % 33 == 0:
                channels.commands.put(("target", _target(tissue=_tissue(4.0, 0.8, 0.0))))
            t0 = time.perf_counter()
            backend.tick(channels)
            times.append(time.perf_counter() - t0)
    finally:
        backend.disconnect()
    median_ms = 1e3 * float(np.median(times))
    print(f"[{name}] tick median {median_ms:.3f} ms, p99 {1e3 * float(np.percentile(times, 99)):.3f} ms")
    assert median_ms < 1.0


# --------------------------------------------------------------------------- server integration
@pytest.fixture(scope="module")
def mujoco_inline_server():
    from surgical_physics_bridge.src.unified_server import UnifiedSurgicalBridgeServer
    server = UnifiedSurgicalBridgeServer(port=18951, gui=False, publish_fps=60.0, scan_mode="orbital_l4",
                                         physics_mode="inline", physics_backend="mujoco", rate_window=2000)
    yield server
    server.shutdown()


def test_server_rejects_unknown_physics_backend():
    from surgical_physics_bridge.src.unified_server import UnifiedSurgicalBridgeServer
    with pytest.raises(ValueError):
        UnifiedSurgicalBridgeServer(port=18952, physics_mode="inline", physics_backend="havok")


def test_mujoco_inline_server_steps_and_tracks_target(mujoco_inline_server):
    s = mujoco_inline_server
    assert type(s.backend).__name__ == "MujocoBackend"
    pose0 = s.env.current_pose_params.copy()
    for _ in range(5):
        s.step_simulation_and_stream(physics_ticks=50)
    assert not np.allclose(s.env.current_pose_params, pose0)
    contact = s.channels.read_contact()
    assert contact["seq"] >= 250
    assert contact["ee_error_m"] < 0.01
    assert s.physics_stats()["total_ticks"] >= 249


def test_mujoco_server_applies_fem_force_extrapolation(mujoco_inline_server):
    s = mujoco_inline_server
    for _ in range(3):
        s.step_simulation_and_stream(physics_ticks=100)
    tis = s.last_target["tissue"]
    assert tis["in_contact"] and tis["stiffness_n_per_mm"] > 0.0
    c = s.channels.read_contact()
    expect = tis["force_n"] + tis["stiffness_n_per_mm"] * (c["tissue_indentation_mm"] - tis["indentation_mm"])
    assert c["tissue_force_n"] == pytest.approx(float(np.clip(expect, 0.0, 40.0)), rel=1e-6, abs=1e-6)
    assert abs(c["tissue_indentation_mm"] - tis["indentation_mm"]) < 5.0
    assert c["tissue_force_n"] > 0.0
