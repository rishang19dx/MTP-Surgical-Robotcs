"""
Tests for the unified server: physics-rate instrumentation, OpenIGTLink payload formats, the
env -> physics target path, and tissue-model-driven skin deformation messages.
"""

import inspect
import time

import numpy as np
import pyigtl
import pytest

from surgical_physics_bridge.src.physics_rate import PhysicsRateMonitor
from surgical_physics_bridge.src.pybullet_backend import PhysicsChannels, run_physics_loop


def _roundtrip(msg):
    raw = msg.pack()
    header = pyigtl.MessageBase.parse_header(raw[:pyigtl.MessageBase.IGTL_HEADER_SIZE])
    out = pyigtl.MessageBase.create_message(header["message_type"])
    out.unpack(header, raw[pyigtl.MessageBase.IGTL_HEADER_SIZE:])
    return out


def test_rate_monitor_detects_overruns_and_rate():
    mon = PhysicsRateMonitor(target_hz=1000.0, window=1000)
    t = 0.0
    for _ in range(500):
        mon.tick(t, t + 0.0002)
        t += 0.001
    s = mon.snapshot()
    assert s["total_ticks"] == 499
    assert s["overruns"] == 0
    assert abs(s["achieved_hz"] - 1000.0) < 1.0
    assert abs(s["period_p50_ms"] - 1.0) < 1e-6
    # one 5 ms stall must be reported as an overrun and as the max period
    t += 0.004
    mon.tick(t, t + 0.0002)
    s = mon.snapshot()
    assert s["overruns"] == 1
    assert abs(s["period_max_ms"] - 5.0) < 1e-6
    assert s["achieved_hz"] < 1000.0


def test_rate_monitor_slow_loop_is_reported_slow():
    mon = PhysicsRateMonitor(target_hz=1000.0, window=100)
    for i in range(50):
        mon.tick(i * 0.004, i * 0.004 + 0.003)  # a 250 Hz loop
    s = mon.snapshot()
    assert abs(s["achieved_hz"] - 250.0) < 1.0
    assert s["overruns"] == 49
    assert s["compute_p99_ms"] == pytest.approx(3.0)


def test_rate_monitor_shared_buffer_reader():
    buf = PhysicsRateMonitor.allocate_shared(window=200)
    writer = PhysicsRateMonitor(target_hz=500.0, window=200, shared_buffer=buf)
    for i in range(100):
        writer.tick(i * 0.002, i * 0.002 + 0.0001)
    s = PhysicsRateMonitor.reader(buf).snapshot()
    assert s["total_ticks"] == 99 and s["target_hz"] == pytest.approx(500.0)
    assert abs(s["achieved_hz"] - 500.0) < 1.0


def test_physics_loop_records_rate_with_fake_backend():
    ticks = []

    class FakeBackend:
        def tick(self, channels):
            ticks.append(time.perf_counter())

    ch = PhysicsChannels(rate_window=2000)
    mon = run_physics_loop(FakeBackend(), ch, target_hz=500.0, duration_s=0.3)
    s = mon.snapshot()
    assert s["total_ticks"] == len(ticks) - 1
    assert len(ticks) > 50  # the loop actually ran
    assert 50.0 < s["achieved_hz"] < 600.0  # measured, not asserted to hit target on CI hardware
    assert s["period_p50_ms"] > 0


def test_array_transport_messages_roundtrip():
    pts = np.random.default_rng(0).uniform(-50, 50, (37, 3)).astype(np.float32)
    out = _roundtrip(pyigtl.ImageMessage(pts[np.newaxis], device_name="ReconstructedBone"))
    assert np.allclose(np.asarray(out.image).reshape(-1, 3), pts)
    rows = np.array([[3, 0.0, -1.5, 0.2], [7, 0.1, -0.7, 0.0]], dtype=np.float32)
    out = _roundtrip(pyigtl.ImageMessage(rows[np.newaxis], device_name="DeformedSkin"))
    assert np.asarray(out.image).shape[-1] == 4
    assert np.allclose(np.asarray(out.image).reshape(-1, 4), rows)


def test_streaming_side_never_imports_pybullet():
    import surgical_physics_bridge.src.unified_server as us
    src = inspect.getsource(us)
    assert "import pybullet" not in src
    assert "pybullet_backend=" not in src  # env must not query PyBullet from the streaming loop


@pytest.fixture(scope="module")
def inline_server():
    from surgical_physics_bridge.src.unified_server import UnifiedSurgicalBridgeServer
    server = UnifiedSurgicalBridgeServer(port=18949, gui=False, publish_fps=60.0, scan_mode="orbital_l4",
                                         physics_mode="inline", rate_window=2000)
    yield server
    server.shutdown()


def test_inline_server_steps_and_tracks_target(inline_server):
    s = inline_server
    pose0 = s.env.current_pose_params.copy()
    for _ in range(5):
        s.step_simulation_and_stream(physics_ticks=50)
    assert s.total_frames_published == 5
    assert not np.allclose(s.env.current_pose_params, pose0)  # the scan moved the probe
    contact = s.channels.read_contact()
    assert contact["seq"] >= 250
    # the robot follows the commanded EE target (measured sub-mm after settling; 1 cm bound while moving)
    assert contact["ee_error_m"] < 0.01
    stats = s.physics_stats()
    assert stats["total_ticks"] >= 249


def test_deformed_skin_message_matches_exported_base_mesh(inline_server):
    """DeformedSkin = (1, S, 3) deformed FEM skin vertices in the vertex order of the OBJ the server exports
    for the Slicer bridge; the probe really deforms it (inward, near the contact point only)."""
    from surgical_physics_bridge.src.obj_io import read_obj
    s = inline_server
    assert type(s.env.tissue_model).__name__ == "FEMTissueModel"
    base_v, base_f = read_obj(s.skin_obj_path)
    ref_v, ref_f = s.env.tissue_model.skin_surface_mesh(None)
    assert np.allclose(base_v, ref_v, atol=1e-4) and np.array_equal(base_f, ref_f)
    s.step_simulation_and_stream(physics_ticks=5)
    ts = s.env.tissue_state
    assert ts is not None and ts.in_contact and ts.reaction_force_n > 1.0
    payload = s.deformed_skin_vertices()[np.newaxis]
    assert payload.shape == (1, len(base_v), 3) and payload.dtype == np.float32
    out = _roundtrip(pyigtl.ImageMessage(np.ascontiguousarray(payload), device_name="DeformedSkin"))
    v = np.asarray(out.image, dtype=np.float32).reshape(-1, 3)
    assert v.shape == base_v.shape
    disp = v - base_v
    moved = np.linalg.norm(disp, axis=1) > 0.2
    assert moved.any(), "probe contact must deform the skin patch"
    r = np.linalg.norm(base_v[:, [0, 2]] - np.asarray(ts.contact_point_mm)[[0, 2]], axis=1)
    k = int(np.argmax(np.linalg.norm(disp, axis=1)))
    assert disp[k, 1] < 0.0 and r[k] < 30.0, "largest displacement is inward (-Y_P) under the probe"
    far = r > 60.0
    assert np.abs(disp[far]).max() < 0.5 * np.abs(disp).max()


def test_physics_loop_applies_fem_force_extrapolation(inline_server):
    """The 1 kHz loop applies F = F0 + K_c (delta - delta0) from the latest FEM tissue state."""
    s = inline_server
    for _ in range(3):
        s.step_simulation_and_stream(physics_ticks=100)
    tis = s.last_target["tissue"]
    assert tis["in_contact"] and tis["stiffness_n_per_mm"] > 0.0
    c = s.channels.read_contact()
    expect = tis["force_n"] + tis["stiffness_n_per_mm"] * (c["tissue_indentation_mm"] - tis["indentation_mm"])
    assert c["tissue_force_n"] == pytest.approx(float(np.clip(expect, 0.0, 40.0)), rel=1e-6, abs=1e-6)
    # the robot holds the face near the commanded indentation, so the applied force is near the FEM force
    assert abs(c["tissue_indentation_mm"] - tis["indentation_mm"]) < 5.0
    assert c["tissue_force_n"] > 0.0
