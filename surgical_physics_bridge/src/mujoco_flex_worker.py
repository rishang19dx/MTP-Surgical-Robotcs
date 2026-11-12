"""
MuJoCo flex tissue worker (DEC-053). Runs ONLY in ``.venv-mj4`` (Python 3.11, MuJoCo 4.x); imports numpy and
mujoco and nothing from the project, so it never needs the project's torch stack.

Units and frame: the patient tet mesh in frame P, converted mm -> m inside MuJoCo. Gravity is off: the FEM
reference configuration is the CT, which already carries gravity, exactly as in the linear FEM.

Material: one flex, Saint Venant-Kirchhoff (``material="svk"``) or MuJoCo's experimental Stable Neo-Hookean
(``material="snh"``, ``elastic3d = 1``). Per-tet Young's moduli are applied by scaling the compiled per-tet
stiffness values (``flex_stiffness``, 24 per tet; values 0-22 are proportional to E for both materials, value 23
is geometric). ``tests/test_mujoco_flex.py`` asserts that this scaling equals changing E.

Two uses:
  * ``python mujoco_flex_worker.py study <in.npz> <out.npz>``: static solves under prescribed nodal forces
    (the DEC-053 gate study; the forces come from ``fem_nonlinear.contact_nodal_forces`` in ``.venv``).
  * ``python mujoco_flex_worker.py live <mesh.npz> <host> <port> <authkey-hex>``: continuous dynamic simulation
    with a probe body (mocap) pressed into the tissue by MuJoCo contact. It receives probe targets and publishes
    the contact force, the probe depth and the nodal displacement field. Messages hold only builtins and raw
    bytes, because the two venvs run different numpy versions (no numpy pickles).
"""

from __future__ import annotations

import os
import struct
import sys
import time
import warnings
import zlib

import numpy as np
import mujoco

STIFFNESS_PER_TET = 24
E_SCALED_SLOTS = 23            # values 0..22 of each tet are proportional to E (both materials)
MATERIALS = {"svk": 0, "snh": 1}
DENSITY = 1000.0               # kg/m^3, soft tissue

# Static (study) solver, chosen on a synthetic slab only (DEC-053 study, item 5).
STATIC_DT = 0.02
STATIC_DAMPING = 0.5           # N s/m per vertex dof
SETTLE_REL = 1e-4              # relative change of the compression over SETTLE_WINDOW steps
SETTLE_WINDOW = 20
SETTLE_SPEED = 1e-5            # m/s, maximum vertex speed
MAX_STEPS = 5000

# Live (demo) simulation; chosen in a scripted press on the 8 mm sample mesh (scratch): with the Stable Neo-Hookean
# material no tet inverted up to 12 mm past first contact (SVK inverted about 80-115), a 2 mm vertex radius keeps
# the 10 mm-wide face from slipping between vertices, and dt = 0.01 s runs close to real time on the laptop.
LIVE_DT = 0.01
LIVE_DAMPING = 0.2
LIVE_RADIUS_M = 0.002
PROBE_FACE_M = (0.045, 0.010)  # lateral x elevation, as the FEM face (face_dims_mm)
PROBE_DEPTH_M = 0.03           # extent of the probe box behind the face
PROBE_MAX_SPEED = 0.03         # m/s, rate limit of the mocap probe (no tunnelling through the elements)
PROBE_MAX_ANG_SPEED = 1.0      # rad/s
PUBLISH_PERIOD_S = 1.0 / 30.0


# --------------------------------------------------------------------------------------------- model
def build_model(nodes_mm: np.ndarray, tets: np.ndarray, fixed: np.ndarray, young_pa: np.ndarray, nu: float,
                material: str = "svk", dt: float = STATIC_DT, damping: float = STATIC_DAMPING,
                probe: bool = False, e_ref: float = None, radius: float = 0.0005):
    """Compile the flex; returns (model, data, info). ``young_pa`` is per tet. The flex is compiled with the
    modulus ``e_ref`` (default: the median) and each tet's stiffness values are then scaled by young / e_ref."""
    nodes = np.asarray(nodes_mm, dtype=np.float64) * 1e-3
    tets = np.asarray(tets, dtype=np.int64)
    young_pa = np.asarray(young_pa, dtype=np.float64)
    e_ref = float(np.median(young_pa)) if e_ref is None else float(e_ref)
    vol = np.abs(np.einsum("ij,ij->i", np.cross(nodes[tets[:, 1]] - nodes[tets[:, 0]],
                                                  nodes[tets[:, 2]] - nodes[tets[:, 0]]),
                           nodes[tets[:, 3]] - nodes[tets[:, 0]])).sum() / 6.0
    pins = " ".join(str(int(i)) for i in np.nonzero(fixed)[0])
    contact = ('<contact condim="3" selfcollide="none" solref="{:.4f} 1" friction="0.3"/>'.format(max(2 * dt, 0.004))
               if probe else '<contact contype="0" conaffinity="0"/>')
    probe_xml = ""
    if probe:
        hx, hz = PROBE_FACE_M[0] / 2, PROBE_FACE_M[1] / 2
        probe_xml = (f'<body name="probe" mocap="true" pos="0 1 0"><geom name="probe_face" type="box" '
                     f'size="{hx} {PROBE_DEPTH_M / 2} {hz}" rgba="0.2 0.4 1 1" friction="0.3"/></body>'
                     '<camera name="demo" pos="0 0 0" fovy="45"/>')
    xml = f"""<mujoco>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/><global offwidth="1280" offheight="960"/></visual>
  <option timestep="{dt}" gravity="0 0 0" integrator="discrete" solver="CG"/>
  <worldbody>
    {probe_xml}
    <flexcomp name="tissue" type="direct" dim="3" radius="{radius}" mass="{vol * DENSITY:.6f}" rgba="0.9 0.7 0.6 1"
              point="{' '.join(f'{v:.6f}' for v in nodes.ravel())}" element="{' '.join(map(str, tets.ravel()))}">
      <edge equality="false"/>
      <elasticity young="{e_ref:.6f}" poisson="{nu}"/>
      {contact}
      <pin id="{pins}"/>
    </flexcomp>
  </worldbody>
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    spec.flexes[0].elastic3d = MATERIALS[material]
    with warnings.catch_warnings():
        # MuJoCo warns that the flex "has no equality constraints or passive forces" because elasticity is set
        # per element after compilation (flex_stiffness); tests/test_mujoco_flex.py shows the elasticity acts.
        warnings.filterwarnings("ignore", message=".*not rigid and has no equality constraints.*")
        model = spec.compile()
    if model.nflexelem != len(tets) or model.nflexstiffness != STIFFNESS_PER_TET * len(tets):
        raise RuntimeError("unexpected flex stiffness layout; per-tet moduli cannot be applied")
    k = model.flex_stiffness.reshape(-1, STIFFNESS_PER_TET)
    k[:, :E_SCALED_SLOTS] *= (young_pa / e_ref)[:, None]
    model.dof_damping[:] = damping
    data = mujoco.MjData(model)
    vb = model.flex_vertbodyid[model.flex_vertadr[0]:model.flex_vertadr[0] + model.flex_vertnum[0]]
    dofadr = np.where(vb > 0, model.body_dofadr[vb], -1)       # pinned vertices are attached to the world
    info = {"nodes_m": nodes, "tets": tets, "dofadr": dofadr, "e_ref": e_ref, "volume_m3": vol,
            "probe_geom": model.geom("probe_face").id if probe else -1,
            "probe_mocap": model.body("probe").mocapid[0] if probe else -1}
    return model, data, info


def displacement_m(data, info) -> np.ndarray:
    return data.flexvert_xpos.copy() - info["nodes_m"]


def inverted_tets(data, info) -> int:
    x = data.flexvert_xpos
    t = info["tets"]
    vol = np.einsum("ij,ij->i", np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]), x[t[:, 3]] - x[t[:, 0]])
    x0 = info["nodes_m"]
    vol0 = np.einsum("ij,ij->i", np.cross(x0[t[:, 1]] - x0[t[:, 0]], x0[t[:, 2]] - x0[t[:, 0]]),
                     x0[t[:, 3]] - x0[t[:, 0]])
    return int((np.sign(vol) != np.sign(vol0)).sum())


def set_nodal_forces(data, info, forces_n: np.ndarray) -> None:
    data.qfrc_applied[:] = 0.0
    da = info["dofadr"]
    on = da >= 0
    for j in range(3):
        data.qfrc_applied[da[on] + j] = forces_n[on, j]


def settle(model, data, info, comp_nodes: np.ndarray, comp_w: np.ndarray, beam: np.ndarray,
           max_steps: int = MAX_STEPS):
    """Step until settled (module constants). Returns (settled, steps, seconds)."""
    t0 = time.perf_counter()
    hist = []
    wsum = float(comp_w.sum()) if len(comp_w) else 1.0
    for n in range(1, max_steps + 1):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return False, n, time.perf_counter() - t0
        u = data.flexvert_xpos[comp_nodes] - info["nodes_m"][comp_nodes]
        hist.append(float((u @ beam) @ comp_w / wsum) if len(comp_w) else 0.0)
        if n > SETTLE_WINDOW:
            c_now, c_old = hist[-1], hist[-1 - SETTLE_WINDOW]
            rel = abs(c_now - c_old) / max(abs(c_now), 1e-9)
            if rel < SETTLE_REL and np.abs(data.qvel).max() < SETTLE_SPEED:
                return True, n, time.perf_counter() - t0
    return False, max_steps, time.perf_counter() - t0


# --------------------------------------------------------------------------------------------- study
def run_study(in_path: str, out_path: str) -> None:
    """Every case in ``in_path`` (one patient mesh, several force sets) for each material."""
    z = np.load(in_path)
    nu = float(z["nu"])
    n_cases = int(z["n_cases"])
    out = {}
    for material in [str(m) for m in z["materials"]]:
        model, data, info = build_model(z["nodes"], z["tets"], z["fixed"], z["young_pa"], nu, material)
        for c in range(n_cases):
            mujoco.mj_resetData(model, data)
            f = z[f"forces_{c}"]
            nodes_c, w_c, beam = z[f"comp_nodes_{c}"], z[f"comp_w_{c}"], z[f"beam_{c}"]
            set_nodal_forces(data, info, f)
            ok, steps, sec = settle(model, data, info, nodes_c, w_c, beam)
            u = displacement_m(data, info) * 1e3
            comp = float((u[nodes_c] @ beam) @ w_c / w_c.sum())
            out[f"{material}_{c}_u_mm"] = u.astype(np.float32)
            out[f"{material}_{c}_meta"] = np.array([comp, float(ok), steps, sec, inverted_tets(data, info),
                                                   float(np.isfinite(u).all())])
            print(f"  {material} case {c}: compression {comp:.3f} mm settled {ok} in {steps} steps "
                  f"({sec:.1f} s), inverted {int(out[f'{material}_{c}_meta'][4])}", flush=True)
        if "g4_forces_a" in z.files:     # cost: warm re-settle after a 0.33 mm indentation step
            mujoco.mj_resetData(model, data)
            nodes_c, w_c, beam = z["g4_comp_nodes"], z["g4_comp_w"], z["g4_beam"]
            set_nodal_forces(data, info, z["g4_forces_a"])
            ok_a, _, _ = settle(model, data, info, nodes_c, w_c, beam)
            set_nodal_forces(data, info, z["g4_forces_b"])
            ok_b, steps, sec = settle(model, data, info, nodes_c, w_c, beam)
            out[f"{material}_g4"] = np.array([float(ok_a and ok_b), steps, sec, model.opt.timestep])
            print(f"  {material} G4 warm re-settle: {sec * 1e3:.0f} ms ({steps} steps, settled {ok_a and ok_b})",
                  flush=True)
    np.savez(out_path, **out)



# --------------------------------------------------------------------------------------------- snapshots
def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal PNG writer (stdlib only; .venv-mj4 has no imaging library)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class FlexSnapshots:
    """Opt-in offscreen renders of the live flex (MTP_FLEX_SNAPSHOT_DIR, every MTP_FLEX_SNAPSHOT_S sim seconds).
    The camera looks at the probe face from the side and above, with dorsal (+Y of frame P) up; a CSV row per
    frame records the contact force and the probe depth so captions come from the simulation, not by hand."""

    def __init__(self, model):
        self.dir = os.environ.get("MTP_FLEX_SNAPSHOT_DIR", "")
        self.period = float(os.environ.get("MTP_FLEX_SNAPSHOT_S", "0.25"))
        self.next_t = 0.0
        self.index = 0
        self.renderer = None
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self.renderer = mujoco.Renderer(model, 480, 640)
            self.cam = model.camera("demo").id
            with open(os.path.join(self.dir, "flex_frames.csv"), "w") as f:
                f.write("file,sim_time_s,force_along_minus_beam_n,probe_face_depth_mm,max_displacement_mm\n")

    def maybe(self, model, data, face_m, beam, force_vec, depth_mm, umax_mm) -> None:
        if self.renderer is None or face_m is None or data.time < self.next_t:
            return
        self.next_t = data.time + self.period
        up = np.array([0.0, 1.0, 0.0])
        lat = np.cross(beam, np.array([0.0, 0.0, 1.0]))
        lat = lat / max(np.linalg.norm(lat), 1e-9)
        view = 0.55 * lat - 0.6 * beam + 0.58 * np.array([0.0, 0.0, 1.0])      # side, above, along the spine
        view /= np.linalg.norm(view)
        pos = face_m + 0.16 * view
        zc = view                                      # camera looks along -z
        xc = np.cross(up, zc)
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        R = np.stack([xc, yc, zc], 1)
        model.cam_pos[self.cam] = pos
        model.cam_quat[self.cam] = _rot_to_quat(R)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        self.renderer.update_scene(data, camera=self.cam)
        name = f"flex_{self.index:04d}.png"
        write_png(os.path.join(self.dir, name), self.renderer.render())
        with open(os.path.join(self.dir, "flex_frames.csv"), "a") as f:
            f.write(f"{name},{data.time:.3f},{float(-force_vec @ beam):.4f},{depth_mm:.3f},{umax_mm:.3f}\n")
        self.index += 1

# --------------------------------------------------------------------------------------------- live
def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).ravel())
    return q


def _probe_contact_force(model, data, probe_geom: int) -> np.ndarray:
    """Total contact force ON the probe (world frame = frame P, N)."""
    total = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom[0] == probe_geom:
            sign = -1.0          # the contact force acts on geom 2 along +normal, so on geom 1 against it
        elif con.geom[1] == probe_geom:
            sign = 1.0
        else:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        total += sign * (con.frame.reshape(3, 3).T @ f6[:3])
    return total


def run_live(mesh_path: str, host: str, port: int, authkey_hex: str) -> None:
    from multiprocessing.connection import Client

    z = np.load(mesh_path)
    model, data, info = build_model(z["nodes"], z["tets"], z["fixed"], z["young_pa"], float(z["nu"]),
                                    str(z["material"]), dt=LIVE_DT, damping=LIVE_DAMPING, probe=True,
                                    radius=LIVE_RADIUS_M)
    conn = Client((host, port), authkey=bytes.fromhex(authkey_hex))
    mocap = info["probe_mocap"]
    half_depth = PROBE_DEPTH_M / 2
    # Park the probe well above the skin until the first target arrives.
    face = None
    goal_face, goal_R = None, np.eye(3)
    data.mocap_pos[mocap] = info["nodes_m"].mean(0) + np.array([0.0, 1.0, 0.0])
    conn.send({"type": "ready", "nodes": int(len(info["nodes_m"])), "tets": int(len(info["tets"])),
               "dt": LIVE_DT, "material": str(z["material"])})
    shots = FlexSnapshots(model)
    skin_top = None
    t_wall0 = time.perf_counter()
    last_pub = 0.0
    step_ms = []
    R = np.eye(3)
    while True:
        while conn.poll(0):
            msg = conn.recv()
            if msg.get("type") == "stop":
                conn.close()
                return
            if msg.get("type") == "target":
                T = np.frombuffer(msg["pose"], dtype=np.float64).reshape(4, 4)
                skin_top = T[:3, 3] * 1e-3
                b = T[:3, 1] / np.linalg.norm(T[:3, 1])
                goal_face = (T[:3, 3] + float(msg["indentation_mm"]) * b) * 1e-3
                goal_R = T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0, keepdims=True)
                if face is None or msg.get("teleport"):
                    # first target: start 5 mm off the skin along the beam, then approach under the rate limit
                    face = goal_face - 0.005 * b - float(max(msg["indentation_mm"], 0.0)) * 1e-3 * b
                    R = goal_R.copy()
        if goal_face is not None:
            dmax = PROBE_MAX_SPEED * LIVE_DT
            step = goal_face - face
            n = np.linalg.norm(step)
            face = face + (step if n <= dmax else step * (dmax / n))
            dR = goal_R @ R.T
            ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            if ang > 1e-9:
                frac = min(1.0, PROBE_MAX_ANG_SPEED * LIVE_DT / ang)
                axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
                a = frac * ang
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K) @ R
            beam = R[:, 1]
            data.mocap_pos[mocap] = face - half_depth * beam
            data.mocap_quat[mocap] = _rot_to_quat(R)
        t0 = time.perf_counter()
        mujoco.mj_step(model, data)
        step_ms.append((time.perf_counter() - t0) * 1e3)
        sim_t = data.time
        # pace to real time (never faster); if slower, the tissue simply runs behind the wall clock
        ahead = sim_t - (time.perf_counter() - t_wall0)
        if ahead > 0:
            time.sleep(ahead)
        if shots.renderer is not None and face is not None and skin_top is not None and data.time >= shots.next_t:
            fv = _probe_contact_force(model, data, info["probe_geom"])
            umax = float(np.linalg.norm(displacement_m(data, info), axis=1).max() * 1e3)
            shots.maybe(model, data, face, R[:, 1], fv, float((face - skin_top) @ R[:, 1] * 1e3), umax)
        now = time.perf_counter() - t_wall0
        if now - last_pub >= PUBLISH_PERIOD_S:
            last_pub = now
            if not np.isfinite(data.qpos).all():
                conn.send({"type": "error", "message": "flex state is not finite"})
                return
            u = (displacement_m(data, info) * 1e3).astype(np.float32)
            fvec = _probe_contact_force(model, data, info["probe_geom"])
            face_mm = (face * 1e3).tolist() if face is not None else None
            conn.send({"type": "state", "sim_time_s": float(sim_t), "wall_time_s": float(now),
                       "force_vec_n": fvec.tolist(), "probe_face_mm": face_mm,
                       "probe_beam": (R[:, 1]).tolist(), "ncon": int(data.ncon),
                       "step_ms_median": float(np.median(step_ms)) if step_ms else 0.0,
                       "u_mm": u.tobytes(), "n_nodes": int(len(u))})
            step_ms = step_ms[-200:]


def main(argv) -> None:
    if len(argv) >= 3 and argv[0] == "study":
        run_study(argv[1], argv[2])
    elif len(argv) >= 5 and argv[0] == "live":
        try:
            run_live(argv[1], argv[2], int(argv[3]), argv[4])
        except (EOFError, ConnectionError):
            pass        # the server went away without a "stop" (killed or crashed): exit quietly
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
