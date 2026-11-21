"""
Run one deformation demo: compute the probe sweep with the FEM, then stream it in real time to 3D Slicer
(OpenIGTLink, port 18945) and render it in PyBullet; optionally record both.

    ./.venv/bin/python -m demos.run_demo --scene liver [--fat-calibration results/vein_calibration_fat_<stamp>.json]
        [--magnify 5] [--frames 240] [--fps 15] [--loops 0] [--no-gui] [--record demos/out/liver]

With --record, PyBullet frames (overview and cutaway close-up) and the B-mode are written per frame together with
frames.csv (time, force, indentation, max displacement); `demos/make_video.py` assembles the video. Slicer runs as
a separate process (`demos/launch_demo.sh` starts both) and saves its own grabs with the frame index in the name.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyigtl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_frames(path, eng, frames, a):
    """Frames as one .npz for `slicer_demo.py` offline mode (MTP_DEMO_OFFLINE): per deforming mesh only the
    moving vertices are stored (index list + positions + |u|); everything else is static."""
    out = {"t": np.array([f.t for f in frames], np.float32),
           "pose_face": np.stack([f.pose_face for f in frames]).astype(np.float64),
           "force_target_n": np.array([f.force_target_n for f in frames], np.float32),
           "force_n": np.array([f.force_n for f in frames], np.float32),
           "indent_mm": np.array([f.indent_mm for f in frames], np.float32),
           "max_u_mm": np.array([f.max_u_mm for f in frames], np.float32),
           "bmode": np.stack([(f.bmode * 255).astype(np.uint8) for f in frames])}
    names = []
    for name in frames[0].verts:
        idx = np.nonzero(eng.meshes[name]["moving"])[0]
        if len(idx) == 0:
            continue
        names.append(name)
        out[f"idx__{name}"] = idx.astype(np.int32)
        out[f"v__{name}"] = np.stack([f.verts[name][idx] for f in frames]).astype(np.float32)
        out[f"d__{name}"] = np.stack([f.disp[name][idx] for f in frames]).astype(np.float16)
    out["meta"] = np.array(json.dumps({"scene": a.scene, "fps": a.fps, "magnify": a.magnify, "names": names,
                                       "fat_e_eff_kpa": eng.e_fat, "sliding": bool(eng.sliding)}))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **out)


def load_frames(path, eng):
    """Frames saved by save_frames, expanded back to full meshes for the PyBullet view and the Slicer stream."""
    from demos.engine import Frame
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    eng.e_fat, eng.sliding = meta.get("fat_e_eff_kpa"), meta.get("sliding")
    out = []
    for k in range(len(z["t"])):
        fr = Frame(t=float(z["t"][k]), pose_skin=z["pose_face"][k], pose_face=z["pose_face"][k],
                   force_target_n=float(z["force_target_n"][k]), force_n=float(z["force_n"][k]),
                   indent_mm=float(z["indent_mm"][k]), bmode=z["bmode"][k].astype(np.float32) / 255.0,
                   max_u_mm=float(z["max_u_mm"][k]))
        for name, m in eng.meshes.items():
            if not m["meta"].get("deforms"):
                continue
            v, d = m["v"].copy(), np.zeros(len(m["v"]))
            if f"idx__{name}" in z.files:
                idx = z[f"idx__{name}"]
                v[idx] = z[f"v__{name}"][k]
                d[idx] = z[f"d__{name}"][k].astype(np.float64)
            fr.verts[name], fr.disp[name] = v, d
        out.append(fr)
    return out


def _server(port, tries=20):
    """OpenIGTLink server; reuses the address (a previous driver may leave the port in TIME_WAIT) and retries."""
    pyigtl.OpenIGTLinkServer.allow_reuse_address = True
    for k in range(tries):
        try:
            return pyigtl.OpenIGTLinkServer(port=port, local_server=True)
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"[demo] port {port} busy ({e}); retrying", flush=True)
            time.sleep(1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fat-calibration", default="")
    ap.add_argument("--magnify", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--loops", type=int, default=0, help="0 = loop forever (unless --record)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--no-slicer", action="store_true")
    ap.add_argument("--record", default="")
    ap.add_argument("--port", type=int, default=18945)
    ap.add_argument("--cache", default="", help="write the computed frames here (.npz) for offline Slicer rendering")
    ap.add_argument("--cache-only", action="store_true", help="compute and cache the frames, then exit")
    ap.add_argument("--from-cache", default="", help="replay frames cached with --cache (no FEM build)")
    a = ap.parse_args(argv)

    from demos.engine import DemoEngine
    from demos.pybullet_view import PyBulletView
    scene_dir = os.path.join(ROOT, "demos", "scenes", a.scene)
    t0 = time.perf_counter()
    if a.from_cache:
        eng = DemoEngine(scene_dir, fat_calibration=a.fat_calibration, build_fem=False)
        frames = load_frames(a.from_cache, eng)
        a.fps = float(json.loads(str(np.load(a.from_cache)["meta"]))["fps"])
    else:
        eng = DemoEngine(scene_dir, fat_calibration=a.fat_calibration)
        print(f"[demo] {a.scene}: FEM ready in {time.perf_counter() - t0:.0f} s (fat "
              f"{eng.e_fat if eng.e_fat is not None else 'shared'} kPa, sliding {eng.sliding})", flush=True)
        t0 = time.perf_counter()
        frames = eng.run(n_frames=a.frames, fps=a.fps)
    print(f"[demo] {len(frames)} frames ready in {time.perf_counter() - t0:.0f} s; "
          f"force {min(f.force_n for f in frames):.1f}-{max(f.force_n for f in frames):.1f} N, "
          f"max |u| {max(f.max_u_mm for f in frames):.2f} mm", flush=True)

    if a.cache:
        save_frames(a.cache, eng, frames, a)
        print(f"[demo] cached {len(frames)} frames to {a.cache}", flush=True)
        if a.cache_only:
            import sys
            sys.stdout.flush()
            os._exit(0)

    srv = None
    if not a.no_slicer:
        srv = _server(a.port)
    view = PyBulletView(eng, gui=not a.no_gui, magnify=a.magnify)
    cut = None
    if a.record:
        os.makedirs(a.record, exist_ok=True)
        csv = open(os.path.join(a.record, "frames.csv"), "w")
        csv.write("frame,t_s,force_target_n,force_n,indent_mm,max_u_mm\n")
    loops = a.loops if a.loops > 0 else (1 if a.record else 10 ** 9)
    dt = 1.0 / a.fps
    try:
        for loop in range(loops):
            for i, fr in enumerate(frames):
                tick = time.perf_counter()
                view.update(fr)
                if srv is not None:
                    srv.send_message(pyigtl.TransformMessage(fr.pose_face, device_name="Probe"), wait=False)
                    srv.send_message(pyigtl.ImageMessage((fr.bmode * 255).astype(np.uint8)[:, :, None],
                                                         device_name="US"), wait=False)
                    for name, v in fr.verts.items():
                        srv.send_message(pyigtl.ImageMessage(v.astype(np.float32)[None], device_name=f"V_{name}"),
                                         wait=False)
                        srv.send_message(pyigtl.ImageMessage(fr.disp[name].astype(np.float32)[None, :, None],
                                                             device_name=f"D_{name}"), wait=False)
                    tel = {"frame": i, "t_s": fr.t, "force_n": fr.force_n, "indent_mm": fr.indent_mm,
                           "max_u_mm": fr.max_u_mm, "scene": a.scene, "magnify": a.magnify}
                    srv.send_message(pyigtl.StringMessage(json.dumps(tel), device_name="Telemetry"), wait=False)
                if a.record:
                    from PIL import Image
                    Image.fromarray(view.snapshot(label="whole body")).save(os.path.join(a.record, f"pb_{i:04d}.png"))
                    Image.fromarray(view.snapshot(yaw=20, pitch=-20, dist=0.30, label="skin hidden", cut=True)).save(
                        os.path.join(a.record, f"cut_{i:04d}.png"))
                    Image.fromarray((fr.bmode * 255).astype(np.uint8)).save(os.path.join(a.record, f"us_{i:04d}.png"))
                    csv.write(f"{i},{fr.t:.3f},{fr.force_target_n:.3f},{fr.force_n:.3f},{fr.indent_mm:.3f},"
                              f"{fr.max_u_mm:.4f}\n")
                else:
                    time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    if a.record:
        csv.close()
        json.dump({"scene": a.scene, "fps": a.fps, "magnify": a.magnify, "frames": len(frames),
                   "fat_e_eff_kpa": eng.e_fat, "sliding": eng.sliding}, open(os.path.join(a.record, "meta.json"), "w"))
    # pyigtl's server thread can block shutdown after the client disconnects: exit hard once outputs are flushed
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
