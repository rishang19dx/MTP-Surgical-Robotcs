"""
Export a demo scene for the interactive web viewer: meshes plus a grid of exact FEM states along the probe path.

    ./.venv/bin/python -m demos.export_web --scene liver [--fat-calibration ...] [--positions 33] [--forces 1,2,3,4,5,6]

Writes `demos/web/data/<scene>/` (each .bin also as base64 .txt, which the viewer loads):
  scene.json   meshes (name, colour, opacity, deforms, vertex/face counts, offsets into mesh.bin), the moving-vertex
               index lists, the state grid (positions x forces) with probe pose, force, indentation, max |u|, and the
               physics settings;
  mesh.bin     float32 vertex positions then uint32 triangle indices of every mesh (scene frame, mm);
  disp.bin     int8 displacements of the moving vertices, per state (state-major), for the deforming meshes;
               value * states[i].disp_scale_mm = displacement in mm (per-state scale = max |component| / 127);
  bmode.bin    uint8 B-mode images (200 x 150), per state.
The viewer interpolates bilinearly between neighbouring states (position, force); at the grid points it shows
exactly what the simulator computed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_FOCUS_CATEGORIES = {"skin", "bone", "liver", "muscle", "vein", "artery", "kidney"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fat-calibration", default="")
    ap.add_argument("--positions", type=int, default=21)
    ap.add_argument("--forces", default="1,3.5,6")
    ap.add_argument("--out", default=os.path.join(ROOT, "demos", "web", "data"))
    a = ap.parse_args(argv)
    from demos.engine import DemoEngine
    eng = DemoEngine(os.path.join(ROOT, "demos", "scenes", a.scene), fat_calibration=a.fat_calibration)
    forces = [float(v) for v in a.forces.split(",")]
    npos = a.positions
    out = os.path.join(a.out, a.scene)
    os.makedirs(out, exist_ok=True)

    # meshes
    meshes, mbin, off = [], [], 0
    moving = {}
    for name, m in eng.meshes.items():
        meta = m["meta"]
        if meta["category"] not in WEB_FOCUS_CATEGORIES:
            continue
        v = m["v"].astype(np.float32)
        f = m["f"].astype(np.uint32)
        entry = {"name": name, "category": meta["category"], "color": meta["color"], "opacity": meta["opacity"],
                 "focus": bool(meta.get("focus")), "deforms": bool(meta.get("deforms")),
                 "nv": int(len(v)), "nf": int(len(f)), "v_off": off}
        mbin.append(v.tobytes())
        off += v.nbytes
        entry["f_off"] = off
        mbin.append(f.tobytes())
        off += f.nbytes
        if meta.get("deforms") and m["moving"].any():
            moving[name] = np.nonzero(m["moving"])[0]
            entry["moving"] = moving[name].tolist()
        meshes.append(entry)
    open(os.path.join(out, "mesh.bin"), "wb").write(b"".join(mbin))

    # state grid: positions along the path (no sweep easing: a = k / (npos - 1)), each at every force
    path = np.asarray(eng.sc["probe_path"], float)
    states, disp, bm = [], [], []
    for k in range(npos):
        s = k / (npos - 1)
        x, z = path[0] + s * (path[-1] - path[0])
        eng.probe_params = (lambda t, x=x, z=z: (float(x), float(z), float(t)))   # t carries the force
        frs = eng.frames(forces, bmode=True)
        for F, fr in zip(forces, frs):
            states.append({"pos": k, "force_target_n": F, "force_n": round(fr.force_n, 3),
                           "indent_mm": round(fr.indent_mm, 3), "max_u_mm": round(fr.max_u_mm, 3),
                           "pose": np.round(fr.pose_face, 4).tolist()})
            parts = [(fr.verts[n][idx] - eng.meshes[n]["v"][idx]) for n, idx in moving.items()]
            u = np.concatenate(parts) if parts else np.zeros((0, 3))
            sc_mm = max(float(np.abs(u).max()) / 127.0, 1e-6) if len(u) else 1.0
            states[-1]["disp_scale_mm"] = sc_mm
            disp.append(np.clip(np.round(u / sc_mm), -127, 127).astype(np.int8).tobytes())
            bm.append((fr.bmode * 255).astype(np.uint8).tobytes())
        print(f"position {k + 1}/{npos}: max |u| {max(f.max_u_mm for f in frs):.2f} mm", flush=True)
    open(os.path.join(out, "disp.bin"), "wb").write(b"".join(disp))
    open(os.path.join(out, "bmode.bin"), "wb").write(b"".join(bm))
    sc = {"scene": a.scene, "title": eng.sc["title"], "frame": eng.sc["frame"], "positions": npos, "forces": forces,
          "bmode_hw": [200, 150], "bmode_mm": [75.0, 50.0], "meshes": meshes,
          "moving_order": list(moving.keys()), "states": states,
          "physics": {"model": "linear FEM, precomputed Green's functions (DEC-040/041)",
                      "fat_e_eff_kpa": eng.e_fat, "sliding_liver": bool(eng.sliding),
                      "roi_mm": [eng.roi[0].tolist(), eng.roi[1].tolist()]},
          "subject": eng.sc["subject"], "source": "SonoGym / TotalSegmentator CT (CC BY 4.0)"}
    json.dump(sc, open(os.path.join(out, "scene.json"), "w"))
    import base64                     # the viewer loads base64 text (artifact hosting does not serve raw .bin)
    for f in ("mesh", "disp", "bmode"):
        with open(os.path.join(out, f + ".bin"), "rb") as src, open(os.path.join(out, f + ".txt"), "wb") as dst:
            dst.write(base64.b64encode(src.read()))
    sizes = {f: os.path.getsize(os.path.join(out, f)) / 1e6 for f in ("mesh.bin", "disp.bin", "bmode.bin", "scene.json")}
    print("wrote", out, {k: f"{v:.1f} MB" for k, v in sizes.items()})


if __name__ == "__main__":
    main()
