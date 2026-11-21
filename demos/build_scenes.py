"""
Build the four demonstration scenes (spine, paraspinal muscle, liver, groin vein) from the SonoGym CTs.

    ./.venv/bin/python -m demos.build_scenes --archive <assets.tar.gz> [--subject s0015] [--out demos/scenes]

Per scene this writes `demos/scenes/<scene>/`:
  display/<name>.obj   surface meshes of the WHOLE region (skin, every bone, organs, muscles, vessels) in the scene's
                       simulation frame (mm), decimated for real-time display;
  scene.json           frame, mesh list (name, category, colour, whether it deforms), probe path, FEM ROI, data paths.

Frames. The lumbar scenes (spine, muscle) use the project's frame P (LPS, mm; dorsal skin on +y) and the existing
`data/patients/sonogym_<id>` FEM crop. The liver and vein scenes use frame Q (= RAS - c: anterior skin on +y) and the
crops of `import_liver_subjects.py` / `import_vein_subjects.py`. The display region is much larger than the FEM crop;
outside the FEM ROI the displacement is zero (the ROI faces are fixed), so the whole surface can be deformed by one
field.

Structure names come from SonoGym's own STLs: each label id is matched to the STL with the same shape (volume and
sorted extents, after the archive-wide uniform scale of 0.677), so no id table is assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STL_SCALE = 0.677
DISPLAY_MM = 2.0

# category -> (name regex, colour RGB 0-1, opacity, target triangles per mesh)
CATEGORIES = [
    ("bone", r"^(vertebrae_|sacrum|rib_|hip_|femur_|sternum|costal_cartilages|scapula_|clavicula_)",
     (0.93, 0.90, 0.80), 1.0, 12000),
    ("liver", r"^liver$", (0.62, 0.22, 0.20), 1.0, 20000),
    ("kidney", r"^kidney_(left|right)$", (0.70, 0.35, 0.35), 1.0, 6000),
    ("muscle", r"^(autochthon_|iliopsoas_)", (0.80, 0.30, 0.30), 0.9, 10000),
    ("vein", r"^(inferior_vena_cava|iliac_vena_|portal_vein_and_splenic_vein)", (0.20, 0.35, 0.85), 1.0, 6000),
    ("artery", r"^(aorta|iliac_artery_)", (0.85, 0.15, 0.15), 1.0, 6000),
]

SCENES = {
    # name: frame, FEM crop, focus structures, probe path (in the scene frame, mm: list of (x, z) waypoints), ROI
    "spine": {"frame": "P", "crop": "data/patients/sonogym_{sid}", "focus": ["vertebrae_L4", "vertebrae_L3", "vertebrae_L5"],
              "path": [(-25.0, -32.0), (25.0, -32.0)], "title": "Lumbar spine (L4)"},
    "muscle": {"frame": "P", "crop": "data/patients/sonogym_{sid}", "focus": ["autochthon_left", "autochthon_right"],
               "path": [(20.0, -55.0), (20.0, -10.0)], "title": "Paraspinal muscle"},
    "liver": {"frame": "Q", "crop": "data/liver/sonogym_{sid}", "focus": ["liver"],
              "path": [(-20.0, -40.0), (30.0, 20.0)], "title": "Liver"},
    "vein": {"frame": "Q", "crop": "data/vein/sonogym_{sid}_left", "focus": ["iliac_vena_left", "iliac_artery_left"],
             "path": [(-15.0, 0.0), (15.0, 0.0)], "title": "Groin vein (left)"},
}
MEMBER_RE = re.compile(r"^assets/data/HumanModels/(selected_dataset/(s\d{4})/ct\.nii\.gz|"
                       r"selected_dataset_stl/(s\d{4})/([A-Za-z0-9_]+)\.(nii\.gz|stl))$")


def extract(archive, sid, dest):
    files = {"stl": {}}
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            mm = MEMBER_RE.match(m.name)
            if not mm or not m.isfile() or (mm.group(2) or mm.group(3)) != sid:
                continue
            if mm.group(2):
                files["ct"] = os.path.join(dest, m.name)
            elif mm.group(5) == "nii.gz" and mm.group(4) == "combined_label_map":
                files["label"] = os.path.join(dest, m.name)
            elif mm.group(5) == "stl":
                files["stl"][mm.group(4)] = os.path.join(dest, m.name)
            else:
                continue
            members.append(m)
        tf.extractall(dest, members=members, filter="data")
    return files


def name_labels(lbl, A, stls):
    """label id -> structure name, by matching shape (volume, sorted extents) to SonoGym's named STLs."""
    import trimesh
    z = np.sqrt((A[:3, :3] ** 2).sum(0))
    objs = ndimage.find_objects(lbl)
    stats = {}
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = lbl[sl] == i
        n = int(m.sum())
        if n < 30:
            continue
        idx = np.argwhere(m) + np.array([s.start for s in sl])
        P = nib.affines.apply_affine(A, idx)
        stats[i] = (n * z.prod(), np.sort(np.ptp(P, 0) + z))
    names, scores = {}, {}
    for name, path in stls.items():
        if name.startswith(("standard_", "combined")):
            continue
        try:
            t = trimesh.load(path, force="mesh")
        except Exception:
            continue
        vs, ext = abs(float(t.volume)), np.sort(np.ptp(np.asarray(t.vertices), 0))
        if vs <= 0:
            continue
        best = min(stats, key=lambda i: abs(np.log(vs / stats[i][0] / STL_SCALE ** 3)) +
                   np.abs(np.log(ext / stats[i][1] / STL_SCALE)).sum())
        sc = abs(np.log(vs / stats[best][0] / STL_SCALE ** 3)) + np.abs(np.log(ext / stats[best][1] / STL_SCALE)).sum()
        if sc < 0.6 and (best not in scores or sc < scores[best]):
            names[best], scores[best] = name, sc
    return names, scores


def scene_transform(frame, meta_crop):
    """RAS -> scene frame (mm) as (R, t): x_scene = R (x_ras - c) + t."""
    if frame == "P":
        R = np.diag([-1.0, -1.0, 1.0])
        return R, np.asarray(meta_crop["l4_centroid_ras_mm"]), np.asarray(meta_crop["l4_centroid_P_mm"])
    return np.eye(3), np.asarray(meta_crop["c_ras_mm"]), np.zeros(3)


def surface(mask, origin, h, tri_target, smooth_sigma=1.0):
    import open3d as o3d
    from skimage import measure
    if mask.sum() < 20:
        return None
    f = ndimage.gaussian_filter(mask.astype(np.float32), smooth_sigma)
    if f.max() < 0.5:
        return None
    v, fc, _, _ = measure.marching_cubes(np.pad(f, 1), 0.5)
    v = (v - 1.0) * h + origin
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(fc[:, ::-1].copy()))
    mesh = mesh.simplify_quadric_decimation(int(tri_target))
    mesh = mesh.filter_smooth_taubin(5)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def write_obj(path, v, f):
    with open(path, "w") as fh:
        fh.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in v))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))


def build(sid, files, out_root, scenes):
    ct_img, li = nib.load(files["ct"]), nib.load(files["label"])
    A = ct_img.affine
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    lbl = np.asarray(li.dataobj).astype(np.int16)
    names, scores = name_labels(lbl, A, files["stl"])
    print(f"{sid}: named {len(names)} labels by STL shape", flush=True)
    for scene in scenes:
        cfg = SCENES[scene]
        crop = os.path.join(ROOT, cfg["crop"].format(sid=sid))
        meta_crop = json.load(open(os.path.join(crop, "meta.json")))
        R, c_ras, t = scene_transform(cfg["frame"], meta_crop)
        # display grid: a box around the FEM crop centre, large enough for the torso cross-section
        g = meta_crop["grid_mm"]
        ctr = np.array([np.mean(g["x"]), np.mean(g["y"]), np.mean(g["z"])])
        half = np.array([170.0, 150.0, 140.0]) if scene != "vein" else np.array([130.0, 120.0, 110.0])
        axes = [np.arange(ctr[d] - half[d], ctr[d] + half[d] + 1e-6, DISPLAY_MM) for d in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        S = np.stack([X, Y, Z], -1).reshape(-1, 3)
        ras = (S - t) @ R + c_ras                                    # R orthonormal & symmetric here (R^T = R)
        vox = nib.affines.apply_affine(np.linalg.inv(A), ras).T.reshape(3, *X.shape)
        ct_s = ndimage.map_coordinates(ct, vox, order=1, mode="constant", cval=-1000.0)
        lb_s = ndimage.map_coordinates(lbl, vox, order=0, mode="constant", cval=0)
        from ReconstructionPipeline.data.import_sonogym_subjects import body_mask
        body = body_mask(ct_s)
        out = os.path.join(out_root, scene)
        os.makedirs(os.path.join(out, "display"), exist_ok=True)
        meshes = []
        origin = np.array([a[0] for a in axes])
        sk = surface(body, origin, DISPLAY_MM, 40000, smooth_sigma=1.2)
        write_obj(os.path.join(out, "display", "skin.obj"), *sk)
        meshes.append({"name": "skin", "category": "skin", "color": [0.87, 0.70, 0.60], "opacity": 0.45,
                       "deforms": True, "file": "display/skin.obj", "vertices": int(len(sk[0]))})
        for lid, name in sorted(names.items(), key=lambda kv: kv[1]):
            cat = next((c for c in CATEGORIES if re.match(c[1], name)), None)
            if cat is None:
                continue
            m = (lb_s == lid) & body
            if m.sum() < 200:
                continue
            tri = cat[4] if name in cfg["focus"] else max(1500, cat[4] // 3)
            sf = surface(m, origin, DISPLAY_MM, tri)
            if sf is None:
                continue
            write_obj(os.path.join(out, "display", f"{name}.obj"), *sf)
            meshes.append({"name": name, "category": cat[0], "color": list(cat[2]), "opacity": cat[3],
                           "deforms": cat[0] != "bone", "focus": name in cfg["focus"], "file": f"display/{name}.obj",
                           "vertices": int(len(sf[0])), "label_id": int(lid), "stl_match_score": round(scores[lid], 3)})
        sc = {"scene": scene, "title": cfg["title"], "subject": sid, "frame": cfg["frame"],
              "frame_doc": "P: LPS mm, dorsal skin +y (Slicer RAS = diag(-1,-1,1) P)" if cfg["frame"] == "P"
              else "Q: RAS - c_ras mm, anterior skin +y (Slicer RAS = Q + c_ras)",
              "to_ras": {"R": R.tolist(), "c_ras": c_ras.tolist(), "t": t.tolist()},
              "fem_crop": os.path.relpath(crop, ROOT), "probe_path": cfg["path"], "focus": cfg["focus"],
              "display_box_mm": [[float(a[0]), float(a[-1])] for a in axes], "meshes": meshes}
        json.dump(sc, open(os.path.join(out, "scene.json"), "w"), indent=1)
        print(f"  {scene}: {len(meshes)} meshes ({sum(m['vertices'] for m in meshes)} vertices)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--subject", default="s0015")
    ap.add_argument("--scenes", default="spine,muscle,liver,vein")
    ap.add_argument("--out", default=os.path.join(ROOT, "demos", "scenes"))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        files = extract(a.archive, a.subject, tmp)
        build(a.subject, files, a.out, a.scenes.split(","))


if __name__ == "__main__":
    main()
