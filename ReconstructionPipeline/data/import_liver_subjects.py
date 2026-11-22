"""
Import an abdominal (right subcostal) crop of the SonoGym TotalSegmentator subjects for the liver headroom bound
(DEC-059, pre-registered).

    ./.venv/bin/python -m ReconstructionPipeline.data.import_liver_subjects --out data/liver \
        --archive <assets.tar.gz> [--subjects s0015]

Source: the same archive as `import_sonogym_subjects.py` (Hugging Face `yunkao/SonoGym_assets_models`,
`assets.tar.gz`, CC BY 4.0; CTs and labels from TotalSegmentator, Wasserthal et al. 2023). Extracted members:
`selected_dataset/<id>/ct.nii.gz`, `selected_dataset_stl/<id>/combined_label_map.nii.gz` and
`selected_dataset_stl/<id>/liver.stl` (QA only).

Frame Q (mm), used only by the liver study: Q = RAS - c + q0, i.e. +x = patient right, +y = ANTERIOR, +z = superior.
Relative to the project's frame P (LPS) this is a 180 deg rotation about the cranio-caudal axis (x -> -x, y -> -y,
det +1), so the anterior abdominal skin takes the place of the dorsal skin and the FEM / contact / probe code, which
expects the skin on +y with the beam along -y, runs unchanged (DEC-059).
  * grid: 1 mm, x in [-70, 70], y in [-50, 120], z in [-90, 85] (the lumbar grid size);
  * c: x, z = the liver centroid; y chosen so that the most anterior body voxel inside the x-z footprint sits at
    y = 115 mm (5 mm below the grid top). The crop therefore reaches 165 mm deep from the most anterior skin.
  * body mask as in `import_sonogym_subjects.body_mask` (the table is now on the deep side and is removed anyway).

Outputs per subject (gitignored NIfTI; voxel (i,j,k) -> Q mm by the affine, arrays stored x,y,z as the lumbar data):
  ct.nii.gz, liver.nii.gz (label 5), fov.nii.gz (1 inside the source CT's field of view), meta.json (transform,
  label QA against SonoGym's liver.stl).
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

from ReconstructionPipeline.data.import_sonogym_subjects import body_mask

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVER = 5                     # TotalSegmentator v2 `total` task: 1 spleen, 2/3 kidneys, 4 gallbladder, 5 liver
GRID_X, GRID_Y, GRID_Z = (-70.0, 70.0), (-50.0, 120.0), (-90.0, 85.0)
WIDE_X, WIDE_Z = (-140.0, 140.0), (-160.0, 155.0)      # DEC-063 (--wide): same depth, wider sides
SKIN_TOP_Y = 115.0
MEMBER_RE = re.compile(r"^assets/data/HumanModels/(selected_dataset/(s\d{4})/ct\.nii\.gz|"
                       r"selected_dataset_stl/(s\d{4})/(combined_label_map\.nii\.gz|liver\.stl))$")


def extract(archive, subjects, dest):
    out = {}
    with tarfile.open(archive, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            mm = MEMBER_RE.match(m.name)
            if not mm or not m.isfile():
                continue
            sid = mm.group(2) or mm.group(3)
            if subjects and sid not in subjects:
                continue
            members.append(m)
            kind = "ct" if mm.group(2) else ("label" if mm.group(4).startswith("combined") else "stl")
            out.setdefault(sid, {})[kind] = os.path.join(dest, m.name)
        tf.extractall(dest, members=members, filter="data")
    return {k: v for k, v in out.items() if {"ct", "label"} <= set(v)}


def _stl_shape(path):
    """Position-free shape of a SonoGym STL: |volume| (mm^3) and sorted bounding-box extents (mm). The STL frame
    is not documented (its centroids are 100-770 mm from the label's in both RAS and LPS), so the label-id check
    compares shape, which does not depend on the frame's translation or axis signs."""
    import trimesh
    m = trimesh.load(path, force="mesh")
    return abs(float(m.volume)), np.sort(np.ptp(np.asarray(m.vertices, float), 0))


def convert(sid, files, out_dir, grid_x=GRID_X, grid_z=GRID_Z):
    GRID_X, GRID_Z = grid_x, grid_z
    ct_img, lbl_img = nib.load(files["ct"]), nib.load(files["label"])
    A = ct_img.affine
    if nib.aff2axcodes(A) != ("R", "A", "S"):
        raise ValueError(f"{sid}: expected RAS axis codes, got {nib.aff2axcodes(A)}")
    if not np.allclose(A, lbl_img.affine, rtol=0.0, atol=1e-3) or ct_img.shape != lbl_img.shape:
        raise ValueError(f"{sid}: CT and label grids differ")
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    lbl = np.asarray(lbl_img.dataobj).astype(np.int16)
    liv_vox = np.argwhere(lbl == LIVER)
    if len(liv_vox) < 10000:
        raise ValueError(f"{sid}: liver label missing or tiny ({len(liv_vox)} voxels)")
    liv_ras = nib.affines.apply_affine(A, liv_vox.mean(0))
    qa = {"liver_voxels_source": int(len(liv_vox)), "liver_centroid_ras_mm": liv_ras.tolist()}
    if "stl" in files:
        vol_stl, ext_stl = _stl_shape(files["stl"])
        zooms = np.asarray(ct_img.header.get_zooms()[:3], float)
        pts = nib.affines.apply_affine(A, liv_vox)
        ext_lbl = np.sort(np.ptp(pts, 0) + zooms)
        qa["stl_volume_ratio"] = float(vol_stl / (len(liv_vox) * zooms.prod()))
        qa["stl_extent_mm"], qa["label_extent_mm"] = ext_stl.tolist(), ext_lbl.tolist()
        qa["stl_extent_max_rel_diff"] = float(np.max(np.abs(ext_stl - ext_lbl) / ext_lbl))

    xs = np.arange(GRID_X[0], GRID_X[1] + 0.5, 1.0)
    zs = np.arange(GRID_Z[0], GRID_Z[1] + 0.5, 1.0)
    ys_wide = np.arange(-200.0, 260.0 + 0.5, 1.0)               # relative to the liver centroid, generous

    def sample(ys, arr, order, cval):
        QX, QY, QZ = np.meshgrid(xs, ys, zs, indexing="ij")
        ras = np.stack([QX + liv_ras[0], QY + liv_ras[1], QZ + liv_ras[2]], -1).reshape(-1, 3)
        vox = nib.affines.apply_affine(np.linalg.inv(A), ras).T.reshape(3, *QX.shape)
        return ndimage.map_coordinates(arr, vox, order=order, mode="constant", cval=cval)

    ct_w = sample(ys_wide, ct, 1, -1000.0).astype(np.float32)
    body_w = body_mask(ct_w)
    top = np.where(body_w.any(1), body_w.shape[1] - 1 - np.argmax(body_w[:, ::-1, :], axis=1), -1)
    y_top = float(ys_wide[int(top.max())])                    # most anterior body voxel (rel. to liver centroid)
    ys = np.arange(GRID_Y[0], GRID_Y[1] + 0.5, 1.0) + y_top - SKIN_TOP_Y   # rel. y; Q_y = rel - y_top + 115
    ct_q = sample(ys, ct, 1, -1000.0).astype(np.float32)
    lb_q = sample(ys, lbl, 0, 0).astype(np.int16)
    fov = sample(ys, np.ones(ct.shape, np.float32), 0, 0.0) > 0.5
    body = body_mask(ct_q)
    ct_q[~body] = -1000.0
    liver = (lb_q == LIVER) & body

    os.makedirs(out_dir, exist_ok=True)
    aff = np.eye(4)
    aff[:3, 3] = [GRID_X[0], GRID_Y[0], GRID_Z[0]]
    nib.save(nib.Nifti1Image(ct_q, aff), os.path.join(out_dir, "ct.nii.gz"))
    nib.save(nib.Nifti1Image(liver.astype(np.uint8), aff), os.path.join(out_dir, "liver.nii.gz"))
    nib.save(nib.Nifti1Image(fov.astype(np.uint8), aff), os.path.join(out_dir, "fov.nii.gz"))
    c_ras = liv_ras + np.array([0.0, y_top - SKIN_TOP_Y, 0.0])       # Q = RAS - c_ras
    meta = {"subject": sid, "study": "DEC-059 liver headroom",
            "source": "huggingface.co/datasets/yunkao/SonoGym_assets_models (assets.tar.gz)",
            "licence": "CC BY 4.0 (SonoGym assets); CTs and labels from TotalSegmentator (Wasserthal et al. 2023)",
            "frame": "Q (mm): Q = RAS - c_ras; +x right, +y anterior, +z superior (P rotated 180 deg about z)",
            "c_ras_mm": c_ras.tolist(), "grid_mm": {"x": GRID_X, "y": GRID_Y, "z": GRID_Z, "spacing": 1.0},
            "source_spacing_mm": [float(v) for v in ct_img.header.get_zooms()[:3]], "source_shape": list(ct.shape),
            "qa": {**qa, "liver_voxels_in_crop": int(liver.sum()),
                   "liver_fraction_in_crop": float(liver.sum() * 1.0 / np.prod(ct_img.header.get_zooms()[:3]) /
                                                   len(liv_vox)),
                   "outside_fov_voxels": int((~fov).sum())}}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "liver"))
    ap.add_argument("--subjects", default="")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--wide", action="store_true", help="DEC-063: x [-140, 140], z [-160, 155] (default out data/liver_wide)")
    args = ap.parse_args(argv)
    grid = (WIDE_X, WIDE_Z) if args.wide else (GRID_X, GRID_Z)
    if args.wide and args.out == os.path.join(ROOT, "data", "liver"):
        args.out = os.path.join(ROOT, "data", "liver_wide")
    subjects = [s for s in args.subjects.split(",") if s] or None
    with tempfile.TemporaryDirectory() as tmp:
        files = extract(args.archive, subjects, tmp)
        for sid in sorted(files):
            meta = convert(sid, files[sid], os.path.join(args.out, f"sonogym_{sid}"), *grid)
            q = meta["qa"]
            print(f"{sid}: liver in crop {q['liver_voxels_in_crop']} vox ({100 * q['liver_fraction_in_crop']:.0f} % of "
                  f"the label volume), STL/label volume {q.get('stl_volume_ratio', float('nan')):.2f}, extents differ <= "
                  f"{100 * q.get('stl_extent_max_rel_diff', float('nan')):.0f} %, outside FOV {q['outside_fov_voxels']}",
                  flush=True)


if __name__ == "__main__":
    main()
