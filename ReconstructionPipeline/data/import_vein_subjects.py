"""
Import groin crops around the lower end of each iliac vein of the SonoGym TotalSegmentator subjects, for the vein
compressibility study (DEC-061, pre-registered).

    ./.venv/bin/python -m ReconstructionPipeline.data.import_vein_subjects --archive <assets.tar.gz> --out data/vein

Same source and frame convention as `import_liver_subjects.py` (frame Q = RAS - c: +x right, +y ANTERIOR, +z superior,
i.e. the project's frame P rotated 180 deg about the cranio-caudal axis, so the anterior skin takes the dorsal skin's
role and the FEM / contact / probe code runs unchanged).

Labels (TotalSegmentator v2 `total`): iliac_vena_left 67, iliac_vena_right 68, iliac_artery_left 65 / right 66. Checked
on 16 Nov against SonoGym's STLs for s0015: each STL has the label's shape up to the archive-wide uniform scale
(volume ratio 0.34-0.37, extents x 0.67-0.68 on all three axes; the next-best label is clearly worse).

Per subject and side: site = centroid (x, z) of the vein voxels in the lowest 15 mm of the label (the common femoral
end); crop x, z in [-40, 40] mm around it; y chosen so that the most anterior body voxel in the footprint sits at
y = 95 mm, grid y in [15, 100] (80 mm below the skin top). Outputs `data/vein/sonogym_<id>_<side>/`: ct.nii.gz,
vein.nii.gz, artery.nii.gz, fov.nii.gz, meta.json.
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
VEIN = {"left": 67, "right": 68}
ARTERY = {"left": 65, "right": 66}
GRID_X, GRID_Y, GRID_Z = (-40.0, 40.0), (15.0, 100.0), (-40.0, 40.0)
SKIN_TOP_Y, LOW_END_MM = 95.0, 15.0
MEMBER_RE = re.compile(r"^assets/data/HumanModels/(selected_dataset/(s\d{4})/ct\.nii\.gz|"
                       r"selected_dataset_stl/(s\d{4})/combined_label_map\.nii\.gz)$")


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
            out.setdefault(sid, {})["ct" if mm.group(2) else "label"] = os.path.join(dest, m.name)
        tf.extractall(dest, members=members, filter="data")
    return {k: v for k, v in out.items() if {"ct", "label"} <= set(v)}


def convert(sid, side, ct, lbl, A, zooms, out_dir):
    vox = np.argwhere(lbl == VEIN[side])
    if len(vox) < 500:
        raise ValueError(f"{sid} {side}: vein label missing or tiny ({len(vox)} voxels)")
    ras = nib.affines.apply_affine(A, vox)
    low = ras[ras[:, 2] <= ras[:, 2].min() + LOW_END_MM]
    c = np.array([low[:, 0].mean(), np.nan, low[:, 2].mean()])
    xs = np.arange(GRID_X[0], GRID_X[1] + 0.5, 1.0)
    zs = np.arange(GRID_Z[0], GRID_Z[1] + 0.5, 1.0)
    cy0 = float(low[:, 1].mean())

    def sample(ys, arr, order, cval):
        QX, QY, QZ = np.meshgrid(xs, ys, zs, indexing="ij")
        r = np.stack([QX + c[0], QY + cy0, QZ + c[2]], -1).reshape(-1, 3)
        v = nib.affines.apply_affine(np.linalg.inv(A), r).T.reshape(3, *QX.shape)
        return ndimage.map_coordinates(arr, v, order=order, mode="constant", cval=cval)

    ys_wide = np.arange(-60.0, 200.0 + 0.5, 1.0)
    body_w = body_mask(sample(ys_wide, ct, 1, -1000.0).astype(np.float32))
    top = np.where(body_w.any(1), body_w.shape[1] - 1 - np.argmax(body_w[:, ::-1, :], axis=1), -1)
    y_top = float(ys_wide[int(top.max())])
    ys = np.arange(GRID_Y[0], GRID_Y[1] + 0.5, 1.0) + y_top - SKIN_TOP_Y
    ct_q = sample(ys, ct, 1, -1000.0).astype(np.float32)
    lb_q = sample(ys, lbl, 0, 0).astype(np.int16)
    fov = sample(ys, np.ones(ct.shape, np.float32), 0, 0.0) > 0.5
    body = body_mask(ct_q)
    ct_q[~body] = -1000.0
    vein = (lb_q == VEIN[side]) & body
    artery = (lb_q == ARTERY[side]) & body
    os.makedirs(out_dir, exist_ok=True)
    aff = np.eye(4)
    aff[:3, 3] = [GRID_X[0], GRID_Y[0], GRID_Z[0]]
    for name, arr in (("ct", ct_q), ("vein", vein.astype(np.uint8)), ("artery", artery.astype(np.uint8)),
                      ("fov", fov.astype(np.uint8))):
        nib.save(nib.Nifti1Image(arr, aff), os.path.join(out_dir, f"{name}.nii.gz"))
    c_ras = np.array([c[0], cy0 + y_top - SKIN_TOP_Y, c[2]])
    meta = {"subject": sid, "side": side, "study": "DEC-061 vein compressibility",
            "source": "huggingface.co/datasets/yunkao/SonoGym_assets_models (assets.tar.gz)",
            "licence": "CC BY 4.0 (SonoGym assets); CTs and labels from TotalSegmentator (Wasserthal et al. 2023)",
            "frame": "Q (mm): Q = RAS - c_ras; +x right, +y anterior, +z superior (P rotated 180 deg about z)",
            "c_ras_mm": c_ras.tolist(), "grid_mm": {"x": GRID_X, "y": GRID_Y, "z": GRID_Z, "spacing": 1.0},
            "labels": {"vein": VEIN[side], "artery": ARTERY[side]}, "source_spacing_mm": [float(v) for v in zooms],
            "qa": {"vein_voxels_in_crop": int(vein.sum()), "artery_voxels_in_crop": int(artery.sum()),
                   "outside_fov_voxels": int((~fov).sum())}}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "vein"))
    ap.add_argument("--subjects", default="")
    ap.add_argument("--archive", required=True)
    args = ap.parse_args(argv)
    subjects = [s for s in args.subjects.split(",") if s] or None
    with tempfile.TemporaryDirectory() as tmp:
        files = extract(args.archive, subjects, tmp)
        for sid in sorted(files):
            img, li = nib.load(files[sid]["ct"]), nib.load(files[sid]["label"])
            if nib.aff2axcodes(img.affine) != ("R", "A", "S") or not np.allclose(img.affine, li.affine, atol=1e-3):
                raise ValueError(f"{sid}: unexpected grid")
            ct = np.asarray(img.dataobj, dtype=np.float32)
            lbl = np.asarray(li.dataobj).astype(np.int16)
            for side in ("left", "right"):
                meta = convert(sid, side, ct, lbl, img.affine, img.header.get_zooms()[:3],
                               os.path.join(args.out, f"sonogym_{sid}_{side}"))
                q = meta["qa"]
                print(f"{sid} {side}: vein {q['vein_voxels_in_crop']} vox, artery {q['artery_voxels_in_crop']} vox, "
                      f"outside FOV {q['outside_fov_voxels']}", flush=True)


if __name__ == "__main__":
    main()
