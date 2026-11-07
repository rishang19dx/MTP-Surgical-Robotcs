"""
Import the SonoGym TotalSegmentator subjects into the project's canonical patient frame (DEC-041, task T4).

    ./.venv/bin/python -m ReconstructionPipeline.data.import_sonogym_subjects --out data/patients
    ./.venv/bin/python -m ReconstructionPipeline.data.import_sonogym_subjects --subjects s0010,s0014 --meshes

Source: Hugging Face dataset `yunkao/SonoGym_assets_models` (`assets.tar.gz`, CC BY 4.0; SonoGym, Ao et al.
2025, arXiv 2507.01152; CTs and labels from TotalSegmentator, Wasserthal et al. 2023).  Only
`selected_dataset/<id>/ct.nii.gz` and `selected_dataset_stl/<id>/combined_label_map.nii.gz` are extracted,
by explicit member name, into a fresh temporary directory.

Why a resampling (and not the files as they are):
  * The SonoGym CTs are true RAS (checked: the spine lies at lower voxel Y than the body centroid in all 12, so
    posterior is -Y).  Frame P is LPS (+Y = dorsal skin, AGENTS.md invariant 4).  The map RAS -> P is the proper
    rotation diag(-1, -1, 1) (det +1, no mirroring).
  * The rest of the code assumes the canonical lumbar frame of `data/sample_totalsegmentator_patient`: a 1 mm
    grid x in [-70, 70], z in [-90, 85] mm with the L4 scan window (c1 in [-32, 32], c2 in [-55, -10]) and the
    target ROI given in absolute P coordinates.  Each subject is therefore placed so that its L4 centroid
    lands on the sample patient's L4 centroid.  The dorsal skin is NOT aligned: soft-tissue thickness stays
    patient-specific (needed for H2).  The y range is extended to [-50, 120] mm so that thick patients keep
    their skin inside the field of view.
  * Patients lie supine in a CT scanner, so the table touches the dorsal skin.  The body is the largest
    connected component of HU > -200 after a small opening; everything else (table, cushions, noise) is set
    to air (-1000 HU), otherwise the skin height map would follow the table.

Label IDs (TotalSegmentator v2 `total` task): sacrum 25, L5 27, L4 28, L3 29, L2 30, L1 31; paraspinal muscle
"autochthon" left 86 / right 87 (DEC-048 Thread B; checked against SonoGym's autochthon_*.stl for s0010, s0014, s0028:
centroid distance 4-15 voxels, next label >= 24 voxels away).  L4 = 28 was
checked against SonoGym's own `vertebrae_L4.stl` for all 12 subjects (centroid distance 3-5 voxels; the
neighbouring labels 27 / 29 are 18-25 voxels away).

The output folder per subject matches the sample patient: `ct.nii.gz`, `segmentations/<structure>.nii.gz`,
`meta.json` (provenance, transform, QA numbers).  NIfTI files are gitignored; `meta.json` is small.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tempfile
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_totalsegmentator_patient")
HF_REPO = "yunkao/SonoGym_assets_models"
LABELS = {"sacrum": 25, "vertebrae_L5": 27, "vertebrae_L4": 28, "vertebrae_L3": 29, "vertebrae_L2": 30,
          "vertebrae_L1": 31, "autochthon_left": 86, "autochthon_right": 87}
GRID_X = (-70.0, 70.0)
GRID_Y = (-50.0, 120.0)
GRID_Z = (-90.0, 85.0)
MEMBER_RE = re.compile(r"^assets/data/HumanModels/(selected_dataset/(s\d{4})/ct\.nii\.gz|"
                       r"selected_dataset_stl/(s\d{4})/combined_label_map\.nii\.gz)$")


def sample_l4_centroid_P() -> np.ndarray:
    """L4 centroid of the sample patient in frame P (its NIfTI affine maps voxels to P directly)."""
    img = nib.load(os.path.join(SAMPLE_DIR, "segmentations", "vertebrae_L4.nii.gz"))
    vox = np.argwhere(np.asarray(img.dataobj) > 0)
    return nib.affines.apply_affine(img.affine, vox.mean(0))


def extract(archive: str, subjects: Optional[List[str]], dest: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
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
    return {k: v for k, v in out.items() if "ct" in v and "label" in v}


def body_mask(ct: np.ndarray) -> np.ndarray:
    """Largest connected HU > -200 component after a 2-voxel opening, then re-grown into the raw mask."""
    raw = ct > -200.0
    opened = ndimage.binary_opening(raw, iterations=2)
    lab, n = ndimage.label(opened)
    if n == 0:
        return raw
    sizes = ndimage.sum(opened, lab, index=np.arange(1, n + 1))
    core = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_dilation(core, iterations=2) & raw


def convert(sid: str, ct_path: str, lbl_path: str, out_dir: str, l4_P: np.ndarray) -> Dict:
    ct_img, lbl_img = nib.load(ct_path), nib.load(lbl_path)
    A = ct_img.affine
    if nib.aff2axcodes(A) != ("R", "A", "S"):
        raise ValueError(f"{sid}: expected RAS axis codes, got {nib.aff2axcodes(A)}")
    # Some scans are slightly oblique (s0004, s0012: up to ~3 deg).  The resampling below uses the full
    # inverse affine, so obliquity is handled exactly; it is recorded in meta.json.
    Rn = A[:3, :3] / np.linalg.norm(A[:3, :3], axis=0)
    oblique_deg = float(np.degrees(np.arccos(np.clip(np.diag(Rn).min(), -1.0, 1.0))))
    if not np.allclose(A, lbl_img.affine, rtol=0.0, atol=1e-3) or ct_img.shape != lbl_img.shape:
        raise ValueError(f"{sid}: CT and label grids differ")
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    lbl = np.asarray(lbl_img.dataobj).astype(np.int16)
    l4_vox = np.argwhere(lbl == LABELS["vertebrae_L4"])
    if len(l4_vox) < 1000:
        raise ValueError(f"{sid}: L4 label missing or tiny ({len(l4_vox)} voxels)")
    l4_ras = nib.affines.apply_affine(A, l4_vox.mean(0))

    # P grid -> RAS: RAS = l4_ras + R^T (P - l4_P) with R = diag(-1,-1,1) (R^T = R)
    R = np.diag([-1.0, -1.0, 1.0])
    xs = np.arange(GRID_X[0], GRID_X[1] + 0.5, 1.0)
    ys = np.arange(GRID_Y[0], GRID_Y[1] + 0.5, 1.0)
    zs = np.arange(GRID_Z[0], GRID_Z[1] + 0.5, 1.0)
    PX, PY, PZ = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([PX, PY, PZ], -1).reshape(-1, 3)
    ras = l4_ras + (P - l4_P) @ R.T
    vox = nib.affines.apply_affine(np.linalg.inv(A), ras).T.reshape(3, *PX.shape)
    ct_P = ndimage.map_coordinates(ct, vox, order=1, mode="constant", cval=-1000.0).astype(np.float32)
    lbl_P = ndimage.map_coordinates(lbl, vox, order=0, mode="constant", cval=0).astype(np.int16)

    body = body_mask(ct_P)
    removed = float(((ct_P > -200.0) & ~body).sum())
    ct_P[~body] = -1000.0

    os.makedirs(os.path.join(out_dir, "segmentations"), exist_ok=True)
    aff = np.eye(4)
    aff[:3, 3] = [GRID_X[0], GRID_Y[0], GRID_Z[0]]   # same convention as the sample patient (voxel -> P mm)
    nib.save(nib.Nifti1Image(ct_P, aff), os.path.join(out_dir, "ct.nii.gz"))
    for name, lid in LABELS.items():
        nib.save(nib.Nifti1Image((lbl_P == lid).astype(np.uint8), aff),
                 os.path.join(out_dir, "segmentations", f"{name}.nii.gz"))
    # skin shell (top 3 voxels of the body along +Y), as the sample export
    top = body.shape[1] - 1 - np.argmax(body[:, ::-1, :], axis=1)            # (nx, nz)
    has = body.any(axis=1)
    jj = np.arange(body.shape[1])[None, :, None]
    skin = has[:, None, :] & (jj <= top[:, None, :]) & (jj >= top[:, None, :] - 2)
    nib.save(nib.Nifti1Image(skin.astype(np.uint8), aff), os.path.join(out_dir, "segmentations", "skin.nii.gz"))

    # QA numbers along the midline at L4 (x = 0, z = L4 centroid): skin height and dorsal bone top
    ix, iz = int(round(l4_P[0] - GRID_X[0])), int(round(l4_P[2] - GRID_Z[0]))
    skin_y = GRID_Y[0] + float(top[ix, iz]) if has[ix, iz] else float("nan")
    bone_col = np.nonzero((lbl_P[ix, :, iz] >= 25) & (lbl_P[ix, :, iz] <= 31))[0]
    bone_top_y = GRID_Y[0] + float(bone_col.max()) if len(bone_col) else float("nan")
    # paraspinal column 20 mm lateral: subcutaneous + muscle thickness over the lamina region
    ixp = int(round(l4_P[0] + 20.0 - GRID_X[0]))
    bone_p = np.nonzero((lbl_P[ixp, :, iz] >= 25) & (lbl_P[ixp, :, iz] <= 31))[0]
    skin_p = GRID_Y[0] + float(top[ixp, iz]) if has[ixp, iz] else float("nan")
    fat_col = (ct_P[ixp, :, iz] < -20.0) & (ct_P[ixp, :, iz] > -200.0)
    meta = {
        "subject": sid, "source": f"huggingface.co/datasets/{HF_REPO} (assets.tar.gz)",
        "licence": "CC BY 4.0 (SonoGym assets); CTs and labels from TotalSegmentator (Wasserthal et al. 2023)",
        "frame": "P (LPS, mm): P = R (RAS - l4_ras) + l4_P, R = diag(-1,-1,1)",
        "l4_centroid_ras_mm": l4_ras.tolist(), "l4_centroid_P_mm": l4_P.tolist(),
        "grid_mm": {"x": GRID_X, "y": GRID_Y, "z": GRID_Z, "spacing": 1.0},
        "source_spacing_mm": [float(v) for v in ct_img.header.get_zooms()[:3]],
        "source_shape": list(ct_img.shape), "source_oblique_deg": oblique_deg, "non_body_voxels_removed": removed,
        "qa": {
            "midline_skin_y_mm": skin_y, "midline_bone_top_y_mm": bone_top_y,
            "midline_soft_tissue_mm": skin_y - bone_top_y,
            "paraspinal20_soft_tissue_mm": (skin_p - (GRID_Y[0] + float(bone_p.max()))) if len(bone_p) else None,
            "paraspinal20_fat_mm": float(fat_col.sum()),
            "skin_touches_fov_top": bool(np.any(has & (top >= body.shape[1] - 1))),
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "patients"))
    ap.add_argument("--subjects", default="", help="comma-separated ids (default: all in the archive)")
    ap.add_argument("--archive", default="", help="local assets.tar.gz (default: download from Hugging Face)")
    ap.add_argument("--meshes", action="store_true", help="also export OBJ meshes (PyBullet / Slicer)")
    args = ap.parse_args(argv)

    archive = args.archive
    if not archive:
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(HF_REPO, "assets.tar.gz", repo_type="dataset")
    subjects = [s for s in args.subjects.split(",") if s] or None
    l4_P = sample_l4_centroid_P()
    with tempfile.TemporaryDirectory() as tmp:
        files = extract(archive, subjects, tmp)
        for sid in sorted(files):
            out = os.path.join(args.out, f"sonogym_{sid}")
            meta = convert(sid, files[sid]["ct"], files[sid]["label"], out, l4_P)
            q = meta["qa"]
            print(f"{sid}: midline soft tissue {q['midline_soft_tissue_mm']:.1f} mm, paraspinal "
                  f"{q['paraspinal20_soft_tissue_mm'] or float('nan'):.1f} mm (fat {q['paraspinal20_fat_mm']:.0f} mm), "
                  f"skin at FOV top: {q['skin_touches_fov_top']}, removed {meta['non_body_voxels_removed']:.0f} vox",
                  flush=True)
            if args.meshes:
                from ReconstructionPipeline.data.prepare_patient_assets import PatientAssetGenerator
                PatientAssetGenerator.export_patient_meshes(out)


if __name__ == "__main__":
    main()
