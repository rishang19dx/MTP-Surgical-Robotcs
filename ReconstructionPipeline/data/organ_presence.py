"""
Liver and iliac-vein presence in every TotalSegmentator v2 subject (DEC-065 Stage 0 eligibility).

    ./.venv/bin/python -m ReconstructionPipeline.data.organ_presence --zip <Totalsegmentator_dataset_v201.zip> \
        [--workers 16]

The liver and vein studies (DEC-059..063) do not need the lumbar spine, so this covers all 1228 subjects, not only
the 399 spine patients. Per subject, from the per-structure masks (reoriented to RAS, no resampling):
  * voxel count of liver, iliac_vena_left, iliac_vena_right;
  * liver_cut: the liver touches the first or last axial slice (organ not fully in the FOV);
  * <side>_end_in_fov: the iliac vein's lowest voxel is >= 15 mm above the lowest axial slice, so the groin end used
    by the vein importer (lowest 15 mm of the label) is inside the CT and not cut by the FOV.
No inclusion decision is taken here; each study fixes its own rule from these columns.
Writes results/organ_presence_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRUCTS = ["liver", "iliac_vena_left", "iliac_vena_right"]
VEIN_END_MM = 15.0


def one(args):
    zpath, sid = args
    rec = {"subject": sid}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zpath) as zf:
        for k in STRUCTS:
            p = os.path.join(tmp, f"{k}.nii.gz")
            try:
                with zf.open(f"{sid}/segmentations/{k}.nii.gz") as src, open(p, "wb") as dst:
                    dst.write(src.read())
            except KeyError:
                rec[k] = None
                continue
            img = nib.as_closest_canonical(nib.load(p))
            m = np.asarray(img.dataobj) > 0
            rec[k] = int(m.sum())
            if not m.any():
                continue
            zs = np.nonzero(m.any(axis=(0, 1)))[0]
            if k == "liver":
                rec["liver_cut"] = bool(zs.min() == 0 or zs.max() == m.shape[2] - 1)
            else:
                rec[f"{k}_end_in_fov"] = bool(zs.min() * float(img.header.get_zooms()[2]) >= VEIN_END_MM)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-vox", type=int, default=1000)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    with zipfile.ZipFile(args.zip) as zf:
        subjects = sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("/ct.nii.gz")})
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, [(args.zip, s) for s in subjects], chunksize=4))
    mv = args.min_vox
    liver_ok = sum((r.get("liver") or 0) >= mv and r.get("liver_cut") is False for r in recs)
    vein = {s: sum((r.get(f"iliac_vena_{s}") or 0) >= mv and r.get(f"iliac_vena_{s}_end_in_fov") is True for r in recs)
            for s in ("left", "right")}
    both = sum(all((r.get(f"iliac_vena_{s}") or 0) >= mv and r.get(f"iliac_vena_{s}_end_in_fov") is True
                   for s in ("left", "right")) for r in recs)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"organ_presence_{stamp}")
    summary = {"zip": os.path.basename(args.zip), "n_subjects": len(subjects), "min_vox": mv,
               "liver_whole_in_fov": liver_ok, "iliac_vein_end_in_fov": vein, "both_veins": both}
    json.dump({"summary": summary, "records": recs}, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write(
        f"# Liver and iliac-vein presence ({stamp}, DEC-065)\n\n"
        f"{len(subjects)} TotalSegmentator v2 subjects; a structure counts if it has >= {mv} voxels.\n\n"
        "| Structure | Subjects |\n| :--- | ---: |\n"
        f"| liver, whole organ inside the FOV | {liver_ok} |\n"
        f"| left iliac vein, groin end inside the FOV | {vein['left']} |\n"
        f"| right iliac vein, groin end inside the FOV | {vein['right']} |\n"
        f"| both veins | {both} |\n")
    print("wrote", base, summary)


if __name__ == "__main__":
    main()
