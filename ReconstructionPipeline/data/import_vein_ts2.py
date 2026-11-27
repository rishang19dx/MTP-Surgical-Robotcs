"""
Import groin-vein crops from TotalSegmentator v2 (DEC-071 vein study), with the DEC-061 crop and frame.

    ./.venv/bin/python -m ReconstructionPipeline.data.import_vein_ts2 --zip <Totalsegmentator_dataset_v201.zip> \
        --presence results/organ_presence_20261126_184554.json --out data/vein_ts2 [--workers 16]

Subjects and sides: every (subject, side) whose iliac vein has >= 1000 voxels and whose groin end lies >= 15 mm above the
lowest axial slice (`organ_presence.py`, DEC-065), so the importer's site (lowest 15 mm of the label) is inside the CT.
The per-structure masks (iliac_vena_left/right, iliac_artery_left/right) are merged into one label map with the v2
`total` IDs of `import_vein_subjects` (67/68 vein, 65/66 artery), both volumes reoriented to RAS (no resampling), and
the crop, frame Q and outputs are exactly `import_vein_subjects.convert`. Output `data/vein_ts2/ts2_<id>_<side>/`.
Writes results/vein_ts2_import_<stamp>.json (counts and per-vein QA).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

from ReconstructionPipeline.data.import_vein_subjects import ARTERY, VEIN, convert

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_VOX = 1000


def one(zpath, sid, sides, out_root):
    recs = []
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zpath) as zf:
        def get(member, name):
            p = os.path.join(tmp, name)
            with zf.open(member) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return nib.as_closest_canonical(nib.load(p))
        ct_img = get(f"{sid}/ct.nii.gz", "ct.nii.gz")
        lbl = np.zeros(ct_img.shape, np.int16)
        for side in ("left", "right"):
            for name, lid in ((f"iliac_vena_{side}", VEIN[side]), (f"iliac_artery_{side}", ARTERY[side])):
                im = get(f"{sid}/segmentations/{name}.nii.gz", name + ".nii.gz")
                if im.shape != ct_img.shape or not np.allclose(im.affine, ct_img.affine, atol=1e-3):
                    raise ValueError(f"{sid}: {name} grid differs from CT")
                lbl[np.asarray(im.dataobj) > 0] = lid
        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        for side in sides:
            out = os.path.join(out_root, f"ts2_{sid}_{side}")
            try:
                meta = convert(sid, side, ct, lbl, ct_img.affine, ct_img.header.get_zooms()[:3], out)
            except ValueError as e:
                recs.append({"subject": sid, "side": side, "status": "reject", "reason": str(e)})
                continue
            meta["source"] = ("TotalSegmentator v2.0.1 Totalsegmentator_dataset_v201.zip (HF mirror "
                              "HajihajihaJimmy/TotalSegmentator_v2; DEC-064)")
            meta["licence"] = "CC BY 4.0 (TotalSegmentator, Wasserthal et al. 2023)"
            meta["study"] = "DEC-071 large-strain vein study"
            json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
            recs.append({"subject": sid, "side": side, "status": "accept", "qa": meta["qa"]})
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--presence", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "vein_ts2"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--subjects", default="")
    a = ap.parse_args(argv)
    pres = json.load(open(a.presence))["records"]
    want = {}
    for r in pres:
        for side in ("left", "right"):
            if (r.get(f"iliac_vena_{side}") or 0) >= MIN_VOX and r.get(f"iliac_vena_{side}_end_in_fov") is True:
                want.setdefault(r["subject"], []).append(side)
    if a.subjects:
        keep = set(a.subjects.split(","))
        want = {k: v for k, v in want.items() if k in keep}
    os.makedirs(a.out, exist_ok=True)
    recs = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, a.zip, sid, sides, a.out): sid for sid, sides in sorted(want.items())}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                recs += f.result()
            except Exception as e:  # counted, never silently dropped
                recs.append({"subject": futs[f], "status": "error", "reason": f"{type(e).__name__}: {e}"})
            if i % 50 == 0:
                print(f"[{i}/{len(futs)}]", flush=True)
    recs.sort(key=lambda r: (r["subject"], r.get("side", "")))
    n_acc = sum(r["status"] == "accept" for r in recs)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ROOT, "results", f"vein_ts2_import_{stamp}.json")
    json.dump({"zip": os.path.basename(a.zip), "presence": os.path.relpath(a.presence, ROOT), "accepted_veins": n_acc,
               "subjects": len(want), "records": recs}, open(path, "w"), indent=1)
    print(f"wrote {path}: {n_acc} veins accepted from {len(want)} subjects")


if __name__ == "__main__":
    main()
