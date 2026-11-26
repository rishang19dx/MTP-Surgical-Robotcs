"""
Import TotalSegmentator v2 subjects into frame P, with a fixed inclusion filter (DEC-064).

    ./.venv/bin/python -m ReconstructionPipeline.data.import_totalseg_v2 \
        --zip <Totalsegmentator_dataset_v201.zip> --out data/patients_ts2 [--subjects s0001,s0002] [--workers 16]
    ./.venv/bin/python -m ReconstructionPipeline.data.import_totalseg_v2 --zip <zip> --verify-against data/patients

Source: TotalSegmentator v2.0.1 (Wasserthal et al. 2023, CC BY 4.0, Zenodo record 10047292), file
`Totalsegmentator_dataset_v201.zip`.  Zenodo blocks the institute network, so the zip comes from the Hugging Face
mirror `HajihajihaJimmy/TotalSegmentator_v2`.  It is accepted as the original only if `--verify-against`
reproduces the 12 SonoGym subjects in `data/patients` (whose CTs came from SonoGym's own copy of the dataset)
voxel for voxel.

Each subject is converted by `import_sonogym_subjects.convert` (same frame, grid, body mask and QA), after the
per-structure masks are merged into one label map with the v2 `total` IDs and both volumes are reoriented to
RAS (`nib.as_closest_canonical`, a voxel permutation, no resampling).

Inclusion rules (fixed before any subject was looked at; counts per rule go to results/):
  R1 labels   : L1-L5 and the sacrum each have >= 1000 voxels; autochthon left and right each have >= 1000.
  R2 complete : no L1-L5 voxel lies on the first or last axial (S-I) slice of the source CT.
  R3 dorsum   : in the P columns |x| <= 40 mm (all z of the grid), >= 99 % of the body columns have their dorsal
                skin + 5 mm inside the source field of view, and the skin never reaches the top of the P grid.
                (Otherwise "air" above the skin may be the edge of the CT reconstruction, not air.)
  R4 new      : subjects already in data/patients (SonoGym) are skipped; they serve as the identity check.
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
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from ReconstructionPipeline.data.import_sonogym_subjects import (GRID_X, GRID_Y, GRID_Z, LABELS, convert,
                                                                 sample_l4_centroid_P)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LUMBAR = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5"]
MIN_VOX = 1000
DORSUM_HALF_X = 40.0
DORSUM_MARGIN_MM = 5.0
DORSUM_MIN_FRAC = 0.99
R = np.diag([-1.0, -1.0, 1.0])


def list_subjects(zpath: str) -> List[str]:
    with zipfile.ZipFile(zpath) as zf:
        return sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("/ct.nii.gz")})


def _extract(zf: zipfile.ZipFile, sid: str, dest: str) -> Dict[str, str]:
    names = {"ct": f"{sid}/ct.nii.gz"}
    names.update({k: f"{sid}/segmentations/{k}.nii.gz" for k in LABELS})
    out = {}
    for k, member in names.items():
        p = os.path.join(dest, f"{k}.nii.gz")
        with zf.open(member) as src, open(p, "wb") as dst:   # explicit member names only, flat dest paths
            shutil.copyfileobj(src, dst)
        out[k] = p
    return out


def dorsum_coverage(out_dir: str, src_shape, src_affine: np.ndarray) -> Dict:
    """R3: fraction of body columns (|x| <= 40 mm) whose skin + 5 mm maps inside the source CT volume."""
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    skin = np.asarray(nib.load(os.path.join(out_dir, "segmentations", "skin.nii.gz")).dataobj) > 0
    xs = np.arange(GRID_X[0], GRID_X[1] + 0.5, 1.0)
    zs = np.arange(GRID_Z[0], GRID_Z[1] + 0.5, 1.0)
    has = skin.any(axis=1)                                                   # (nx, nz)
    top = skin.shape[1] - 1 - np.argmax(skin[:, ::-1, :], axis=1)
    sel = has & (np.abs(xs)[:, None] <= DORSUM_HALF_X)
    ix, iz = np.nonzero(sel)
    P = np.stack([xs[ix], GRID_Y[0] + top[ix, iz] + DORSUM_MARGIN_MM, zs[iz]], -1)
    l4_ras, l4_P = np.asarray(meta["l4_centroid_ras_mm"]), np.asarray(meta["l4_centroid_P_mm"])
    vox = nib.affines.apply_affine(np.linalg.inv(src_affine), l4_ras + (P - l4_P) @ R.T)
    inside = np.all((vox >= -0.5) & (vox <= np.asarray(src_shape) - 0.5), axis=1)
    frac = float(inside.mean()) if len(inside) else 0.0
    return {"columns": int(len(inside)), "inside_fraction": frac,
            "skin_touches_fov_top": bool(meta["qa"]["skin_touches_fov_top"])}


def process(zpath: str, sid: str, out_root: str, l4_P: np.ndarray, keep_rejected: bool = False) -> Dict:
    rec: Dict = {"subject": sid}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zpath) as zf:
        try:
            files = _extract(zf, sid, tmp)
        except KeyError as e:
            rec.update(status="reject", rule="R1", reason=f"missing member {e}")
            return rec
        ct_img = nib.as_closest_canonical(nib.load(files["ct"]))
        lbl = np.zeros(ct_img.shape, np.int16)
        counts, touches = {}, []
        for name, lid in LABELS.items():
            m_img = nib.as_closest_canonical(nib.load(files[name]))
            if m_img.shape != ct_img.shape or not np.allclose(m_img.affine, ct_img.affine, atol=1e-3):
                rec.update(status="reject", rule="R1", reason=f"{name} grid differs from CT")
                return rec
            m = np.asarray(m_img.dataobj) > 0
            counts[name] = int(m.sum())
            lbl[m] = lid
            if name in LUMBAR and (m[:, :, 0].any() or m[:, :, -1].any()):
                touches.append(name)
        rec["label_voxels"] = counts
        low = [k for k, v in counts.items() if v < MIN_VOX]
        if low:
            rec.update(status="reject", rule="R1", reason=f"labels < {MIN_VOX} voxels: {low}")
            return rec
        if touches:
            rec.update(status="reject", rule="R2", reason=f"on first/last axial slice: {touches}")
            return rec
        ct_path, lbl_path = os.path.join(tmp, "ct_ras.nii.gz"), os.path.join(tmp, "lbl_ras.nii.gz")
        nib.save(ct_img, ct_path)
        nib.save(nib.Nifti1Image(lbl, ct_img.affine), lbl_path)
        out_dir = os.path.join(out_root, f"ts2_{sid}")
        try:
            meta = convert(sid, ct_path, lbl_path, out_dir, l4_P)
        except ValueError as e:
            shutil.rmtree(out_dir, ignore_errors=True)
            rec.update(status="reject", rule="convert", reason=str(e))
            return rec
        cov = dorsum_coverage(out_dir, ct_img.shape, ct_img.affine)
        rec["qa"], rec["dorsum"] = meta["qa"], cov
        if cov["inside_fraction"] < DORSUM_MIN_FRAC or cov["skin_touches_fov_top"]:
            rec.update(status="reject", rule="R3",
                       reason=f"dorsum inside {cov['inside_fraction']:.3f}, top {cov['skin_touches_fov_top']}")
            if not keep_rejected:
                shutil.rmtree(out_dir, ignore_errors=True)
            return rec
        meta["source"] = ("TotalSegmentator v2.0.1 Totalsegmentator_dataset_v201.zip "
                          "(HF mirror HajihajihaJimmy/TotalSegmentator_v2; DEC-064)")
        meta["licence"] = "CC BY 4.0 (TotalSegmentator, Wasserthal et al. 2023)"
        meta["dorsum_coverage"] = cov
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        rec["status"] = "accept"
        return rec


def verify(zpath: str, ref_root: str, l4_P: np.ndarray) -> List[Dict]:
    """Re-import the SonoGym subjects from the zip and compare with data/patients voxel for voxel."""
    rows = []
    for d in sorted(os.listdir(ref_root)):
        if not d.startswith("sonogym_s"):
            continue
        sid = d.split("_", 1)[1]
        ref = os.path.join(ref_root, d)
        with tempfile.TemporaryDirectory() as tmp:
            rec = process(zpath, sid, tmp, l4_P, keep_rejected=True)
            new = os.path.join(tmp, f"ts2_{sid}")
            row = {"subject": sid, "status": rec.get("status"), "rule": rec.get("rule")}
            if os.path.isdir(new):
                a = np.asarray(nib.load(os.path.join(ref, "ct.nii.gz")).dataobj)
                b = np.asarray(nib.load(os.path.join(new, "ct.nii.gz")).dataobj)
                row["ct_max_abs_diff"] = float(np.abs(a.astype(np.float64) - b).max())
                lab = {}
                for name in list(LABELS) + ["skin"]:
                    la = np.asarray(nib.load(os.path.join(ref, "segmentations", f"{name}.nii.gz")).dataobj)
                    lb = np.asarray(nib.load(os.path.join(new, "segmentations", f"{name}.nii.gz")).dataobj)
                    lab[name] = int((la != lb).sum())
                row["label_voxels_differing"] = lab
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "patients_ts2"))
    ap.add_argument("--subjects", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-against", default="")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    l4_P = sample_l4_centroid_P()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.verify_against:
        rows = verify(args.zip, args.verify_against, l4_P)
        path = os.path.join(args.results, f"totalseg_v2_identity_{stamp}.json")
        json.dump({"zip": os.path.basename(args.zip), "reference": args.verify_against, "rows": rows},
                  open(path, "w"), indent=2)
        print("wrote", path)
        return

    existing = {d.split("_", 1)[1] for d in os.listdir(os.path.join(ROOT, "data", "patients"))
                if d.startswith("sonogym_s")}
    subjects = [s for s in args.subjects.split(",") if s] or list_subjects(args.zip)
    todo = [s for s in subjects if s not in existing]
    os.makedirs(args.out, exist_ok=True)
    recs = [{"subject": s, "status": "reject", "rule": "R4", "reason": "already in data/patients"}
            for s in subjects if s in existing]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process, args.zip, s, args.out, l4_P): s for s in todo}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                r = f.result()
            except Exception as e:  # a corrupt file must not stop the import; it is counted
                r = {"subject": s, "status": "error", "reason": f"{type(e).__name__}: {e}"}
            recs.append(r)
            if i % 25 == 0 or i == len(todo):
                n_acc = sum(x["status"] == "accept" for x in recs)
                print(f"[{i}/{len(todo)}] accepted so far {n_acc}", flush=True)
    recs.sort(key=lambda r: r["subject"])
    by_rule: Dict[str, int] = {}
    for r in recs:
        key = "accept" if r["status"] == "accept" else (r.get("rule") or r["status"])
        by_rule[key] = by_rule.get(key, 0) + 1
    summary = {"zip": os.path.basename(args.zip), "n_subjects": len(subjects), "counts": by_rule,
               "rules": {"min_vox": MIN_VOX, "dorsum_half_x_mm": DORSUM_HALF_X,
                         "dorsum_margin_mm": DORSUM_MARGIN_MM, "dorsum_min_frac": DORSUM_MIN_FRAC},
               "records": recs}
    jpath = os.path.join(args.results, f"totalseg_v2_import_{stamp}.json")
    json.dump(summary, open(jpath, "w"), indent=2)
    acc = [r for r in recs if r["status"] == "accept"]
    soft = sorted(r["qa"]["midline_soft_tissue_mm"] for r in acc if r.get("qa"))
    with open(jpath[:-5] + ".md", "w") as fh:
        fh.write(f"# TotalSegmentator v2 import ({stamp}, DEC-064)\n\n")
        fh.write(f"Source zip `{summary['zip']}`, {len(subjects)} subjects. Output `{os.path.relpath(args.out, ROOT)}`.\n\n")
        fh.write("| Outcome | Subjects |\n| :--- | ---: |\n")
        for k in sorted(by_rule):
            fh.write(f"| {k} | {by_rule[k]} |\n")
        if soft:
            q = np.percentile(soft, [0, 25, 50, 75, 100])
            fh.write("\nMidline soft tissue at L4 over accepted subjects (mm): min / q1 / median / q3 / max = "
                     + " / ".join(f"{v:.0f}" for v in q) + "\n")
    print("wrote", jpath, by_rule)


if __name__ == "__main__":
    main()
