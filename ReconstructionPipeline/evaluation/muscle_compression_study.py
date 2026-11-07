"""
Thread B (DEC-048): paraspinal muscle ("autochthon") thickness under probe compression.  Gate + H4 + H5 (DEC-050).

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_compression_study.py [--patients sonogym]

Per patient (the SonoGym subjects; the sample patient has no muscle label), 6 paraspinal poses (x = +-20 mm over the
erector spinae, z in {-45, -32.5, -20} mm, untilted) and forces F (default 1-5 N, plus 8 and 10 N reported with the
DEC-047 caveat that the linear FEM is within 5 % only up to ~2-5 N).  At each (pose, F), with the indentation found
by bisection on the batched FEM (shared moduli):
  * GT thickness: the muscle label (autochthon_left | right), sliced through the FEM warp at the indented face pose,
    counted along the central scanlines (+-5 mm laterally), median over those columns.  Ground truth: metric only.
  * Reference thickness: the same along the undeformed beam with the probe at first contact (no compression).
  * Image thickness (B-mode only, legacy B-mode as T9, fixed speckle seed): deep boundary = the B-mode bone surface
    (`segment_bmode_batch`) median depth over the central scanlines; superficial boundary = the strongest local
    maximum of the central column-averaged B-mode profile between 5 mm below the face (detector v2) and 5 mm above the bone
    echo (the fat / muscle fascia interface).  Thickness = bone depth - fascia depth.
Outputs per patient and pooled; the gate, H4 and H5 criteria are those pre-registered in DECISIONS DEC-050:
  gate (i)  detection rate >= 50 % and median |image - GT| < 3 mm at 1-2 N;
  gate (ii) median GT compression (1 N -> 5 N) > one axial pixel (0.377 mm);
  H4        median slope of image thickness vs F over 1-5 N < 0 (thinner with force), per patient and pooled;
  H5        a force-linear correction t_corr = t_img + k (F - 1 N), k fitted on TRAIN patients (least squares
            against GT-at-1-N... i.e. the uncompressed reference), reduces the median |t - reference| on TEST
            patients compared with no correction.
Writes results/muscle_compression_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim  # noqa: E402
from mtp_gym.perception import segment_bmode_batch  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter  # noqa: E402
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator  # noqa: E402
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor  # noqa: E402
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator  # noqa: E402

AX = PHYSICAL_SIZE_MM[0] / (IMAGE_SIZE[0] - 1)
LAT = PHYSICAL_SIZE_MM[1] / (IMAGE_SIZE[1] - 1)
SITES = [(x, z) for x in (-20.0, 20.0) for z in (-45.0, -32.5, -20.0)]
# Fascia detector v2 (DEC-050): v1 searched from 2 mm and locked onto the skin ring-down (+~8 mm bias on the train
# patient s0015).  v2 skips the first 5 mm and averages the profile over +-10 mm laterally.  Developed on TRAIN
# patients only (s0015, s0010); fixed before the full run.
FASCIA_MIN_DEPTH_MM = 5.0
FASCIA_HALF_WIDTH_MM = 10.0


def muscle_volume(pdir: str) -> np.ndarray:
    """autochthon_left | right as a (z, y, x) float volume, the same axis order the loader gives the CT."""
    seg = os.path.join(pdir, "segmentations")
    m = None
    for side in ("left", "right"):
        p = os.path.join(seg, f"autochthon_{side}.nii.gz")
        if os.path.exists(p):
            v = np.asarray(nib.load(p).dataobj) > 0
            m = v if m is None else (m | v)
    if m is None:
        raise FileNotFoundError(f"{pdir}: no autochthon labels (re-run import_sonogym_subjects)")
    return m.transpose(2, 1, 0).astype(np.float32)


def central_cols(half_mm: float = 5.0):
    w = IMAGE_SIZE[1]
    c = (w - 1) / 2.0
    k = int(round(half_mm / LAT))
    return slice(int(c) - k, int(c) + k + 1)


@torch.no_grad()
def measure(ps: PatientSim, mus: VolumeSliceExtractor, forces, seed):
    dev = ps.device
    poses = np.stack([ps.model.dorsal_pose(x, z) for x, z in SITES])
    nP, nF = len(poses), len(forces)
    T = torch.as_tensor(np.repeat(poses, nF, 0), dtype=torch.float64, device=dev)
    Ft = torch.as_tensor(np.tile(forces, nP), dtype=torch.float64, device=dev)
    d, d0 = indentation_for_force(ps, T, Ft)
    st = ps.solver.solve(T, d, stiffness=False)
    face = st.probe_face_poses(T)
    lab = mus.extract_slices(face, interpolation="nearest", batched_warp_fn=st.warp_to_reference) > 0.5
    # reference: probe at first contact, undeformed
    T0 = torch.as_tensor(poses, dtype=torch.float64, device=dev)
    d0p = ps.solver.first_contact(T0)
    d0p = torch.where(torch.isfinite(d0p), d0p, torch.zeros_like(d0p))
    face0 = T0.clone()
    face0[:, :3, 3] += d0p[:, None] * T0[:, :3, 1] / T0[:, :3, 1].norm(dim=1, keepdim=True)
    lab0 = mus.extract_slices(face0, interpolation="nearest") > 0.5
    cs = central_cols()
    gt = (lab[:, :, cs].float().sum(1) * AX).median(1).values.view(nP, nF).cpu().numpy()
    ref = (lab0[:, :, cs].float().sum(1) * AX).median(1).values.cpu().numpy()
    # B-mode (legacy, fixed speckle) and image-only measurement
    ct = ps.ct.extract_slices(face, interpolation="bilinear", batched_warp_fn=st.warp_to_reference)
    sim = ModelBasedUltrasoundSimulator(device=dev, axial_resolution_mm=AX, lateral_resolution_mm=LAT)
    gen = torch.Generator(device=dev)
    bm = []
    for i in range(len(ct)):
        gen.manual_seed(seed)
        bm.append(sim.simulate_bmode(ct[i:i + 1], generator=gen)["bmode"])
    bm = torch.cat(bm)
    conf = UltrasoundConfidenceEstimator(device=dev, axial_resolution_mm=AX).compute_confidence_map(bm)
    has, y = segment_bmode_batch(bm, conf, BoneSurfaceSegmenter(axial_resolution_mm=AX))
    img = np.full(len(bm), np.nan)
    cw = central_cols(FASCIA_HALF_WIDTH_MM)
    prof_all = torch.nn.functional.avg_pool1d(bm[:, :, cw].mean(2)[:, None], 5, 1, 2)[:, 0].cpu().numpy()
    h_c, y_c = has[:, cs].cpu().numpy(), y[:, cs].cpu().numpy()
    for i in range(len(bm)):
        if h_c[i].sum() < 3:
            continue
        bone = float(np.median(y_c[i][h_c[i]])) * AX
        lo, hi = int(round(FASCIA_MIN_DEPTH_MM / AX)), int(round((bone - 5.0) / AX))
        if hi - lo < 5:
            continue
        p = prof_all[i, lo:hi]
        peaks = np.nonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
        if len(peaks) == 0:
            continue
        fascia = (lo + peaks[np.argmax(p[peaks])]) * AX
        img[i] = bone - fascia
    return {"gt": gt, "ref": ref, "img": img.reshape(nP, nF),
            "indent_mm": (d - d0).view(nP, nF).cpu().numpy(), "force_n": st.reaction_force_n.view(nP, nF).cpu().numpy()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forces", default="1,2,3,4,5,8,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--only", default="", help="comma-separated patient names (smoke tests)")
    args = ap.parse_args(argv)
    forces = np.array([float(v) for v in args.forces.split(",")])
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    names = [n for n in split["train"] + split["test"] if n.startswith("sonogym_")]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i5 = int(np.argmin(np.abs(forces - 5.0)))
    low = forces <= 2.0
    rows = {}
    for n in names:
        pdir = os.path.join(ROOT, "data", "patients", n)
        ps = PatientSim(pdir, dev)
        d = ps.data
        mus = VolumeSliceExtractor(muscle_volume(pdir), spacing=d["spacing_mm"], origin=d["origin_mm"],
                                   image_size=IMAGE_SIZE, physical_size_mm=PHYSICAL_SIZE_MM, device=dev)
        r = measure(ps, mus, forces, args.seed)
        err_low = np.abs(r["img"][:, low] - r["gt"][:, low])
        det_low = float(np.isfinite(r["img"][:, low]).mean())
        comp5 = r["gt"][:, 0] - r["gt"][:, i5]
        within = forces <= 5.0
        slopes = []
        for k in range(r["img"].shape[0]):
            yv, xv = r["img"][k, within], forces[within]
            ok = np.isfinite(yv)
            if ok.sum() >= 3:
                slopes.append(float(np.polyfit(xv[ok], yv[ok], 1)[0]))
        rows[n] = {"split": "train" if n in split["train"] else "test", "raw": {k: np.asarray(v).tolist() for k, v in r.items()},
                   "detection_rate_low_force": det_low,
                   "median_abs_err_low_force_mm": float(np.nanmedian(err_low)) if np.isfinite(err_low).any() else None,
                   "median_gt_compression_1_to_5N_mm": float(np.median(comp5)),
                   "median_ref_thickness_mm": float(np.median(r["ref"])),
                   "median_img_slope_mm_per_N": float(np.median(slopes)) if slopes else None}
        q = rows[n]
        print(f"{n:16s} ref {q['median_ref_thickness_mm']:.1f} mm  GT compression 1->5 N {q['median_gt_compression_1_to_5N_mm']:.2f} mm  "
              f"detect@<=2N {q['detection_rate_low_force']:.2f}  |err|@<=2N {q['median_abs_err_low_force_mm']}  "
              f"img slope {q['median_img_slope_mm_per_N']}", flush=True)
        del ps, mus
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Gate, pooled
    det = np.median([q["detection_rate_low_force"] for q in rows.values()])
    errs = [q["median_abs_err_low_force_mm"] for q in rows.values() if q["median_abs_err_low_force_mm"] is not None]
    comp = np.median([q["median_gt_compression_1_to_5N_mm"] for q in rows.values()])
    gate_i = bool(det >= 0.5 and errs and np.median(errs) < 3.0)
    gate_ii = bool(comp > AX)
    slopes = [q["median_img_slope_mm_per_N"] for q in rows.values() if q["median_img_slope_mm_per_N"] is not None]
    h4 = bool(slopes and np.median(slopes) < 0)
    # H5: force-linear correction fitted on train, evaluated on test, against the uncompressed reference
    def pairs(split_name):
        X, Y = [], []
        for q in rows.values():
            if q["split"] != split_name:
                continue
            img, ref = np.array(q["raw"]["img"]), np.array(q["raw"]["ref"])
            for k in range(img.shape[0]):
                for j, f in enumerate(forces):
                    if f <= 5.0 and np.isfinite(img[k, j]):
                        X.append(f - 1.0)
                        Y.append(ref[k] - img[k, j])
        return np.array(X), np.array(Y)
    Xtr, Ytr = pairs("train")
    Xte, Yte = pairs("test")
    k_fit = float((Xtr @ Ytr) / (Xtr @ Xtr)) if len(Xtr) and (Xtr @ Xtr) > 0 else float("nan")
    h5 = None
    if len(Xte) and np.isfinite(k_fit):
        before = float(np.median(np.abs(Yte)))
        after = float(np.median(np.abs(Yte - k_fit * Xte)))
        h5 = {"k_mm_per_N": k_fit, "test_median_abs_err_uncorrected_mm": before,
              "test_median_abs_err_corrected_mm": after, "supported": bool(after < before)}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"meta": {"script": "ReconstructionPipeline/evaluation/muscle_compression_study.py", "commit": commit,
                    "args": vars(args), "forces_n": forces.tolist(), "preregistration": "DEC-050"},
           "gate": {"median_detection_rate_low_force": float(det), "median_abs_err_low_force_mm": float(np.median(errs)) if errs else None,
                    "median_gt_compression_1_to_5N_mm": float(comp), "gate_i_measurable": gate_i,
                    "gate_ii_compression_gt_1px": gate_ii, "pass": gate_i and gate_ii},
           "H4": {"median_img_slope_mm_per_N": float(np.median(slopes)) if slopes else None, "supported": h4},
           "H5": h5, "patients": rows}
    base = os.path.join(args.out_dir, f"muscle_compression_{stamp}")
    json.dump(out, open(base + ".json", "w"), indent=2)
    g = out["gate"]
    L = [f"# Paraspinal muscle thickness under compression ({stamp})", "", f"commit `{commit}` · pre-registration "
         "DEC-050 · forces " + ", ".join(f"{f:g}" for f in forces) + " N (above 5 N: linear-FEM caveat, DEC-047)", "",
         f"**Gate:** measurable (detect ≥ 0.5 and median |err| < 3 mm at ≤ 2 N): {'PASS' if g['gate_i_measurable'] else 'FAIL'} "
         f"(detect {g['median_detection_rate_low_force']:.2f}, |err| {g['median_abs_err_low_force_mm']}); compression 1→5 N > "
         f"{AX:.3f} mm: {'PASS' if g['gate_ii_compression_gt_1px'] else 'FAIL'} ({g['median_gt_compression_1_to_5N_mm']:.2f} mm).", "",
         f"**H4** (image thickness falls with force, 1–5 N): {'supported' if out['H4']['supported'] else 'NOT supported'}, "
         f"median slope {out['H4']['median_img_slope_mm_per_N']} mm/N.", "",
         f"**H5** (force-linear correction fitted on train, test): {json.dumps(h5)}", "",
         "| Patient | Split | Reference thickness (mm) | GT compression 1→5 N (mm) | Detect ≤ 2 N | Median abs err ≤ 2 N (mm) | Image slope (mm/N) |",
         "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for n, q in rows.items():
        L.append(f"| {n} | {q['split']} | {q['median_ref_thickness_mm']:.1f} | {q['median_gt_compression_1_to_5N_mm']:.2f} | "
                 f"{q['detection_rate_low_force']:.2f} | {q['median_abs_err_low_force_mm']} | {q['median_img_slope_mm_per_N']} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
