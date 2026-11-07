"""
Gate T9 (DEC-041): does probe force change the simulated image, and does the change depend on the patient?

    ./.venv/bin/python ReconstructionPipeline/evaluation/compression_sweep.py [--patients all|train|test|<names>]

For every patient (shared tissue moduli, `PatientSim` default), at a 3 x 3 grid of untilted poses over the L4 scan
window, the indentation giving each target force F in --forces is found by bisection on the batched FEM
(`BatchedFEMSolver.force`).  At each (pose, F) the deformed B-mode is rendered with a FIXED speckle seed, so
image differences come from the deformation only.  Measured from the IMAGE only (no labels):
  * bone echo depth per scanline = row of the B-mode bone surface (`segment_bmode_batch`) x axial pixel spacing;
    the echo shift is paired per scanline (detected at both the lowest and the highest force), because the set of
    visible structures changes with force and would move a per-image median on its own;
  * mean image-only confidence;
  * mean absolute B-mode difference to the F = 1 N image at the same pose.
Plus the physical indentation past first contact and the soft-tissue thickness under the probe (FEM geometry).

The gate passes for a patient if force changes the image measurably: the paired echo moves toward the probe by more
than one axial pixel (median, and at >= 80 % of the paired poses), or bone visibility changes by >= 2 of 9 poses.
It fails otherwise; both outcomes are written.  A first version averaged the depth over whichever poses had an
echo at each force; a second used the per-image median depth.  Both mix structures and were replaced before any
result was used.
Writes results/compression_sweep_<stamp>.json and .md.

Limitation (stated, not hidden): speckle is generated in image coordinates, so it does not move with the tissue.
The fixed seed removes it from the comparison but it is not a model of decorrelation under compression.
"""

from __future__ import annotations

import argparse
import gc
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim  # noqa: E402
from mtp_gym.perception import segment_bmode_batch  # noqa: E402
from ReconstructionPipeline.reconstruction.bone_segmentation import BoneSurfaceSegmenter  # noqa: E402
from ReconstructionPipeline.simulation.confidence_map import UltrasoundConfidenceEstimator  # noqa: E402
from ReconstructionPipeline.simulation.ultrasound_model_based import ModelBasedUltrasoundSimulator  # noqa: E402

AXIAL_MM = PHYSICAL_SIZE_MM[0] / (IMAGE_SIZE[0] - 1)


def patient_dirs(spec: str):
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    base = os.path.join(ROOT, split["patients_root"])
    if spec in ("all", "train", "test"):
        names = split["train"] + split["test"] if spec == "all" else split[spec]
    else:
        names = spec.split(",")
    return [os.path.normpath(os.path.join(base, n)) for n in names]


def indentation_for_force(ps: PatientSim, T: torch.Tensor, F: torch.Tensor, iters: int = 40):
    s = ps.solver
    d0 = s.first_contact(T)
    d0 = torch.where(torch.isfinite(d0), d0, torch.zeros_like(d0))
    lo, hi = d0.clone(), d0 + 60.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = s.force(T, mid) < F
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return hi, d0


@torch.no_grad()
def sweep_patient(ps: PatientSim, forces, seed: int):
    dev = ps.device
    poses = np.stack(ps.model.window_poses(ps.c1_bounds, ps.c2_bounds))                       # (9,4,4)
    nP, nF = len(poses), len(forces)
    T = torch.as_tensor(np.repeat(poses, nF, 0), dtype=torch.float64, device=dev)              # (9*nF,4,4)
    Ft = torch.as_tensor(np.tile(forces, nP), dtype=torch.float64, device=dev)
    d, d0 = indentation_for_force(ps, T, Ft)
    st = ps.solver.solve(T, d, stiffness=False)
    sl = ps.ct.extract_slices(st.probe_face_poses(T), batched_warp_fn=st.warp_to_reference)
    sim = ModelBasedUltrasoundSimulator(device=dev, axial_resolution_mm=AXIAL_MM,
                                        lateral_resolution_mm=PHYSICAL_SIZE_MM[1] / (IMAGE_SIZE[1] - 1))
    gen = torch.Generator(device=dev)
    bm = []
    for i in range(len(sl)):                       # same speckle draw for every image
        gen.manual_seed(seed)
        bm.append(sim.simulate_bmode(sl[i:i + 1], generator=gen)["bmode"])
    bm = torch.cat(bm)
    conf = UltrasoundConfidenceEstimator(device=dev, axial_resolution_mm=AXIAL_MM).compute_confidence_map(bm)
    seg = BoneSurfaceSegmenter(axial_resolution_mm=AXIAL_MM)
    has, y = segment_bmode_batch(bm, conf, seg)
    col_depth = torch.where(has, y.float() * AXIAL_MM, torch.full_like(y, float("nan"), dtype=torch.float32))
    depth = torch.full((len(bm),), float("nan"), device=dev)
    for i in range(len(bm)):
        if int(has[i].sum()) >= 5:
            depth[i] = y[i][has[i]].float().median() * AXIAL_MM
    bm_r = bm.view(nP, nF, *bm.shape[1:])
    diff = (bm_r - bm_r[:, :1]).abs().mean((2, 3))
    out = {
        "force_achieved_n": st.reaction_force_n.view(nP, nF).cpu().numpy(),
        "indent_past_contact_mm": (d - d0).view(nP, nF).cpu().numpy(),
        "echo_depth_mm": depth.view(nP, nF).cpu().numpy(),
        "column_depth_mm": col_depth.view(nP, nF, -1).cpu().numpy(),
        "confidence_mean": conf.mean((1, 2)).view(nP, nF).cpu().numpy(),
        "bmode_mad_vs_first": diff.cpu().numpy(),
        "thickness_mm": np.array([ps.model.tissue_thickness(Tp) for Tp in poses]),
    }
    return out


def summarise(name, r, forces):
    """Paired per pose: the echo shift is measured only at poses with a bone echo at both the first and the last
    force (averaging over whichever poses have an echo at each force mixes poses and is not a shift).  Bone
    visibility (fraction of poses with an echo) is reported per force on its own."""
    ed = r["echo_depth_mm"]                                        # (poses, forces)
    cd = r["column_depth_mm"]                                      # (poses, forces, columns)
    # per pose: median over scanlines with bone detected at both the lowest and the highest force.  The
    # per-image median is NOT used for the shift: which structures are visible changes with force (e.g. laminae
    # entering the field of view under compression), which moves the image median without any tissue motion.
    shift = []
    for k in range(cd.shape[0]):
        both = np.isfinite(cd[k, 0]) & np.isfinite(cd[k, -1])
        if both.sum() >= 5:
            shift.append(float(np.median(cd[k, 0, both] - cd[k, -1, both])))
    shift = np.array(shift)
    paired = np.zeros(cd.shape[0], bool)
    paired[:len(shift)] = True
    vis = np.isfinite(ed).mean(0)
    med = float(np.median(shift)) if len(shift) else float("nan")
    frac_pos = float(np.mean(shift > AXIAL_MM)) if len(shift) else float("nan")
    return {
        "patient": name, "thickness_mm_median": float(np.nanmedian(r["thickness_mm"])),
        "bone_visible_fraction_per_force": vis.tolist(),
        "paired_poses": int(paired.sum()),
        "echo_shift_mm_median_paired": med,
        "echo_shift_mm_per_pose": shift.tolist(),
        "fraction_paired_poses_shift_gt_1px": frac_pos,
        "indent_past_contact_mm_mean_per_force": np.nanmean(r["indent_past_contact_mm"], 0).tolist(),
        "confidence_mean_per_force": np.nanmean(r["confidence_mean"], 0).tolist(),
        "bmode_mad_vs_first_per_force": np.nanmean(r["bmode_mad_vs_first"], 0).tolist(),
        "force_rel_error_max": float(np.nanmax(np.abs(r["force_achieved_n"] - forces[None]) / forces[None])),
        # pass: force changes the image measurably at this patient, either by moving the paired echo
        # (median > 1 px and > 1 px at >= 80 % of paired poses) or by changing bone visibility by >= 2 of 9 poses
        "gate_pass": bool((len(shift) > 0 and med > AXIAL_MM and frac_pos >= 0.8) or (vis[-1] - vis[0] >= 2 / 9)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default="all")
    ap.add_argument("--forces", default="1,2,4,6,8,10,12,15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    forces = np.array([float(v) for v in args.forces.split(",")])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pdir in patient_dirs(args.patients):
        name = os.path.basename(pdir)
        ps = PatientSim(pdir, dev)
        r = sweep_patient(ps, forces, args.seed)
        s = summarise(name, r, forces)
        rows.append(s)
        v = s["bone_visible_fraction_per_force"]
        print(f"{name:34s} thick {s['thickness_mm_median']:5.1f} mm  visible {v[0]:.2f}->{v[-1]:.2f}  paired "
              f"{s['paired_poses']}  shift {s['echo_shift_mm_median_paired']:5.2f} mm ({s['fraction_paired_poses_shift_gt_1px']:.2f} "
              f"> 1px)  indent@max {s['indent_past_contact_mm_mean_per_force'][-1]:5.1f}"
              f"  gate {'PASS' if s['gate_pass'] else 'FAIL'}", flush=True)
        del ps
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/compression_sweep.py", "args": vars(args), "commit": commit,
            "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "forces_n": forces.tolist(),
            "axial_pixel_mm": AXIAL_MM, "stiffness": "shared"}
    base = os.path.join(args.out_dir, f"compression_sweep_{stamp}")
    json.dump({"meta": meta, "rows": rows}, open(base + ".json", "w"), indent=2)
    lines = [f"# Compression sweep ({stamp})", "", f"commit `{commit}` · {meta['gpu']} · forces {forces.tolist()} N · "
             "shared moduli · 3x3 untilted poses over the L4 window · fixed speckle seed", "",
             "Echo shift: per pose, median over scanlines with bone detected at both the lowest and the highest force; then the median over poses.", "",
             "| Patient | Thickness (mm) | Bone visible, lowest F | highest F | Paired poses | Median echo shift (mm) | Paired poses shift > 1 px | Indent past contact at max F (mm) | B-mode MAD vs lowest F | Gate |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for s in rows:
        v = s["bone_visible_fraction_per_force"]
        lines.append(f"| {s['patient']} | {s['thickness_mm_median']:.1f} | {v[0]:.2f} | {v[-1]:.2f} | {s['paired_poses']} | "
                     f"{s['echo_shift_mm_median_paired']:.2f} | {s['fraction_paired_poses_shift_gt_1px']:.2f} | "
                     f"{s['indent_past_contact_mm_mean_per_force'][-1]:.1f} | {s['bmode_mad_vs_first_per_force'][-1]:.3f} | "
                     f"{'pass' if s['gate_pass'] else 'FAIL'} |")
    open(base + ".md", "w").write("\n".join(lines) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
