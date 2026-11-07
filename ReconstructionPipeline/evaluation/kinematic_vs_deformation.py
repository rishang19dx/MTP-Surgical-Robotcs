"""
How much of the force-dependent echo shift is deformation, and how much is probe kinematics? (DEC-048 thread C)

    ./.venv/bin/python ReconstructionPipeline/evaluation/kinematic_vs_deformation.py [--patients all]

The T9 compression sweep reported that the bone echo moves 6-10 mm toward the probe from 1 to 15 N and called that
something a rigid simulator cannot produce.  But the probe face itself moves deeper as force rises, and bone is
fixed, so part (possibly all) of the shift is kinematic.  This script renders, at the SAME indented face poses and
forces as `compression_sweep.py` (same seed, legacy B-mode, image-only measurement):
  * deformable: CT sliced through the FEM warp (as T9);
  * rigid:      CT sliced at the same face pose without the warp (rigid tissue, probe pushed in).
Per scanline paired between the lowest and highest force, it reports the median echo shift in each condition and
their difference (the part attributable to tissue deformation), plus the probe indentation.
Writes results/kinematic_vs_deformation_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import PatientSim  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs, sweep_patient  # noqa: E402


def paired_shift(cd: np.ndarray) -> list:
    """cd (poses, forces, columns) -> per-pose median shift over scanlines detected at the first and last force."""
    out = []
    for k in range(cd.shape[0]):
        both = np.isfinite(cd[k, 0]) & np.isfinite(cd[k, -1])
        if both.sum() >= 5:
            out.append(float(np.median(cd[k, 0, both] - cd[k, -1, both])))
    return out


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
        ps = PatientSim(pdir, dev)
        rd = sweep_patient(ps, forces, args.seed, warp=True)
        rr = sweep_patient(ps, forces, args.seed, warp=False)
        sd, sr = paired_shift(rd["column_depth_mm"]), paired_shift(rr["column_depth_mm"])
        row = {"patient": os.path.basename(pdir),
               "shift_deformable_mm": float(np.median(sd)) if sd else None,
               "shift_rigid_mm": float(np.median(sr)) if sr else None,
               "poses_deformable": len(sd), "poses_rigid": len(sr),
               "indent_past_contact_max_force_mm": float(np.nanmean(rd["indent_past_contact_mm"][:, -1]))}
        if sd and sr:
            row["deformation_part_mm"] = row["shift_deformable_mm"] - row["shift_rigid_mm"]
        rows.append(row)
        print(f"{row['patient']:34s} shift deformable {row['shift_deformable_mm']}  rigid {row['shift_rigid_mm']}  "
              f"indent {row['indent_past_contact_max_force_mm']:.1f} mm", flush=True)
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
    meta = {"script": "ReconstructionPipeline/evaluation/kinematic_vs_deformation.py", "args": vars(args),
            "commit": commit, "forces_n": forces.tolist(), "bmode": "legacy (as compression_sweep.py)"}
    base = os.path.join(args.out_dir, f"kinematic_vs_deformation_{stamp}")
    json.dump({"meta": meta, "rows": rows}, open(base + ".json", "w"), indent=2)
    L = [f"# Echo shift: kinematic vs deformation ({stamp})", "", f"commit `{commit}` · forces {forces[0]:.0f}→"
         f"{forces[-1]:.0f} N · same face poses and speckle seed in both conditions · per-scanline paired", "",
         "| Patient | Shift, deformable (mm) | Shift, rigid tissue same pose (mm) | Deformation part (mm) | Indentation at max F (mm) |",
         "| :--- | ---: | ---: | ---: | ---: |"]
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    for r in rows:
        L.append(f"| {r['patient']} | {fmt(r['shift_deformable_mm'])} | {fmt(r['shift_rigid_mm'])} | "
                 f"{fmt(r.get('deformation_part_mm'))} | {r['indent_past_contact_max_force_mm']:.1f} |")
    parts = [r["deformation_part_mm"] for r in rows if r.get("deformation_part_mm") is not None]
    if parts:
        L += ["", f"Median deformation part over {len(parts)} patients: {np.median(parts):.2f} mm "
              f"(range {min(parts):.2f} to {max(parts):.2f})."]
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
