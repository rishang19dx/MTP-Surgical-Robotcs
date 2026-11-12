"""
Is the ground-truth muscle thickness of the thread B study (DEC-050) cut off by the 75 mm image depth?

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_window_check.py [--forces 1,2,3,4,5]

The study (`muscle_compression_study.py`) measures ground-truth thickness by counting muscle-label pixels in the
central columns of the warped label slice. The slice covers PHYSICAL_SIZE_MM[0] = 75 mm below the probe face. If the
muscle's deep edge lies below that, the count is truncated, and pressing the probe deeper brings more muscle into
the window: the thickness then appears to GROW with force. DEC-051 reported such "negative compression" in three
thick backs and left it unexplained.

Per patient, pose (the study's 6 sites) and force (same bisection on the batched FEM, shared moduli), this reports
the muscle's top and bottom depth in the central columns, and flags a (pose, force) as TRUNCATED when at least half of
the central columns have muscle in the last 3 rows of the window. It then recomputes the median ground-truth
compression 1 -> 5 N over all poses and over the poses that are untruncated at every force.

Writes results/muscle_window_check_<stamp>.json / .md.
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

from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.evaluation.muscle_compression_study import AX, SITES, central_cols, muscle_volume  # noqa: E402
from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor  # noqa: E402

BOTTOM_ROWS = 3
TRUNCATED_COL_FRACTION = 0.5


@torch.no_grad()
def check(ps: PatientSim, mus: VolumeSliceExtractor, forces: np.ndarray):
    dev = ps.device
    poses = np.stack([ps.model.dorsal_pose(x, z) for x, z in SITES])
    nP, nF = len(poses), len(forces)
    T = torch.as_tensor(np.repeat(poses, nF, 0), dtype=torch.float64, device=dev)
    Ft = torch.as_tensor(np.tile(forces, nP), dtype=torch.float64, device=dev)
    d, _ = indentation_for_force(ps, T, Ft)
    st = ps.solver.solve(T, d, stiffness=False)
    lab = mus.extract_slices(st.probe_face_poses(T), interpolation="nearest", batched_warp_fn=st.warp_to_reference) > 0.5
    L = lab[:, :, central_cols()].cpu().numpy()                   # (nP*nF, rows, cols)
    has = L.any(1)
    top = np.where(has, np.argmax(L, 1), np.nan) * AX
    bottom = np.where(has, L.shape[1] - 1 - np.argmax(L[:, ::-1], 1), np.nan) * AX
    touch = L[:, -BOTTOM_ROWS:, :].any(1).mean(1)
    gt = (L.sum(1) * AX)
    return {"top_mm": np.nanmedian(top, 1).reshape(nP, nF), "bottom_mm": np.nanmedian(bottom, 1).reshape(nP, nF),
            "bottom_touch_fraction": touch.reshape(nP, nF), "gt_mm": np.median(gt, 1).reshape(nP, nF)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forces", default="1,2,3,4,5")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    forces = np.array([float(v) for v in args.forces.split(",")])
    i1, i5 = int(np.argmin(np.abs(forces - 1.0))), int(np.argmin(np.abs(forces - 5.0)))
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    names = [n for n in split["train"] + split["test"] if n.startswith("sonogym_")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for n in names:
        pdir = os.path.join(ROOT, "data", "patients", n)
        ps = PatientSim(pdir, dev)
        d = ps.data
        mus = VolumeSliceExtractor(muscle_volume(pdir), spacing=d["spacing_mm"], origin=d["origin_mm"],
                                   image_size=IMAGE_SIZE, physical_size_mm=PHYSICAL_SIZE_MM, device=dev)
        r = check(ps, mus, forces)
        trunc = r["bottom_touch_fraction"] >= TRUNCATED_COL_FRACTION          # (nP, nF)
        clean = ~trunc.any(1)
        comp = r["gt_mm"][:, i1] - r["gt_mm"][:, i5]
        rows[n] = {"split": "train" if n in split["train"] else "test",
                   "truncated_fraction": float(trunc.mean()),
                   "poses_untruncated_at_all_forces": int(clean.sum()),
                   "median_gt_compression_1_to_5N_all_mm": float(np.median(comp)),
                   "median_gt_compression_1_to_5N_untruncated_mm": float(np.median(comp[clean])) if clean.any() else None,
                   "median_muscle_top_1N_mm": float(np.nanmedian(r["top_mm"][:, i1])),
                   "median_muscle_bottom_1N_mm": float(np.nanmedian(r["bottom_mm"][:, i1])),
                   "raw": {k: np.asarray(v).tolist() for k, v in r.items()}}
        q = rows[n]
        print(f"{n:16s} truncated {q['truncated_fraction']:.2f}  clean poses {q['poses_untruncated_at_all_forces']}/6  "
              f"GT comp all {q['median_gt_compression_1_to_5N_all_mm']:+.2f}  clean "
              f"{q['median_gt_compression_1_to_5N_untruncated_mm']}  muscle {q['median_muscle_top_1N_mm']:.1f}-"
              f"{q['median_muscle_bottom_1N_mm']:.1f} mm", flush=True)
        del ps, mus
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/muscle_window_check.py", "commit": commit,
            "forces_n": forces.tolist(), "window_depth_mm": PHYSICAL_SIZE_MM[0], "bottom_rows": BOTTOM_ROWS,
            "truncated_if_col_fraction_at_least": TRUNCATED_COL_FRACTION, "sites": SITES}
    base = os.path.join(args.out_dir, f"muscle_window_check_{stamp}")
    json.dump({"meta": meta, "patients": rows}, open(base + ".json", "w"), indent=1)
    L = [f"# Muscle ground truth vs the {PHYSICAL_SIZE_MM[0]:.0f} mm image window ({stamp})", "",
         f"commit `{commit}` · forces {forces.tolist()} N · a (pose, force) is truncated when ≥ "
         f"{TRUNCATED_COL_FRACTION:.0%} of the central columns have muscle in the last {BOTTOM_ROWS} rows", "",
         "| Patient | Split | Muscle depth at 1 N (mm) | Truncated (pose, force) | Poses clean at all forces | "
         "GT compression 1→5 N, all poses (mm) | … clean poses only (mm) |",
         "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |"]
    for n, q in rows.items():
        c = q["median_gt_compression_1_to_5N_untruncated_mm"]
        L.append(f"| {n} | {q['split']} | {q['median_muscle_top_1N_mm']:.1f}–{q['median_muscle_bottom_1N_mm']:.1f} | "
                 f"{q['truncated_fraction']:.0%} | {q['poses_untruncated_at_all_forces']}/6 | "
                 f"{q['median_gt_compression_1_to_5N_all_mm']:+.2f} | {'—' if c is None else format(c, '+.2f')} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
