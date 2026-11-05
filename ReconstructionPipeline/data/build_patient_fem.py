"""
Build (or load) and calibrate the FEM for every patient folder, and write a QA table (DEC-041, task T4).

    ./.venv/bin/python -m ReconstructionPipeline.data.build_patient_fem --patients-dir data/patients \
        [--include-sample] [--out results/patients_fem_<stamp>.json]

Per patient: mesh size, cold precompute or cache-load time, Maxwell-Betti asymmetry (solver check), calibrated
e_scale and moduli, calibration log-error, the field size on the device, the fitted scan window, and the soft-
tissue thickness under the probe on a 3 x 3 grid of window poses (`FEMTissueModel.tissue_thickness`).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    from mtp_gym.core import PatientSim

    ap = argparse.ArgumentParser()
    ap.add_argument("--patients-dir", default=os.path.join(ROOT, "data", "patients"))
    ap.add_argument("--include-sample", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    dirs = sorted(glob.glob(os.path.join(args.patients_dir, "*", "ct.nii.gz")))
    dirs = [os.path.dirname(d) for d in dirs]
    if args.include_sample:
        dirs = [os.path.join(ROOT, "data", "sample_totalsegmentator_patient")] + dirs
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for d in dirs:
        t = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        ps = PatientSim(d, dev)
        m = ps.model
        build_s = time.perf_counter() - t
        poses = m.window_poses(ps.c1_bounds, ps.c2_bounds)
        thick = [m.tissue_thickness(T) for T in poses]
        row = {
            "patient": os.path.relpath(d, ROOT), "build_s": build_s, "from_cache": m.loaded_from_cache,
            "mesh": m.mesh.summary(), "free_skin_columns": int(len(m.free_skin)),
            "field_mb": m.G_lat.numel() * m.G_lat.element_size() / 1e6,
            "compliance_asymmetry": m.timings.get("compliance_asymmetry"),
            "pcg_worst_rel_residual": m.timings.get("pcg_worst_rel_residual"),
            "e_scale": m.e_scale, "moduli_kpa": list(m.effective_moduli_kpa),
            "calib_rms_log_err_max": float(np.max(m.calibration["rms_log_error_per_pose"])),
            "window_c1": list(ps.c1_bounds), "window_c2": list(ps.c2_bounds),
            "thickness_mm_min_med_max": [float(np.nanmin(thick)), float(np.nanmedian(thick)), float(np.nanmax(thick))],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del ps, m
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(ROOT, "results", f"patients_fem_{stamp}.json")
    with open(out, "w") as fh:
        json.dump({"gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "rows": rows}, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
