"""
Vein compressibility across patients (DEC-061, PRE-REGISTERED; frozen with the first commit of this file).

    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_compress_study.py --calibration results/vein_calibration_<stamp>.json
    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_compress_study.py --calibration ... --dev --veins sonogym_s0015_left

Question: does the probe force needed to compress a healthy groin vein vary enough between patients that a single
fixed protocol force is a poor choice (which would give an adaptive, deformation-aware policy something to learn)?

Per vein (`data/vein/<subject>_<side>`, from `import_vein_subjects.py`; frame Q, anterior skin on +y):
  * site: x = centroid of the vein voxels in the slice z = 0 (the crop is centred on the vein's common femoral end),
    z = 0; untilted probe (`dorsal_pose`), long axis along x, i.e. transverse to the vein.
  * measurable: the central beam crosses the vein; no bone (HU > 250) and no air (HU < -200) between the skin and
    the vein; the vein's top is <= 60 mm deep and its bottom (end of the first run) inside the 75 mm window; the vein
    does not touch the source CT's field of view within 10 mm of the probe footprint.
  * FEM: DEC-040/041 linear FEM at h = 2 mm, ROI x = site +- 35 mm, z +- 17.5 mm, y from the crop bottom (80 mm below
    the skin top) to the top; bone fixed; shared e_scale; the vein label is a region with the CALIBRATED effective
    modulus (vein_calibration.py, from the popliteal probe measurement) and Poisson ratio 0. The artery is ordinary
    tissue (HU rule). No disk cache.
  * outcome: AP diameter ratio on the central beam (d0 + u_y(top wall) - u_y(bottom wall)) / d0 over forces
    0.25-20 N (0.25 N steps; force by bisection on the FEM); F50 = smallest force with ratio <= 0.5 (linear
    interpolation between steps), Fc = smallest force with ratio <= 0.1 (reported only: far outside small strain).
    A vein that is not halved by 20 N gets F50 = inf.
  * gate: S = P90 / P10 of F50 over measurable veins (numpy linear percentiles; inf propagates). PASS iff S >= 2.0 with
    >= 8 measurable veins; fewer than 8 measurable veins = INCONCLUSIVE.
  * reported: F50 vs skin-to-vein depth and vs fat thickness on the path (Spearman); the femoral cross-check
    median Fc / face area (450 mm^2) against 82 mmHg = 10.9 kPa (Eitminavicius et al. 2026: median full femoral
    collapse under external thigh compression) - within a factor 2 is called "plausible", otherwise "implausible".
`--dev` writes to .agents/vein/ and is for TRAINING subjects only.
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
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import SHARED_E_SCALE  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver  # noqa: E402
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel  # noqa: E402

PREREGISTRATION = "DEC-061"
FORCES = np.arange(0.25, 20.0 + 1e-9, 0.25)
H_MM, NU_LUMEN = 2.0, 0.0
ROI_HALF_X, ROI_HALF_Z = 35.0, 17.5
MAX_TOP_MM, WINDOW_MM, FOV_MARGIN_MM = 60.0, 75.0, 10.0
FACE_HALF = (22.5, 5.0)
BONE_HU, BODY_HU, FAT_HU = 250.0, -200.0, -20.0
GATE_RATIO, MIN_VEINS = 2.0, 8
FACE_AREA_MM2, FEMORAL_KPA = 450.0, 82.0 * 0.133322


class _PS:
    def __init__(self, solver):
        self.solver = solver


def load(d):
    def arr(name):
        img = nib.load(os.path.join(d, name))
        return np.asarray(img.dataobj).transpose(2, 1, 0), img.affine
    ct, aff = arr("ct.nii.gz")
    vein, _ = arr("vein.nii.gz")
    fov, _ = arr("fov.nii.gz")
    data = {"ct_volume": ct.astype(np.float32), "origin_mm": tuple(float(v) for v in aff[:3, 3]),
            "spacing_mm": (1.0, 1.0, 1.0)}
    return data, vein > 0, fov > 0


def site_and_path(data, vein, fov):
    ct = data["ct_volume"]
    ox, oy, oz = data["origin_mm"]
    iz = int(round(0.0 - oz))
    sl = np.argwhere(vein[iz])                                   # (y, x)
    if len(sl) == 0:
        return None, "no vein voxel in the site slice"
    x = float(np.round(sl[:, 1].mean() + ox))
    ix = int(round(x - ox))
    col = ct[iz, :, ix]
    body = np.nonzero(col > BODY_HU)[0]
    if len(body) == 0:
        return None, "no body on the central beam"
    top = int(body.max())
    j = np.arange(top, -1, -1)
    v = vein[iz, j, ix]
    if not v.any():
        return None, "central beam misses the vein"
    k1 = int(np.argmax(v))
    k2 = k1 + (int(np.argmin(v[k1:])) if not v[k1:].all() else len(v) - k1)
    if (col[j[:k1]] > BONE_HU).any() or (col[j[:k1]] < BODY_HU).any():
        return None, "bone or air between skin and vein"
    if k1 > MAX_TOP_MM or k2 > WINDOW_MM:
        return None, f"vein too deep (top {k1} mm, bottom {k2} mm)"
    edge = ndimage.binary_dilation(~fov, iterations=1)
    xs = slice(max(0, int(round(x - FACE_HALF[0] - FOV_MARGIN_MM - ox))), int(round(x + FACE_HALF[0] + FOV_MARGIN_MM - ox)) + 1)
    zs = slice(max(0, int(round(-FACE_HALF[1] - FOV_MARGIN_MM - oz))), int(round(FACE_HALF[1] + FOV_MARGIN_MM - oz)) + 1)
    if (vein[zs, :, xs] & edge[zs, :, xs]).any():
        return None, "vein touches the CT field of view near the probe"
    y_skin = oy + top
    fat = float(((col[j[:k1]] < FAT_HU) & (col[j[:k1]] > BODY_HU)).sum())
    # wall points: half a voxel inside the lumen at both ends of the first run
    return {"x": x, "z": 0.0, "y_skin": y_skin, "depth_top_mm": float(k1), "d0_mm": float(k2 - k1),
            "fat_on_path_mm": fat, "top_wall": [x, y_skin - k1, 0.0], "bottom_wall": [x, y_skin - (k2 - 1), 0.0]}, None


@torch.no_grad()
def measure(data, vein, site, e_lumen, dev, e_fat=None):
    ox, oy, oz = data["origin_mm"]
    ct = data["ct_volume"]
    lo = np.array([max(site["x"] - ROI_HALF_X, ox), oy, -ROI_HALF_Z])
    hi = np.array([min(site["x"] + ROI_HALF_X, ox + ct.shape[2] - 1), oy + ct.shape[1] - 1 + 4.0, ROI_HALF_Z])
    fat = {} if e_fat is None else {"e_fat_kpa": e_fat / SHARED_E_SCALE}
    model = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=H_MM, cache_dir=None, device=dev,
                           verbose=False, region_modulus=(vein, e_lumen / SHARED_E_SCALE, NU_LUMEN), **fat)
    model.set_parameters(e_scale=SHARED_E_SCALE)
    s = BatchedFEMSolver(model)
    nF = len(FORCES)
    T = torch.as_tensor(np.repeat(model.dorsal_pose(site["x"], site["z"])[None], nF, 0), dtype=torch.float64, device=dev)
    d, d0 = indentation_for_force(_PS(s), T, torch.as_tensor(FORCES, dtype=torch.float64, device=dev))
    st = s.solve(T, d, stiffness=False)
    X = torch.tensor([[site["top_wall"], site["bottom_wall"]]] * nF, dtype=torch.float32, device=dev)
    u = st.displacement(X).cpu().numpy()                        # (nF, 2, 3)
    dw = site["bottom_wall"][1] - site["top_wall"][1]           # negative: bottom below top
    ratio = (-(dw + u[:, 1, 1] - u[:, 0, 1])) / (-dw)
    F = st.reaction_force_n.cpu().numpy()

    def first_below(th):
        k = np.nonzero(ratio <= th)[0]
        if len(k) == 0:
            return float("inf")
        k = int(k[0])
        if k == 0:
            return float(FORCES[0])
        f0, f1, r0, r1 = FORCES[k - 1], FORCES[k], ratio[k - 1], ratio[k]
        return float(f0 + (r0 - th) * (f1 - f0) / max(r0 - r1, 1e-12))
    out = {"ratio_by_force": ratio.tolist(), "force_reached_n": F.tolist(), "indent_mm": (d - d0).cpu().numpy().tolist(),
           "F50_n": first_below(0.5), "Fc_n": first_below(0.1), "lumen_cells": int(model.cells_in_region().sum()),
           "pcg_s": model.timings.get("pcg_s")}
    del model, s, st
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--veins", default="")
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args(argv)
    cal = json.load(open(args.calibration))
    if not cal.get("converged"):
        raise SystemExit("calibration did not converge")
    e_lumen = float(cal["lumen_e_eff_kpa"])
    # DEC-062: a fat-fit calibration also sets the (effective) fat modulus of the patient models
    e_fat = float(cal["fat_e_eff_kpa"]) if cal.get("meta", {}).get("fit") == "fat" else None
    base = os.path.join(ROOT, "data", "vein")
    names = [s for s in args.veins.split(",") if s] or sorted(os.listdir(base))
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    subj = lambda n: n.rsplit("_", 1)[0]  # noqa: E731
    if args.dev and any(subj(n) not in split["train"] for n in names):
        raise SystemExit("--dev is for TRAINING subjects only (DEC-061)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for n in names:
        data, vein, fov = load(os.path.join(base, n))
        site, why = site_and_path(data, vein, fov)
        row = {"split": "train" if subj(n) in split["train"] else "test"}
        if site is None:
            row.update(measurable=False, reason=why)
            print(f"{n} ({row['split']}): not measurable ({why})", flush=True)
        else:
            row.update(measurable=True, site=site, **measure(data, vein, site, e_lumen, dev, e_fat=e_fat))
            print(f"{n} ({row['split']}): depth {site['depth_top_mm']:.0f} mm, d0 {site['d0_mm']:.0f} mm, fat "
                  f"{site['fat_on_path_mm']:.0f} mm -> F50 {row['F50_n']:.2f} N, Fc {row['Fc_n']:.2f} N", flush=True)
        rows[n] = row
        del data, vein, fov
        gc.collect()
    meas = [r for r in rows.values() if r.get("measurable")]
    f50 = np.array([r["F50_n"] for r in meas], float)
    with np.errstate(all="ignore"):
        p10, p90 = (float(np.percentile(f50, q)) if len(f50) else float("nan") for q in (10, 90))
        S = p90 / p10 if len(f50) else float("nan")
    verdict = "INCONCLUSIVE" if len(meas) < MIN_VEINS else ("PASS" if S >= GATE_RATIO else "FAIL")
    fin = np.isfinite(f50)
    rep = {}
    if fin.sum() >= 3:
        dep = np.array([r["site"]["depth_top_mm"] for r in meas])[fin]
        fat = np.array([r["site"]["fat_on_path_mm"] for r in meas])[fin]
        rep["spearman_F50_depth"] = float(spearmanr(f50[fin], dep).statistic)
        rep["spearman_F50_fat"] = float(spearmanr(f50[fin], fat).statistic)
    fc = np.array([r["Fc_n"] for r in meas], float)
    if np.isfinite(fc).any():
        kpa = float(np.median(fc[np.isfinite(fc)]) / FACE_AREA_MM2 * 1e3)
        rep["median_Fc_pressure_kpa"] = kpa
        rep["femoral_cross_check"] = "plausible" if 0.5 <= kpa / FEMORAL_KPA <= 2.0 else "implausible"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, ".agents", "vein") if args.dev else os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"vein_compress_{'dev_' if args.dev else ''}{stamp}")
    meta = {"script": "ReconstructionPipeline/evaluation/vein_compress_study.py", "commit": commit,
            "preregistration": PREREGISTRATION, "dev": bool(args.dev), "calibration": os.path.relpath(args.calibration, ROOT),
            "lumen_e_eff_kpa": e_lumen, "fat_e_eff_kpa": e_fat, "h_mm": H_MM, "forces_n": [float(FORCES[0]), float(FORCES[-1]), 0.25]}
    gate = {"P10_F50_n": p10, "P90_F50_n": p90, "spread": S, "measurable": len(meas), "verdict": verdict}
    json.dump({"meta": meta, "gate": gate, "reported": rep, "veins": rows}, open(path + ".json", "w"), indent=1)
    L = [f"# Vein compressibility across patients ({stamp})", "",
         f"pre-registration {PREREGISTRATION} · commit `{commit}` · lumen E {e_lumen:.4f} kPa (calibration "
         f"`{meta['calibration']}`) · h {H_MM} mm", "", "## Gate", "",
         f"P90/P10 of F50 over {len(meas)} measurable veins: {p90:.2f} / {p10:.2f} = **{S:.2f}** (pass if ≥ {GATE_RATIO}, "
         f"≥ {MIN_VEINS} veins) → **{verdict}**", "", "## Per vein", "",
         "| Vein | Split | Depth (mm) | Diameter (mm) | Fat on path (mm) | F50 (N) | Fc (N, reported) |",
         "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for n, r in rows.items():
        if not r.get("measurable"):
            L.append(f"| {n} | {r['split']} | — | — | — | not measurable: {r['reason']} | |")
            continue
        s = r["site"]
        L.append(f"| {n} | {r['split']} | {s['depth_top_mm']:.0f} | {s['d0_mm']:.0f} | {s['fat_on_path_mm']:.0f} | "
                 f"{r['F50_n']:.2f} | {r['Fc_n']:.2f} |")
    L += ["", "Reported: " + ", ".join(f"{k} = {v:.2f}" if isinstance(v, float) else f"{k} = {v}" for k, v in rep.items()),
          f"(femoral reference {FEMORAL_KPA:.1f} kPa = 82 mmHg)"]
    open(path + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", path + ".json / .md")


if __name__ == "__main__":
    main()
