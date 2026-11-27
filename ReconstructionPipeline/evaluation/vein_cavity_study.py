"""
Large-strain vein model with a CAVITY lumen (DEC-075, PRE-REGISTERED; frozen with the DEC-075 commit): calibration,
mesh check and validation gate. Same protocol as DEC-073 (`vein_nl_study.py`, whose constants and helpers are reused
unchanged) except the lumen model and how the AP ratio is measured.

    ./.venv/bin/python -m ReconstructionPipeline.evaluation.vein_cavity_study calibrate [--workers 12]
    ./.venv/bin/python -m ReconstructionPipeline.evaluation.vein_cavity_study meshcheck --calibration results/vein_cavity_calibration_<stamp>.json
    ./.venv/bin/python -m ReconstructionPipeline.evaluation.vein_cavity_study gate --calibration results/vein_cavity_calibration_<stamp>.json [--workers 16]

Model: `NeoHookeanContact(model, cavity=True)` (fem_contact_nl.py): the lumen cells (the `region_modulus` mask) are
removed from the mesh; the cavity walls get frictionless penalty contact with reference-configuration pairs along
the beam. Muscle at its shared modulus, shared e_scale, fat modulus fitted.
AP ratio: `cavity_ratio` = mean current/reference wall gap of the +b wall pairs whose node lies within h of the line
along the beam through the lumen centre (phantom: the vessel axis point under the probe; patient vein: the midpoint
of the DEC-061 site's top and bottom wall points).

calibrate: Sato & Karino 2022 phantom (vein_calibration.py), h = 2 mm, probe untilted at the centre; 0.25 mm
increments until the force exceeds 1.87 N; ratio at 1.87 N interpolated linearly in force; fat modulus fitted so the
ratio is 0.50 +- 0.01 (12-point log grid in [0.3, 12] kPa, 12 points inside the bracket, log-linear interpolation, a
confirming run). Same as DEC-073.
meshcheck (reported, not gating): the calibrated modulus on the phantom at h = 1 mm: ratio at 1.87 N.
gate: the first 40 measurable TRAINING veins in sorted name order (DEC-061 measurability and site, plus: at least one
wall pair within h of the site line), h = 2 mm, DEC-061 ROI; 0.5 mm increments (halved on failure) until AP ratio
<= 0.1 (Fc), or force > 40 N, or indentation > 40 mm (Fc = inf). GATE PASS iff the median Fc / 450 mm^2 is within a
factor 2 of 82 mmHg = 10.93 kPa and >= 8 veins are measurable. Also reported: veins whose sweep stalled before
collapse or a cap (solver failures, counted as Fc = inf). Test veins are not touched.
Writes results/vein_cavity_{calibration,meshcheck,gate}_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ReconstructionPipeline.evaluation.vein_nl_study import (D_CAP, E_LUMEN_VOID, F_CAP, FACE_AREA_MM2, FACTOR,  # noqa: E402
                                                             FC_RATIO, FEMORAL_KPA, GATE_N, MIN_VEINS, NU_LUMEN,
                                                             TARGET_F, TARGET_R, TOL, _commit, _interp_first,
                                                             vein_split)

PREREGISTRATION = "DEC-075"
H_CAL_MM, H_CHECK_MM = 2.0, 1.0


# ----------------------------------------------------------------------------------------------- calibration
def phantom_ratio(args) -> dict:
    e_fat_eff, h = args if isinstance(args, tuple) else (args, H_CAL_MM)
    import torch
    torch.set_num_threads(2)
    from mtp_gym.core import SHARED_E_SCALE
    from ReconstructionPipeline.evaluation import vein_calibration as vc
    from ReconstructionPipeline.simulation.fem_contact_nl import NeoHookeanContact
    from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
    data, lumen = vc.phantom()
    lo = np.array([-vc.HALF_X, 0.0, -vc.HALF_Z])
    hi = np.array([vc.HALF_X, vc.SKIN_Y + 4.0, vc.HALF_Z])
    m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=h, cache_dir=None,
                       device=torch.device("cpu"), verbose=False, precompute=False, e_fat_kpa=e_fat_eff / SHARED_E_SCALE,
                       region_modulus=(lumen, E_LUMEN_VOID / SHARED_E_SCALE, NU_LUMEN))
    m.set_parameters(e_scale=SHARED_E_SCALE)
    nh = NeoHookeanContact(m, cavity=True)
    centre = np.array([0.0, vc.SKIN_Y - vc.DEPTH_MM, 0.0])
    Fs, Rs = [0.0], [1.0]

    def rec(dd, F, u):
        Fs.append(F)
        Rs.append(nh.cavity_ratio(u.ravel(), centre))

    res = nh.sweep(m.dorsal_pose(0.0, 0.0), d_max_mm=20.0, step_mm=0.25, f_stop_n=TARGET_F, on_step=rec)
    r = float(np.interp(TARGET_F, Fs, Rs)) if Fs[-1] >= TARGET_F else float("nan")
    return {"e_fat_eff_kpa": e_fat_eff, "h_mm": h, "ratio_at_target": r, "converged": res["converged"],
            "force_reached_n": Fs[-1], "pairs": int(len(nh._pairs["node"])), "curve": list(zip(Fs, Rs))}


def calibrate(workers: int):
    log = []
    with ProcessPoolExecutor(workers) as ex:
        E1 = np.exp(np.linspace(np.log(0.3), np.log(12.0), 12))
        r1 = list(ex.map(phantom_ratio, E1))
        log += r1
        ok = sorted([r for r in r1 if np.isfinite(r["ratio_at_target"])], key=lambda r: r["e_fat_eff_kpa"])
        br = next(((a, b) for a, b in zip(ok, ok[1:]) if a["ratio_at_target"] <= TARGET_R <= b["ratio_at_target"]), None)
        if br is None:
            return {"converged": False, "reason": "target not bracketed in [0.3, 12] kPa", "log": log}
        E2 = np.exp(np.linspace(np.log(br[0]["e_fat_eff_kpa"]), np.log(br[1]["e_fat_eff_kpa"]), 12))
        r2 = list(ex.map(phantom_ratio, E2))
        log += r2
    pts = sorted([r for r in r1 + r2 if np.isfinite(r["ratio_at_target"])], key=lambda r: r["e_fat_eff_kpa"])
    br = next(((a, b) for a, b in zip(pts, pts[1:]) if a["ratio_at_target"] <= TARGET_R <= b["ratio_at_target"]))
    la, lb = np.log(br[0]["e_fat_eff_kpa"]), np.log(br[1]["e_fat_eff_kpa"])
    ra, rb = br[0]["ratio_at_target"], br[1]["ratio_at_target"]
    e_fit = float(np.exp(la + (TARGET_R - ra) * (lb - la) / max(rb - ra, 1e-12)))
    final = phantom_ratio(e_fit)
    log.append(final)
    return {"converged": bool(abs(final["ratio_at_target"] - TARGET_R) <= TOL), "fat_e_eff_kpa": e_fit,
            "final_ratio": final["ratio_at_target"], "lumen": "cavity", "log": log}


# ----------------------------------------------------------------------------------------------- gate
def sweep_vein(args) -> dict:
    name, e_fat = args
    import torch
    torch.set_num_threads(2)
    from mtp_gym.core import SHARED_E_SCALE
    from ReconstructionPipeline.evaluation import vein_compress_study as vs
    from ReconstructionPipeline.simulation.fem_contact_nl import NeoHookeanContact
    from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
    d = os.path.join(ROOT, "data", "vein_ts2", name)
    data, vein, fov = vs.load(d)
    site, why = vs.site_and_path(data, vein, fov)
    if site is None:
        return {"vein": name, "measurable": False, "reason": why}
    ox, oy, oz = data["origin_mm"]
    ct = data["ct_volume"]
    lo = np.array([max(site["x"] - vs.ROI_HALF_X, ox), oy, -vs.ROI_HALF_Z])
    hi = np.array([min(site["x"] + vs.ROI_HALF_X, ox + ct.shape[2] - 1), oy + ct.shape[1] - 1 + 4.0, vs.ROI_HALF_Z])
    m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=vs.H_MM, cache_dir=None,
                       device=torch.device("cpu"), verbose=False, precompute=False, e_fat_kpa=e_fat / SHARED_E_SCALE,
                       region_modulus=(vein, E_LUMEN_VOID / SHARED_E_SCALE, NU_LUMEN))
    m.set_parameters(e_scale=SHARED_E_SCALE)
    nh = NeoHookeanContact(m, cavity=True)
    T = m.dorsal_pose(site["x"], site["z"])
    centre = 0.5 * (np.asarray(site["top_wall"], float) + np.asarray(site["bottom_wall"], float))
    nh._pairs = nh.cavity_pairs(T)
    if not np.isfinite(nh.cavity_ratio(np.zeros(nh.n), centre)):
        return {"vein": name, "measurable": False, "reason": "no cavity wall pair within h of the site line"}
    Fs, Rs = [0.0], [1.0]

    def rec(dd, F, u):
        Fs.append(F)
        Rs.append(nh.cavity_ratio(u.ravel(), centre))
        return Rs[-1] <= FC_RATIO

    res = nh.sweep(T, d_max_mm=D_CAP, step_mm=0.5, f_stop_n=F_CAP, on_step=rec)
    fc = _interp_first(Fs, Rs, FC_RATIO)
    return {"vein": name, "measurable": True, "site": site, "F50_n": _interp_first(Fs, Rs, 0.5), "Fc_n": fc,
            "converged": res["converged"], "stalled": bool(not np.isfinite(fc) and not res["converged"]),
            "indent_reached_mm": res["indent_reached_mm"], "force_reached_n": Fs[-1], "curve": list(zip(Fs, Rs)),
            "tets": int(len(nh.tets)), "pairs": int(len(nh._pairs["node"]))}


def gate(cal_path: str, workers: int):
    cal = json.load(open(cal_path))
    if not cal.get("converged") or cal.get("lumen") != "cavity":
        raise SystemExit("needs a converged cavity calibration")
    e_fat = float(cal["fat_e_eff_kpa"])
    names = sorted(os.listdir(os.path.join(ROOT, "data", "vein_ts2")))
    sp = vein_split(names)
    train = [n for n in names if sp[n] == "train"]
    recs, i = [], 0
    with ProcessPoolExecutor(workers) as ex:
        while sum(r["measurable"] for r in recs) < GATE_N and i < len(train):
            need = GATE_N - sum(r["measurable"] for r in recs)
            batch = train[i:i + max(need, workers)]
            i += len(batch)
            recs += list(ex.map(sweep_vein, [(n, e_fat) for n in batch]))
    meas = [r for r in recs if r["measurable"]][:GATE_N]
    fc = np.array([r["Fc_n"] for r in meas])
    med = float(np.median(fc)) if len(fc) else float("nan")
    p_kpa = med / FACE_AREA_MM2 * 1e3
    passed = bool(len(meas) >= MIN_VEINS and np.isfinite(p_kpa) and FEMORAL_KPA / FACTOR <= p_kpa <= FEMORAL_KPA * FACTOR)
    return {"calibration": os.path.relpath(cal_path, ROOT), "fat_e_eff_kpa": e_fat, "veins_tried": len(recs),
            "veins_measurable": len(meas), "stalled": int(sum(r["stalled"] for r in meas)), "median_Fc_n": med,
            "median_pressure_kpa": p_kpa, "femoral_kpa": FEMORAL_KPA, "factor": FACTOR, "PASS": passed,
            "records": recs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calibrate", "meshcheck", "gate"])
    ap.add_argument("--calibration", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/vein_cavity_study.py", "commit": _commit(),
            "preregistration": PREREGISTRATION, "mode": a.mode}
    head = f"pre-registration {PREREGISTRATION} · commit `{meta['commit']}` · Neo-Hookean, cavity lumen with wall contact"
    if a.mode == "calibrate":
        res = calibrate(a.workers)
        base = os.path.join(ROOT, "results", f"vein_cavity_calibration_{stamp}")
        L = [f"# Cavity-lumen vein calibration on the Sato & Karino phantom ({stamp})", "",
             f"{head}, h = {H_CAL_MM:g} mm · target AP ratio {TARGET_R} at {TARGET_F:.2f} N", "",
             f"**converged: {res['converged']}** · fitted fat modulus {res.get('fat_e_eff_kpa', float('nan')):.3f} kPa "
             f"(effective) · final ratio {res.get('final_ratio', float('nan')):.4f}", "",
             "| E_fat (kPa) | AP ratio at 1.87 N | sweep converged |", "| ---: | ---: | :---: |"]
        L += [f"| {r['e_fat_eff_kpa']:.4f} | {r['ratio_at_target']:.4f} | {r['converged']} |" for r in res["log"]]
    elif a.mode == "meshcheck":
        cal = json.load(open(a.calibration))
        res = {"calibration": os.path.relpath(a.calibration, ROOT),
               **phantom_ratio((float(cal["fat_e_eff_kpa"]), H_CHECK_MM))}
        base = os.path.join(ROOT, "results", f"vein_cavity_meshcheck_{stamp}")
        L = [f"# Cavity-lumen phantom at h = {H_CHECK_MM:g} mm with the h = {H_CAL_MM:g} mm fit ({stamp})", "",
             f"{head} · calibration `{res['calibration']}` (fat {res['e_fat_eff_kpa']:.3f} kPa)", "",
             f"AP ratio at {TARGET_F:.2f} N: **{res['ratio_at_target']:.4f}** (h = 2 mm fit: {TARGET_R}); sweep converged "
             f"{res['converged']}; wall pairs {res['pairs']} (reported, not gating)"]
    else:
        res = gate(a.calibration, a.workers)
        base = os.path.join(ROOT, "results", f"vein_cavity_gate_{stamp}")
        L = [f"# Cavity-lumen vein model: held-out femoral collapse gate ({stamp})", "",
             f"{head} · calibration `{res['calibration']}` (fat {res['fat_e_eff_kpa']:.3f} kPa) · training veins only", "",
             f"Measurable veins {res['veins_measurable']} (tried {res['veins_tried']}; stalled {res['stalled']}); median "
             f"full-collapse force {res['median_Fc_n']:.2f} N = **{res['median_pressure_kpa']:.2f} kPa** over the 450 mm^2 "
             f"face; femoral reference {FEMORAL_KPA:.2f} kPa, pass band ×{FACTOR}: **{'PASS' if res['PASS'] else 'FAIL'}**", "",
             "| Vein | F50 (N) | Fc (N) | force reached (N) | indent (mm) | converged | stalled |",
             "| :--- | ---: | ---: | ---: | ---: | :---: | :---: |"]
        L += [f"| {r['vein']} | {r['F50_n']:.2f} | {r['Fc_n']:.2f} | {r['force_reached_n']:.2f} | "
              f"{r['indent_reached_mm']:.1f} | {r['converged']} | {r['stalled']} |" for r in res["records"] if r["measurable"]]
    json.dump({"meta": meta, **res}, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
