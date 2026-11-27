"""
Large-strain vein model (DEC-071/073, PRE-REGISTERED; frozen with the DEC-073 commit): calibration and validation gate.

    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_nl_study.py calibrate [--workers 12]
    ./.venv/bin/python ReconstructionPipeline/evaluation/vein_nl_study.py gate --calibration results/vein_nl_calibration_<stamp>.json [--workers 16]

Model: `NeoHookeanContact` (fem_contact_nl.py; compressible Neo-Hookean, probe contact re-solved on the deformed skin),
on the same mesh, materials and region rule as the linear FEM; near-empty lumen (0.1 kPa effective, Poisson 0, as
DEC-062); muscle at its shared modulus; shared e_scale.

calibrate: the Sato & Karino 2022 popliteal phantom of vein_calibration.py (fat block, 5 mm lumen with its centre 11.5 mm
deep, h = H_CAL_MM = 2 mm, the gate's mesh size (h = 1 mm stalls before 1.87 N, DEC-073); probe untilted at the
centre transverse to the vessel). For a fat modulus E, the probe is swept
(0.25 mm increments) until the force exceeds 1.87 N; the AP ratio at 1.87 N is interpolated linearly in force. E is
fitted in log space so that the ratio is 0.50 +- 0.01: round 1 evaluates 12 log-spaced values in [0.3, 12] kPa in
parallel, round 2 twelve values inside the bracketing pair, then the final E is the log-linear interpolation at 0.50
and is evaluated once more; converged iff that final ratio is within 0.01.

gate (validation, held-out literature value): TRAINING veins only of the vein split below, the first 40 measurable in
sorted name order (measurable and site exactly as DEC-061, `vein_compress_study.site_and_path`), h = 2 mm and the
DEC-061 ROI. Each vein is swept (0.5 mm increments, halved on failure) until the AP ratio is <= 0.1 (full collapse,
Fc), or the force exceeds 40 N, or the indentation exceeds 40 mm (Fc = inf). F50 (ratio <= 0.5) is recorded too.
GATE PASS iff median over the veins of Fc / 450 mm^2 lies within a factor 2 of 82 mmHg = 10.93 kPa (Eitminavicius et
al. 2026; the DEC-061 "plausible" rule, fixed 16 Nov) and at least 8 veins are measurable; inf propagates (a median of
inf fails).

Vein split: a vein inherits its subject's split in data/splits_v2.json; other subjects, sorted by id, are test when
their position i % 3 == 1, else train. Test veins are not touched here.
Writes results/vein_nl_calibration_<stamp>.json / .md and results/vein_nl_gate_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PREREGISTRATION = "DEC-073"
TARGET_F, TARGET_R, TOL = 0.191 * 9.80665, 0.50, 0.01
E_LUMEN_VOID, NU_LUMEN = 0.1, 0.0
H_CAL_MM = 2.0          # calibration mesh = gate mesh (DEC-073)
GATE_N, FC_RATIO, F_CAP, D_CAP = 40, 0.1, 40.0, 40.0
FACE_AREA_MM2, FEMORAL_KPA, FACTOR, MIN_VEINS = 450.0, 82.0 * 0.133322, 2.0, 8


def _commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        return c + ("-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else "")
    except Exception:
        return "unknown"


def _interp_first(F, R, th):
    """Force at which R first drops to th (linear in force between recorded steps); inf if never."""
    F, R = np.asarray(F), np.asarray(R)
    k = np.nonzero(R <= th)[0]
    if len(k) == 0:
        return float("inf")
    k = int(k[0])
    if k == 0:
        return float(F[0])
    return float(F[k - 1] + (R[k - 1] - th) * (F[k] - F[k - 1]) / max(R[k - 1] - R[k], 1e-12))


# ----------------------------------------------------------------------------------------------- calibration
def phantom_ratio(e_fat_eff: float) -> dict:
    import torch
    torch.set_num_threads(2)
    from mtp_gym.core import SHARED_E_SCALE
    from ReconstructionPipeline.evaluation import vein_calibration as vc
    from ReconstructionPipeline.simulation.fem_contact_nl import NeoHookeanContact
    from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
    data, lumen = vc.phantom()
    lo = np.array([-vc.HALF_X, 0.0, -vc.HALF_Z])
    hi = np.array([vc.HALF_X, vc.SKIN_Y + 4.0, vc.HALF_Z])
    m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, element_size_mm=H_CAL_MM, cache_dir=None,
                       device=torch.device("cpu"), verbose=False, precompute=False, e_fat_kpa=e_fat_eff / SHARED_E_SCALE,
                       region_modulus=(lumen, E_LUMEN_VOID / SHARED_E_SCALE, NU_LUMEN))
    m.set_parameters(e_scale=SHARED_E_SCALE)
    nh = NeoHookeanContact(m)
    yc = vc.SKIN_Y - vc.DEPTH_MM
    walls = np.array([[0.0, yc + vc.DIAM_MM / 2, 0.0], [0.0, yc - vc.DIAM_MM / 2, 0.0]])
    Fs, Rs = [0.0], [1.0]

    def rec(dd, F, u):
        du = nh.displacement_at(u, walls)
        Fs.append(F)
        Rs.append((vc.DIAM_MM + du[0, 1] - du[1, 1]) / vc.DIAM_MM)

    res = nh.sweep(m.dorsal_pose(0.0, 0.0), d_max_mm=20.0, step_mm=0.25, f_stop_n=TARGET_F, on_step=rec)
    r = float(np.interp(TARGET_F, Fs, Rs)) if Fs[-1] >= TARGET_F else float("nan")
    return {"e_fat_eff_kpa": e_fat_eff, "ratio_at_target": r, "converged": res["converged"],
            "force_reached_n": Fs[-1], "curve": list(zip(Fs, Rs))}


def calibrate(workers: int):
    log = []
    with ProcessPoolExecutor(workers) as ex:
        E1 = np.exp(np.linspace(np.log(0.3), np.log(12.0), 12))
        r1 = list(ex.map(phantom_ratio, E1))
        log += r1
        ok = [r for r in r1 if np.isfinite(r["ratio_at_target"])]
        ok.sort(key=lambda r: r["e_fat_eff_kpa"])
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
            "final_ratio": final["ratio_at_target"], "lumen_e_eff_kpa": E_LUMEN_VOID, "log": log}


# ----------------------------------------------------------------------------------------------- gate
def vein_split(names):
    s = json.load(open(os.path.join(ROOT, "data", "splits_v2.json")))
    known = {}
    for sp in ("train", "test"):
        for n in s[sp]:
            known[n.split("/")[-1].replace("ts2_", "").replace("sonogym_", "")] = sp
    subs = sorted({n.split("_")[1] for n in names})
    other = [x for x in subs if x not in known]
    rest = {x: ("test" if i % 3 == 1 else "train") for i, x in enumerate(other)}
    return {n: known.get(n.split("_")[1], rest.get(n.split("_")[1])) for n in names}


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
    nh = NeoHookeanContact(m)
    walls = np.array([site["top_wall"], site["bottom_wall"]])
    dw = walls[1, 1] - walls[0, 1]
    Fs, Rs = [0.0], [1.0]

    def rec(dd, F, u):
        du = nh.displacement_at(u, walls)
        Fs.append(F)
        Rs.append((-(dw + du[1, 1] - du[0, 1])) / (-dw))
        return Rs[-1] <= FC_RATIO

    res = nh.sweep(m.dorsal_pose(site["x"], site["z"]), d_max_mm=D_CAP, step_mm=0.5, f_stop_n=F_CAP, on_step=rec)
    return {"vein": name, "measurable": True, "site": site, "F50_n": _interp_first(Fs, Rs, 0.5),
            "Fc_n": _interp_first(Fs, Rs, FC_RATIO), "converged": res["converged"],
            "indent_reached_mm": res["indent_reached_mm"], "force_reached_n": Fs[-1], "curve": list(zip(Fs, Rs)),
            "tets": int(len(m.mesh.tets))}


def gate(cal_path: str, workers: int):
    cal = json.load(open(cal_path))
    if not cal.get("converged"):
        raise SystemExit("calibration did not converge")
    e_fat = float(cal["fat_e_eff_kpa"])
    names = sorted(os.listdir(os.path.join(ROOT, "data", "vein_ts2")))
    sp = vein_split(names)
    train = [n for n in names if sp[n] == "train"]
    recs, i = [], 0
    with ProcessPoolExecutor(workers) as ex:       # stream until GATE_N measurable training veins are done
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
            "veins_measurable": len(meas), "median_Fc_n": med, "median_pressure_kpa": p_kpa,
            "femoral_kpa": FEMORAL_KPA, "factor": FACTOR, "PASS": passed, "records": recs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calibrate", "gate"])
    ap.add_argument("--calibration", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/vein_nl_study.py", "commit": _commit(),
            "preregistration": PREREGISTRATION, "mode": a.mode}
    if a.mode == "calibrate":
        res = calibrate(a.workers)
        base = os.path.join(ROOT, "results", f"vein_nl_calibration_{stamp}")
        L = [f"# Large-strain vein calibration on the Sato & Karino phantom ({stamp})", "",
             f"pre-registration {PREREGISTRATION} · commit `{meta['commit']}` · Neo-Hookean + deformed-skin contact, "
             f"h = {H_CAL_MM:g} mm, lumen {E_LUMEN_VOID} kPa (Poisson 0) · target AP ratio {TARGET_R} at {TARGET_F:.2f} N", "",
             f"**converged: {res['converged']}** · fitted fat modulus "
             f"{res.get('fat_e_eff_kpa', float('nan')):.3f} kPa (effective) · final ratio {res.get('final_ratio', float('nan')):.4f}", "",
             "| E_fat (kPa) | AP ratio at 1.87 N | sweep converged |", "| ---: | ---: | :---: |"]
        L += [f"| {r['e_fat_eff_kpa']:.4f} | {r['ratio_at_target']:.4f} | {r['converged']} |" for r in res["log"]]
    else:
        res = gate(a.calibration, a.workers)
        base = os.path.join(ROOT, "results", f"vein_nl_gate_{stamp}")
        L = [f"# Large-strain vein model: held-out femoral collapse gate ({stamp})", "",
             f"pre-registration {PREREGISTRATION} · commit `{meta['commit']}` · calibration `{res['calibration']}` "
             f"(fat {res['fat_e_eff_kpa']:.3f} kPa) · training veins only", "",
             f"Measurable veins {res['veins_measurable']} (tried {res['veins_tried']}); median full-collapse force "
             f"{res['median_Fc_n']:.2f} N = **{res['median_pressure_kpa']:.2f} kPa** over the 450 mm^2 face; femoral "
             f"reference {FEMORAL_KPA:.2f} kPa, pass band ×{FACTOR}: **{'PASS' if res['PASS'] else 'FAIL'}**", "",
             "| Vein | F50 (N) | Fc (N) | force reached (N) | indent (mm) | converged |", "| :--- | ---: | ---: | ---: | ---: | :---: |"]
        L += [f"| {r['vein']} | {r['F50_n']:.2f} | {r['Fc_n']:.2f} | {r['force_reached_n']:.2f} | {r['indent_reached_mm']:.1f} | "
              f"{r['converged']} |" for r in res["records"] if r["measurable"]]
    json.dump({"meta": meta, **res}, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
