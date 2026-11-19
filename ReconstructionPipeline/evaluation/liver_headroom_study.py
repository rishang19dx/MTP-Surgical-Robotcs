"""
Liver target-shift headroom bound (DEC-059, PRE-REGISTERED; frozen with the first commit of this file).

    ./.venv/bin/python ReconstructionPipeline/evaluation/liver_headroom_study.py              # all 12, results/
    ./.venv/bin/python ReconstructionPipeline/evaluation/liver_headroom_study.py --dev --subjects sonogym_s0015

Input: `data/liver/<subject>` from `ReconstructionPipeline/data/import_liver_subjects.py` (frame Q: +y anterior, the
anterior abdominal skin plays the dorsal skin's role, so the FEM / contact / probe code runs unchanged).

Per subject, as registered:
  * sites: 10 mm grid over the crop's anterior skin, untilted probe (`dorsal_pose`); valid if along the central beam no
    bone (HU > 250) and no air (HU < -200; DEC-059 addendum) lies between the skin and the first liver voxel, that voxel is 10-50 mm deep, and the liver has
    >= 20 mm (voxels) inside the 75 mm window. Up to 6 sites by farthest-point sampling from the valid candidate
    closest to the crop centre.
  * measurable: >= 1 site, and the liver does not touch the source CT's field-of-view boundary within 10 mm of the
    footprint of any selected site.
  * FEM: DEC-040/041 linear FEM, 4 mm, bone fixed, shared e_scale; cells in the liver label get the modulus whose
    EFFECTIVE value (after e_scale) is E_L in {2.3, 4.1, 5.9} kPa (Fung et al. 2013). No disk cache.
  * targets: material points on the central beam inside the liver label, every 5 mm from 5 mm below the first liver
    voxel to the end of that liver run or 70 mm deep, whichever is shallower.
  * outcome: |u(X)| at the indented pose for forces 1-5 N (8, 10 reported), force set by bisection on the FEM.
    In kinematic mode the same point does not move, so |u| IS the deformable-minus-kinematic target position.
  * gate: per patient median over sites x targets of |u| at 5 N; PASS iff >= 2.0 mm in >= 2/3 of the measurable
    patients at EACH E_L.
`--dev` writes to .agents/liver/ (never results/) and is meant for TRAINING subjects only (DEC-059 development rule).

`--sliding` (DEC-060, pre-registered separately): the liver slides without friction against the abdominal wall
(`FEMTissueModel(sliding_region=liver)`, normal-only penalty tie, k_n = PENALTY x E_muscle x h); targets read the
liver-side field. Everything else, including the gate, is unchanged. `--penalty` (sensitivity runs only) changes PENALTY.
Without `--sliding` the script does exactly what DEC-059 registered.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import SHARED_E_SCALE  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force  # noqa: E402
from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver  # noqa: E402
from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel  # noqa: E402

PREREGISTRATION = "DEC-059"
PREREGISTRATION_SLIDING = "DEC-060"
PENALTY = 100.0
E_LIVER_KPA = (2.3, 4.1, 5.9)          # effective; primary 4.1 (Fung et al. 2013, healthy range and mean)
FORCES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0])
GATE_MM, GATE_FRACTION, GATE_FORCE = 2.0, 2.0 / 3.0, 5.0
BONE_HU, BODY_HU = 250.0, -200.0
WINDOW_MM, MAX_TARGET_DEPTH, TARGET_STEP = 75.0, 70.0, 5.0
FIRST_LIVER_MM, MIN_LIVER_MM = (10.0, 50.0), 20.0
FACE_HALF = (22.5, 5.0)                 # probe face 45 x 10 mm (x along the probe, z across)
FOV_MARGIN_MM, ROI_MARGIN_MM, MAX_SITES = 10.0, 30.0, 6
GRID_X = np.arange(-40.0, 40.0 + 0.5, 10.0)      # keeps the 45 mm face inside the 140 mm crop
GRID_Z = np.arange(-80.0, 75.0 + 0.5, 10.0)


class _PS:
    """Minimal stand-in for PatientSim: `indentation_for_force` only needs `.solver`."""

    def __init__(self, solver):
        self.solver = solver


def load(subject_dir):
    def arr(name):
        img = nib.load(os.path.join(subject_dir, name))
        return np.asarray(img.dataobj).transpose(2, 1, 0), img.affine          # (x,y,z) -> (z,y,x)
    ct, aff = arr("ct.nii.gz")
    liver, _ = arr("liver.nii.gz")
    fov, _ = arr("fov.nii.gz")
    data = {"ct_volume": ct.astype(np.float32), "origin_mm": tuple(float(v) for v in aff[:3, 3]),
            "spacing_mm": (1.0, 1.0, 1.0)}
    return data, liver > 0, fov > 0


def beam_profile(data, liver, x, z):
    """Central-beam column at (x, z): (skin y, depth, liver, bone, air) on the 1 mm grid, from the skin down."""
    ct = data["ct_volume"]
    ox, oy, oz = data["origin_mm"]
    ix, iz = int(round(x - ox)), int(round(z - oz))
    col = ct[iz, :, ix]
    body = np.nonzero(col > BODY_HU)[0]
    if len(body) == 0:
        return None
    top = int(body.max())
    j = np.arange(top, -1, -1)                                   # from the skin downwards (-y)
    return oy + top, (top - j).astype(float), liver[iz, j, ix], col[j] > BONE_HU, col[j] < BODY_HU


def site_ok(prof):
    if prof is None:
        return False, None
    _, depth, liv, bone, air = prof
    if not liv.any():
        return False, None
    k1 = int(np.argmax(liv))
    d1 = depth[k1]
    # DEC-059 addendum (16 Nov, before any displacement was viewed): no air (lung, bowel gas) on the path either
    if bone[:k1].any() or air[:k1].any() or not (FIRST_LIVER_MM[0] <= d1 <= FIRST_LIVER_MM[1]):
        return False, d1
    return bool(liv[depth <= WINDOW_MM].sum() >= MIN_LIVER_MM), d1


def select_sites(data, liver):
    cand = []
    for z in GRID_Z:
        for x in GRID_X:
            ok, _ = site_ok(beam_profile(data, liver, x, z))
            if ok:
                cand.append((x, z))
    if not cand:
        return [], 0
    c = np.array(cand)
    zc = 0.5 * (GRID_Z[0] + GRID_Z[-1])
    first = int(np.argmin(np.hypot(c[:, 0], c[:, 1] - zc)))
    sel = [first]
    while len(sel) < min(MAX_SITES, len(c)):
        d = np.min(np.hypot(c[:, None, 0] - c[None, sel, 0], c[:, None, 1] - c[None, sel, 1]), 1)
        d[sel] = -1.0
        sel.append(int(np.argmax(d)))
    return [tuple(map(float, c[i])) for i in sel], len(c)


def touches_fov(data, liver, fov, sites):
    ox, _, oz = data["origin_mm"]
    edge = ndimage.binary_dilation(~fov, iterations=1)
    for x, z in sites:
        xs = slice(max(0, int(round(x - FACE_HALF[0] - FOV_MARGIN_MM - ox))), int(round(x + FACE_HALF[0] + FOV_MARGIN_MM - ox)) + 1)
        zs = slice(max(0, int(round(z - FACE_HALF[1] - FOV_MARGIN_MM - oz))), int(round(z + FACE_HALF[1] + FOV_MARGIN_MM - oz)) + 1)
        if (liver[zs, :, xs] & edge[zs, :, xs]).any():
            return True
    return False


def targets_for(prof, x, z):
    y_skin, depth, liv, _, _ = prof
    k1 = int(np.argmax(liv))
    run_end = k1 + int(np.argmin(liv[k1:])) if not liv[k1:].all() else len(liv)
    d_end = min(depth[run_end - 1], MAX_TARGET_DEPTH)
    out = []
    for d in np.arange(depth[k1] + TARGET_STEP, d_end + 1e-6, TARGET_STEP):
        k = int(round(d))
        if k < len(liv) and liv[k]:
            out.append((x, y_skin - d, z, d))
    return out


@torch.no_grad()
def run_modulus(data, liver, sites, targets, e_liver, dev, sliding=False, penalty=PENALTY, e_fat=None):
    xs = [s[0] for s in sites]
    zs = [s[1] for s in sites]
    ct = data["ct_volume"]
    top = data["origin_mm"][1] + ct.shape[1] - 1
    lo = np.array([max(min(xs) - FACE_HALF[0] - ROI_MARGIN_MM, -70.0), data["origin_mm"][1],
                   max(min(zs) - FACE_HALF[1] - ROI_MARGIN_MM, -90.0)])
    hi = np.array([min(max(xs) + FACE_HALF[0] + ROI_MARGIN_MM, 70.0), top + 4.0,
                   min(max(zs) + FACE_HALF[1] + ROI_MARGIN_MM, 85.0)])
    t0 = time.perf_counter()
    slide = dict(sliding_region=liver, sliding_penalty=penalty) if sliding else {}
    if e_fat is not None:                       # DEC-062: fat modulus from the vein calibration (effective kPa)
        slide["e_fat_kpa"] = e_fat / SHARED_E_SCALE
    model = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, cache_dir=None, device=dev, verbose=False,
                           region_modulus=(liver, e_liver / SHARED_E_SCALE), **slide)
    model.set_parameters(e_scale=SHARED_E_SCALE)
    build_s = time.perf_counter() - t0
    solver = BatchedFEMSolver(model)
    poses = np.stack([model.dorsal_pose(x, z) for x, z in sites])
    nS, nF = len(sites), len(FORCES)
    T = torch.as_tensor(np.repeat(poses, nF, 0), dtype=torch.float64, device=dev)
    Ft = torch.as_tensor(np.tile(FORCES, nS), dtype=torch.float64, device=dev)
    d, d0 = indentation_for_force(_PS(solver), T, Ft)
    st = solver.solve(T, d, stiffness=False)
    nT = max(len(t) for t in targets)
    X = np.zeros((nS * nF, nT, 3))
    for i in range(nS):
        P = np.array([t[:3] for t in targets[i]])
        P = np.concatenate([P, np.repeat(P[-1:], nT - len(P), 0)], 0)
        X[i * nF:(i + 1) * nF] = P[None]
    u = st.displacement(torch.as_tensor(X, dtype=torch.float32, device=dev), inner=sliding).double().cpu().numpy()
    # sanity (reported only): displacement of the skin point under the face centre, 1 mm below the skin
    Xs = T[:, None, :3, 3].float().clone()
    Xs[..., 1] -= 1.0
    us = st.displacement(Xs).double().cpu().numpy()[:, 0]
    R = T[:, :3, :3].cpu().numpy()
    lat, beam, elev = (np.einsum("btk,bk->bt", u, R[:, :, j]) for j in range(3))
    res = {"build_s": build_s, "region_cells": int(model.cells_in_region().sum()),
           "interface_nodes": int(len(model.if_pairs)),
           "roi_min_mm": lo.tolist(), "roi_max_mm": hi.tolist(),
           "force_n": st.reaction_force_n.view(nS, nF).cpu().numpy().tolist(),
           "indent_mm": (d - d0).view(nS, nF).cpu().numpy().tolist(),
           "skin_centre_u_mm": np.linalg.norm(us, axis=1).reshape(nS, nF).tolist(), "sites": []}
    for i in range(nS):
        k = len(targets[i])
        sl = slice(i * nF, (i + 1) * nF)
        res["sites"].append({"u_abs_mm": np.linalg.norm(u[sl, :k], axis=2).tolist(),
                             "u_inplane_mm": np.hypot(lat[sl, :k], beam[sl, :k]).tolist(),
                             "u_outplane_mm": np.abs(elev[sl, :k]).tolist()})
    del model, solver, st
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def patient_stat(r):
    j = int(np.nonzero(FORCES == GATE_FORCE)[0][0])
    vals = np.concatenate([np.asarray(s["u_abs_mm"])[j] for s in r["sites"]])
    by_f = [float(np.median(np.concatenate([np.asarray(s["u_abs_mm"])[f] for s in r["sites"]])))
            for f in range(len(FORCES))]
    inpl = np.concatenate([np.asarray(s["u_inplane_mm"])[j] for s in r["sites"]])
    outp = np.concatenate([np.asarray(s["u_outplane_mm"])[j] for s in r["sites"]])
    return {"median_u_5n_mm": float(np.median(vals)), "median_u_by_force_mm": by_f,
            "median_inplane_5n_mm": float(np.median(inpl)), "median_outplane_5n_mm": float(np.median(outp))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--moduli", default=",".join(str(v) for v in E_LIVER_KPA), help="(dev/sensitivity) subset of E_L")
    ap.add_argument("--sliding", action="store_true", help="DEC-060: liver slides against the abdominal wall")
    ap.add_argument("--penalty", type=float, default=PENALTY, help="(sensitivity runs only) sliding penalty factor")
    ap.add_argument("--sensitivity", action="store_true", help="DEC-060 penalty sensitivity: reported, never the gate")
    ap.add_argument("--fat-calibration", default="", help="DEC-062: results/vein_calibration_fat_<stamp>.json")
    args = ap.parse_args(argv)
    moduli = [float(v) for v in args.moduli.split(",")]
    e_fat = None
    if args.fat_calibration:
        cal = json.load(open(args.fat_calibration))
        if cal.get("meta", {}).get("fit") != "fat" or not cal.get("converged"):
            raise SystemExit("--fat-calibration needs a converged DEC-062 fat fit")
        e_fat = float(cal["fat_e_eff_kpa"])
    if not (args.dev or args.sensitivity) and (moduli != list(E_LIVER_KPA) or args.penalty != PENALTY):
        raise SystemExit("the registered run uses all three moduli and the registered penalty")
    if args.sensitivity and not args.sliding:
        raise SystemExit("--sensitivity is a DEC-060 (sliding) option")
    base = os.path.join(ROOT, "data", "liver")
    names = [s for s in args.subjects.split(",") if s] or sorted(os.listdir(base))
    split = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    if args.dev and any(n not in split["train"] for n in names):
        raise SystemExit("--dev is for TRAINING subjects only (DEC-059)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for n in names:
        data, liver, fov = load(os.path.join(base, n))
        sites, n_cand = select_sites(data, liver)
        row = {"split": "train" if n in split["train"] else "test", "valid_candidates": n_cand, "sites": sites}
        if not sites:
            row.update(measurable=False, reason="no valid site")
        elif touches_fov(data, liver, fov, sites):
            row.update(measurable=False, reason="liver touches the CT field of view within 10 mm of a site")
        else:
            row["measurable"] = True
            profs = [beam_profile(data, liver, x, z) for x, z in sites]
            tg = [targets_for(p, x, z) for p, (x, z) in zip(profs, sites)]
            row["targets_depth_mm"] = [[t[3] for t in ts] for ts in tg]
            row["first_liver_depth_mm"] = [float(site_ok(p)[1]) for p in profs]
            if any(len(t) == 0 for t in tg):
                keep = [i for i, t in enumerate(tg) if t]
                row["sites_without_targets"] = [sites[i] for i in range(len(sites)) if i not in keep]
                sites, tg = [sites[i] for i in keep], [tg[i] for i in keep]
            row["by_modulus"] = {}
            for e in moduli:
                r = run_modulus(data, liver, sites, tg, e, dev, sliding=args.sliding, penalty=args.penalty, e_fat=e_fat)
                r.update(patient_stat(r))
                row["by_modulus"][str(e)] = r
                print(f"{n} ({row['split']}) E_L {e} kPa: {len(sites)} sites, median |u| at 5 N "
                      f"{r['median_u_5n_mm']:.2f} mm (in-plane {r['median_inplane_5n_mm']:.2f}, out {r['median_outplane_5n_mm']:.2f}); "
                      f"build {r['build_s']:.0f} s", flush=True)
        if not row.get("measurable"):
            print(f"{n} ({row['split']}): not measurable ({row['reason']}; {n_cand} valid candidates)", flush=True)
        rows[n] = row
        del data, liver, fov
        gc.collect()
    gate = {}
    meas = [n for n, r in rows.items() if r.get("measurable")]
    for e in moduli:
        vals = [rows[n]["by_modulus"][str(e)]["median_u_5n_mm"] for n in meas]
        k = sum(v >= GATE_MM for v in vals)
        gate[str(e)] = {"patients": len(vals), "at_or_above": k,
                        "pass": bool(len(vals) > 0 and k / len(vals) >= GATE_FRACTION)}
    overall = bool(gate and all(g["pass"] for g in gate.values()) and moduli == list(E_LIVER_KPA))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, ".agents", "liver") if args.dev else os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = ("fatcal_" if e_fat is not None else "") + ("sliding_" if args.sliding else "") + ("sens_" if args.sensitivity else "") + ("dev_" if args.dev else "")
    path = os.path.join(out_dir, f"liver_headroom_{tag}{stamp}")
    meta = {"script": "ReconstructionPipeline/evaluation/liver_headroom_study.py", "commit": commit,
            "preregistration": ("DEC-062" if e_fat is not None else PREREGISTRATION_SLIDING if args.sliding else
                                PREREGISTRATION), "dev": bool(args.dev),
            "sliding": bool(args.sliding), "penalty": args.penalty, "fat_e_eff_kpa": e_fat,
            "fat_calibration": os.path.relpath(args.fat_calibration, ROOT) if args.fat_calibration else None, "sensitivity": bool(args.sensitivity),
            "moduli_kpa": moduli,
            "forces_n": FORCES.tolist(), "gate": {"mm": GATE_MM, "fraction": GATE_FRACTION, "force_n": GATE_FORCE},
            "shared_e_scale": SHARED_E_SCALE}
    json.dump({"meta": meta, "gate": gate, "pass": overall, "patients": rows}, open(path + ".json", "w"), indent=1)
    L = [f"# Liver target-shift headroom{' — SLIDING liver (DEC-060)' if args.sliding else ''}"
         f"{' — penalty sensitivity ' + str(args.penalty) if args.sensitivity else ''} ({stamp})", "",
         f"pre-registration {meta['preregistration']} · commit `{commit}` · deformation-only shift |u| of liver targets "
         f"(deformable − kinematic) · gate: median at {GATE_FORCE:.0f} N ≥ {GATE_MM} mm in ≥ 2/3 of measurable patients at "
         "every liver modulus", "", "## Gate", "", "| Liver E (kPa) | Measurable patients | ≥ 2 mm | Pass |",
         "| ---: | ---: | ---: | :---: |"]
    for e, g in gate.items():
        L.append(f"| {e} | {g['patients']} | {g['at_or_above']} | {'**pass**' if g['pass'] else '**FAIL**'} |")
    L += ["", f"**Overall: {'PASS' if overall else 'FAIL'}**", "", "## Per patient (median over sites × targets)", "",
          "| Patient | Split | Sites | " + " | ".join(f"|u| 5 N, E {e}" for e in moduli) +
          f" | in-plane / out-of-plane 5 N, E {moduli[len(moduli) // 2]} | |u| by force 1–10 N, E {moduli[len(moduli) // 2]} |",
          "| :--- | :--- | ---: | " + " | ".join("---:" for _ in moduli) + " | ---: | :--- |"]
    for n, r in rows.items():
        if not r.get("measurable"):
            L.append(f"| {n} | {r['split']} | — | " + " | ".join("—" for _ in moduli) + f" | — | not measurable: {r['reason']} |")
            continue
        bm = r["by_modulus"]
        mid = bm[str(moduli[len(moduli) // 2])]
        L.append(f"| {n} | {r['split']} | {len(mid['sites'])} | " +
                 " | ".join(f"{bm[str(e)]['median_u_5n_mm']:.2f}" for e in moduli) +
                 f" | {mid['median_inplane_5n_mm']:.2f} / {mid['median_outplane_5n_mm']:.2f} | " +
                 " ".join(f"{v:.2f}" for v in mid["median_u_by_force_mm"]) + " |")
    L += ["", "Reference (not decisive): one axial pixel 0.377 mm; lumbar muscle force effect ≈ 1.0–1.3 mm over 1–5 N "
          "(DEC-056)."]
    open(path + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", path + ".json / .md")


if __name__ == "__main__":
    main()
