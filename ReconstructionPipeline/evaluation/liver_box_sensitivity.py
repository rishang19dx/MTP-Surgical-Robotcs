"""
DEC-063 (pre-registered): is the liver target-shift null an artefact of the FEM's clamped box walls?

    ./.venv/bin/python ReconstructionPipeline/evaluation/liver_box_sensitivity.py \
        --fat-calibration results/vein_calibration_fat_<stamp>.json [--margins 30,60,90]

Input: `data/liver_wide/<subject>` (`import_liver_subjects.py --wide`: x [-140, 140], z [-160, 155] mm, same depth
and frame Q as DEC-059). Per subject: one site (the first DEC-059 site, nearest the grid centre), the DEC-059
targets, a box = probe face + margin m on all four sides (deep face unchanged), tied liver 4.1 kPa, fitted fat.
Outcome: median |u| of the targets at 5 N (as DEC-059). Subjects: training only, in the registered order; the
first 3 eligible are used (eligible = measurable under DEC-059 and no body voxel on the CT field-of-view edge
inside the largest box). Decision: R = |u|(90)/|u|(30); artefact if R >= 1.5 in >= 2/3, null stands if R < 1.2 in
>= 2/3, otherwise inconclusive. Writes results/liver_box_sensitivity_<stamp>.json / .md.
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
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ReconstructionPipeline.evaluation.liver_headroom_study import (  # noqa: E402
    BODY_HU, FACE_HALF, FORCES, GATE_FORCE, GATE_MM, beam_profile, load, patient_stat, run_modulus, select_sites,
    site_ok, targets_for, touches_fov)

PREREGISTRATION = "DEC-063"
ORDER = ["sonogym_s0004", "sonogym_s0010", "sonogym_s0012", "sonogym_s0038", "sonogym_s0024", "sonogym_s0028",
         "sonogym_s0014", "sonogym_s0015"]
N_SUBJECTS, E_LIVER = 3, 4.1
ARTEFACT_R, NULL_R = 1.5, 1.2


def box(data, site, m):
    x, z = site
    top = data["origin_mm"][1] + data["ct_volume"].shape[1] - 1
    lo = np.array([x - FACE_HALF[0] - m, data["origin_mm"][1], z - FACE_HALF[1] - m])
    hi = np.array([x + FACE_HALF[0] + m, top + 4.0, z + FACE_HALF[1] + m])
    return lo, hi


def fov_edge_body_voxels(data, fov, lo, hi):
    """Body voxels inside the box's x-z footprint that touch the CT field-of-view edge (would be a false free surface)."""
    ct = data["ct_volume"]
    ox, _, oz = data["origin_mm"]
    xs = slice(max(0, int(np.floor(lo[0] - ox))), min(ct.shape[2], int(np.ceil(hi[0] - ox)) + 1))
    zs = slice(max(0, int(np.floor(lo[2] - oz))), min(ct.shape[0], int(np.ceil(hi[2] - oz)) + 1))
    inside_crop = (lo[0] >= ox and hi[0] <= ox + ct.shape[2] - 1 and lo[2] >= oz and hi[2] <= oz + ct.shape[0] - 1)
    edge = ndimage.binary_dilation(~fov, iterations=1)
    n = int(((ct[zs, :, xs] > BODY_HU) & edge[zs, :, xs]).sum())
    return n, inside_crop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fat-calibration", required=True)
    ap.add_argument("--margins", default="30,60,90")
    ap.add_argument("--subjects", default="", help="(debug) override the registered order; output goes to .agents/")
    a = ap.parse_args(argv)
    cal = json.load(open(a.fat_calibration))
    if cal.get("meta", {}).get("fit") != "fat" or not cal.get("converged"):
        raise SystemExit("needs a converged DEC-062 fat fit")
    e_fat = float(cal["fat_e_eff_kpa"])
    margins = [float(v) for v in a.margins.split(",")]
    debug = bool(a.subjects) or margins != [30.0, 60.0, 90.0]
    names = [s for s in a.subjects.split(",") if s] or ORDER
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.join(ROOT, "data", "liver_wide")
    rows, used = {}, []
    for n in names:
        if len(used) >= N_SUBJECTS:
            break
        data, liver, fov = load(os.path.join(base, n))
        sites, _ = select_sites(data, liver)
        row = {}
        if not sites or touches_fov(data, liver, fov, sites[:1]):
            row.update(eligible=False, reason="not measurable under DEC-059 rules")
        else:
            site = sites[0]
            prof = beam_profile(data, liver, *site)
            tg = targets_for(prof, *site)
            nb, inside = fov_edge_body_voxels(data, fov, *box(data, site, max(margins)))
            row.update(site=site, targets_depth_mm=[t[3] for t in tg], first_liver_depth_mm=float(site_ok(prof)[1]),
                       fov_edge_body_voxels=nb, box_inside_crop=inside)
            if not tg:
                row.update(eligible=False, reason="no targets at the site")
            elif nb > 0 or not inside:
                row.update(eligible=False, reason=f"largest box reaches the CT field-of-view edge ({nb} body voxels)"
                           if inside else "largest box leaves the crop")
            else:
                row["eligible"] = True
                row["by_margin"] = {}
                for m in margins:
                    lo, hi = box(data, site, m)
                    r = run_modulus(data, liver, [site], [tg], E_LIVER, dev, e_fat=e_fat, roi=(lo, hi))
                    r.update(patient_stat(r))
                    row["by_margin"][str(m)] = r
                    print(f"{n} margin {m:.0f} mm: box {np.round(hi - lo).tolist()} mm, median |u| at 5 N "
                          f"{r['median_u_5n_mm']:.2f} mm, build {r['build_s']:.0f} s", flush=True)
                used.append(n)
        if not row.get("eligible"):
            print(f"{n}: not eligible ({row['reason']})", flush=True)
        rows[n] = row
        del data, liver, fov
        gc.collect()

    R = {n: rows[n]["by_margin"][str(max(margins))]["median_u_5n_mm"] /
         rows[n]["by_margin"][str(min(margins))]["median_u_5n_mm"] for n in used}
    conv = {}
    if len(margins) >= 2:
        m1, m2 = sorted(margins)[-2:]
        conv = {n: rows[n]["by_margin"][str(m2)]["median_u_5n_mm"] / rows[n]["by_margin"][str(m1)]["median_u_5n_mm"]
                for n in used}
    k_art = sum(v >= ARTEFACT_R for v in R.values())
    k_null = sum(v < NULL_R for v in R.values())
    if len(used) < N_SUBJECTS:
        verdict = f"INCOMPLETE ({len(used)} eligible subjects)"
    elif k_art >= 2:
        verdict = "BOX ARTEFACT CONFIRMED"
    elif k_null >= 2:
        verdict = "NULL STANDS"
    else:
        verdict = "INCONCLUSIVE"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, ".agents", "liver") if debug else os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"liver_box_sensitivity_{stamp}")
    meta = {"script": "ReconstructionPipeline/evaluation/liver_box_sensitivity.py", "commit": commit,
            "preregistration": PREREGISTRATION, "debug": debug, "margins_mm": margins, "e_liver_kpa": E_LIVER,
            "fat_e_eff_kpa": e_fat, "fat_calibration": os.path.relpath(a.fat_calibration, ROOT),
            "forces_n": FORCES.tolist(), "outcome": f"median |u| of liver targets at {GATE_FORCE} N"}
    json.dump({"meta": meta, "ratio_max_over_min": R, "convergence_last_two": conv, "verdict": verdict,
               "subjects": rows}, open(path + ".json", "w"), indent=1)
    L = [f"# Liver clamped-wall margin sensitivity ({stamp})", "",
         f"pre-registration {PREREGISTRATION} · commit `{commit}` · tied liver {E_LIVER} kPa · fitted fat "
         f"{e_fat:.2f} kPa · one site per subject · outcome: median |u| of liver targets at 5 N", "",
         f"**Verdict: {verdict}** (artefact if R = |u|({max(margins):.0f})/|u|({min(margins):.0f}) ≥ {ARTEFACT_R} in "
         f"≥ 2/3; null stands if R < {NULL_R} in ≥ 2/3)", "",
         "| Subject | Site (x, z) | " + " | ".join(f"|u| m={m:.0f} (mm)" for m in margins) +
         " | R | |u|(last)/|u|(prev) | ≥ 2 mm at any m |", "| :--- | :--- | " + " ---: |" * (len(margins) + 2) + " :---: |"]
    for n, r in rows.items():
        if not r.get("eligible"):
            L.append(f"| {n} | — | " + " | ".join("—" for _ in margins) + f" | — | — | not eligible: {r['reason']} |")
            continue
        us = [r["by_margin"][str(m)]["median_u_5n_mm"] for m in margins]
        L.append(f"| {n} | ({r['site'][0]:.0f}, {r['site'][1]:.0f}) | " + " | ".join(f"{u:.2f}" for u in us) +
                 f" | {R[n]:.2f} | {conv.get(n, float('nan')):.2f} | {'yes' if max(us) >= GATE_MM else 'no'} |")
    L += ["", "Box size (x × z mm) and build time per margin:", ""]
    for n in used:
        L.append(f"- {n}: " + "; ".join(
            f"m={m:.0f}: {np.round(np.subtract(r['roi_max_mm'], r['roi_min_mm'])[[0, 2]]).astype(int).tolist()}, "
            f"{r['build_s']:.0f} s" for m, r in ((float(k), v) for k, v in rows[n]["by_margin"].items())))
    open(path + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", path + ".json / .md")


if __name__ == "__main__":
    main()
