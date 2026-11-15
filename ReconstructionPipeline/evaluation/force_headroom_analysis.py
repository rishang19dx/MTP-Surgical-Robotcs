"""
Force-choice headroom (DEC-058; POST HOC, exploratory): how much could ANY choice of probe force change the error of
the image-measured paraspinal muscle thickness, scored against the UNCOMPRESSED thickness (the quantity a clinician
wants)? It bounds the policy-level effect that a force-choosing task ("P1") could show on the spine, without training.

    ./.venv/bin/python ReconstructionPipeline/evaluation/force_headroom_analysis.py results/muscle_thickness_<stamp>.json

Input: the thread B v2 study (DEC-056). Only scored sites, primary forces 1-5 N, and measurable patients (detection
at <= 2 N >= 0.5, as in DEC-056) are used. Per patient and condition c in {deformable, rigid}:
    e_c(F)  = median over sites of |img_c(F) - ref|       (ref = fascia-to-bone thickness at first contact)
    span_c  = max_F e_c(F) - min_F e_c(F)                  (how much the force choice moves the error)
    F*_c    = argmin_F e_c(F)
Headroom = span_deformable. Noise floor = span_rigid: in rigid tissue the thickness does not depend on force, so its
span is measurement noise (speckle, pixel quantisation). The headroom counts as a real force effect only where it
exceeds the noise floor; it is a meaningful effect only where it also exceeds one axial pixel (AX = 0.377 mm). Also reported: the detector's constant bias at <= 2 N (from DEC-056) against
the best force, because compression can cancel a positive bias.

POST HOC: the thresholds were not fixed before the data were seen (the input was produced on 12 Nov). Results are
reported, not used as a pass/fail test. Writes results/force_headroom_<stamp>.json / .md.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AX = 75.0 / 199
G_DETECT = 0.5


def _curve(r, cond, prim):
    ok = np.asarray(r["scored_site"], bool)
    img = np.asarray(r[cond]["img"], float)[ok][:, prim]
    ref = np.asarray(r["ref"], float)[ok][:, None]
    with np.errstate(all="ignore"):
        return np.nanmedian(np.abs(img - ref), 0)


def analyse(d):
    F = np.asarray(d["meta"]["forces_n"], float)
    prim = F <= 5.0
    Fp = F[prim]
    pp = d.get("analysis", {}).get("per_patient", {})
    out = {}
    for n, r in d["patients"].items():
        v = pp.get(n)
        if not v or v["detect_low"] < G_DETECT:
            continue
        row = {"split": r["split"], "sites": int(np.sum(r["scored_site"])), "bias_low_mm": v["bias_low_mm"]}
        for c in ("deformable", "rigid"):
            e = _curve(r, c, prim)
            if not np.isfinite(e).any():
                row[c] = None
                continue
            row[c] = {"err_by_force_mm": e.tolist(), "span_mm": float(np.nanmax(e) - np.nanmin(e)),
                      "best_force_n": float(Fp[np.nanargmin(e)]), "err_at_1n_mm": float(e[0]),
                      "err_at_5n_mm": float(e[-1])}
        dfm, rig = row["deformable"], row["rigid"]
        if dfm and rig:
            row["headroom_mm"] = dfm["span_mm"]
            row["noise_floor_mm"] = rig["span_mm"]
            row["above_noise"] = bool(dfm["span_mm"] > rig["span_mm"])
            row["above_one_pixel"] = bool(dfm["span_mm"] > AX)
        out[n] = row
    return {"forces_n": Fp.tolist(), "axial_pixel_mm": AX, "patients": out}


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    d = json.load(open(path))
    a = analyse(d)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    res = {"meta": {"script": "ReconstructionPipeline/evaluation/force_headroom_analysis.py", "decision": "DEC-058",
                    "post_hoc": True, "input": os.path.relpath(path, ROOT), "input_commit": d["meta"]["commit"]}, **a}
    base = os.path.join(ROOT, "results", f"force_headroom_{stamp}")
    json.dump(res, open(base + ".json", "w"), indent=1)
    f = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    L = [f"# Force-choice headroom for the muscle measurement ({stamp})", "",
         f"**POST HOC (DEC-058).** Input `{res['meta']['input']}` (commit `{d['meta']['commit']}`); scored sites, 1–5 N, "
         "measurable patients only. Error = median over sites of |image thickness − uncompressed thickness|. "
         f"One axial pixel = {AX:.3f} mm.", "",
         "| Patient | Split | Sites | Detector bias ≤ 2 N (mm, DEC-056) | Deformable: error at 1 N → 5 N (mm) | Best force (N) | **Headroom** (span, mm) | "
         "Noise floor (rigid span, mm) | > noise | > 1 pixel |",
         "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | :---: | :---: |"]
    for n, r in a["patients"].items():
        dm = r["deformable"]
        L.append(f"| {n} | {r['split']} | {r['sites']} | {r['bias_low_mm']:+.2f} | {f(dm and dm['err_at_1n_mm'])} → {f(dm and dm['err_at_5n_mm'])} | "
                 f"{f(dm and dm['best_force_n'])} | {f(r.get('headroom_mm'))} | {f(r.get('noise_floor_mm'))} | "
                 f"{'yes' if r.get('above_noise') else 'no'} | {'yes' if r.get('above_one_pixel') else 'no'} |")
    for split in ("train", "test"):
        h = [r["headroom_mm"] for r in a["patients"].values() if r["split"] == split and "headroom_mm" in r]
        z = [r["noise_floor_mm"] for r in a["patients"].values() if r["split"] == split and "noise_floor_mm" in r]
        if h:
            L.append(f"\n{split}: median headroom {np.median(h):.2f} mm (max {np.max(h):.2f}) vs median noise floor "
                     f"{np.median(z):.2f} mm, over {len(h)} patients.")
    pts = [r for r in a["patients"].values() if r["deformable"]]
    b = np.array([r["bias_low_mm"] for r in pts])
    bf = np.array([r["deformable"]["best_force_n"] for r in pts])
    rho = float(spearmanr(b, bf).statistic)
    res["bias_vs_best_force_spearman"] = rho
    json.dump(res, open(base + ".json", "w"), indent=1)
    L.append(f"\nDetector bias (image − truth at ≤ 2 N) vs best force, all {len(pts)} patients: Spearman ρ = {rho:+.2f} "
             "(ties averaged; 8 points, descriptive only). A positive bias is cancelled by compressing the muscle, so the "
             "'best' force partly exploits the detector's constant offset rather than avoiding compression.")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
