"""
Analysis of the thread B v2 muscle-thickness study, exactly as pre-registered in DEC-056 (frozen with it).

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_thickness_analysis.py results/muscle_thickness_<stamp>.json

Writes results/muscle_thickness_<stamp>.md and adds the criteria to the JSON as "analysis".
Criteria (scored sites only; per patient = median over its scored sites):
  G    test: median detection at <= 2 N (deformable) >= 0.5 and median |img - GT| at <= 2 N < 3 mm
  H4a  test: deformable image slope over 1-5 N < 0 in every measurable patient (detection at <= 2 N >= 0.5)
  H4b  test: |rigid image slope| <= 0.15 mm/N in every measurable patient
  H4c  test: median over measurable patients of |deformable image slope - GT slope| <= 0.10 mm/N
  H5   fit ref - img = a + k (F - 1) and = a0 on TRAIN (deformable, 1-5 N); on TEST the force-aware correction's
       median |corrected - ref| must beat the intercept-only one by >= one axial pixel (0.377 mm)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

AX = 75.0 / 199
H4B_MAX = 0.15
H4C_MAX = 0.10
G_DETECT, G_ERR = 0.5, 3.0


def _slope(F, y):
    ok = np.isfinite(y)
    return float(np.polyfit(F[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan


def per_patient(r, F):
    ok = np.asarray(r["scored_site"], bool)
    if not ok.any():
        return None
    low, prim = F <= 2.0, F <= 5.0
    dimg, dgt = np.asarray(r["deformable"]["img"], float)[ok], np.asarray(r["deformable"]["gt"], float)[ok]
    rimg = np.asarray(r["rigid"]["img"], float)[ok]
    err = dimg[:, low] - dgt[:, low]
    s_d = [_slope(F[prim], y[prim]) for y in dimg]
    s_g = [_slope(F[prim], y[prim]) for y in dgt]
    s_r = [_slope(F[prim], y[prim]) for y in rimg]
    with np.errstate(all="ignore"):
        return {"scored_sites": int(ok.sum()), "detect_low": float(np.isfinite(dimg[:, low]).mean()),
                "abs_err_low_mm": float(np.nanmedian(np.abs(err))) if np.isfinite(err).any() else np.nan,
                "bias_low_mm": float(np.nanmedian(err)) if np.isfinite(err).any() else np.nan,
                "slope_img_def": float(np.nanmedian(s_d)), "slope_gt_def": float(np.nanmedian(s_g)),
                "slope_img_rigid": float(np.nanmedian(s_r)),
                "track_gap": float(np.nanmedian(np.abs(np.array(s_d) - np.array(s_g))))}


def _pairs(rows, split, F):
    X, Y = [], []
    prim = F <= 5.0
    for r in rows.values():
        if r["split"] != split:
            continue
        ok = np.asarray(r["scored_site"], bool)
        img = np.asarray(r["deformable"]["img"], float)[ok][:, prim]
        ref = np.asarray(r["ref"], float)[ok]
        for k in range(len(img)):
            for j, f in enumerate(F[prim]):
                if np.isfinite(img[k, j]) and np.isfinite(ref[k]):
                    X.append(f - 1.0)
                    Y.append(ref[k] - img[k, j])
    return np.array(X), np.array(Y)


def analyse(d):
    F = np.asarray(d["meta"]["forces_n"], float)
    rows = d["patients"]
    pp = {n: per_patient(r, F) for n, r in rows.items()}
    out = {"per_patient": pp}
    for split in ("test", "train"):
        q = [v for n, v in pp.items() if v is not None and rows[n]["split"] == split]
        if not q:
            out[split] = None
            continue
        det = float(np.median([v["detect_low"] for v in q]))
        errs = [v["abs_err_low_mm"] for v in q if np.isfinite(v["abs_err_low_mm"])]
        err = float(np.median(errs)) if errs else np.nan
        # H4 is scored on MEASURABLE patients only (detection at <= 2 N >= G_DETECT): a slope is meaningless where the
        # thickness cannot be read. Others are reported, not scored (DEC-056).
        m = [v for v in q if v["detect_low"] >= G_DETECT]
        h4a = bool(m) and all(np.isfinite(v["slope_img_def"]) and v["slope_img_def"] < 0 for v in m)
        h4b = bool(m) and all(np.isfinite(v["slope_img_rigid"]) and abs(v["slope_img_rigid"]) <= H4B_MAX for v in m)
        gaps = [v["track_gap"] for v in m if np.isfinite(v["track_gap"])]
        gap = float(np.median(gaps)) if gaps else np.nan
        h4c = bool(np.isfinite(gap) and gap <= H4C_MAX)
        out[split] = {"patients_scored": len(q), "patients_measurable": len(m), "median_detect_low": det,
                      "median_abs_err_low_mm": err, "G": bool(det >= G_DETECT and np.isfinite(err) and err < G_ERR),
                      "H4a": bool(h4a), "H4b": bool(h4b), "H4c": h4c, "median_track_gap": gap,
                      "H4": bool(h4a and h4b and h4c)}
    Xtr, Ytr = _pairs(rows, "train", F)
    Xte, Yte = _pairs(rows, "test", F)
    h5 = None
    if len(Xtr) >= 3 and len(Xte):
        k, a = np.polyfit(Xtr, Ytr, 1)
        a0 = float(np.mean(Ytr))
        e_force = float(np.median(np.abs(Yte - (a + k * Xte))))
        e_const = float(np.median(np.abs(Yte - a0)))
        e_none = float(np.median(np.abs(Yte)))
        h5 = {"k_mm_per_N": float(k), "a_mm": float(a), "a0_mm": a0, "test_err_uncorrected_mm": e_none,
              "test_err_intercept_only_mm": e_const, "test_err_force_aware_mm": e_force,
              "gain_mm": e_const - e_force, "H5": bool(e_const - e_force >= AX)}
    out["H5"] = h5
    return out


def _f(v, spec):
    return "—" if v is None or not np.isfinite(v) else format(v, spec)


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    d = json.load(open(path))
    a = analyse(d)
    d["analysis"] = a
    json.dump(d, open(path, "w"), indent=1)
    t, h5 = a["test"], a["H5"]
    L = [f"# Thread B v2: muscle thickness under probe force ({os.path.basename(path)[16:-5]})", "",
         f"commit `{d['meta']['commit']}` · pre-registration {d['meta']['preregistration']} · imaging "
         f"{d['meta'].get('bmode_physics')} + fascia (DEC-055) · forces {d['meta']['forces_n']} N "
         "(primary 1–5 N; 8–10 N reported, DEC-047 caveat)", "", "## Criteria (test patients)", "",
         "| Criterion | Value | Pass |", "| :--- | :--- | :---: |"]
    if t:
        L += [f"| G: detection ≤ 2 N ≥ 0.5 | {t['median_detect_low']:.2f} | |",
              f"| G: median abs error ≤ 2 N < 3 mm | {_f(t['median_abs_err_low_mm'], '.2f')} mm | "
              f"{'**pass**' if t['G'] else '**FAIL**'} |",
              f"| H4a: deformable image slope < 0 in every measurable patient ({t['patients_measurable']}) | see table | {'pass' if t['H4a'] else 'FAIL'} |",
              f"| H4b: rigid image slope within ±{H4B_MAX} mm/N | see table | {'pass' if t['H4b'] else 'FAIL'} |",
              f"| H4c: median tracking gap ≤ {H4C_MAX} mm/N | {_f(t['median_track_gap'], '.3f')} mm/N | "
              f"{'pass' if t['H4c'] else 'FAIL'} |",
              f"| **H4** (a and b and c) | | {'**pass**' if t['H4'] else '**FAIL**'} |"]
    else:
        L.append("| — | no scored test patient | — |")
    if h5:
        L.append(f"| **H5**: force-aware beats intercept-only by ≥ {AX:.3f} mm | {h5['test_err_force_aware_mm']:.2f} vs "
                 f"{h5['test_err_intercept_only_mm']:.2f} mm (uncorrected {h5['test_err_uncorrected_mm']:.2f}); "
                 f"k = {h5['k_mm_per_N']:.3f} mm/N | {'**pass**' if h5['H5'] else '**FAIL**'} |")
    L += ["", "## Per patient (scored sites; slopes over 1–5 N)", "",
          "| Patient | Split | Scored sites | Detect ≤ 2 N | Median abs err ≤ 2 N (mm) | Bias (mm) | Image slope, deformable | "
          "GT slope, deformable | Image slope, rigid (mm/N) |",
          "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, v in a["per_patient"].items():
        sp = d["patients"][n]["split"]
        if v is None:
            L.append(f"| {n} | {sp} | 0 | — | — | — | — | — | — |")
            continue
        L.append(f"| {n} | {sp} | {v['scored_sites']} | {v['detect_low']:.2f} | {_f(v['abs_err_low_mm'], '.2f')} | "
                 f"{_f(v['bias_low_mm'], '+.2f')} | {_f(v['slope_img_def'], '+.3f')} | {_f(v['slope_gt_def'], '+.3f')} | "
                 f"{_f(v['slope_img_rigid'], '+.3f')} |")
    tr = a["train"]
    if tr:
        L += ["", f"Train (development set, not scored): G {'pass' if tr['G'] else 'fail'} (detect "
              f"{tr['median_detect_low']:.2f}, err {_f(tr['median_abs_err_low_mm'], '.2f')} mm), H4 "
              f"{'pass' if tr['H4'] else 'fail'} (a {tr['H4a']}, b {tr['H4b']}, c {tr['H4c']})."]
    md = path[:-5] + ".md"
    open(md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", md)


if __name__ == "__main__":
    main()
