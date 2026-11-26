"""
DEC-067 analysis (PRE-REGISTERED, frozen with the DEC-067 commit): the DEC-056 muscle-thickness study at scale
(S1.1), its stiffness sensitivity (S1.2) and the spine force-headroom gate (S1.3).

    ./.venv/bin/python ReconstructionPipeline/evaluation/muscle_scale_analysis.py results/muscle_thickness_<tag>.json \
        [--compare results/muscle_thickness_<baseline>.json]

Per-patient quantities are DEC-056's (`muscle_thickness_analysis.per_patient`, unchanged): scored sites, detection
at <= 2 N, error, and the median over sites of the 1-5 N slopes. MEASURABLE = detection at <= 2 N >= 0.5 (DEC-056).
Scored set = TEST patients of the split. Bootstrap: patients resampled with replacement, B = 10,000, seed 0,
percentile 95 % interval of the median.

Primary criteria (test, measurable patients; at least N_MIN = 20, otherwise every criterion is "underpowered"):
  G    median detection <= 2 N >= 0.5 and median |img - GT| <= 2 N < 3 mm              (DEC-056, already population)
  H4a  deformable image slope: median < 0, bootstrap upper bound < 0, and >= 75 % of patients with slope < 0
  H4b  rigid image slope: bootstrap upper bound of median |slope| <= 0.15 mm/N
  H4c  tracking gap |img slope - GT slope|: bootstrap upper bound of the median <= 0.10 mm/N
  H4   = H4a and H4b and H4c
  H5   DEC-056 rule unchanged: fit on TRAIN (deformable, 1-5 N), force-aware beats intercept-only on TEST by >= 0.377 mm
S1.3 gate (spine force-choosing policy, P1), from `force_headroom_analysis.analyse` per measurable test patient:
  excess = headroom (deformable error span over 1-5 N) - noise floor (rigid span).
  P1 gated IN if median excess >= 1.0 mm AND >= 50 % of measurable test patients have excess >= 1.0 mm; else OUT.
S1.2 (`--compare BASELINE`): ROBUST if G, H4a, H4b, H4c, H5 and the P1 gate have the same verdicts as the baseline
run; SENSITIVE otherwise (each differing verdict listed).
Reported, never decisive: DEC-056's original per-patient rules ("every measurable patient") on the same data; the
same criteria without flagged patients (splits `flags`) and on the new test patients only; train-set criteria; Spearman rho of the deformable slope
against midline soft tissue and paraspinal fat at L4 (patient meta.json qa).
Writes <input>_dec067.md and adds "analysis" (DEC-056 per patient) and "analysis_dec067" to the input JSON.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from ReconstructionPipeline.evaluation import force_headroom_analysis as fha
from ReconstructionPipeline.evaluation.muscle_thickness_analysis import (AX, G_DETECT, G_ERR, H4B_MAX, H4C_MAX,
                                                                         analyse as analyse_dec056)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_MIN, B, FRAC_NEG = 20, 10_000, 0.75
P1_EXCESS_MM, P1_FRAC = 1.0, 0.5


def _boot_median(x, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def criteria(pp, rows, names, h5, headroom):
    q = [pp[n] for n in names if pp.get(n) is not None]
    m = [v for v in q if v["detect_low"] >= G_DETECT]
    out = {"patients_scored": len(q), "patients_measurable": len(m)}
    if not q:
        return out | {"verdict": "no scored patient"}
    det = float(np.median([v["detect_low"] for v in q]))
    errs = [v["abs_err_low_mm"] for v in q if np.isfinite(v["abs_err_low_mm"])]
    err = float(np.median(errs)) if errs else np.nan
    out["G"] = bool(det >= G_DETECT and np.isfinite(err) and err < G_ERR)
    out["median_detect_low"], out["median_abs_err_low_mm"] = det, err
    s_def = [v["slope_img_def"] for v in m]
    s_rig = [abs(v["slope_img_rigid"]) for v in m]
    gaps = [v["track_gap"] for v in m]
    md, lo, hi = _boot_median(s_def)
    fneg = float(np.mean([np.isfinite(s) and s < 0 for s in s_def])) if m else np.nan
    mr, _, hr = _boot_median(s_rig)
    mg, _, hg = _boot_median(gaps)
    out.update(slope_def_median=md, slope_def_ci=[lo, hi], frac_negative=fneg, rigid_abs_median=mr,
               rigid_abs_ci_hi=hr, gap_median=mg, gap_ci_hi=hg)
    sl = [v["slope_gt_def"] for v in m]
    out["slope_gt_median"], *_ = _boot_median(sl)
    if len(m) < N_MIN:
        out.update(H4a="underpowered", H4b="underpowered", H4c="underpowered", H4="underpowered",
                   P1_gate="underpowered")
    else:
        out["H4a"] = bool(np.isfinite(md) and md < 0 and hi < 0 and fneg >= FRAC_NEG)
        out["H4b"] = bool(np.isfinite(hr) and hr <= H4B_MAX)
        out["H4c"] = bool(np.isfinite(hg) and hg <= H4C_MAX)
        out["H4"] = bool(out["H4a"] and out["H4b"] and out["H4c"])
        ex = [headroom[n]["headroom_mm"] - headroom[n]["noise_floor_mm"] for n in names
              if n in headroom and "headroom_mm" in headroom[n]]
        me, elo, ehi = _boot_median(ex)
        fr = float(np.mean([e >= P1_EXCESS_MM for e in ex])) if ex else np.nan
        out.update(excess_median_mm=me, excess_ci=[elo, ehi], excess_frac_ge_1mm=fr, excess_n=len(ex))
        out["P1_gate"] = "IN" if (np.isfinite(me) and me >= P1_EXCESS_MM and fr >= P1_FRAC) else "OUT"
    out["H5"] = None if h5 is None else bool(h5["H5"])
    return out


def patient_qa(name, rows_root):
    p = os.path.normpath(os.path.join(rows_root, name, "meta.json"))
    try:
        q = json.load(open(p))["qa"]
        return q.get("midline_soft_tissue_mm"), q.get("paraspinal20_fat_mm")
    except (OSError, KeyError, ValueError):
        return None, None


def analyse(d):
    a56 = analyse_dec056(d)
    d["analysis"] = a56                               # force_headroom_analysis reads analysis.per_patient
    headroom = fha.analyse(d)["patients"]
    pp, rows = a56["per_patient"], d["patients"]
    split = json.load(open(os.path.join(ROOT, d["meta"].get("splits", "data/splits.json"))))
    flags = set(split.get("flags", {}))
    test = [n for n, r in rows.items() if r["split"] == "test"]
    train = [n for n, r in rows.items() if r["split"] == "train"]
    out = {"primary_test": criteria(pp, rows, test, a56["H5"], headroom),
           "test_without_flagged": criteria(pp, rows, [n for n in test if n not in flags], a56["H5"], headroom),
           "test_new_only": criteria(pp, rows, [n for n in test if n.startswith("../patients_ts2/")], a56["H5"],
                                     headroom),
           "train": criteria(pp, rows, train, a56["H5"], headroom),
           "dec056_rules_test": a56["test"], "H5_detail": a56["H5"]}
    root = os.path.join(ROOT, split["patients_root"])
    xs, soft, fat = [], [], []
    for n in test:
        v = pp.get(n)
        if v is None or v["detect_low"] < G_DETECT or not np.isfinite(v["slope_img_def"]):
            continue
        st, ft = patient_qa(n, root)
        if st is not None and ft is not None:
            xs.append(v["slope_img_def"]); soft.append(st); fat.append(ft)
    if len(xs) >= 5:
        out["spearman_slope_vs_soft_tissue"] = float(spearmanr(xs, soft).statistic)
        out["spearman_slope_vs_paraspinal_fat"] = float(spearmanr(xs, fat).statistic)
        out["spearman_n"] = len(xs)
    return out


VERDICT_KEYS = ("G", "H4a", "H4b", "H4c", "H5", "P1_gate")


def _f(v, spec=".3f"):
    if v is None or isinstance(v, str):
        return "—" if v is None else v
    return "—" if not np.isfinite(v) else format(v, spec)


def _yn(v):
    return v if isinstance(v, str) else ("—" if v is None else ("**pass**" if v else "**FAIL**"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--compare", default="")
    a = ap.parse_args(argv)
    d = json.load(open(a.path))
    res = analyse(d)
    if a.compare:
        base = json.load(open(a.compare))
        b = base.get("analysis_dec067") or analyse(base)
        diff = [k for k in VERDICT_KEYS if b["primary_test"].get(k) != res["primary_test"].get(k)]
        res["S1_2"] = {"baseline": os.path.relpath(a.compare, ROOT), "differing_verdicts": diff,
                       "verdict": "ROBUST" if not diff else "SENSITIVE"}
    d["analysis_dec067"] = res
    json.dump(d, open(a.path, "w"), indent=1)
    t = res["primary_test"]
    m = d["meta"]
    L = [f"# DEC-067: muscle thickness under force at scale ({os.path.basename(a.path)})", "",
         f"commit `{m['commit']}` · pre-registration {m['preregistration']} · split `{m.get('splits')}` · "
         f"stiffness draw {m.get('e_scale_draw') or 'none (shared moduli)'} · forces {m['forces_n']} N", "",
         f"Test patients scored {t['patients_scored']}, measurable {t['patients_measurable']} (detection ≤ 2 N ≥ 0.5). "
         f"Bootstrap {B} resamples of patients, 95 % percentile intervals.", "",
         "| Criterion | Value | Verdict |", "| :--- | :--- | :---: |"]
    if "G" in t:
        L += [f"| G: median detection ≤ 2 N ≥ 0.5 and median abs error < 3 mm | {t['median_detect_low']:.2f}, "
              f"{_f(t['median_abs_err_low_mm'], '.2f')} mm | {_yn(t['G'])} |",
              f"| H4a: deformable image slope median < 0 (CI upper < 0), ≥ 75 % negative | {_f(t['slope_def_median'])} "
              f"[{_f(t['slope_def_ci'][0])}, {_f(t['slope_def_ci'][1])}] mm/N, {_f(t['frac_negative'], '.0%')} negative "
              f"(GT slope median {_f(t['slope_gt_median'])}) | {_yn(t['H4a'])} |",
              f"| H4b: median abs rigid slope, CI upper ≤ {H4B_MAX} mm/N | {_f(t['rigid_abs_median'])} (upper "
              f"{_f(t['rigid_abs_ci_hi'])}) | {_yn(t['H4b'])} |",
              f"| H4c: median tracking gap, CI upper ≤ {H4C_MAX} mm/N | {_f(t['gap_median'])} (upper "
              f"{_f(t['gap_ci_hi'])}) | {_yn(t['H4c'])} |",
              f"| **H4** | | {_yn(t['H4'])} |"]
    h5 = res["H5_detail"]
    if h5:
        L.append(f"| **H5**: force-aware beats intercept-only by ≥ {AX:.3f} mm | {h5['test_err_force_aware_mm']:.2f} vs "
                 f"{h5['test_err_intercept_only_mm']:.2f} mm (uncorrected {h5['test_err_uncorrected_mm']:.2f}); k = "
                 f"{h5['k_mm_per_N']:.3f} mm/N | {_yn(h5['H5'])} |")
    if "excess_median_mm" in t:
        L.append(f"| **S1.3 P1 gate**: median (headroom − noise floor) ≥ {P1_EXCESS_MM} mm and ≥ 50 % of patients | "
                 f"{_f(t['excess_median_mm'], '.2f')} [{_f(t['excess_ci'][0], '.2f')}, {_f(t['excess_ci'][1], '.2f')}] mm, "
                 f"{_f(t['excess_frac_ge_1mm'], '.0%')} ≥ 1 mm (n = {t['excess_n']}) | **{t['P1_gate']}** |")
    elif "P1_gate" in t:
        L.append(f"| **S1.3 P1 gate** | | {t['P1_gate']} |")
    if "S1_2" in res:
        s = res["S1_2"]
        L.append(f"| **S1.2** vs `{s['baseline']}` | differing: {', '.join(s['differing_verdicts']) or 'none'} | "
                 f"**{s['verdict']}** |")
    o = res["dec056_rules_test"] or {}
    L += ["", "## Reported, not decisive", "",
          f"- DEC-056 original rules on these test patients: G {o.get('G')}, H4a (every patient) {o.get('H4a')}, "
          f"H4b (every patient) {o.get('H4b')}, H4c {o.get('H4c')}.",
          f"- Without flagged patients: measurable {res['test_without_flagged']['patients_measurable']}, H4 "
          f"{res['test_without_flagged'].get('H4')}, P1 gate {res['test_without_flagged'].get('P1_gate')}.",
          f"- New test patients only (the 4 old test patients were analysed in DEC-056): measurable "
          f"{res['test_new_only']['patients_measurable']}, H4 {res['test_new_only'].get('H4')}, P1 gate "
          f"{res['test_new_only'].get('P1_gate')}.",
          f"- Train (development set): measurable {res['train']['patients_measurable']}, H4 {res['train'].get('H4')}."]
    if "spearman_n" in res:
        L.append(f"- Spearman ρ of the deformable image slope vs midline soft tissue {res['spearman_slope_vs_soft_tissue']:+.2f}, "
                 f"vs paraspinal fat {res['spearman_slope_vs_paraspinal_fat']:+.2f} (n = {res['spearman_n']}).")
    md = a.path[:-5] + "_dec067.md"
    open(md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", md)


if __name__ == "__main__":
    main()
