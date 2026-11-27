"""
DEC-072 analysis (PRE-REGISTERED, frozen with the DEC-072 commit): navigation with all 238 eligible training patients
against DEC-068's N = 64.

    ./.venv/bin/python ReconstructionPipeline/evaluation/nav_scale_ext_analysis.py "results/vec_eval_nav_*_s31_*.json"

Inputs: DEC-068 evaluation files, tag s31_N<n>_aware_s<seed>_dep-aware_<test|train>, now including N = 238.
H1c: s(238) - s(64) >= 0.05 and the paired bootstrap 95 % interval over test patients excludes 0 (seeds averaged per
patient first; `nav_scale_analysis.boot_diff`, B = 10,000, seed 0). Reported: the full curve N = 1, 4, 16, 64, 238
with train success. Writes results/nav_scale_ext_<stamp>.json / .md.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import sys

import numpy as np

from ReconstructionPipeline.evaluation.nav_scale_analysis import boot_diff, load, per_patient_mean

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NS = (1, 4, 16, 64, 238)
MIN_GAIN = 0.05


def main(argv=None):
    paths = argv or sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = glob.glob(paths[0])
    runs, base = load(paths)
    curve, pp = {}, {}
    for N in NS:
        m, per_seed = per_patient_mean(runs, N, "aware", "aware", "test")
        if m is None:
            continue
        tr, _ = per_patient_mean(runs, N, "aware", "aware", "train")
        pp[N] = m
        curve[str(N)] = {"heldout_success": float(np.mean(list(m.values()))), "per_seed": per_seed,
                         "train_success": None if tr is None else float(np.mean(list(tr.values())))}
    if 238 in pp and 64 in pp:
        d, lo, hi = boot_diff(pp[238], pp[64])
        verdict = {"diff_238_minus_64": d, "ci": [lo, hi], "H1c": bool(d >= MIN_GAIN and lo > 0)}
    else:
        verdict = {"H1c": "incomplete"}
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(ROOT, "results", f"nav_scale_ext_{stamp}")
    json.dump({"meta": {"script": "ReconstructionPipeline/evaluation/nav_scale_ext_analysis.py",
                        "preregistration": "DEC-072"}, "curve": curve, "verdict": verdict},
              open(outp + ".json", "w"), indent=1)
    L = [f"# Navigation with all eligible training patients (DEC-072) ({stamp})", "",
         "| Training patients N | Held-out success | Per seed | Train success |", "| ---: | ---: | :--- | ---: |"]
    for N in NS:
        r = curve.get(str(N))
        if r:
            tr = "—" if r["train_success"] is None else f"{r['train_success']:.3f}"
            L.append(f"| {N} | {r['heldout_success']:.3f} | {', '.join(f'{v:.3f}' for v in r['per_seed'])} | {tr} |")
    if "ci" in verdict:
        L += ["", f"H1c: s(238) − s(64) = {verdict['diff_238_minus_64']:+.3f}, 95 % CI [{verdict['ci'][0]:+.3f}, "
              f"{verdict['ci'][1]:+.3f}] (needs ≥ +{MIN_GAIN} and CI > 0): **{'PASS' if verdict['H1c'] else 'FAIL'}**"]
    else:
        L += ["", "H1c: incomplete"]
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", outp + ".json / .md")


if __name__ == "__main__":
    main()
