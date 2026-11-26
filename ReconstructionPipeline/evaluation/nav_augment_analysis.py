"""
DEC-070 (S3.2): does physical FEM augmentation improve held-out navigation more than non-physical controls?
PRE-REGISTERED, frozen with the DEC-070 commit.

    ./.venv/bin/python ReconstructionPipeline/evaluation/nav_augment_analysis.py "results/vec_eval_nav_*_s3*.json"

Inputs: `eval_vec_policies.py` outputs tagged  s31_N16_aware_s<seed>_dep-aware_test  (arm "none", shared with
DEC-068) and  s32_N16_aug-<fem|warp|imaging>_s<seed>_dep-aware_test. Held-out success per run = mean over the 137
test patients; per patient, seeds are averaged first; bootstrap over test patients, paired, B = 10,000, seed 0.
  H2   fem - none    >= 0.05 (mean over seeds) and its 95 % interval excludes 0
  H3   fem - warp    >= 0.05 with interval > 0  AND  fem - imaging >= 0.05 with interval > 0
Not informative (no verdict) if every arm's held-out success is >= 0.95 or <= 0.05.
Writes results/nav_augment_<stamp>.json / .md.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import sys

import numpy as np

from ReconstructionPipeline.evaluation.nav_scale_analysis import boot_diff

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = re.compile(r"^(?:s31_N16_aware|s32_N16_aug-(fem|warp|imaging))_s(\d+)_dep-aware_test$")
MIN_GAIN = 0.05


def load(paths):
    runs = {}
    for p in paths:
        d = json.load(open(p))
        m = TAG.match(d["meta"]["args"].get("tag", ""))
        if not m:
            continue
        arm = m.group(1) or "none"
        runs[(arm, int(m.group(2)))] = {r["patient"]: r["success"] for r in d["rows"] if r["policy"].startswith("ppo:")}
    return runs


def arm_mean(runs, arm):
    seeds = sorted(s for a, s in runs if a == arm)
    if not seeds:
        return None
    pats = sorted(set.intersection(*[set(runs[(arm, s)]) for s in seeds]))
    return {p: float(np.mean([runs[(arm, s)][p] for s in seeds])) for p in pats}, \
        [float(np.mean([runs[(arm, s)][p] for p in pats])) for s in seeds]


def main(argv=None):
    paths = argv or sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = glob.glob(paths[0])
    runs = load(paths)
    arms = {a: arm_mean(runs, a) for a in ("none", "fem", "warp", "imaging")}
    res = {"arms": {a: (None if v is None else {"heldout_success": float(np.mean(list(v[0].values()))),
                                                 "per_seed": v[1]}) for a, v in arms.items()}}
    comp = {}
    for other in ("none", "warp", "imaging"):
        if arms["fem"] and arms[other]:
            d, lo, hi = boot_diff(arms["fem"][0], arms[other][0])
            comp[f"fem_minus_{other}"] = {"diff": d, "ci": [lo, hi], "counts": bool(d >= MIN_GAIN and lo > 0)}
    res["comparisons"] = comp
    have = [v["heldout_success"] for v in res["arms"].values() if v]
    if have and all(x >= 0.95 or x <= 0.05 for x in have):
        res["verdict"] = {"H2": "not informative", "H3": "not informative"}
    else:
        res["verdict"] = {
            "H2": comp["fem_minus_none"]["counts"] if "fem_minus_none" in comp else "incomplete",
            "H3": (comp["fem_minus_warp"]["counts"] and comp["fem_minus_imaging"]["counts"])
            if {"fem_minus_warp", "fem_minus_imaging"} <= set(comp) else "incomplete"}
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(ROOT, "results", f"nav_augment_{stamp}")
    json.dump({"meta": {"script": "ReconstructionPipeline/evaluation/nav_augment_analysis.py", "preregistration": "DEC-070",
                        "inputs": [os.path.relpath(p, ROOT) for p in sorted(paths) if TAG.match(
                            json.load(open(p))["meta"]["args"].get("tag", ""))]}, **res},
              open(outp + ".json", "w"), indent=1)
    L = [f"# Augmentation and held-out navigation, N = 16 (DEC-070) ({stamp})", "",
         "| Arm | Held-out success | Per seed |", "| :--- | ---: | :--- |"]
    for a, v in res["arms"].items():
        succ = "—" if not v else format(v["heldout_success"], ".3f")
        seeds = "" if not v else ", ".join(format(x, ".3f") for x in v["per_seed"])
        L.append(f"| {a} | {succ} | {seeds} |")
    L += ["", "| Comparison | Difference | 95 % CI | Counts (≥ 0.05, CI > 0) |", "| :--- | ---: | :--- | :---: |"]
    for k, c in comp.items():
        L.append(f"| {k} | {c['diff']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {'yes' if c['counts'] else 'no'} |")
    L += ["", f"**H2** (FEM augmentation beats none): {res['verdict']['H2']}; **H3** (FEM beats both non-physical "
          f"controls): {res['verdict']['H3']}."]
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", outp + ".json / .md")


if __name__ == "__main__":
    main()
