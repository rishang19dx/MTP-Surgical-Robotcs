"""
DEC-068 (S3.1, navigation generalisation vs number of training patients) and DEC-069 (S3.3, navigation transfer
across tissue models on held-out patients): analysis, PRE-REGISTERED and frozen with the DEC-068/069 commit.

    ./.venv/bin/python ReconstructionPipeline/evaluation/nav_scale_analysis.py results/vec_eval_nav_*_s3*.json

Inputs: `eval_vec_policies.py` outputs whose `--tag` is  <study>_N<n>_<trainmode>_s<seed>_dep-<deploymode>_<set>
(study s31 | s33; set test | train). Each file has one ppo:* policy plus baselines, rows per patient (success in
[0, 1] = fraction of that patient's 16 episodes). Per run, held-out success = mean over test patients.
Bootstrap: test patients resampled with replacement (paired across runs: the same draw for every run), B = 10,000,
seed 0, percentile 95 % interval; per patient, seeds are averaged first.

DEC-068 H1 (deploy aware, test set): s(N) = mean over seeds of held-out success.
  H1a  s(1) <= s(4) <= s(16) <= s(64)                                      (monotone non-decreasing, mean over seeds)
  H1b  s(64) - s(1) >= 0.10 and its bootstrap 95 % interval excludes 0
  H1   = H1a and H1b.   Reported: train-set success and generalisation gap per N, per-seed values, baselines.
DEC-069 (N = 16, deploy aware, test set), DEC-057 rule with 3 seeds:
  deformation-only gap = success(aware-trained) - success(kinematic-trained), per seed
  SonoGym gap          = success(aware-trained) - success(rigid-trained), per seed
  A gap COUNTS if |mean over seeds| >= 0.15 and all seeds have the same sign. Bootstrap interval reported.
  Not informative (reported as such, no verdict) if the aware-trained held-out success is >= 0.95 or <= 0.05.
Writes results/nav_scale_<stamp>.json / .md.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = re.compile(r"^(s31|s33)_N(\d+)_(aware|rigid|kinematic)_s(\d+)_dep-(aware|rigid|kinematic)_(test|train)$")
B, NS = 10_000, (1, 4, 16, 64)
GAP_COUNT, CEIL_HI, CEIL_LO = 0.15, 0.95, 0.05


def load(paths):
    runs = {}          # (N, trainmode, seed, deploy, set) -> {patient: success}
    base = defaultdict(dict)
    for p in paths:
        d = json.load(open(p))
        tag = d["meta"]["args"].get("tag", "")
        m = TAG.match(tag)
        if not m:
            continue
        _, n, tm, sd, dep, st = m.groups()
        key = (int(n), tm, int(sd), dep, st)
        runs[key] = {r["patient"]: r["success"] for r in d["rows"] if r["policy"].startswith("ppo:")}
        for r in d["rows"]:
            if not r["policy"].startswith("ppo:") and dep == "aware" and st == "test":
                base[r["policy"]][r["patient"]] = r["success"]
    return runs, base


def per_patient_mean(runs, N, tm, dep, st):
    seeds = sorted(k[2] for k in runs if k[0] == N and k[1] == tm and k[3] == dep and k[4] == st)
    if not seeds:
        return None, []
    pats = sorted(set.intersection(*[set(runs[(N, tm, s, dep, st)]) for s in seeds]))
    M = np.array([[runs[(N, tm, s, dep, st)][p] for p in pats] for s in seeds], float)   # seeds x patients
    return dict(zip(pats, M.mean(0))), [float(v) for v in M.mean(1)]


def boot_diff(a: dict, b: dict, seed=0):
    pats = sorted(set(a) & set(b))
    if not pats:
        return np.nan, np.nan, np.nan
    x = np.array([a[p] - b[p] for p in pats])
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(runs, base):
    out = {"H1": {}, "transfer": {}}
    s, pp, seeds_s = {}, {}, {}
    for N in NS:
        m, per_seed = per_patient_mean(runs, N, "aware", "aware", "test")
        if m is None:
            continue
        pp[N], s[N], seeds_s[N] = m, float(np.mean(list(m.values()))), per_seed
        tr, _ = per_patient_mean(runs, N, "aware", "aware", "train")
        out["H1"][str(N)] = {"heldout_success": s[N], "per_seed": per_seed,
                             "train_success": None if tr is None else float(np.mean(list(tr.values()))),
                             "n_test_patients": len(m)}
    if all(N in s for N in NS):
        h1a = all(s[NS[i]] <= s[NS[i + 1]] for i in range(len(NS) - 1))
        d, lo, hi = boot_diff(pp[64], pp[1])
        h1b = bool(d >= 0.10 and lo > 0)
        out["H1"]["verdict"] = {"H1a": bool(h1a), "H1b": h1b, "diff_64_minus_1": d, "ci": [lo, hi],
                                "H1": bool(h1a and h1b)}
    else:
        out["H1"]["verdict"] = {"H1": "incomplete", "have_N": sorted(s)}
    # DEC-069: N = 16
    a, aseeds = per_patient_mean(runs, 16, "aware", "aware", "test")
    for name, other in (("deformation_only", "kinematic"), ("sonogym", "rigid")):
        o, oseeds = per_patient_mean(runs, 16, other, "aware", "test")
        if a is None or o is None:
            out["transfer"][name] = {"verdict": "incomplete"}
            continue
        seeds = sorted(set(k[2] for k in runs if k[0] == 16 and k[1] == "aware" and k[3] == "aware" and k[4] == "test")
                       & set(k[2] for k in runs if k[0] == 16 and k[1] == other and k[3] == "aware" and k[4] == "test"))
        g = []
        for sd in seeds:
            ra, ro = runs[(16, "aware", sd, "aware", "test")], runs[(16, other, sd, "aware", "test")]
            pats = sorted(set(ra) & set(ro))
            g.append(float(np.mean([ra[p] for p in pats]) - np.mean([ro[p] for p in pats])))
        d, lo, hi = boot_diff(a, o)
        sa = float(np.mean(list(a.values())))
        if sa >= CEIL_HI or sa <= CEIL_LO:
            verdict = "not informative (ceiling/floor)"
        else:
            counts = abs(np.mean(g)) >= GAP_COUNT and (all(x > 0 for x in g) or all(x < 0 for x in g))
            verdict = "COUNTS" if counts else "does not count"
        out["transfer"][name] = {"per_seed_gap": g, "mean_gap": float(np.mean(g)), "ci": [lo, hi],
                                 "aware_success": sa, "verdict": verdict}
    out["baselines_test"] = {k: float(np.mean(list(v.values()))) for k, v in base.items()}
    return out


def main(argv=None):
    paths = argv or sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = glob.glob(paths[0])
    runs, base = load(paths)
    res = analyse(runs, base)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(ROOT, "results", f"nav_scale_{stamp}")
    json.dump({"meta": {"script": "ReconstructionPipeline/evaluation/nav_scale_analysis.py",
                        "preregistration": "DEC-068, DEC-069",
                        "inputs": [os.path.relpath(p, ROOT) for p in sorted(paths)]}, **res},
              open(outp + ".json", "w"), indent=1)
    f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    L = [f"# Navigation at scale: patients (DEC-068) and tissue-model transfer (DEC-069) ({stamp})", "",
         "Held-out = test patients of `data/splits_v2.json`, deployed on deformable tissue; success averaged over "
         "patients (16 episodes each), then over seeds.", "",
         "| Training patients N | Held-out success | Per seed | Train success |", "| ---: | ---: | :--- | ---: |"]
    for N in NS:
        r = res["H1"].get(str(N))
        if r:
            L.append(f"| {N} | {r['heldout_success']:.3f} | {', '.join(f'{v:.3f}' for v in r['per_seed'])} | "
                     f"{f(r['train_success'])} |")
    v = res["H1"]["verdict"]
    if "H1a" in v:
        L += ["", f"H1a (monotone): **{'pass' if v['H1a'] else 'FAIL'}**; H1b (s(64) − s(1) = {v['diff_64_minus_1']:+.3f}, "
              f"95 % CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], needs ≥ +0.10 and CI > 0): "
              f"**{'pass' if v['H1b'] else 'FAIL'}**; **H1 {'PASS' if v['H1'] else 'FAIL'}**."]
    else:
        L += ["", f"H1: incomplete (have N = {v.get('have_N')})."]
    L += ["", "## Transfer across tissue models (N = 16, DEC-069)", "",
          "| Gap | Per seed | Mean | 95 % CI | Aware-trained success | Verdict |", "| :--- | :--- | ---: | :--- | ---: | :--- |"]
    for name, r in res["transfer"].items():
        if "per_seed_gap" in r:
            L.append(f"| {name} | {', '.join(f'{g:+.3f}' for g in r['per_seed_gap'])} | {r['mean_gap']:+.3f} | "
                     f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['aware_success']:.3f} | **{r['verdict']}** |")
        else:
            L.append(f"| {name} | | | | | {r['verdict']} |")
    if res["baselines_test"]:
        L += ["", "Baselines on test (deploy aware): " + ", ".join(f"{k} {v:.3f}" for k, v in res["baselines_test"].items())]
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", outp + ".json / .md")


if __name__ == "__main__":
    main()
