"""
DEC-076 analysis (PRE-REGISTERED, frozen with the DEC-076 commit): reconstruction at server scale, the navigation
lessons applied (many training patients, budget grown with N, checkpoint chosen on training patients only).

    ./.venv/bin/python -m ReconstructionPipeline.evaluation.recon_scale_analysis "results/vec_eval_recon_*_s35_*.json"

Inputs (eval_vec_policies.py --task recon JSON; metric: coverage@3 per patient, mean over its episodes):
  * PPO runs:  tag s35_N<N>_T<T>_s<seed>_<test|train>   (N in 16, 64, 238; T = checkpoint index x 999,936 steps)
  * baselines: tag s35_baselines_test (policies random and scripted raster, the 137 test patients)
Per patient, PPO coverage is averaged over the seeds first.
T_sel(N) = the checkpoint with the highest seed-mean TRAINING-patient coverage (ties: the smaller T); test data play
no part in it. c(N) = held-out (test) coverage at T_sel(N).
  R1 (more patients help): c(238) - c(16) >= 0.05 and the paired bootstrap 95 % CI over test patients excludes 0.
  R2 (learned scanning beats the scripted raster on unseen patients): c(238) - raster >= 0.05, CI lower > 0.
Reported: the coverage curve over T for every N (test and train), the train - test gap, mean sensed force of PPO vs
raster (PPO pressing harder would be a confound for coverage), random baseline, per-seed values, collapse per seed
(train coverage at a later T more than 0.10 below that seed's best). Bootstrap: `nav_scale_analysis.boot_diff`
(B = 10,000, seed 0). Writes results/recon_scale_<stamp>.json / .md.
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
TAG = re.compile(r"^s35_N(\d+)_T(\d+)_s(\d+)_(test|train)$")
NS = (16, 64, 238)
GAIN, BEAT, COLLAPSE = 0.05, 0.05, 0.10


def load(paths):
    runs, force, base, bforce = {}, {}, {}, {}
    for p in paths:
        d = json.load(open(p))
        tag = d["meta"]["args"].get("tag", "")
        if tag == "s35_baselines_test":
            for r in d["rows"]:
                base.setdefault(r["policy"], {})[r["patient"]] = r["coverage"]
                bforce.setdefault(r["policy"], {})[r["patient"]] = r["force_mean"]
            continue
        m = TAG.match(tag)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
        rows = [r for r in d["rows"] if r["policy"].startswith("ppo:")]
        runs[key] = {r["patient"]: r["coverage"] for r in rows}
        force[key] = {r["patient"]: r["force_mean"] for r in rows}
    return runs, force, base, bforce


def seed_mean(runs, N, T, st):
    seeds = sorted(k[2] for k in runs if k[0] == N and k[1] == T and k[3] == st)
    if not seeds:
        return None, []
    pats = sorted(set.intersection(*[set(runs[(N, T, s, st)]) for s in seeds]))
    M = np.array([[runs[(N, T, s, st)][p] for p in pats] for s in seeds], float)
    return dict(zip(pats, M.mean(0))), [float(v) for v in M.mean(1)]


def mean(d):
    return None if not d else float(np.nanmean(list(d.values())))


def main(argv=None):
    paths = argv or sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = glob.glob(paths[0])
    runs, force, base, bforce = load(paths)
    curve, sel, test_at = {}, {}, {}
    for N in NS:
        Ts = sorted({k[1] for k in runs if k[0] == N})
        rows = []
        for T in Ts:
            te, pte = seed_mean(runs, N, T, "test")
            tr, ptr = seed_mean(runs, N, T, "train")
            fte, _ = seed_mean(force, N, T, "test")
            rows.append({"T": T, "test": mean(te), "test_per_seed": pte, "train": mean(tr), "train_per_seed": ptr,
                         "force_test": mean(fte)})
        curve[str(N)] = rows
        trs = {r["T"]: r["train"] for r in rows if r["train"] is not None}
        if trs:
            best = max(trs.values())
            t_sel = min(T for T, x in trs.items() if x == best)
            sel[N] = t_sel
            test_at[N] = seed_mean(runs, N, t_sel, "test")[0]
    col = {}
    for N in NS:
        for s in sorted({k[2] for k in runs if k[0] == N}):
            ser = [mean(runs[(N, T, s, "train")]) for T in sorted({k[1] for k in runs if k[0] == N})
                   if (N, T, s, "train") in runs]
            if ser:
                ib = int(np.argmax(ser))
                col[f"N{N}_s{s}"] = bool(any(x < ser[ib] - COLLAPSE for x in ser[ib + 1:]))
    v = {"T_sel": {str(k): t for k, t in sel.items()}, "collapse": col}
    if test_at.get(238) and test_at.get(16):
        d, lo, hi = boot_diff(test_at[238], test_at[16])
        v["R1"] = {"diff": d, "ci": [lo, hi], "pass": bool(d >= GAIN and lo > 0)}
    if test_at.get(238) and base.get("scripted"):
        d, lo, hi = boot_diff(test_at[238], base["scripted"])
        v["R2"] = {"diff": d, "ci": [lo, hi], "pass": bool(d >= BEAT and lo > 0)}
    v["baselines"] = {p: {"coverage": mean(c), "force": mean(bforce.get(p))} for p, c in base.items()}
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(ROOT, "results", f"recon_scale_{stamp}")
    json.dump({"meta": {"script": "ReconstructionPipeline/evaluation/recon_scale_analysis.py",
                        "preregistration": "DEC-076"}, "curve": curve, "verdict": v}, open(outp + ".json", "w"), indent=1)
    f = lambda x: "—" if x is None else f"{x:.3f}"
    L = [f"# Reconstruction at server scale (DEC-076) ({stamp})", "",
         "| N | T (x 999,936 steps) | Held-out Cov@3 | Per seed | Train Cov@3 | Mean force held-out (N) |",
         "| ---: | ---: | ---: | :--- | ---: | ---: |"]
    for N in NS:
        for r in curve[str(N)]:
            mark = " **(T_sel)**" if sel.get(N) == r["T"] else ""
            L.append(f"| {N} | {r['T']}{mark} | {f(r['test'])} | {', '.join(f(x) for x in r['test_per_seed'])} | "
                     f"{f(r['train'])} | {f(r['force_test'])} |")
    L.append("")
    for p, b in v["baselines"].items():
        L.append(f"Baseline {p}: held-out Cov@3 {f(b['coverage'])}, mean force {f(b['force'])} N")
    for k, need in (("R1", f"c(238) − c(16) ≥ +{GAIN}"), ("R2", f"c(238) − raster ≥ +{BEAT}")):
        if k in v:
            r = v[k]
            L.append(f"{k}: {r['diff']:+.3f}, 95 % CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] (needs {need} and CI > 0): "
                     f"**{'PASS' if r['pass'] else 'FAIL'}**")
        else:
            L.append(f"{k}: incomplete")
    L.append(f"Collapse per run: {col}")
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", outp + ".json / .md")


if __name__ == "__main__":
    main()
