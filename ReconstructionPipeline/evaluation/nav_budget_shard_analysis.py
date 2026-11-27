"""
DEC-074 analysis (PRE-REGISTERED, frozen with the DEC-074 commit): why N = 238 navigation fell below N = 64 (DEC-072):
the sharded env, or the fixed 2M-step budget?

    ./.venv/bin/python -m ReconstructionPipeline.evaluation.nav_budget_shard_analysis "results/vec_eval_nav_*_s3[14]_*.json"

Inputs (eval_vec_policies.py JSON, PPO rows only; patient success averaged over the seeds first):
  * DEC-068 N = 64, unsharded:   tag s31_N64_aware_s<seed>_dep-aware_<test|train>
  * arm S, N = 64, 4 shards, 2M: tag s34_N64sh_T2_s<seed>_<test|train>
  * arm L, N = 238, 4 shards, checkpoints at T = 2, 4, 6, 8 (x 999,936 steps): tag s34_N238sh_T<T>_s<seed>_<test|train>

Shard effect (arm S vs DEC-068 N = 64, held-out): d_S = s(S64) - s(64), paired bootstrap 95 % CI over test patients
(`nav_scale_analysis.boot_diff`, B = 10,000, seed 0).
  SHARD-HARMFUL iff d_S <= -0.05 and CI upper < 0;  SHARD-HELPFUL iff d_S >= +0.05 and CI lower > 0;
  SHARD-EQUIVALENT iff the CI lies inside [-0.05, +0.05];  else INCONCLUSIVE.
Checkpoint for arm L, chosen without test data: T_sel = the T with the highest seed-mean TRAINING-patient success
(ties: the smaller T).
H1d: s(L238, T_sel) - s(S64) >= 0.05 and CI lower > 0 (same env implementation on both sides, so only N and the
budget differ). Secondary, reported: s(L238, T_sel) - s(64) against unsharded DEC-068.
FIT: training-patient success at T_sel >= 0.90 (else the 238-patient policy is still under-fitted at 8M).
COLLAPSE (reported per seed): training-patient success at a later T more than 0.20 below that seed's best.
Writes results/nav_budget_shard_<stamp>.json / .md.
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
TAG_OLD = re.compile(r"^s31_N64_aware_s(\d+)_dep-aware_(test|train)$")
TAG_NEW = re.compile(r"^s34_(N64sh|N238sh)_T(\d+)_s(\d+)_(test|train)$")
TS = (2, 4, 6, 8)
MARGIN, MIN_GAIN, FIT, COLLAPSE = 0.05, 0.05, 0.90, 0.20


def load(paths):
    runs = {}          # (arm, T, seed, set) -> {patient: success}; arm in {"U64", "S64", "L238"}
    for p in paths:
        d = json.load(open(p))
        tag = d["meta"]["args"].get("tag", "")
        rows = {r["patient"]: r["success"] for r in d["rows"] if r["policy"].startswith("ppo:")}
        m = TAG_OLD.match(tag)
        if m:
            runs[("U64", 2, int(m.group(1)), m.group(2))] = rows
            continue
        m = TAG_NEW.match(tag)
        if m:
            arm = "S64" if m.group(1) == "N64sh" else "L238"
            runs[(arm, int(m.group(2)), int(m.group(3)), m.group(4))] = rows
    return runs


def seed_mean(runs, arm, T, st):
    seeds = sorted(k[2] for k in runs if k[0] == arm and k[1] == T and k[3] == st)
    if not seeds:
        return None, []
    pats = sorted(set.intersection(*[set(runs[(arm, T, s, st)]) for s in seeds]))
    M = np.array([[runs[(arm, T, s, st)][p] for p in pats] for s in seeds], float)
    return dict(zip(pats, M.mean(0))), [float(v) for v in M.mean(1)]


def mean(d):
    return None if d is None else float(np.mean(list(d.values())))


def main(argv=None):
    paths = argv or sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = glob.glob(paths[0])
    runs = load(paths)
    rows, out = [], {"meta": {"script": "ReconstructionPipeline/evaluation/nav_budget_shard_analysis.py",
                              "preregistration": "DEC-074"}}
    u64, _ = seed_mean(runs, "U64", 2, "test")
    s64, s64_seeds = seed_mean(runs, "S64", 2, "test")
    for name, arm, T in [("N=64 unsharded (DEC-068)", "U64", 2), ("N=64 sharded (arm S)", "S64", 2)] + \
            [(f"N=238 sharded (arm L), {T}M", "L238", T) for T in TS]:
        te, ps = seed_mean(runs, arm, T, "test")
        tr, ps_tr = seed_mean(runs, arm, T, "train")
        rows.append({"run": name, "arm": arm, "T": T, "heldout": mean(te), "heldout_per_seed": ps,
                     "train": mean(tr), "train_per_seed": ps_tr})
    out["rows"] = rows
    v = {}
    if u64 and s64:
        d, lo, hi = boot_diff(s64, u64)
        v["shard"] = {"diff": d, "ci": [lo, hi],
                      "verdict": "SHARD-HARMFUL" if (d <= -MARGIN and hi < 0) else
                      "SHARD-HELPFUL" if (d >= MARGIN and lo > 0) else
                      "SHARD-EQUIVALENT" if (lo >= -MARGIN and hi <= MARGIN) else "INCONCLUSIVE"}
    Ltr = {r["T"]: r["train"] for r in rows if r["arm"] == "L238" and r["train"] is not None}
    if Ltr:
        best = max(Ltr.values())
        t_sel = min(T for T, x in Ltr.items() if x == best)
        v["T_sel"] = t_sel
        v["FIT"] = bool(Ltr[t_sel] >= FIT)
        lte, _ = seed_mean(runs, "L238", t_sel, "test")
        if lte and s64:
            d, lo, hi = boot_diff(lte, s64)
            v["H1d"] = {"diff_vs_S64": d, "ci": [lo, hi], "pass": bool(d >= MIN_GAIN and lo > 0)}
        if lte and u64:
            d, lo, hi = boot_diff(lte, u64)
            v["secondary_vs_U64"] = {"diff": d, "ci": [lo, hi]}
        col = {}
        for s in sorted({k[2] for k in runs if k[0] == "L238"}):
            ser = [(T, mean(runs.get(("L238", T, s, "train")))) for T in TS if ("L238", T, s, "train") in runs]
            if ser:
                b = max(x for _, x in ser)
                ib = [x for _, x in ser].index(b)
                col[str(s)] = bool(any(x < b - COLLAPSE for _, x in ser[ib + 1:]))
        v["collapse_per_seed"] = col
    out["verdict"] = v
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(ROOT, "results", f"nav_budget_shard_{stamp}")
    json.dump(out, open(outp + ".json", "w"), indent=1)
    f = lambda x: "—" if x is None else f"{x:.3f}"
    L = [f"# N = 238 navigation: sharded env vs training budget (DEC-074) ({stamp})", "",
         "| Run | Held-out success | Per seed | Train success | Per seed |", "| :--- | ---: | :--- | ---: | :--- |"]
    for r in rows:
        L.append(f"| {r['run']} | {f(r['heldout'])} | {', '.join(f(x) for x in r['heldout_per_seed'])} | "
                 f"{f(r['train'])} | {', '.join(f(x) for x in r['train_per_seed'])} |")
    L.append("")
    if "shard" in v:
        s = v["shard"]
        L.append(f"Shard effect: s(S64) − s(64) = {s['diff']:+.3f}, 95 % CI [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] "
                 f"(margin ±{MARGIN}): **{s['verdict']}**")
    if "T_sel" in v:
        L.append(f"T_sel (best training-patient success) = {v['T_sel']}M; FIT (train ≥ {FIT}): "
                 f"**{'yes' if v['FIT'] else 'no'}**; collapse per seed: {v['collapse_per_seed']}")
    if "H1d" in v:
        h = v["H1d"]
        L.append(f"H1d: s(L238, T_sel) − s(S64) = {h['diff_vs_S64']:+.3f}, 95 % CI [{h['ci'][0]:+.3f}, "
                 f"{h['ci'][1]:+.3f}] (needs ≥ +{MIN_GAIN} and CI > 0): **{'PASS' if h['pass'] else 'FAIL'}**")
    if "secondary_vs_U64" in v:
        h = v["secondary_vs_U64"]
        L.append(f"Secondary: s(L238, T_sel) − s(64 unsharded) = {h['diff']:+.3f}, 95 % CI [{h['ci'][0]:+.3f}, "
                 f"{h['ci'][1]:+.3f}]")
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", outp + ".json / .md")


if __name__ == "__main__":
    main()
