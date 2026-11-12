"""
Step-2 check (DEC-057, pre-registered): does a navigation policy behave differently depending on the tissue model it
was trained on? Frozen with the pre-registration; computed from the three evaluation files of `scripts/run_nav_transfer.sh`.

    ./.venv/bin/python ReconstructionPipeline/evaluation/nav_transfer_analysis.py \
        --aware results/vec_eval_nav_A.json --rigid results/vec_eval_nav_B.json --kinematic results/vec_eval_nav_C.json

Training conditions (2 seeds each, 2 patients: sample, s0015): aware (deformable), rigid (SonoGym-style: probe on the
skin, undeformed CT), kinematic (probe pressed in, undeformed CT). Every policy is evaluated in every deployment
condition on the same 2 patients, 32 episodes each (64 per policy and condition).

Primary (deployment on deformable tissue, "aware"), per seed s:
    SonoGym gap_s           = success(aware-trained s) - success(rigid-trained s)
    deformation-only gap_s  = success(aware-trained s) - success(kinematic-trained s)
A gap COUNTS if |mean over seeds| >= 0.15 and both seeds have the same sign. Decision for step 3: move on to policy
outcomes only if the deformation-only gap counts.
Reported, not decisive: the other deployments, final position / yaw error, episode length, mean force.
Writes results/nav_transfer_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP = 0.15
CONDITIONS = ("aware", "rigid", "kinematic")
METRICS = ("success", "dist", "yaw_err", "steps", "force_mean")


def _policy_key(name: str):
    """'ppo:nav_<cond>_depth_s<seed>_<tag>.zip' -> (cond, seed); baselines -> (name, None)."""
    m = re.match(r"ppo:nav_(aware|rigid|kinematic)_depth_s(\d+)_", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def table(path: str):
    d = json.load(open(path))
    out = {}
    for r in d["rows"]:
        k = _policy_key(r["policy"])
        out.setdefault(k, []).append(r)
    return {k: {m: float(np.mean([r[m] for r in rs if r[m] is not None])) for m in METRICS} for k, rs in out.items()}, d


def gap(tab, ref_cond, seeds=(0, 1)):
    g = [tab[("aware", s)]["success"] - tab[(ref_cond, s)]["success"] for s in seeds]
    mean = float(np.mean(g))
    counts = bool(abs(mean) >= GAP and (all(x > 0 for x in g) or all(x < 0 for x in g)))
    return {"per_seed": g, "mean": mean, "counts": counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in CONDITIONS:
        ap.add_argument(f"--{c}", required=True, help=f"vec_eval_nav JSON deployed in '{c}'")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    tabs, metas = {}, {}
    for c in CONDITIONS:
        tabs[c], d = table(getattr(args, c))
        metas[c] = {"file": os.path.relpath(getattr(args, c), ROOT), "commit": d["meta"]["commit"],
                    "eval_deformation": d["meta"]["args"]["deformation"]}
        assert d["meta"]["args"]["deformation"] == c, f"{getattr(args, c)} was not deployed in '{c}'"
    son, dfm = gap(tabs["aware"], "rigid"), gap(tabs["aware"], "kinematic")
    decision = "move on to policy outcomes" if dfm["counts"] else "do not move on (deformation-only gap does not count)"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    res = {"meta": {"script": "ReconstructionPipeline/evaluation/nav_transfer_analysis.py", "preregistration": "DEC-057",
                    "inputs": metas, "gap_threshold": GAP},
           "sonogym_gap": son, "deformation_only_gap": dfm, "decision": decision,
           "tables": {c: {f"{k[0]}{'' if k[1] is None else f'_s{k[1]}'}": v for k, v in t.items()} for c, t in tabs.items()}}
    base = os.path.join(args.out_dir, f"nav_transfer_{stamp}")
    json.dump(res, open(base + ".json", "w"), indent=1)
    pct = lambda v: f"{100 * v:+.1f}"  # noqa: E731
    L = [f"# Step-2 check: navigation transfer across tissue models ({stamp})", "",
         "pre-registration DEC-057 · patients sample, s0015 (training patients) · 64 episodes per policy and condition · "
         "inputs: " + ", ".join(f"`{m['file']}` ({c})" for c, m in metas.items()), "",
         "## Primary (deployed on deformable tissue)", "", "| Gap | Seed 0 | Seed 1 | Mean | Counts (≥ 15 points, same sign) |",
         "| :--- | ---: | ---: | ---: | :---: |",
         f"| SonoGym gap (aware-trained − rigid-trained) | {pct(son['per_seed'][0])} | {pct(son['per_seed'][1])} | "
         f"{pct(son['mean'])} | {'**yes**' if son['counts'] else 'no'} |",
         f"| Deformation-only gap (aware-trained − kinematic-trained) | {pct(dfm['per_seed'][0])} | {pct(dfm['per_seed'][1])} | "
         f"{pct(dfm['mean'])} | {'**yes**' if dfm['counts'] else 'no'} |", "", f"**Decision:** {decision}.", "",
         "## Success by training condition × deployment condition", "",
         "| Trained \\ deployed | " + " | ".join(CONDITIONS) + " |", "| :--- | " + " | ".join("---:" for _ in CONDITIONS) + " |"]
    keys = sorted({k for t in tabs.values() for k in t if k[1] is not None}) + \
        sorted({k for t in tabs.values() for k in t if k[1] is None})
    for k in keys:
        name = k[0] if k[1] is None else f"{k[0]} s{k[1]}"
        L.append(f"| {name} | " + " | ".join(f"{tabs[c][k]['success']:.3f}" if k in tabs[c] else "—" for c in CONDITIONS) + " |")
    L += ["", "## Behaviour on deformable tissue (secondary)", "",
          "| Policy | Success | Final position error (mm) | Final yaw error (deg) | Episode length | Mean force (N) |",
          "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for k in keys:
        if k in tabs["aware"]:
            v = tabs["aware"][k]
            name = k[0] if k[1] is None else f"{k[0]} s{k[1]}"
            L.append(f"| {name} | {v['success']:.3f} | {v['dist']:.1f} | {v['yaw_err']:.1f} | {v['steps']:.1f} | "
                     f"{v['force_mean']:.2f} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
