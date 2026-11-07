"""
FEM fidelity: precomputed linear model vs Neo-Hookean finite-strain reference on the same mesh (task T14, H4).

    ./.venv/bin/python ReconstructionPipeline/evaluation/fem_fidelity.py [--patients <names>] [--forces 2,5,10,15]

For each patient (shared moduli) and probe site, the linear model's contact problem is solved at the indentation
giving each target LINEAR contact force (sum of nodal forces, before the strain-stiffening factor).  The same
nodal forces are applied to the Neo-Hookean reference (`simulation/fem_nonlinear.py`).  Reported per load:
  * compression under the probe: force-weighted normal displacement of the loaded skin nodes, linear vs NL;
  * displacement difference |u_NL - u_lin| over nodes within 30 mm of the probe centre (RMS and max);
  * Newton convergence of the reference.
Identical loads isolate the elasticity model; contact is not re-solved on the nonlinear geometry (limitation).
Writes results/fem_fidelity_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import PatientSim  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402
from ReconstructionPipeline.simulation.fem_nonlinear import NeoHookeanReference, contact_nodal_forces  # noqa: E402

SITES = {"paraspinal (-20, -45)": (-20.0, -45.0), "midline (0, -32.5)": (0.0, -32.5)}


def linear_force(m, T, d):
    ct = m._contact(T, d)
    return float(ct["f"].sum()) if len(ct["f"]) else 0.0


def indentation_for_linear_force(m, T, target):
    lo = m.first_contact_indentation(T)
    hi = lo + 60.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if linear_force(m, T, mid) < target else (lo, mid)
    return hi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default="../sample_totalsegmentator_patient,sonogym_s0015,sonogym_s0014")
    ap.add_argument("--forces", default="2,5,10,15")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    forces = [float(v) for v in args.forces.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pdir in patient_dirs(args.patients):
        ps = PatientSim(pdir, dev)
        m = ps.model
        ref = NeoHookeanReference(m)
        sk = m.mesh.skin_nodes
        for site, (x, z) in SITES.items():
            T = m.dorsal_pose(x, z)
            for F in forces:
                d = indentation_for_linear_force(m, T, F)
                f, loaded, b = contact_nodal_forces(m, T, d)
                u_lin = ref.linear_solve(f)
                t0 = time.perf_counter()
                r = ref.solve(f, load_steps=max(2, math.ceil(F / 2.5)))
                w = f[sk[loaded]] @ b
                comp = lambda u: float((u[sk[loaded]] @ b) @ w / w.sum())  # noqa: E731
                near = np.linalg.norm(m.mesh.nodes - (T[:3, 3] + d * b), axis=1) <= 30.0
                du = np.linalg.norm(r["u"][near] - u_lin[near], axis=1)
                row = {"patient": os.path.basename(pdir), "site": site, "linear_force_n": F, "indentation_mm": d,
                       "compression_linear_mm": comp(u_lin), "compression_nl_mm": comp(r["u"]),
                       "rms_diff_near_mm": float(np.sqrt((du ** 2).mean())), "max_diff_near_mm": float(du.max()),
                       "max_disp_linear_mm": float(np.linalg.norm(u_lin, axis=1).max()),
                       "newton": r["history"], "seconds": time.perf_counter() - t0}
                row["compression_rel_diff"] = (row["compression_nl_mm"] - row["compression_linear_mm"]) \
                    / row["compression_linear_mm"]
                rows.append(row)
                print(f"{row['patient']:32s} {site:22s} F={F:4.1f} N  comp lin {row['compression_linear_mm']:6.2f} "
                      f"NL {row['compression_nl_mm']:6.2f} mm ({100 * row['compression_rel_diff']:+5.1f} %)  "
                      f"|du| near rms {row['rms_diff_near_mm']:.2f} max {row['max_diff_near_mm']:.2f} mm  "
                      f"res {r['history'][-1]['rel_residual']:.1e}  {row['seconds']:.0f}s", flush=True)
        del ps, m, ref
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
    except Exception:
        commit = "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/fem_fidelity.py", "args": vars(args), "commit": commit,
            "material": "compressible Neo-Hookean, same E (fat/muscle by HU, shared e_scale) and nu = 0.45",
            "comparison": "identical nodal contact forces (linear contact solution, before strain stiffening)"}
    base = os.path.join(args.out_dir, f"fem_fidelity_{stamp}")
    json.dump({"meta": meta, "rows": rows}, open(base + ".json", "w"), indent=2)
    L = [f"# FEM fidelity: linear vs Neo-Hookean reference ({stamp})", "", f"commit `{commit}` · {meta['material']} · "
         f"{meta['comparison']}", "",
         "| Patient | Site | Linear force (N) | Compression linear (mm) | Compression Neo-Hookean (mm) | Rel. diff | RMS diff ≤30 mm (mm) | Max diff ≤30 mm (mm) |",
         "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        L.append(f"| {r['patient']} | {r['site']} | {r['linear_force_n']:.0f} | {r['compression_linear_mm']:.2f} | "
                 f"{r['compression_nl_mm']:.2f} | {100 * r['compression_rel_diff']:+.1f} % | {r['rms_diff_near_mm']:.2f} | "
                 f"{r['max_diff_near_mm']:.2f} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
