"""
MuJoCo flex fidelity: the DEC-053 gate study (pre-registered in DECISIONS.md before this script was run).

    ./.venv/bin/python ReconstructionPipeline/evaluation/mujoco_flex_fidelity.py [--patients ...] [--materials svk,snh]

Cases are those of DEC-047 (`fem_fidelity.py`): three patients, two sites, linear contact force 2/5/10/15 N, on the
same 4 mm mesh, Dirichlet set and per-cell moduli (shared `e_scale`), nu = 0.45. The SAME nodal forces
(`fem_nonlinear.contact_nodal_forces`) are applied to a MuJoCo flex in `.venv-mj4`
(`surgical_physics_bridge/src/mujoco_flex_worker.py study`), which isolates the material model (contact is not
part of the gate).

Reference: `compression_nl_mm` (Neo-Hookean) of REFERENCE_JSON. The linear compression is recomputed here; a case
whose value differs from the JSON by more than 0.1 % has no valid reference and is reported, not scored. Rows
where the Neo-Hookean solve did not converge have no reference and are excluded, as in DEC-047. A flex case that
does not settle counts as a failure.

Gate (per material, all must hold): G1 every scored 2/5 N case within 10 % of Neo-Hookean; G2 median absolute
relative error at 10/15 N <= the linear FEM's on the same cases; G3 finite and no inverted tet in every case;
G4 warm re-settle after a 0.33 mm indentation step <= 33 ms (or <= 100 ms, "10 Hz tissue"); G5 per-tet moduli
represented (asserted by `tests/test_mujoco_flex.py`, recorded here as the test's name).

Writes results/mujoco_flex_fidelity_<stamp>.json / .md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtp_gym.core import PatientSim  # noqa: E402
from ReconstructionPipeline.evaluation.compression_sweep import patient_dirs  # noqa: E402
from ReconstructionPipeline.evaluation.fem_fidelity import SITES, indentation_for_linear_force  # noqa: E402
from ReconstructionPipeline.simulation.fem_nonlinear import NeoHookeanReference, contact_nodal_forces  # noqa: E402

REFERENCE_JSON = os.path.join(ROOT, "results", "fem_fidelity_20261107_234638.json")
MJ4_PYTHON = os.path.join(ROOT, ".venv-mj4", "bin", "python")
WORKER = os.path.join(ROOT, "surgical_physics_bridge", "src", "mujoco_flex_worker.py")
G4_STEP_MM = 0.33
G4_CASE = ("sample_totalsegmentator_patient", "midline (0, -32.5)", 5.0)
LINEAR_CHECK_REL = 1e-3
G5_TEST = "ReconstructionPipeline/tests/test_mujoco_flex.py::test_per_tet_scaling_equals_changing_young"


def per_tet_young_pa(m) -> np.ndarray:
    """The Neo-Hookean reference's moduli (fem_nonlinear.NeoHookeanReference) in Pa."""
    return np.where(m.mesh.cell_hu < m.fat_hu, m.e_fat, m.e_muscle)[m.mesh.tet_cell] * 1e3 * m.e_scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default="../sample_totalsegmentator_patient,sonogym_s0015,sonogym_s0014")
    ap.add_argument("--forces", default="2,5,10,15")
    ap.add_argument("--materials", default="svk,snh")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args(argv)
    forces = [float(v) for v in args.forces.split(",")]
    materials = args.materials.split(",")
    ref_rows = {(r["patient"], r["site"], r["linear_force_n"]): r for r in json.load(open(REFERENCE_JSON))["rows"]}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, g4 = [], {}
    tmp = tempfile.mkdtemp(prefix="mj_flex_")
    for pdir in patient_dirs(args.patients):
        name = os.path.basename(pdir)
        ps = PatientSim(pdir, dev)
        m = ps.model
        ref = NeoHookeanReference(m)
        sk = m.mesh.skin_nodes
        payload = {"nodes": m.mesh.nodes, "tets": m.mesh.tets, "fixed": m.mesh.fixed,
                   "young_pa": per_tet_young_pa(m), "nu": m.nu, "materials": np.array(materials)}
        cases = []
        for site, (x, z) in SITES.items():
            T = m.dorsal_pose(x, z)
            for F in forces:
                d = indentation_for_linear_force(m, T, F)
                f, loaded, b = contact_nodal_forces(m, T, d)
                w = f[sk[loaded]] @ b
                u_lin = ref.linear_solve(f)
                comp_lin = float((u_lin[sk[loaded]] @ b) @ w / w.sum())
                c = len(cases)
                payload.update({f"forces_{c}": f, f"comp_nodes_{c}": sk[loaded], f"comp_w_{c}": w, f"beam_{c}": b})
                cases.append({"patient": name, "site": site, "linear_force_n": F, "indentation_mm": d,
                              "compression_linear_mm": comp_lin})
                if (name, site, F) == G4_CASE:
                    fb, loaded_b, _ = contact_nodal_forces(m, T, d + G4_STEP_MM)
                    payload.update({"g4_forces_a": f, "g4_forces_b": fb, "g4_comp_nodes": sk[loaded],
                                    "g4_comp_w": w, "g4_beam": b})
        payload["n_cases"] = len(cases)
        in_path, out_path = os.path.join(tmp, f"{name}_in.npz"), os.path.join(tmp, f"{name}_out.npz")
        np.savez(in_path, **payload)
        print(f"[{name}] {len(cases)} cases, {len(m.mesh.nodes)} nodes, {len(m.mesh.tets)} tets -> .venv-mj4", flush=True)
        t0 = time.perf_counter()
        subprocess.run([MJ4_PYTHON, WORKER, "study", in_path, out_path], check=True)
        res = np.load(out_path)
        print(f"[{name}] flex solves took {time.perf_counter() - t0:.0f} s", flush=True)
        for c, case in enumerate(cases):
            r_ref = ref_rows.get((name, case["site"], case["linear_force_n"]))
            for mat in materials:
                comp, settled, steps, sec, inverted, finite = res[f"{mat}_{c}_meta"].tolist()
                row = dict(case, material=mat, compression_flex_mm=comp, settled=bool(settled), steps=int(steps),
                           seconds=sec, inverted_tets=int(inverted), finite=bool(finite))
                if r_ref is None:
                    row["scored"], row["reason"] = False, "no DEC-047 row"
                elif not r_ref["converged"]:
                    row["scored"], row["reason"] = False, "Neo-Hookean not converged in DEC-047"
                elif abs(case["compression_linear_mm"] - r_ref["compression_linear_mm"]) \
                        > LINEAR_CHECK_REL * abs(r_ref["compression_linear_mm"]):
                    row["scored"], row["reason"] = False, (f"linear compression {case['compression_linear_mm']:.4f} "
                                                           f"!= JSON {r_ref['compression_linear_mm']:.4f}")
                else:
                    row["scored"], row["reason"] = True, ""
                if r_ref is not None:
                    nl = r_ref["compression_nl_mm"]
                    row["compression_nl_mm"] = nl
                    row["rel_err_flex"] = (comp - nl) / nl
                    row["rel_err_linear"] = (r_ref["compression_linear_mm"] - nl) / nl
                rows.append(row)
        for mat in materials:
            if f"{mat}_g4" in res.files:
                ok, steps, sec, dt = res[f"{mat}_g4"].tolist()
                g4[mat] = {"settled": bool(ok), "steps": int(steps), "wall_ms": sec * 1e3, "timestep_s": dt}
        del ps, m, ref
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    gate = {}
    for mat in materials:
        mr = [r for r in rows if r["material"] == mat and r["scored"]]
        low = [r for r in mr if r["linear_force_n"] <= 5.0]
        high = [r for r in mr if r["linear_force_n"] >= 10.0]
        g1 = bool(low) and all(r["settled"] and abs(r["rel_err_flex"]) <= 0.10 for r in low)
        if high and all(r["settled"] for r in high):
            med_flex = float(np.median([abs(r["rel_err_flex"]) for r in high]))
            med_lin = float(np.median([abs(r["rel_err_linear"]) for r in high]))
            g2 = med_flex <= med_lin
        else:
            med_flex = med_lin = float("nan")
            g2 = False
        allm = [r for r in rows if r["material"] == mat]
        g3 = all(r["finite"] and r["inverted_tets"] == 0 for r in allm)
        c4 = g4.get(mat)
        g4_ms = c4["wall_ms"] if c4 and c4["settled"] else float("inf")
        g4v = "pass (30 Hz)" if g4_ms <= 33.0 else ("pass (10 Hz tissue)" if g4_ms <= 100.0 else "fail")
        gate[mat] = {"G1": g1, "G2": g2, "G2_median_abs_err_flex": med_flex, "G2_median_abs_err_linear": med_lin,
                     "G3": g3, "G4": g4v, "G4_wall_ms": g4_ms, "G5": f"see {G5_TEST}",
                     "n_scored": len(mr), "n_unsettled": sum(not r["settled"] for r in allm)}
        gate[mat]["pass_G1_to_G4"] = bool(g1 and g2 and g3 and g4v != "fail")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        commit += "-dirty" if subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) else ""
        mj = subprocess.check_output([MJ4_PYTHON, "-c", "import mujoco; print(mujoco.__version__)"], text=True).strip()
    except Exception:
        commit, mj = "unknown", "unknown"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"script": "ReconstructionPipeline/evaluation/mujoco_flex_fidelity.py", "args": vars(args), "commit": commit,
            "mujoco": mj, "reference": os.path.relpath(REFERENCE_JSON, ROOT), "gate": "DEC-053",
            "comparison": "identical nodal contact forces (linear contact solution), as DEC-047"}
    base = os.path.join(args.out_dir, f"mujoco_flex_fidelity_{stamp}")
    json.dump({"meta": meta, "rows": rows, "g4": g4, "gate": gate}, open(base + ".json", "w"), indent=2)
    L = [f"# MuJoCo flex fidelity vs Neo-Hookean reference ({stamp})", "",
         f"commit `{commit}` · MuJoCo {mj} · gate DEC-053 · reference `{meta['reference']}` · {meta['comparison']}", "",
         "| Patient | Site | Force (N) | Material | Comp. linear (mm) | Comp. Neo-Hookean (mm) | Comp. flex (mm) | "
         "Rel. err flex | Rel. err linear | Settled (steps) | Inverted | Scored |",
         "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |"]
    for r in rows:
        nl = r.get("compression_nl_mm")
        L.append(f"| {r['patient']} | {r['site']} | {r['linear_force_n']:.0f} | {r['material']} | "
                 f"{r['compression_linear_mm']:.2f} | {nl if nl is None else format(nl, '.2f')} | "
                 f"{r['compression_flex_mm']:.2f} | "
                 f"{'' if 'rel_err_flex' not in r else format(100 * r['rel_err_flex'], '+.1f') + ' %'} | "
                 f"{'' if 'rel_err_linear' not in r else format(100 * r['rel_err_linear'], '+.1f') + ' %'} | "
                 f"{'yes' if r['settled'] else 'NO'} ({r['steps']}) | {r['inverted_tets']} | "
                 f"{'yes' if r['scored'] else 'no: ' + r['reason']} |")
    L += ["", "## Gate (DEC-053)", "", "| Material | G1 (≤5 N, all within 10 %) | G2 (10–15 N median vs linear) | "
          "G3 (finite, no inversion) | G4 (warm re-settle) | G5 | Passes |", "| :--- | :---: | :--- | :---: | :--- | :--- | :---: |"]
    for mat, g in gate.items():
        L.append(f"| {mat} | {'pass' if g['G1'] else 'FAIL'} | {'pass' if g['G2'] else 'FAIL'} "
                 f"({100 * g['G2_median_abs_err_flex']:.1f} % vs {100 * g['G2_median_abs_err_linear']:.1f} %) | "
                 f"{'pass' if g['G3'] else 'FAIL'} | {g['G4']} ({g['G4_wall_ms']:.0f} ms) | {g['G5']} | "
                 f"{'**yes**' if g['pass_G1_to_G4'] else '**no**'} |")
    open(base + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-2 - len(gate):]))
    print("wrote", base + ".json / .md")


if __name__ == "__main__":
    main()
