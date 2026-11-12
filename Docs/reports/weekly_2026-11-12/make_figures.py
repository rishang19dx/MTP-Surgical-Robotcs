"""Figures for the weekly report (5-12 Nov 2026), read from results/ only.

    ./.venv/bin/python Docs/reports/weekly_2026-11-12/make_figures.py

fig_flex_gate.pdf: absolute relative compression error against the Neo-Hookean reference, per scored case, for
the two MuJoCo flex materials and the project's linear FEM (results/mujoco_flex_fidelity_20261112_112537.json).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "results", "mujoco_flex_fidelity_20261112_112537.json")

# Reference palette, categorical slots 1-3 in fixed order (dataviz skill, references/palette.md), light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("Linear FEM (project)", "#2a78d6", "^"), ("Stable Neo-Hookean (MuJoCo flex)", "#1baf7a", "o"),
          ("SVK (MuJoCo flex)", "#eb6834", "s")]


def main():
    rows = [r for r in json.load(open(SRC))["rows"] if r["scored"]]
    data = {
        SERIES[0][0]: [(r["linear_force_n"], abs(r["rel_err_linear"])) for r in rows if r["material"] == "svk"],
        SERIES[1][0]: [(r["linear_force_n"], abs(r["rel_err_flex"])) for r in rows if r["material"] == "snh"],
        SERIES[2][0]: [(r["linear_force_n"], abs(r["rel_err_flex"])) for r in rows if r["material"] == "svk"],
    }
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = (-0.45, 0.0, 0.45)
    for (name, color, marker), dx in zip(SERIES, offsets):
        f, e = np.array(data[name]).T
        ax.scatter(f + dx, 100 * e, s=34, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.0,
                   label=name, zorder=3)
        med = {F: np.median(e[f == F]) for F in sorted(set(f))}
        ax.plot([F + dx for F in med], [100 * v for v in med.values()], color=color, lw=1.2, alpha=0.6, zorder=2)
    ax.plot([1.2, 5.8], [10, 10], ls="--", lw=1.0, color=INK2, zorder=1, label="Gate G1 bound (every case, 2–5 N)")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 800)
    ax.set_xlim(0.8, 16.2)
    ax.set_xticks([2, 5, 10, 15])
    ax.set_xlabel("Linear contact force (N)")
    ax.set_ylabel("|compression error| vs Neo-Hookean (%)")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("MuJoCo flex vs the linear FEM, same nodal forces (22 scored cases each; lines = medians)",
                 fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_flex_gate.{ext}"), dpi=200, facecolor=SURFACE)
    print("wrote fig_flex_gate.pdf/.png")


if __name__ == "__main__":
    main()
