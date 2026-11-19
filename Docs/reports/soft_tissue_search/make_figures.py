"""Figures for the "Soft-Tissue Search" page (liver and vein routes). Illustrations use TRAINING subjects only.

    ./.venv/bin/python Docs/reports/soft_tissue_search/make_figures.py [--skip-fem]

fig_liver_results.png   per-patient median liver target shift at 5 N vs the 2 mm gate
                        (results/liver_headroom_20261116_202126.json; sliding points from the interrupted DEC-060 run log)
fig_liver_anatomy.png   s0015 (train): liver, probe sites and targets in frame Q (anterior up)
fig_liver_field.png     s0015 (train), one site, 5 N, E_L 4.1 kPa: |u| tied vs sliding (illustration, one site)
fig_vein_anatomy.png    groin veins of three training subjects as a transverse probe would see them
"""

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
LIVER_RES = os.path.join(ROOT, "results", "liver_headroom_20261116_202126.json")
SLIDING_LOG = os.path.join(ROOT, ".agents", "sliding_queue.log")
PLATE, INK, INK2, GRID = "#ffffff", "#15222b", "#56636c", "#e4e8eb"
TEAL, GATE, BLUE, RED = "#0d6f78", "#b4561f", "#2a78d6", "#c0392b"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})


def _save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=150, facecolor=PLATE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_liver_results():
    d = json.load(open(LIVER_RES))
    rows = [(n.replace("sonogym_", ""), r) for n, r in d["patients"].items() if r.get("measurable")]
    rows.sort(key=lambda t: t[1]["by_modulus"]["4.1"]["median_u_5n_mm"])
    slide = {}
    if os.path.exists(SLIDING_LOG):
        for m in re.finditer(r"sonogym_(s\d+) \(\w+\) E_L ([\d.]+) kPa: .*?median \|u\| at 5 N ([\d.]+) mm",
                             open(SLIDING_LOG).read()):
            slide.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=PLATE)
    ax.set_facecolor(PLATE)
    y = np.arange(len(rows))
    for i, (n, r) in enumerate(rows):
        v = [r["by_modulus"][k]["median_u_5n_mm"] for k in ("2.3", "4.1", "5.9")]
        ax.plot([min(v), max(v)], [i, i], color=TEAL, lw=2.2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter([v[1]], [i], s=46, color=TEAL, zorder=3, label="glued liver (4.1 kPa; bar = 2.3–5.9 kPa)" if i == 0 else None)
        if n in slide and "4.1" in slide[n]:
            ax.scatter([slide[n]["4.1"]], [i], s=60, facecolor="none", edgecolor=INK, lw=1.6, zorder=4,
                       label="sliding liver (4.1 kPa; 2 of 12 patients run)" if n == next(iter(slide)) else None)
    ax.axvline(2.0, color=GATE, lw=2.4, zorder=1)
    ax.text(1.97, len(rows) - 0.4, "2 mm gate", color=GATE, ha="right", va="center", fontsize=10, fontweight="bold")
    ax.axvline(75 / 199, color=INK2, lw=1, ls=(0, (3, 4)), zorder=1)
    ax.text(75 / 199 + 0.03, -0.9, "1 image pixel", color=INK2, fontsize=9, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({r['split']})" for n, r in rows])
    ax.set_xlim(0, 2.3)
    ax.set_ylim(-1.4, len(rows) - 0.2)
    ax.set_xlabel("median liver target shift at 5 N (mm)")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_liver_results.png")


def _liver_subject(name="sonogym_s0015"):
    from ReconstructionPipeline.evaluation.liver_headroom_study import load, select_sites
    data, liver, fov = load(os.path.join(ROOT, "data", "liver", name))
    sites, n = select_sites(data, liver)
    return data, liver, sites


def fig_liver_anatomy():
    from ReconstructionPipeline.evaluation.liver_headroom_study import beam_profile, targets_for
    data, liver, sites = _liver_subject()
    ct = data["ct_volume"]
    ox, oy, oz = data["origin_mm"]
    x0, z0 = sites[0]
    iz, ix = int(z0 - oz), int(x0 - ox)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=PLATE)
    views = ((axes[0], ct[iz], liver[iz], (ox, ox + ct.shape[2], oy, oy + ct.shape[1]), "axial slice through the probe",
              "patient left ← x (mm) → right"),
             (axes[1], ct[:, :, ix].T, liver[:, :, ix].T, (oz, oz + ct.shape[0], oy, oy + ct.shape[1]),
              "sagittal slice through the probe", "feet ← z (mm) → head"))
    for ax, img, lv, ext, title, xl in views:
        ax.imshow(img, origin="lower", cmap="gray", vmin=-160, vmax=240, extent=ext, aspect="equal")
        h, w = lv.shape
        ax.contour(np.linspace(ext[0], ext[1], w), np.linspace(ext[2], ext[3], h), lv, [0.5], colors=[TEAL], linewidths=1.8)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("anterior ↑ y (mm)")
    prof = beam_profile(data, liver, x0, z0)
    tg = targets_for(prof, x0, z0)
    ys = prof[0]
    axes[0].plot([x0 - 22.5, x0 + 22.5], [ys + 1.5] * 2, color=GATE, lw=5, solid_capstyle="butt")
    axes[1].plot([z0 - 5, z0 + 5], [ys + 1.5] * 2, color=GATE, lw=5, solid_capstyle="butt")
    for ax, c in ((axes[0], x0), (axes[1], z0)):
        ax.plot([c, c], [ys, ys - 75], color=GATE, lw=1, ls="--")
        ax.scatter([c] * len(tg), [t[1] for t in tg], s=16, color="#ffd34d", zorder=5, edgecolor=INK, lw=0.4)
    axes[1].text(oz + 3, oy + ct.shape[1] - 8, "orange: probe face and beam   yellow: liver targets   teal: liver",
                 color="white", fontsize=8.5, va="top")
    fig.tight_layout()
    _save(fig, "fig_liver_anatomy.png")


def fig_liver_field():
    import torch

    from mtp_gym.core import SHARED_E_SCALE
    from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force
    from ReconstructionPipeline.simulation.fem_batched import BatchedFEMSolver
    from ReconstructionPipeline.simulation.fem_tissue import FEMTissueModel
    data, liver, sites = _liver_subject()
    x0, z0 = sites[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = data["ct_volume"]
    ox, oy, oz = data["origin_mm"]
    lo = np.array([max(x0 - 52.5, -70.0), oy, max(z0 - 40.0, -90.0)])
    hi = np.array([min(x0 + 52.5, 70.0), oy + ct.shape[1] - 1 + 4.0, min(z0 + 40.0, 85.0)])
    zs = np.arange(lo[2] + 2, hi[2] - 2, 1.0)
    yv = np.arange(oy + 2, oy + ct.shape[1] - 2, 1.0)
    Zg, Yg = np.meshgrid(zs, yv)
    P = np.stack([np.full_like(Zg, x0), Yg, Zg], -1).reshape(1, -1, 3)
    iz = np.clip(np.rint(Zg - oz).astype(int), 0, ct.shape[0] - 1)
    iy = np.clip(np.rint(Yg - oy).astype(int), 0, ct.shape[1] - 1)
    in_liver = liver[iz, iy, int(x0 - ox)]
    body = ct[iz, iy, int(x0 - ox)] > -200
    out = {}
    for mode in ("tied", "sliding"):
        kw = dict(sliding_region=liver) if mode == "sliding" else {}
        m = FEMTissueModel(data, roi_min_mm=lo, roi_max_mm=hi, cache_dir=None, device=dev, verbose=False,
                           region_modulus=(liver, 4.1 / SHARED_E_SCALE), **kw)
        m.set_parameters(e_scale=SHARED_E_SCALE)
        s = BatchedFEMSolver(m)
        T = torch.as_tensor(m.dorsal_pose(x0, z0)[None], dtype=torch.float64, device=dev)
        d, _ = indentation_for_force(type("PS", (), {"solver": s})(), T, torch.tensor([5.0], dtype=torch.float64, device=dev))
        st = s.solve(T, d, stiffness=False)
        X = torch.as_tensor(P, dtype=torch.float32, device=dev)
        uo = st.displacement(X).cpu().numpy()[0]
        ui = st.displacement(X, inner=True).cpu().numpy()[0] if mode == "sliding" else uo
        u = np.where(in_liver.reshape(-1, 1), ui, uo)
        mag = np.linalg.norm(u, axis=1).reshape(Zg.shape)
        out[mode] = np.where(body, mag, np.nan)
        print(mode, "force", float(st.reaction_force_n[0]), "max |u|", np.nanmax(out[mode]))
        del m, s, st
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    vmax = float(np.nanpercentile(np.concatenate([out["tied"].ravel(), out["sliding"].ravel()]), 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), facecolor=PLATE, gridspec_kw={"width_ratios": [1, 1, 1]})
    ext = (zs[0], zs[-1], yv[0], yv[-1])
    for ax, mode in zip(axes[:2], ("tied", "sliding")):
        im = ax.imshow(out[mode], origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        ax.contour(Zg, Yg, in_liver.astype(float), [0.5], colors=[TEAL], linewidths=1.6)
        ax.set_title(f"{'glued to the wall' if mode == 'tied' else 'sliding along the wall'}: |u| at 5 N",
                     fontsize=10, loc="left", color=INK)
        ax.set_xlabel("feet ← z (mm) → head")
    axes[0].set_ylabel("anterior ↑ y (mm)")
    fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.02, label="displacement (mm)")
    diff = out["sliding"] - out["tied"]
    lim = float(np.nanpercentile(np.abs(diff), 99.5))
    im2 = axes[2].imshow(diff, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    axes[2].contour(Zg, Yg, in_liver.astype(float), [0.5], colors=[TEAL], linewidths=1.6)
    axes[2].set_title("sliding − glued (red = moves more)", fontsize=10, loc="left", color=INK)
    axes[2].set_xlabel("feet ← z (mm) → head")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, label="mm")
    _save(fig, "fig_liver_field.png")


def fig_vein_anatomy():
    from ReconstructionPipeline.evaluation.vein_compress_study import load, site_and_path
    import nibabel as nib
    names = ["sonogym_s0012_left", "sonogym_s0015_left", "sonogym_s0010_left"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), facecolor=PLATE)
    for ax, n in zip(axes, names):
        d = os.path.join(ROOT, "data", "vein", n)
        data, vein, fov = load(d)
        art = np.asarray(nib.load(os.path.join(d, "artery.nii.gz")).dataobj).transpose(2, 1, 0) > 0
        site, _ = site_and_path(data, vein, fov)
        ct = data["ct_volume"]
        ox, oy, oz = data["origin_mm"]
        iz = int(0 - oz)
        ext = (ox, ox + ct.shape[2], oy, oy + ct.shape[1])
        ax.imshow(ct[iz], origin="lower", cmap="gray", vmin=-160, vmax=240, extent=ext, aspect="equal")
        xx = np.linspace(ext[0], ext[1], ct.shape[2])
        yy = np.linspace(ext[2], ext[3], ct.shape[1])
        ax.contour(xx, yy, vein[iz].astype(float), [0.5], colors=[BLUE], linewidths=2)
        ax.contour(xx, yy, art[iz].astype(float), [0.5], colors=[RED], linewidths=2)
        ys = site["y_skin"]
        ax.plot([site["x"] - 22.5, site["x"] + 22.5], [ys + 1.5] * 2, color=GATE, lw=5, solid_capstyle="butt")
        ax.plot([site["x"]] * 2, [ys, max(ys - 75, oy)], color=GATE, lw=1, ls="--")
        ax.set_ylim(oy, oy + ct.shape[1])
        ax.set_title(f"{n.replace('sonogym_', '').replace('_', ' ')}: vein {site['depth_top_mm']:.0f} mm deep, "
                     f"{site['fat_on_path_mm']:.0f} mm fat above", fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("anterior ↑ y (mm)")
    axes[2].text(0.98, 0.04, "blue: vein   red: artery   orange: probe", transform=axes[2].transAxes, ha="right",
                 color="white", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig_vein_anatomy.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fem", action="store_true")
    a = ap.parse_args()
    fig_liver_results()
    fig_liver_anatomy()
    fig_vein_anatomy()
    if not a.skip_fem:
        fig_liver_field()
