"""Figures for the RSS 2027 Stage 1 extended abstract, sized for the IEEEtran two-column layout
(column 3.5 in, page 7.16 in).

    ./.venv/bin/python Docs/papers/rss2027_stage1/make_figures.py [--skip-sim]

Data figures read results/ only:
  fig_thickness.pdf  results/muscle_thickness_20261112_151349.json (thread B v2, pre-registered DEC-056)
  fig_nav.pdf        results/nav_transfer_20261112_183531.json (pre-registered DEC-057)
Illustrations are rendered by the simulator at the current commit (training patient s0015; one site/pose; speckle
seed fixed); they are not results:
  fig_modes.pdf      one frame in the rigid / kinematic / deformable modes and their differences
  fig_muscle.pdf     B-mode at 1 N and 5 N, deformable vs rigid, with the image-only detections and the label truth
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
THICK = os.path.join(ROOT, "results", "muscle_thickness_20261112_151349.json")
NAV = os.path.join(ROOT, "results", "nav_transfer_20261112_183531.json")
COL, PAGE = 3.45, 7.1

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman",
                     "DejaVu Serif"], "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def _clean(ax):
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", name)


def fig_thickness():
    d = json.load(open(THICK))
    F = np.asarray(d["meta"]["forces_n"], float)
    pp = d["analysis"]["per_patient"]
    test = [n for n, r in d["patients"].items() if r["split"] == "test" and pp.get(n) and pp[n]["detect_low"] >= 0.5]
    fig, axes = plt.subplots(1, len(test), figsize=(COL, 1.75), sharey=True)
    series = (("deformable", "img", "image, deformable", BLUE, "o", "-"),
              ("deformable", "gt", "truth, deformable", GREEN, "^", "--"),
              ("rigid", "img", "image, rigid", ORANGE, "s", "-"))
    for ax, n in zip(axes, test):
        r = d["patients"][n]
        ok = np.asarray(r["scored_site"], bool)
        for cond, key, label, c, m, ls in series:
            y = np.asarray(r[cond][key], float)[ok]
            ax.plot(F, np.nanmedian(y - y[:, :1], 0), ls=ls, marker=m, color=c, ms=3, lw=1.0, label=label, zorder=3)
        ax.axvspan(5.5, 10.5, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.set_title(f"held-out {n.replace('sonogym_', '')}", fontsize=7, loc="left", pad=2)
        ax.set_xlabel("probe force (N)")
        ax.set_xticks([1, 2, 3, 4, 5, 8, 10])
        _clean(ax)
    axes[0].set_ylabel("thickness change\nfrom 1 N (mm)")
    axes[0].legend(frameon=False, fontsize=6, loc="lower left", handlelength=2.2)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_thickness")


def fig_nav():
    d = json.load(open(NAV))
    deps = ("aware", "kinematic", "rigid")
    dep_label = {"aware": "deformable", "kinematic": "kinematic", "rigid": "rigid"}
    trains = (("aware", "trained deformable", BLUE), ("kinematic", "trained kinematic", GREEN),
              ("rigid", "trained rigid", ORANGE))
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    w = 0.26
    for j, (tc, label, c) in enumerate(trains):
        for i, dep in enumerate(deps):
            s = [100 * d["tables"][dep][f"{tc}_s{k}"]["success"] for k in (0, 1)]
            x = i + (j - 1) * w
            ax.bar(x, np.mean(s), w * 0.9, color=c, alpha=0.85, label=label if i == 0 else None, zorder=2)
            ax.scatter([x - 0.05, x + 0.05], s, s=5, color=INK, zorder=3, label="seeds" if (i == 0 and j == 0) else None)
    blind = 100 * d["tables"]["aware"]["blind"]["success"]
    ax.axhline(blind, color=INK2, ls=":", lw=0.8, zorder=1, label="blind")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"tested {dep_label[x]}" for x in deps])
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105)
    _clean(ax)
    ax.legend(frameon=False, fontsize=6, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=0.8)
    fig.tight_layout()
    _save(fig, "fig_nav")


def fig_modes():
    import torch

    from mtp_gym.core import PHYSICAL_SIZE_MM, PatientSim, UltrasoundVectorEnv
    from mtp_gym.tasks import FreeScanTask

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = PatientSim(os.path.join(ROOT, "data", "patients", "sonogym_s0015"), dev)
    frames = {}
    for mode in ("rigid", "kinematic", "aware"):
        env = UltrasoundVectorEnv(1, FreeScanTask(), [p], deformation_mode=mode, seed=4)
        env.reset(seed=4, options={"init_pose": [0.0, -30.0, 0.0, 0.0]})
        fr = env.last_frame
        frames[mode] = (fr.bmode[0].squeeze().detach().cpu().numpy(), float(fr.force_n[0]),
                        float(fr.indentation_mm[0]))
    ext = [-PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[0], 0]
    fig, axes = plt.subplots(1, 5, figsize=(PAGE, 1.75), sharey=True)
    titles = {"rigid": "(a) rigid: probe on skin", "kinematic": "(b) kinematic: pressed in",
              "aware": "(c) deformable: pressed in"}
    for ax, mode in zip(axes[:3], ("rigid", "kinematic", "aware")):
        im, f, ind = frames[mode]
        ax.imshow(im, cmap="gray", extent=ext, aspect="auto", vmin=0, vmax=1)
        sub = f"{f:.1f} N, image at skin" if mode == "rigid" else f"{f:.1f} N, indentation {ind:.1f} mm"
        ax.set_title(f"{titles[mode]}\n{sub}", fontsize=6.5, loc="left", pad=2)
    d1 = np.abs(frames["kinematic"][0] - frames["rigid"][0])
    d2 = np.abs(frames["aware"][0] - frames["kinematic"][0])
    vmax = float(max(d1.max(), d2.max()))
    for ax, dimg, t in ((axes[3], d1, "(d) |b − a|: indentation"), (axes[4], d2, "(e) |c − b|: deformation")):
        ax.imshow(dimg, cmap="magma", extent=ext, aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{t}\nmean {dimg.mean():.3f}", fontsize=6.5, loc="left", pad=2)
    for ax in axes:
        ax.set_xlabel("lateral (mm)")
    axes[0].set_ylabel("depth (mm)")
    fig.tight_layout(w_pad=0.3)
    _save(fig, "fig_modes")


def fig_muscle():
    import torch

    from mtp_gym.core import IMAGE_SIZE, PHYSICAL_SIZE_MM, PatientSim
    from ReconstructionPipeline.evaluation.compression_sweep import indentation_for_force
    from ReconstructionPipeline.evaluation.muscle_compression_study import AX, LAT, SITES, muscle_volume
    from ReconstructionPipeline.evaluation.muscle_thickness_study import _gt, measure_thickness
    from ReconstructionPipeline.simulation.slice_extractor import VolumeSliceExtractor
    from ReconstructionPipeline.simulation.ultrasound_model_based import (ModelBasedUltrasoundSimulator,
                                                                         superficial_fascia_mask)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = os.path.join(ROOT, "data", "patients", "sonogym_s0015")     # a TRAINING patient
    ps = PatientSim(pdir, dev)
    dd = ps.data
    mus = VolumeSliceExtractor(muscle_volume(pdir), spacing=dd["spacing_mm"], origin=dd["origin_mm"],
                               image_size=IMAGE_SIZE, physical_size_mm=PHYSICAL_SIZE_MM, device=dev)
    forces = np.array([1.0, 5.0])
    pose = ps.model.dorsal_pose(*SITES[0])
    T = torch.as_tensor(np.repeat(pose[None], 2, 0), dtype=torch.float64, device=dev)
    d, _ = indentation_for_force(ps, T, torch.as_tensor(forces, dtype=torch.float64, device=dev))
    st = ps.solver.solve(T, d, stiffness=False)
    face = st.probe_face_poses(T)
    sim = ModelBasedUltrasoundSimulator.depth_physics(device=dev, axial_resolution_mm=AX, lateral_resolution_mm=LAT)
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.2), sharex=True, sharey=True)
    ext = [-PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[0], 0]
    for row, (cond, warp) in enumerate((("deformable", st.warp_to_reference), ("rigid", None))):
        kw = {"batched_warp_fn": warp} if warp is not None else {}
        with torch.no_grad():
            lab = mus.extract_slices(face, interpolation="nearest", **kw) > 0.5
            ct = ps.ct.extract_slices(face, interpolation="bilinear", **kw)
            fas = superficial_fascia_mask(lab, AX)
            bm = []
            for i in range(2):
                g = torch.Generator(device=dev)
                g.manual_seed(0)
                bm.append(sim.simulate_bmode(ct[i:i + 1], fascia_mask=fas[i:i + 1], generator=g)["bmode"])
            bm = torch.cat(bm)
        gt, _, top, _ = _gt(lab, ct)
        img, f_mm, b_mm = measure_thickness(bm, components=True)
        for col in range(2):
            ax = axes[row, col]
            ax.imshow(bm[col].cpu().numpy(), cmap="gray", extent=ext, aspect="auto", vmin=0, vmax=1)
            for y, c, ls, lab_ in ((f_mm[col], ORANGE, "-", "fascia (image)"), (b_mm[col], BLUE, "-", "bone (image)"),
                                   (top[col], GREEN, "--", "muscle top (truth)")):
                if np.isfinite(y):
                    ax.plot([-10, 10], [y, y], color=c, ls=ls, lw=1.4, label=lab_)
            ax.set_title(f"{cond}, {forces[col]:.0f} N\nimage {img[col]:.1f} mm, truth {gt[col]:.1f} mm", fontsize=6.3,
                         loc="left", pad=2)
            ax.set_ylim(52, 0)
    for ax in axes[1]:
        ax.set_xlabel("lateral (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth (mm)")
    axes[0, 0].legend(frameon=True, fontsize=5.5, loc="center left", facecolor="white", framealpha=0.85)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    _save(fig, "fig_muscle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    fig_thickness()
    fig_nav()
    if not a.skip_sim:
        fig_modes()
        fig_muscle()


if __name__ == "__main__":
    main()
