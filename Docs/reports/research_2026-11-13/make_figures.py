"""Figures for the research report of 13 Nov 2026.

    ./.venv/bin/python Docs/reports/research_2026-11-13/make_figures.py [--skip-sim]

Data figures (read from results/ only):
  fig_thickness_vs_force.pdf  change in muscle thickness from 1 N, per held-out patient, deformable image / deformable
                              ground truth / rigid image (results/muscle_thickness_20261112_151349.json, DEC-056)
  fig_nav_transfer.pdf        navigation success by training x deployment tissue model, both seeds
                              (results/nav_transfer_20261112_183531.json, DEC-057)
Illustrations (rendered by the simulator at the current commit; one site, one speckle seed; not results):
  fig_demo_muscle.png         B-mode at 1 N and 5 N, deformable vs rigid, training patient s0015, with the image-only
                              fascia and bone detections and the label ground truth
  fig_demo_modes.png          one gym frame in the rigid / kinematic / deformable modes and their differences
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

# Reference palette slots 1-3 (dataviz skill), light surface; the same as the weekly report.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})


def _clean(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name, exts=("pdf", "png")):
    for ext in exts:
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_thickness():
    d = json.load(open(THICK))
    F = np.asarray(d["meta"]["forces_n"], float)
    pp = d["analysis"]["per_patient"]          # measurable = detection at <= 2 N >= 0.5 (DEC-056, as scored)
    test = [n for n, r in d["patients"].items() if r["split"] == "test" and pp.get(n) and pp[n]["detect_low"] >= 0.5]
    fig, axes = plt.subplots(1, len(test), figsize=(3.3 * len(test), 3.0), facecolor=SURFACE, sharey=True)
    series = (("deformable", "img", "Image, deformable tissue", BLUE, "o", "-"),
              ("deformable", "gt", "Ground truth, deformable", GREEN, "^", "--"),
              ("rigid", "img", "Image, rigid tissue", ORANGE, "s", "-"))
    for ax, n in zip(np.atleast_1d(axes), test):
        r = d["patients"][n]
        ok = np.asarray(r["scored_site"], bool)
        for cond, key, label, c, m, ls in series:
            y = np.asarray(r[cond][key], float)[ok]
            dy = np.nanmedian(y - y[:, :1], 0)
            ax.plot(F, dy, ls=ls, marker=m, color=c, ms=4.5, lw=1.4, label=label, zorder=3)
        ax.axvspan(5.5, 10.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text(8.0, -0.75, "reported only\n(linear FEM\nless accurate >5 N)", ha="center", fontsize=7, color=INK2)
        ax.set_title(f"{n.replace('sonogym_', '')} (held out, {int(ok.sum())} sites)", fontsize=9, loc="left")
        ax.set_xlabel("Probe force (N)")
        ax.set_xticks(F)
        _clean(ax)
    np.atleast_1d(axes)[0].set_ylabel("Thickness change from 1 N (mm)\nmedian over sites")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    _save(fig, "fig_thickness_vs_force")


def fig_nav():
    d = json.load(open(NAV))
    deps = ("aware", "kinematic", "rigid")
    dep_label = {"aware": "Deployed on\ndeformable", "kinematic": "Deployed on\nkinematic", "rigid": "Deployed on\nrigid (skin)"}
    trains = (("aware", "Trained deformable", BLUE), ("kinematic", "Trained kinematic", GREEN),
              ("rigid", "Trained rigid (SonoGym-style)", ORANGE))
    fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor=SURFACE)
    w = 0.26
    for j, (tc, label, c) in enumerate(trains):
        for i, dep in enumerate(deps):
            t = d["tables"][dep]
            s = [100 * t[f"{tc}_s{k}"]["success"] for k in (0, 1)]
            x = i + (j - 1) * w
            ax.bar(x, np.mean(s), w * 0.9, color=c, alpha=0.85, label=label if i == 0 else None, zorder=2)
            ax.scatter([x - 0.05, x + 0.05], s, s=14, color=INK, zorder=3, marker="o",
                       label="Seeds 0 and 1" if (i == 0 and j == 0) else None)
    blind = 100 * d["tables"]["aware"]["blind"]["success"]
    ax.axhline(blind, color=INK2, ls=":", lw=1.0, zorder=1, label=f"Blind baseline ({blind:.0f} %)")
    ax.set_xticks(range(3))
    ax.set_xticklabels([dep_label[x] for x in deps])
    ax.set_ylabel("Success (%), 64 episodes")
    ax.set_ylim(0, 108)
    _clean(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "fig_nav_transfer")


def fig_demo_muscle():
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
    d, d0 = indentation_for_force(ps, T, torch.as_tensor(forces, dtype=torch.float64, device=dev))
    st = ps.solver.solve(T, d, stiffness=False)
    face = st.probe_face_poses(T)
    sim = ModelBasedUltrasoundSimulator.depth_physics(device=dev, axial_resolution_mm=AX, lateral_resolution_mm=LAT)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), facecolor=SURFACE)
    ext = [-PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[1] / 2, PHYSICAL_SIZE_MM[0], 0]
    for row, (cond, warp) in enumerate((("Deformable tissue", st.warp_to_reference), ("Rigid tissue", None))):
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
        gt, _, top, bone = _gt(lab, ct)
        img, f_mm, b_mm = measure_thickness(bm, components=True)
        for col in range(2):
            ax = axes[row, col]
            ax.imshow(bm[col].cpu().numpy(), cmap="gray", extent=ext, aspect="auto", vmin=0, vmax=1)
            for y, c, ls, lab_ in ((f_mm[col], ORANGE, "-", "fascia (image)"), (b_mm[col], BLUE, "-", "bone (image)"),
                                   (top[col], GREEN, "--", "muscle top (label, GT)")):
                if np.isfinite(y):
                    ax.plot([-10, 10], [y, y], color=c, ls=ls, lw=1.6, label=lab_)
            ax.set_title(f"{cond}, {forces[col]:.0f} N\nimage {img[col]:.1f} mm · ground truth {gt[col]:.1f} mm",
                         fontsize=8.5, loc="left")
            ax.set_xlabel("lateral (mm)")
            ax.set_ylabel("depth from probe face (mm)")
            ax.set_ylim(52, 0)
    axes[0, 0].legend(frameon=True, fontsize=7, loc="center left", facecolor="white", framealpha=0.8)
    fig.tight_layout()
    _save(fig, "fig_demo_muscle", exts=("png",))


def fig_demo_modes():
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
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3), facecolor=SURFACE)
    names = {"rigid": "Rigid (SonoGym-style):\nprobe on the skin", "kinematic": "Kinematic: probe pressed in,\ntissue not deformed",
             "aware": "Deformable: probe pressed in,\nFEM-deformed tissue"}
    for ax, mode in zip(axes[:3], ("rigid", "kinematic", "aware")):
        im, f, ind = frames[mode]
        ax.imshow(im, cmap="gray", extent=ext, aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{names[mode]}\n{f:.1f} N, indentation {ind:.1f} mm" if mode != "rigid" else
                     f"{names[mode]}\n{f:.1f} N (force simulated, image at skin)", fontsize=8, loc="left")
    d1 = np.abs(frames["kinematic"][0] - frames["rigid"][0])
    d2 = np.abs(frames["aware"][0] - frames["kinematic"][0])
    vmax = float(max(d1.max(), d2.max()))
    for ax, dimg, t in ((axes[3], d1, "|kinematic − rigid|\n(indentation only)"),
                        (axes[4], d2, "|deformable − kinematic|\n(deformation only)")):
        ax.imshow(dimg, cmap="magma", extent=ext, aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{t}\nmean {dimg.mean():.3f}", fontsize=8, loc="left")
    for ax in axes:
        ax.set_xlabel("lateral (mm)")
    axes[0].set_ylabel("depth (mm)")
    fig.tight_layout()
    _save(fig, "fig_demo_modes", exts=("png",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    fig_thickness()
    fig_nav()
    if not a.skip_sim:
        fig_demo_muscle()
        fig_demo_modes()


if __name__ == "__main__":
    main()
