"""
Reconstruction metrics (contract C6), vectorized with scipy cKDTree.

    coverage  = fraction of target ROI points (US-visible ground-truth surface) within tol of the reconstruction
    precision = fraction of reconstructed points within tol of ANY ground-truth bone point
    f_score   = 2 P C / (P + C)
    chamfer_mm = 0.5 * (mean recon->bone + mean target->recon)
    mad_mm    = mean recon->bone distance
"""

from typing import Dict, Iterable, Optional, Sequence, Union
import numpy as np
from scipy.spatial import cKDTree


def _nn_dist(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return cKDTree(ref).query(query, k=1, workers=-1)[0]


def compute_reconstruction_metrics(
    recon_pts: np.ndarray,
    target_roi_pts: np.ndarray,
    all_bone_pts: Optional[np.ndarray] = None,
    tol_mm: Union[float, Sequence[float]] = 3.0,
    extra_tols_mm: Iterable[float] = (2.0, 3.0),
) -> Dict[str, float]:
    """
    Args:
        recon_pts: (N, 3) reconstructed points [mm].
        target_roi_pts: (M, 3) target (US-visible) surface points [mm] for coverage.
        all_bone_pts: (K, 3) all ground-truth bone points [mm] for precision / MAD. Defaults to target_roi_pts.
        tol_mm: Primary tolerance (float), or a list whose first element is primary; every listed value is
            also reported with an @tol suffix.
        extra_tols_mm: Additional tolerances reported as 'coverage@t', 'precision@t', 'f_score@t'.

    Returns:
        dict with coverage, precision, f_score, chamfer_mm, mad_mm, num_points, plus @tol variants
        (e.g. 'coverage@2', 'precision@3'). With no reconstructed points: coverage/precision/f_score 0,
        distances inf.
    """
    tols = [float(t) for t in (tol_mm if isinstance(tol_mm, (list, tuple, np.ndarray)) else [tol_mm])]
    primary = tols[0]
    all_tols = sorted(set(tols) | {float(t) for t in extra_tols_mm})

    recon = np.asarray(recon_pts, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_roi_pts, dtype=np.float64).reshape(-1, 3)
    bone = target if all_bone_pts is None else np.asarray(all_bone_pts, dtype=np.float64).reshape(-1, 3)

    out: Dict[str, float] = {"num_points": int(len(recon))}
    if len(recon) == 0 or len(target) == 0 or len(bone) == 0:
        out.update(coverage=0.0, precision=0.0, f_score=0.0, chamfer_mm=float("inf"), mad_mm=float("inf"))
        for t in all_tols:
            out[f"coverage@{t:g}"] = out[f"precision@{t:g}"] = out[f"f_score@{t:g}"] = 0.0
        return out

    d_r2b = _nn_dist(recon, bone)    # recon -> any bone
    d_t2r = _nn_dist(target, recon)  # target ROI -> recon

    def prf(t):
        c = float(np.mean(d_t2r <= t))
        p = float(np.mean(d_r2b <= t))
        f = 2.0 * p * c / (p + c) if (p + c) > 0 else 0.0
        return c, p, f

    c, p, f = prf(primary)
    out.update(
        coverage=c, precision=p, f_score=f,
        chamfer_mm=float(0.5 * (d_r2b.mean() + d_t2r.mean())),
        mad_mm=float(d_r2b.mean()),
        median_r2b_mm=float(np.median(d_r2b)),
    )
    for t in all_tols:
        c, p, f = prf(t)
        out[f"coverage@{t:g}"], out[f"precision@{t:g}"], out[f"f_score@{t:g}"] = c, p, f
    return out
