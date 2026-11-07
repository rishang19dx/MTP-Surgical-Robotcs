"""Force-regulated scanning task (DEC-041, task T7).

Run: ./.venv/bin/python -m pytest ReconstructionPipeline/tests/test_vec_force_scan.py -q -s
Needs the imported SonoGym subjects s0015 (thin back) and s0014 (thick back).
"""
import os

import numpy as np
import pytest
import torch

from mtp_gym.core import PatientSim, UltrasoundVectorEnv
from mtp_gym.tasks import ForceScanTask

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
THIN, THICK = (os.path.join(ROOT, "data", "patients", n) for n in ("sonogym_s0015", "sonogym_s0014"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not (os.path.isdir(THIN) and os.path.isdir(THICK)),
                                reason="imported SonoGym subjects not available")
GRID = [(x, z) for z in (-45.0, -32.5, -20.0) for x in (-16.0, 0.0, 16.0)]
FORCES = (2.0, 4.0, 8.0, 12.0, 15.0)


@pytest.fixture(scope="module")
def patients():
    return [PatientSim(THIN, DEV), PatientSim(THICK, DEV)]


def _hold(patients, mode, force_n, steps=8, avg=3):
    """Both patients, 9 window poses each, probe held still at commanded `force_n`.
    Returns per patient: mean visibility and mean (w_vis * visibility - w_fc * sensed force) over the last `avg` steps."""
    task = ForceScanTask(target_coverage=1.01)
    env = UltrasoundVectorEnv(18, task, patients, deformation_mode=mode, max_steps=100, seed=0)
    init = np.array([[x, z, 0.0, 0.0] for _ in range(2) for x, z in GRID], dtype=np.float32)
    env.reset(seed=0, options={"init_pose": init})
    a = np.zeros((18, 5), dtype=np.float32)
    a[:, 4] = task.force_action(force_n)
    vis, img = [], []
    for k in range(steps):
        _, _, _, _, info = env.step(a)
        if k >= steps - avg:
            v = info["bone_visibility"]
            vis.append(v)
            img.append(task.w_vis * v - task.w_fc * info["normal_force"])
    vis, img = torch.stack(vis).mean(0), torch.stack(img).mean(0)
    return [float(vis[:9].mean()), float(vis[9:].mean())], [float(img[:9].mean()), float(img[9:].mean())]


def test_force_command_mapping():
    t = ForceScanTask()
    for f in (1.0, 4.0, 15.0):
        assert float(t.force_command(torch.tensor([t.force_action(f)]))[0]) == pytest.approx(f)


def test_force_changes_visibility_only_with_deformation(patients):
    v_def = {f: _hold(patients, "aware", f)[0] for f in (4.0, 12.0)}
    v_rig = {f: _hold(patients, "rigid", f)[0] for f in (4.0, 12.0)}
    print(f"\n[force] visibility thin/thick  aware 4N {v_def[4.0]}  12N {v_def[12.0]}  |  rigid 4N {v_rig[4.0]}  12N {v_rig[12.0]}")
    # Only the pre-registered control (DEC-042 P3) is asserted.  The deformable visibilities are printed, not
    # asserted: P2 (thick-back gain > 0.10 from 4 to 12 N) failed, and an un-registered "pressing harder never
    # hides bone" check was dropped when the DEC-044 perception preset made the thin back flat (0.608 -> 0.598).
    for p in (0, 1):
        assert abs(v_rig[12.0][p] - v_rig[4.0][p]) < 0.03   # rigid control: force cannot change the image


@pytest.mark.xfail(strict=True, reason="Pre-registered prediction P1 of DEC-042 FAILED (2026-11-07): with the "
                   "legacy B-mode the best force was 4 N for both backs; with the literature depth physics (per-tissue "
                   "attenuation, receiver noise before TGC) it is 2 N for both. H2 is reframed (DEC-042 outcome). "
                   "Kept as a strict xfail so that any later change that flips it is noticed.")
def test_best_force_is_higher_for_the_thick_back(patients):
    """H2 sanity on the reward landscape (not a learned result): the force maximising the image term minus the
    force cost is larger for the thick back than for the thin back."""
    img = np.array([_hold(patients, "aware", f)[1] for f in FORCES])          # (forces, 2)
    best = [FORCES[int(np.argmax(img[:, p]))] for p in (0, 1)]
    print(f"\n[force] image reward per force {dict(zip(FORCES, np.round(img, 3).tolist()))}; best thin {best[0]} N, thick {best[1]} N")
    assert best[1] > best[0]
