"""
Patient split v2 (DEC-065): the 13 old patients keep their DEC-041 split; the 399 TotalSegmentator v2 patients
(DEC-064) are split by the same rule.

    ./.venv/bin/python -m ReconstructionPipeline.data.make_splits_v2 [--out data/splits_v2.json]

Rule (fixed by the user on 27 Nov 2026, before any study used the new patients):
  * sort the new patients by midline soft-tissue thickness at L4 (meta.json qa.midline_soft_tissue_mm), ties by id;
  * positions 1, 4, 7, ... (every 3rd, 0-based i % 3 == 1) are TEST, the rest TRAIN (as DEC-041);
  * DEV = every 10th TRAIN patient in the same order (j % 10 == 5): for tuning only; DEV is part of TRAIN and is
    never a held-out set.
  * flags (patients kept, marked): old patients that fail the DEC-064 filter, and new patients with partial skin
    coverage of the scan window (results/patients_fem_ts2_shard*of4_20261126.json, free_skin_columns < 744).
Entries are paths relative to `patients_root` (data/patients), as in data/splits.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAGS_OLD = {
    "sonogym_s0004": "fails DEC-064 R3 (dorsal skin + 5 mm outside the source FOV in > 1 % of columns)",
    "sonogym_s0012": "fails DEC-064 R3 (dorsal skin + 5 mm outside the source FOV in > 1 % of columns)",
    "sonogym_s0006": "fails DEC-064 R2 (lumbar vertebra on the first/last axial slice)",
}
MIN_SKIN_COLUMNS = 744


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "splits_v2.json"))
    ap.add_argument("--ts2-dir", default=os.path.join(ROOT, "data", "patients_ts2"))
    args = ap.parse_args(argv)

    old = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    new = []
    for m in sorted(glob.glob(os.path.join(args.ts2_dir, "ts2_s*", "meta.json"))):
        meta = json.load(open(m))
        name = "../patients_ts2/" + os.path.basename(os.path.dirname(m))
        new.append((float(meta["qa"]["midline_soft_tissue_mm"]), name))
    new.sort()
    test = [n for i, (_, n) in enumerate(new) if i % 3 == 1]
    train = [n for i, (_, n) in enumerate(new) if i % 3 != 1]
    dev = [n for j, n in enumerate(train) if j % 10 == 5]

    flags = dict(FLAGS_OLD)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "patients_fem_ts2_shard*of4_20261126.json"))):
        for r in json.load(open(f))["rows"]:
            if r["free_skin_columns"] < MIN_SKIN_COLUMNS:
                name = "../patients_ts2/" + os.path.basename(r["patient"])
                flags[name] = f"partial scan-window skin coverage ({r['free_skin_columns']} free skin columns)"

    out = {
        "description": __doc__.strip().splitlines()[0] + " See DEC-065.",
        "patients_root": old["patients_root"],
        "train": old["train"] + train,
        "test": old["test"] + test,
        "dev": dev,
        "old_train": old["train"], "old_test": old["test"],
        "flags": flags,
        "thickness_order_mm": {**old["thickness_order_mm"], **{n: t for t, n in new}},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}: train {len(out['train'])} (new {len(train)}), test {len(out['test'])} "
          f"(new {len(test)}), dev {len(dev)}, flags {len(flags)}")


if __name__ == "__main__":
    main()
