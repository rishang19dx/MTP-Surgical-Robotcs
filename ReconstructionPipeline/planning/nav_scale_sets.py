"""
Nested training-patient sets for DEC-068 / DEC-069 (frozen with them).

    ./.venv/bin/python -m ReconstructionPipeline.planning.nav_scale_sets --n 16      # prints a comma list

Pool: the NEW (TotalSegmentator v2) TRAIN patients of data/splits_v2.json, minus DEV and minus flagged patients,
sorted by name and permuted with numpy default_rng(0). The set for N is the first N of that order, so the sets are
nested (N=1 within 4 within 16 within 64).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def order(splits=os.path.join(ROOT, "data", "splits_v2.json")):
    s = json.load(open(splits))
    bad = set(s["dev"]) | set(s.get("flags", {}))
    pool = sorted(n for n in s["train"] if n.startswith("../patients_ts2/") and n not in bad)
    return [pool[i] for i in np.random.default_rng(0).permutation(len(pool))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args(argv)
    print(",".join(order()[:a.n]))


if __name__ == "__main__":
    main()
