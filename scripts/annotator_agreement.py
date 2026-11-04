"""Inter-annotator agreement measured with the competition metric (a reference ceiling).

For files annotated by at least two people, annotator A's set is scored as if it were a
prediction of annotator B's set. The resulting PQ (about 0.35 with the default settings)
shows how far the "ground truth" itself is from being unambiguous.

Run:  python scripts/annotator_agreement.py [--data DIR] [--n 60] [--seed 0]
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.data import find_dataset_root, instance_masks, load_samples  # noqa: E402
from filament.metric import pq_from_stats, pq_stats  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT / "data", help="folder containing the dataset")
    ap.add_argument("--n", type=int, default=60, help="number of multi-annotated files to score")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = load_samples(find_dataset_root(args.data))
    by_file: dict[str, list] = defaultdict(list)
    for s in samples:
        by_file[s.file_name].append(s)
    multi = [v for v in by_file.values() if len(v) >= 2]
    random.Random(args.seed).shuffle(multi)
    chosen = multi[:args.n]

    stats = []
    for group in chosen:
        a, b = group[0], group[1]
        stats.append(pq_stats(instance_masks(a), instance_masks(b)))
    res = pq_from_stats(stats)

    print(f"files with >= 2 annotator sets: {len(multi)} (of {len(by_file)}); scored {len(stats)}")
    print("inter-annotator agreement (set 1 scored as a prediction of set 2):")
    for k in ("PQ", "SQ", "RQ", "PQ_per_image_mean", "TP", "FP", "FN"):
        v = res[k]
        print(f"  {k:<18} {v:.3f}" if isinstance(v, float) else f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
