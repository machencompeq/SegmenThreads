"""Reproducible dataset measurements quoted in the README and the technical report.

(a) Filament width: 2 x the maximum of the Euclidean distance transform inside each
    instance mask (median and 5th percentile over random instances).
(b) GT instance area percentiles (p1, p5, p50) at 2048x2048, from the polygon rasterisation.
(c) Resolution / soft-target oracle: the GT union of random annotation sets is downscaled
    with INTER_AREA to a working resolution and brought back to 2048 either
      (i)  hard:  binarised at 0.5 first, then upsampled with INTER_LINEAR, or
      (ii) soft:  kept as area fraction and upsampled with INTER_CUBIC,
    then thresholded at 0.5, split into instances with prob_to_labels(min_area=100) and
    scored with PQ against the original instances. This is the PQ ceiling a perfect
    network could reach at that resolution.

Run:  python scripts/measure_filaments.py [--data DIR] [--n-width 300] [--n-oracle 40]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.data import IMG_SIZE, Sample, find_dataset_root, instance_masks, load_samples, polygon_to_mask, semantic_mask  # noqa: E402
from filament.metric import gt_crops, pq_from_stats, pq_stats_labels  # noqa: E402
from filament.postprocess import prob_to_labels  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def filament_widths(samples: list[Sample], n: int, rng: random.Random) -> np.ndarray:
    """2 * max distance-transform value per instance, for n random (set, polygon) pairs."""
    pairs = [(s, i) for s in samples for i in range(len(s.polygons))]
    widths = []
    for s, i in rng.sample(pairs, min(n, len(pairs))):
        m = polygon_to_mask(s.polygons[i])
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        widths.append(2.0 * float(dist.max()))
    return np.asarray(widths)


def instance_areas(samples: list[Sample]) -> np.ndarray:
    """Rasterised area of every polygon (identical to mask.sum(), but without decoding)."""
    areas = []
    for s in samples:
        rles = mask_utils.frPyObjects(s.polygons, IMG_SIZE, IMG_SIZE)
        areas.extend(int(a) for a in mask_utils.area(rles))
    return np.asarray(areas)


def oracle_pq(samples: list[Sample], size: int, soft: bool, min_area: int = 100) -> float:
    """PQ of the GT union after a down/up-scaling round trip through `size`."""
    stats = []
    for s in samples:
        gts = instance_masks(s)
        union = semantic_mask(s).astype(np.float32)
        small = cv2.resize(union, (size, size), interpolation=cv2.INTER_AREA)
        if soft:
            prob = cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        else:
            hard = (small >= 0.5).astype(np.float32)
            prob = cv2.resize(hard, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        labels = prob_to_labels(np.clip(prob, 0, 1), thr=0.5, min_area=min_area, link_px=0)
        stats.append(pq_stats_labels(gt_crops(gts), labels))
    return pq_from_stats(stats)["PQ"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT / "data", help="folder containing the dataset")
    ap.add_argument("--n-width", type=int, default=300, help="random instances for the width estimate")
    ap.add_argument("--n-oracle", type=int, default=40, help="random annotation sets for the oracle")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 1536], help="working resolutions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = load_samples(find_dataset_root(args.data))
    rng = random.Random(args.seed)

    widths = filament_widths(samples, args.n_width, rng)
    areas = instance_areas(samples)
    oracle_sets = rng.sample(samples, min(args.n_oracle, len(samples)))

    print(f"annotation sets: {len(samples)}, instances: {len(areas)}, seed: {args.seed}")
    print()
    print(f"(a) filament width at 2048 (2*max distance transform, n={len(widths)})")
    print(f"    {'p5':>6} {'median':>8} {'p95':>6}")
    print(f"    {np.percentile(widths, 5):6.1f} {np.median(widths):8.1f} {np.percentile(widths, 95):6.1f} px")
    print()
    print(f"(b) GT instance area at 2048 (all {len(areas)} instances)")
    print(f"    {'p1':>6} {'p5':>6} {'p50':>6}")
    print(f"    {np.percentile(areas, 1):6.0f} {np.percentile(areas, 5):6.0f} {np.percentile(areas, 50):6.0f} px")
    print()
    print(f"(c) resolution oracle PQ on {len(oracle_sets)} random sets (thr 0.5, min_area 100, link 0)")
    print(f"    {'size':>6} {'hard+linear':>12} {'soft+cubic':>11}")
    for size in args.sizes:
        hard = oracle_pq(oracle_sets, size, soft=False)
        soft = oracle_pq(oracle_sets, size, soft=True)
        print(f"    {size:>6} {hard:12.3f} {soft:11.3f}")


if __name__ == "__main__":
    main()
