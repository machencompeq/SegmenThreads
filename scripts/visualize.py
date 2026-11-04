"""Draw training images with their filament annotations overlaid and save them to viz/.

Each output figure has three panels: left = raw image, middle = annotation outlines,
right = zoom on the largest filament (raw on top, annotated below).

Picks the 3 annotation sets with the most filaments plus one with a median count.

Run:  python scripts/visualize.py [--data DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.data import (IMG_SIZE, Sample, find_dataset_root, instance_masks, load_samples,  # noqa: E402
                           read_gray, train_image_dir)

PROJECT = Path(__file__).resolve().parents[1]


def render(sample: Sample, image_dir: Path, rng: random.Random) -> np.ndarray:
    """Return the 3-panel BGR figure for one annotation set."""
    gray = read_gray(image_dir / sample.file_name)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    drawn = base.copy()
    polys = [np.array(p, dtype=np.float32).reshape(-1, 2).astype(np.int32) for p in sample.polygons]
    for p in polys:
        color = (rng.randint(0, 120), rng.randint(150, 255), 255)
        cv2.polylines(drawn, [p], True, color, 4)

    # zoom on the filament with the largest rasterised area (same convention as the host)
    areas = [int(m.sum()) for m in instance_masks(sample)]
    big = polys[int(np.argmax(areas))]
    x, y, w, h = cv2.boundingRect(big)
    pad = max(w, h) // 2 + 40
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
    x1, y1 = min(cx + pad, IMG_SIZE - 1), min(cy + pad, IMG_SIZE - 1)
    crop_raw = base[y0:y1, x0:x1]
    crop_ann = drawn[y0:y1, x0:x1]
    zoom = cv2.resize(np.vstack([crop_raw, crop_ann]), (512, 1024), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(drawn, (x0, y0), (x1, y1), (0, 255, 255), 4)

    row = np.hstack([cv2.resize(base, (1024, 1024)), cv2.resize(drawn, (1024, 1024)), zoom])
    cv2.putText(row, f"{sample.set_id}  filaments={len(polys)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    cv2.putText(row, "zoom: raw / annotated", (2060, 1000), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT / "data", help="folder containing the dataset")
    ap.add_argument("--out", type=Path, default=PROJECT / "viz", help="output folder for the figures")
    args = ap.parse_args()

    root = find_dataset_root(args.data)
    image_dir = train_image_dir(root)
    samples = load_samples(root)

    args.out.mkdir(exist_ok=True)
    for old in args.out.glob("*.jpg"):
        old.unlink()

    # the 3 most crowded annotation sets + one with a median filament count
    ordered = sorted(samples, key=lambda s: len(s.polygons), reverse=True)
    picks = ordered[:3] + [ordered[len(ordered) // 2]]

    rng = random.Random(0)
    for s in picks:
        out = args.out / f"{s.set_id}.jpg"
        cv2.imwrite(str(out), render(s, image_dir, rng))
        print(out.name, len(s.polygons))


if __name__ == "__main__":
    main()
