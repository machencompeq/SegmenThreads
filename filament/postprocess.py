"""Probability map -> instances (one per filament).

Instances are the connected components of the thresholded map. PQ punishes fragmentation,
so components may optionally be labelled on a slightly *dilated* mask (pieces up to
2*link_px apart join into one filament) and are then cut back to the undilated pixels.
Components smaller than `min_area` pixels are dropped.

Everything is done on a single int32 label map (0 = background, 1..K = instances), so
the cost is independent of the number of components. A noisy map with thousands of blobs
used to take seconds and gigabytes per call when every instance was a full 2048x2048 array.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessParams:
    """The three tuned post-processing knobs (chosen on the validation split by PQ)."""
    thr: float = 0.5        # probability threshold for 'filament' pixels
    min_area: int = 200     # drop components smaller than this (pixels at 2048x2048)
    link_px: int = 0        # dilation radius used only to decide which pieces belong together

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PostprocessParams":
        return cls(thr=float(d["thr"]), min_area=int(d["min_area"]), link_px=int(d["link_px"]))


# Grid searched on the validation split. GT filament areas at 2048: 1st percentile 209 px,
# 5th 317, median 1228 (scripts/measure_filaments.py), so min_area above ~300 can only lose true
# filaments; the oracle experiment also showed link_px > 0 merges neighbours, so 0 is
# included and expected to win unless the model fragments filaments.
DEFAULT_GRID = {
    "thr": (0.3, 0.4, 0.5, 0.6),
    "min_area": (50, 100, 150, 200, 300),
    "link_px": (0, 3, 6),
}


def prob_to_labels(prob: np.ndarray, thr: float = 0.5, min_area: int = 200,
                   link_px: int = 0) -> np.ndarray:
    """prob: float32 HxW in [0,1] at full resolution. Returns an int32 label map."""
    fg = (prob >= thr).astype(np.uint8)
    lab_src = fg
    if link_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * link_px + 1, 2 * link_px + 1))
        lab_src = cv2.dilate(fg, k)
    n, labels = cv2.connectedComponents(lab_src, connectivity=8)
    labels = labels * fg                        # cut back to the undilated pixels
    area = np.bincount(labels.ravel(), minlength=n)
    keep = area >= max(int(min_area), 1)
    keep[0] = False
    lut = np.zeros(n, np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[labels]


def labels_to_instances(labels: np.ndarray):
    """Yield one uint8 HxW mask per instance (1..K), in label order."""
    for k in range(1, int(labels.max()) + 1):
        yield (labels == k).astype(np.uint8)


def prob_to_instances(prob: np.ndarray, thr: float = 0.5, min_area: int = 200,
                      link_px: int = 0) -> list[np.ndarray]:
    """Convenience wrapper: list of binary instance masks (same rule as prob_to_labels)."""
    return list(labels_to_instances(prob_to_labels(prob, thr, min_area, link_px)))
