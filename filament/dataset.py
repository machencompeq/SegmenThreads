"""PyTorch dataset: downscaled whole-disk images + soft filament targets, cached in RAM.

Design (measured on the 2026 release, see README "Design decisions"):
- One item per *file*, not per annotator set. Up to 3 annotators mark the same image and
  their union masks agree only moderately (median IoU 0.47), so the target is the pixelwise
  mean of the annotators' masks. Same optimum as feeding the sets separately, but with
  less gradient noise and without weighting multi-annotated files 2-3x.
- The 2048 mask is downscaled with INTER_AREA and kept *soft* (area fraction per pixel)
  instead of thresholded: thin filaments (widest point median 15.6 px, 5th pct 8 px at 2048) lose ~10 % IoU
  when binarised at 1024, but almost nothing when the fraction is kept and the prediction
  is upsampled bicubically (oracle PQ ceiling 0.90 -> 0.99).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Sample, read_gray, semantic_mask

# Fixed normalisation; the exact values do not matter much because the encoder starts
# with BatchNorm, but train and inference must use the same ones.
MEAN, STD = 0.5, 0.25

# Photometric jitter ranges: GONG sites differ in exposure/seeing. Disk mean varies 99-130
# across train+test (ratio ~1.3); filaments are ~12 % darker than their surroundings.
GAIN_RANGE = (0.80, 1.25)
OFFSET_RANGE = (-0.08, 0.08)
GAMMA_RANGE = (0.80, 1.25)


def normalize(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HxW -> float tensor 1xHxW with the training normalisation."""
    return torch.from_numpy(np.ascontiguousarray(img_u8)).float().div(255).sub(MEAN).div(STD)[None]


def to_tensor(img_small: np.ndarray) -> torch.Tensor:
    """uint8 HxW -> 1x1xHxW tensor, for inference."""
    return normalize(img_small)[None]


def soft_target(samples_of_file: list[Sample], size: int) -> np.ndarray:
    """Mean over annotator sets of the INTER_AREA-downscaled filament mask, as uint8 (x255)."""
    acc = np.zeros((size, size), np.float32)
    for s in samples_of_file:
        m = semantic_mask(s).astype(np.float32)
        acc += cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
    acc /= len(samples_of_file)
    return np.rint(acc * 255).astype(np.uint8)


class FilamentDataset(Dataset):
    """Items: (normalised image 1xSxS, soft target 1xSxS in [0,1]). One item per image file."""

    def __init__(self, samples: list[Sample], image_dir: Path, size: int = 1024, augment: bool = False):
        self.size, self.augment = size, augment
        by_file: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_file[s.file_name].append(s)
        self.files = sorted(by_file)
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for f in self.files:
            img = cv2.resize(read_gray(image_dir / f), (size, size), interpolation=cv2.INTER_AREA)
            self.items.append((img, soft_target(by_file[f], size)))

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, img: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(4)                      # the 8 symmetries of the square
        img, m = np.rot90(img, k), np.rot90(m, k)
        if np.random.rand() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        f = img.astype(np.float32) / 255.0
        gain = np.random.uniform(*GAIN_RANGE)
        offset = np.random.uniform(*OFFSET_RANGE)
        gamma = np.random.uniform(*GAMMA_RANGE)
        # keep the off-disk background exactly black, as it is in every real image
        f = np.where(f > 0, np.clip(f * gain + offset, 0, 1) ** gamma, 0.0)
        return (f * 255).astype(np.uint8), m

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, m = self.items[i]
        if self.augment:
            img, m = self._augment(img, m)
        x = normalize(img)
        y = torch.from_numpy(np.ascontiguousarray(m)).float().div(255)[None]
        return x, y
