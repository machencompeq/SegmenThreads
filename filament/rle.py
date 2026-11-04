"""Submission encoding: COCO compressed RLE *counts* string for a 2048x2048 binary mask.

The host decodes with pycocotools using size [2048, 2048]; we encode with the same library
in Fortran order, so a mask survives the round trip exactly (checked in tests).
"""
import numpy as np
from pycocotools import mask as mask_utils

from .data import IMG_SIZE

SIZE = (IMG_SIZE, IMG_SIZE)


def encode_counts(mask: np.ndarray) -> str:
    """Binary HxW mask -> COCO compressed RLE counts (ASCII 48..111; may contain '\')."""
    if mask.shape != SIZE:
        raise ValueError(f"mask must be {SIZE}, got {mask.shape}")
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def decode_counts(counts: str) -> np.ndarray:
    """Inverse of encode_counts: counts string -> uint8 HxW mask (the host's decode path)."""
    return mask_utils.decode({"size": list(SIZE), "counts": counts.encode("ascii")})
