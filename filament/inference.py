"""Model inference at reduced resolution with test-time augmentation, upsampled to 2048x2048."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .dataset import to_tensor


def _d4_transforms(x: torch.Tensor, full: bool):
    """Yield (transformed input, function that maps a prediction back).

    `full=False`: the 4 flip combinations. `full=True`: all 8 symmetries of the square
    (rotations + flips), matching the rot90/flip augmentation used in training.
    """
    rots = range(4) if full else (0,)
    for k in rots:
        r = torch.rot90(x, k, (2, 3)) if k else x
        yield r, (lambda p, k=k: torch.rot90(p, -k, (2, 3)) if k else p)
        yield torch.flip(r, [3]), (lambda p, k=k: torch.rot90(torch.flip(p, [3]), -k, (2, 3)) if k
                                   else torch.flip(p, [3]))
        if not full:  # the two remaining flips of the 4-flip set
            yield torch.flip(r, [2]), (lambda p: torch.flip(p, [2]))
            yield torch.flip(r, [2, 3]), (lambda p: torch.flip(p, [2, 3]))


@torch.no_grad()
def predict_prob(model: nn.Module, img_full: np.ndarray, size: int, device: str,
                 tta: bool = True, tta_full: bool = False) -> np.ndarray:
    """Return a float32 probability map at the input image's resolution (2048x2048).

    The network runs at `size` x `size`; the probability map is upsampled with bicubic
    interpolation, which preserves the sub-pixel information in the soft output
    (measured on GT round trips: PQ ceiling 0.991 bicubic vs 0.975 bilinear at 1024).
    """
    small = cv2.resize(img_full, (size, size), interpolation=cv2.INTER_AREA)
    x = to_tensor(small).to(device)
    acc, n = None, 0
    with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        for xi, back in (_d4_transforms(x, tta_full) if tta else [(x, lambda p: p)]):
            p = back(torch.sigmoid(model(xi).float()))
            acc = p if acc is None else acc + p
            n += 1
    prob = (acc / n)[0, 0].cpu().numpy().astype(np.float32)
    prob = cv2.resize(prob, (img_full.shape[1], img_full.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(prob, 0.0, 1.0)
