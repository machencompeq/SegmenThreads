"""U-Net (segmentation_models_pytorch) predicting a 1-channel 'filament' probability map,
the training loss, and weight save/load helpers."""
from __future__ import annotations

import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,      # images are grayscale; smp sums the RGB weights of the first conv
        classes=1,
    )


class BceDiceLoss(nn.Module):
    """BCE (accepts soft targets) + soft Dice against the heavy class imbalance (~0.4 % foreground).

    The logits are cast to fp32 first so the loss is safe even if called outside autocast
    (a 1024x1024 sum overflows fp16).
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.w = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return self.w * bce + (1 - self.w) * dice.mean()


def save_weights(model: nn.Module, path: Path, encoder: str, size: int, half: bool = True) -> None:
    """Save the state_dict (fp16 by default: 47 MB instead of 93 MB, fits in a git repo)
    plus a sidecar JSON with what is needed to rebuild the network."""
    sd = model.state_dict()
    if half:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, path)
    with Path(str(path) + ".json").open("w", encoding="utf-8") as fh:
        json.dump({"encoder": encoder, "size": size, "in_channels": 1, "classes": 1, "half": half}, fh)


def load_weights(path: Path, device: str) -> tuple[nn.Module, dict]:
    """Rebuild the model from the sidecar JSON and load the (possibly fp16) state_dict."""
    with Path(str(path) + ".json").open(encoding="utf-8") as fh:
        meta = json.load(fh)
    model = build_model(meta["encoder"], pretrained=False)
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return model.to(device).eval(), meta
