"""Loading MAGFiLO COCO annotations and rasterising them into masks.

Key facts (verified on the 2026 Kaggle release):
- 707 train JPEGs, 2048x2048, 8-bit grayscale.
- 1154 COCO "image" entries: one per (annotator set, file). A file can have up to 3 sets.
- Each annotation is a single polygon; rasterisation uses pycocotools, the same library the
  host uses for RLE, so our masks match the official ground-truth convention.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

IMG_SIZE = 2048
ANN_JSON = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


@dataclass
class Sample:
    """One annotator's annotation set for one image file."""
    set_id: str                    # e.g. "010401-20160920230134Lh"
    file_name: str                 # e.g. "20160920230134Lh.jpeg"
    polygons: list[list[float]] = field(default_factory=list)  # each [x0,y0,x1,y1,...]

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem


def find_dataset_root(start: str | Path) -> Path:
    """Return the MAGFiLO_1.0_Kaggle_2026 folder under `start` (works locally and on Kaggle)."""
    start = Path(start)
    direct = start / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026" / "train" / ANN_JSON
    hits = [direct] if direct.exists() else list(start.rglob(ANN_JSON))
    if not hits:
        raise FileNotFoundError(f"annotation json not found under {start}")
    return hits[0].parent.parent  # .../MAGFiLO_1.0_Kaggle_2026


def train_image_dir(root: Path) -> Path:
    return root / "train" / "train_images"


def test_image_dir(root: Path) -> Path:
    return root / "test" / "test_images"


def load_samples(root: Path) -> list[Sample]:
    """One Sample per COCO image entry (= one annotator's set for one file), polygons attached.

    Every annotation in the 2026 release is a single polygon (checked: 8199/8199, no
    iscrowd/RLE, none with < 3 points); the check below makes a future format change loud
    instead of silently dropping pieces.
    """
    with (root / "train" / ANN_JSON).open(encoding="utf-8") as fh:
        coco = json.load(fh)
    by_set: dict[str, Sample] = {
        im["id"]: Sample(set_id=im["id"], file_name=im["file_name"]) for im in coco["images"]
    }
    for a in coco["annotations"]:
        seg = a["segmentation"]
        if a.get("iscrowd", 0) or not isinstance(seg, list) or len(seg) != 1 or len(seg[0]) < 6:
            raise ValueError(f"unexpected segmentation format in annotation {a['id']}")
        by_set[a["image_id"]].polygons.append(seg[0])
    return list(by_set.values())


def split_by_file(samples: list[Sample], val_frac: float = 0.15,
                  seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    """Hold out ~val_frac of the *files*, grouped by observation date (YYYYMMDD).

    All annotator sets of a file land on the same side, and so do files taken the same
    day from different GONG sites (two val files used to have a same-minute twin in train
    with image correlation 0.97, i.e. a leaked duplicate).
    """
    files = sorted({s.file_name for s in samples})
    days = sorted({f[:8] for f in files})
    rng = random.Random(seed)
    rng.shuffle(days)
    target = int(round(len(files) * val_frac))
    val_files: set[str] = set()
    for d in days:
        if len(val_files) >= target:
            break
        val_files |= {f for f in files if f[:8] == d}
    train = [s for s in samples if s.file_name not in val_files]
    val = [s for s in samples if s.file_name in val_files]
    return train, val


def polygon_to_mask(poly: list[float], h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    """Rasterise one COCO polygon with pycocotools (matches the host's `area` field exactly)."""
    rles = mask_utils.frPyObjects([poly], h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def instance_masks(sample: Sample) -> list[np.ndarray]:
    """One uint8 2048x2048 mask per filament in the annotation set."""
    return [polygon_to_mask(p) for p in sample.polygons]


def semantic_mask(sample: Sample) -> np.ndarray:
    """Union of all filaments in one annotation set, uint8 {0,1}, 2048x2048."""
    m = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    for p in sample.polygons:
        m |= polygon_to_mask(p)
    return m


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img
