"""Local re-implementation of Panoptic Quality (Kirillov et al., CVPR 2019) for one class.

PQ = sum_{TP} IoU / (|TP| + 0.5|FP| + 0.5|FN|), a pair is TP when IoU > 0.5
(IoU > 0.5 guarantees the matching is unique, as proven in the PQ paper).

Checked against the host's public self-evaluation notebook (Kaggle, azimahmadzadeh): same
strict IoU > 0.5 rule, same FP/FN counting, same dataset-level aggregation; identical
TP/FP/FN/PQ on real annotator-vs-annotator data. The host additionally applies unspecified
"verification strategies" for edge cases, so treat the number as a very close estimate.
"""
from __future__ import annotations

import numpy as np


def _bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _iou_matrix(gts: list[np.ndarray], preds: list[np.ndarray]) -> np.ndarray:
    """IoU for every (gt, pred) pair; only pairs whose bounding boxes overlap are computed."""
    iou = np.zeros((len(gts), len(preds)))
    gb = [_bbox(m) for m in gts]
    pb = [_bbox(m) for m in preds]
    g_area = [int(m.sum()) for m in gts]
    p_area = [int(m.sum()) for m in preds]
    for i, b1 in enumerate(gb):
        if b1 is None:
            continue
        for j, b2 in enumerate(pb):
            if b2 is None:
                continue
            y0, y1 = max(b1[0], b2[0]), min(b1[1], b2[1])
            x0, x1 = max(b1[2], b2[2]), min(b1[3], b2[3])
            if y0 >= y1 or x0 >= x1:
                continue
            inter = int(np.logical_and(gts[i][y0:y1, x0:x1], preds[j][y0:y1, x0:x1]).sum())
            if inter:
                iou[i, j] = inter / (g_area[i] + p_area[j] - inter)
    return iou


def pq_stats(gts: list[np.ndarray], preds: list[np.ndarray]) -> dict:
    """Return TP/FP/FN counts and summed TP IoU for one image."""
    iou = _iou_matrix(gts, preds)
    tp_mask = iou > 0.5
    tp = int(tp_mask.sum())
    return {
        "tp": tp,
        "fp": len(preds) - int(tp_mask.any(0).sum()),
        "fn": len(gts) - int(tp_mask.any(1).sum()),
        "iou_sum": float(iou[tp_mask].sum()),
    }


def gt_crops(gts: list[np.ndarray]) -> list[tuple]:
    """Precompute (y0, y1, x0, x1, bool crop, area) per GT mask; reuse across many evaluations."""
    out = []
    for m in gts:
        b = _bbox(m)
        if b is None:  # empty GT mask: keep it so it still counts as a FN (host behaviour)
            out.append((0, 0, 0, 0, np.zeros((0, 0), bool), 0))
            continue
        y0, y1, x0, x1 = b
        crop = m[y0:y1, x0:x1].astype(bool)
        out.append((y0, y1, x0, x1, crop, int(crop.sum())))
    return out


def pq_stats_labels(gt: list[tuple], labels: np.ndarray) -> dict:
    """Fast PQ stats when predictions are a label map (0 = background, 1..K = disjoint instances).

    Gives the same result as pq_stats() for disjoint predictions (checked in tests).
    """
    k = int(labels.max())
    p_area = np.bincount(labels.ravel(), minlength=k + 1)
    matched_p = set(); tp = 0; iou_sum = 0.0
    for y0, y1, x0, x1, crop, g_area in gt:
        if g_area == 0:
            continue
        inter = np.bincount(labels[y0:y1, x0:x1][crop], minlength=k + 1)
        inter[0] = 0
        j = int(inter.argmax())          # IoU > 0.5 can only happen for the largest overlap
        if inter[j] == 0:
            continue
        iou = inter[j] / (g_area + p_area[j] - inter[j])
        if iou > 0.5:
            tp += 1; iou_sum += float(iou); matched_p.add(j)
    return {"tp": tp, "fp": k - len(matched_p), "fn": len(gt) - tp, "iou_sum": iou_sum}


def pq_from_stats(stats: list[dict]) -> dict:
    """Aggregate over the dataset (the leaderboard definition) plus diagnostics.

    `PQ` is what the host computes. `PQ_per_image_mean`, `SQ`, `RQ` are diagnostics only
    (an image with no GT and no prediction counts as 1.0 in the per-image mean).
    """
    tp = sum(s["tp"] for s in stats); fp = sum(s["fp"] for s in stats); fn = sum(s["fn"] for s in stats)
    iou_sum = sum(s["iou_sum"] for s in stats)
    denom = tp + 0.5 * fp + 0.5 * fn
    per_img = [s["iou_sum"] / d if (d := s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]) > 0 else 1.0 for s in stats]
    return {
        "PQ": iou_sum / denom if denom else 0.0,
        "PQ_per_image_mean": float(np.mean(per_img)) if per_img else 0.0,
        "SQ": iou_sum / tp if tp else 0.0,
        "RQ": tp / denom if denom else 0.0,
        "TP": tp, "FP": fp, "FN": fn,
    }
