"""Unit tests for the filament pipeline (RLE, PQ, post-processing, TTA, data split,
soft targets, submission file, weight I/O, loss).

Run with pytest, or without it:  python -m tests.test_core   (from the project root)

Data-free tests build synthetic masks and finish in a few seconds. Tests that need the
MAGFiLO dataset are skipped when the `data/` folder is missing.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from filament.data import (IMG_SIZE, Sample, find_dataset_root, instance_masks, load_samples,
                           polygon_to_mask, semantic_mask, split_by_file)
from filament.dataset import normalize, soft_target
from filament.inference import _d4_transforms, predict_prob
from filament.metric import gt_crops, pq_from_stats, pq_stats, pq_stats_labels
from filament.model import BceDiceLoss, build_model, load_weights, save_weights
from filament.postprocess import labels_to_instances, prob_to_instances, prob_to_labels
from filament.rle import decode_counts, encode_counts
from filament.submission import rows_for_image, verify_submission, write_submission

PROJECT = Path(__file__).resolve().parent.parent
N_REAL_SETS = 1154
N_REAL_POLYGONS = 8199


# ----------------------------------------------------------------------------- helpers
def _data_root() -> Path | None:
    """The MAGFiLO_1.0_Kaggle_2026 folder under the project's data/, or None if absent."""
    try:
        return find_dataset_root(PROJECT / "data")
    except FileNotFoundError:
        return None


def _skip(msg: str) -> None:
    """pytest.skip when pytest is available; otherwise the caller just returns."""
    try:
        import pytest
    except ImportError:
        return
    pytest.skip(msg)


def _rect(y0: int, y1: int, x0: int, x1: int, h: int = IMG_SIZE, w: int = IMG_SIZE) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _random_mask(seed: int, p: float = 0.3) -> np.ndarray:
    return (np.random.default_rng(seed).random((IMG_SIZE, IMG_SIZE)) < p).astype(np.uint8)


def _same_stats(a: dict, b: dict) -> bool:
    return (a["tp"], a["fp"], a["fn"]) == (b["tp"], b["fp"], b["fn"]) and abs(a["iou_sum"] - b["iou_sum"]) < 1e-6


def _twelve_rects() -> list[np.ndarray]:
    return [_rect(100 + 150 * i, 180 + 150 * i, 50 + 120 * i, 150 + 120 * i) for i in range(12)]


# ----------------------------------------------------------------------------- 1. RLE
def test_rle_round_trip_random_masks():
    for seed in range(3):
        m = _random_mask(seed, p=0.2 + 0.2 * seed)
        c = encode_counts(m)
        assert '"' not in c and "'" not in c and "," not in c
        assert np.array_equal(decode_counts(c), m)
    # an all-zero and an all-one mask
    for v in (0, 1):
        m = np.full((IMG_SIZE, IMG_SIZE), v, np.uint8)
        assert np.array_equal(decode_counts(encode_counts(m)), m)


def test_rle_round_trip_with_backslash():
    found = None
    for seed in range(200):
        m = _random_mask(seed)
        c = encode_counts(m)
        if "\\" in c:
            found = (m, c)
            break
    assert found is not None, "no seed produced a counts string with a backslash"
    m, c = found
    assert np.array_equal(decode_counts(c), m)


def test_rle_wrong_shape_raises():
    try:
        encode_counts(np.zeros((1024, 1024), np.uint8))
    except ValueError:
        return
    raise AssertionError("encode_counts accepted a 1024x1024 mask")


# ----------------------------------------------------------------------------- 2. PQ
def test_pq_identical_sets_is_one():
    gts = _twelve_rects()
    r = pq_from_stats([pq_stats(gts, gts)])
    assert abs(r["PQ"] - 1.0) < 1e-9 and r["TP"] == 12 and r["FP"] == 0 and r["FN"] == 0


def test_pq_empty_prediction_is_zero():
    gts = _twelve_rects()
    s = pq_stats(gts, [])
    assert s == {"tp": 0, "fp": 0, "fn": 12, "iou_sum": 0.0}
    assert pq_from_stats([s])["PQ"] == 0.0


def test_pq_drop_one_of_twelve():
    gts = _twelve_rects()
    r = pq_from_stats([pq_stats(gts, gts[1:])])
    assert r["TP"] == 11 and r["FP"] == 0 and r["FN"] == 1
    assert abs(r["PQ"] - 11 / 11.5) < 1e-9
    assert abs(r["PQ"] - 0.9565) < 1e-4


def test_pq_iou_exactly_half_is_not_tp():
    gt = _rect(100, 110, 100, 110)          # area 100
    pred = _rect(100, 110, 100, 105)        # half of it: inter 50, union 100 -> IoU 0.5
    s = pq_stats([gt], [pred])
    assert s == {"tp": 0, "fp": 1, "fn": 1, "iou_sum": 0.0}
    # one pixel more and it becomes a TP
    pred2 = _rect(100, 110, 100, 106)
    s2 = pq_stats([gt], [pred2])
    assert s2["tp"] == 1 and s2["fp"] == 0 and s2["fn"] == 0 and abs(s2["iou_sum"] - 0.6) < 1e-9


def test_pq_overlapping_predictions():
    gt = _rect(100, 200, 100, 200)
    good = gt.copy()
    partial = _rect(150, 260, 150, 260)     # overlaps gt (IoU < 0.5) and overlaps `good`
    s = pq_stats([gt], [good, partial])
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0 and abs(s["iou_sum"] - 1.0) < 1e-9
    # a prediction far away is a plain FP and a second GT without prediction is a FN
    far = _rect(1500, 1600, 1500, 1600)
    gt2 = _rect(900, 950, 900, 950)
    s = pq_stats([gt, gt2], [good, partial, far])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1)


# ----------------------------------------------------------------------------- 3. fast PQ
def _touching_components_prob() -> tuple[list[np.ndarray], np.ndarray]:
    """GT rectangles, several of them touching (8-connected), plus tiny blobs."""
    gts = [
        _rect(100, 200, 100, 300),
        _rect(200, 260, 300, 420),      # touches the first one at a corner
        _rect(100, 160, 300, 400),      # touches the first one along an edge
        _rect(500, 560, 100, 400),
        _rect(566, 620, 100, 400),      # 6 px gap: merged by link_px=6, not by 0
        _rect(800, 900, 800, 900),
        _rect(1000, 1004, 1000, 1004),  # 16 px: dropped by min_area 50
        _rect(1200, 1205, 1200, 1206),  # 30 px: dropped by min_area 50
        _rect(1500, 1700, 1500, 1520),
    ]
    prob = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    for g in gts:
        prob = np.maximum(prob, g.astype(np.float32))
    return gts, prob


def test_pq_stats_labels_matches_pq_stats_synthetic():
    gts, prob = _touching_components_prob()
    crops = gt_crops(gts)
    seen = set()
    for link_px in (0, 6):
        for min_area in (1, 50):
            ref = pq_stats(gts, prob_to_instances(prob, 0.5, min_area, link_px))
            fast = pq_stats_labels(crops, prob_to_labels(prob, 0.5, min_area, link_px))
            assert _same_stats(ref, fast), (link_px, min_area, ref, fast)
            seen.add((ref["tp"], ref["fp"], ref["fn"]))
    assert len(seen) > 1, "settings should produce different statistics"


def test_pq_stats_labels_empty_gt_mask_counts_as_fn():
    gts = [_rect(100, 200, 100, 200), np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)]
    labels = prob_to_labels(gts[0].astype(np.float32), 0.5, 1, 0)
    fast = pq_stats_labels(gt_crops(gts), labels)
    assert (fast["tp"], fast["fp"], fast["fn"]) == (1, 0, 1)


# ----------------------------------------------------------------------------- 4. postprocess
def _two_l_shapes(h: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Two L-shapes whose bounding boxes coincide but whose pixels are 10 px apart."""
    a = np.zeros((h, h), np.float32)
    a[10:100, 10:20] = 1          # vertical bar
    a[90:100, 10:100] = 1         # horizontal bar at the bottom
    b = np.zeros((h, h), np.float32)
    b[10:20, 30:100] = 1          # horizontal bar at the top
    b[10:80, 90:100] = 1          # vertical bar at the right
    assert not np.any(np.logical_and(a > 0, b > 0))
    return a, b


def test_prob_to_labels_overlapping_bboxes_disjoint_pixels():
    a, b = _two_l_shapes()
    prob = np.maximum(a, b)
    labels = prob_to_labels(prob, 0.5, 1, 0)
    assert labels.max() == 2
    labels3 = prob_to_labels(prob, 0.5, 1, 3)   # 10 px gap is not bridged by radius 3
    assert labels3.max() == 2


def test_prob_to_labels_link_px_merges_four_px_gap():
    prob = np.zeros((256, 256), np.float32)
    prob[50:100, 50:100] = 1
    prob[104:150, 50:100] = 1       # 4 px gap
    assert prob_to_labels(prob, 0.5, 1, 0).max() == 2
    assert prob_to_labels(prob, 0.5, 1, 1).max() == 2   # radius 1 closes only 2 px
    merged = prob_to_labels(prob, 0.5, 1, 3)
    assert merged.max() == 1
    # the merged label is cut back to the undilated pixels
    assert int((merged > 0).sum()) == int((prob >= 0.5).sum())


def test_prob_to_labels_min_area_and_contiguity():
    prob = np.zeros((256, 256), np.float32)
    prob[10:20, 10:20] = 0.9      # 100 px
    prob[50:53, 50:53] = 0.9      # 9 px
    prob[100:130, 100:130] = 0.9  # 900 px
    prob[200:202, 200:202] = 0.9  # 4 px
    prob[0:5, 200:256] = 0.2      # below threshold
    labels = prob_to_labels(prob, 0.5, 1, 0)
    assert labels.dtype == np.int32
    fg = prob >= 0.5
    assert labels.max() == 4
    assert np.array_equal(labels > 0, fg)
    labels = prob_to_labels(prob, 0.5, 50, 0)
    assert labels.max() == 2
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    areas = np.bincount(labels.ravel())[1:]
    assert sorted(areas.tolist()) == [100, 900]


def test_labels_to_instances_disjoint_and_cover_fg():
    gts, prob = _touching_components_prob()
    labels = prob_to_labels(prob, 0.5, 1, 0)
    inst = list(labels_to_instances(labels))
    assert len(inst) == labels.max()
    total = np.zeros_like(labels)
    for m in inst:
        assert m.dtype == np.uint8 and m.max() == 1
        total += m
    assert total.max() == 1                      # disjoint
    assert np.array_equal(total > 0, prob >= 0.5)  # cover the foreground exactly


def test_prob_to_labels_all_nan_gives_zero_labels():
    prob = np.full((256, 256), np.nan, np.float32)
    labels = prob_to_labels(prob, 0.5, 1, 0)
    assert labels.shape == prob.shape and labels.max() == 0
    assert list(labels_to_instances(labels)) == []
    assert prob_to_instances(prob) == []


# ----------------------------------------------------------------------------- 5. TTA
def test_d4_transforms_counts_and_inverses():
    x = torch.rand(1, 1, 8, 8)
    for full, n in ((False, 4), (True, 8)):
        pairs = list(_d4_transforms(x, full))
        assert len(pairs) == n
        for xi, back in pairs:
            assert xi.shape == x.shape
            assert torch.equal(back(xi), x)
        for i in range(n):
            for j in range(i + 1, n):
                assert not torch.equal(pairs[i][0], pairs[j][0]), (full, i, j)


def test_predict_prob_identity_module():
    model = nn.Identity()
    img = np.random.default_rng(0).integers(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for tta, full in ((False, False), (True, False), (True, True)):
        prob = predict_prob(model, img, size=128, device="cpu", tta=tta, tta_full=full)
        assert prob.shape == (IMG_SIZE, IMG_SIZE) and prob.dtype == np.float32
        assert prob.min() >= 0.0 and prob.max() <= 1.0
        assert prob.std() > 0.0


def test_normalize_shape_and_scale():
    img = np.zeros((16, 16), np.uint8)
    img[0, 0] = 255
    t = normalize(img)
    assert t.shape == (1, 16, 16) and t.dtype == torch.float32
    assert abs(float(t[0, 0, 0]) - 2.0) < 1e-6 and abs(float(t[0, 1, 1]) + 2.0) < 1e-6


# ----------------------------------------------------------------------------- 6. split
def _fake_samples(seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    samples = []
    sites = ["Lh", "Bh", "Th", "Uh", "Mh", "Ch"]
    for d in range(60):
        day = f"2015{1 + d // 28:02d}{1 + d % 28:02d}"
        for _ in range(rng.randint(1, 4)):
            fname = f"{day}{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}{rng.choice(sites)}.jpeg"
            for k in range(rng.randint(1, 3)):
                samples.append(Sample(set_id=f"0{k + 1}0101-{Path(fname).stem}", file_name=fname))
    return samples


def test_split_by_file_synthetic():
    samples = _fake_samples()
    train, val = split_by_file(samples, val_frac=0.15, seed=42)
    assert len(train) + len(val) == len(samples)
    tf = {s.file_name for s in train}
    vf = {s.file_name for s in val}
    assert not (tf & vf)
    assert not ({f[:8] for f in tf} & {f[:8] for f in vf})
    frac = len(vf) / len(tf | vf)
    assert 0.08 <= frac <= 0.25, frac
    # all annotator sets of a file are on the same side
    for s in val:
        assert s.file_name in vf
    # deterministic
    train2, val2 = split_by_file(samples, val_frac=0.15, seed=42)
    assert [s.set_id for s in val2] == [s.set_id for s in val]


# ----------------------------------------------------------------------------- 7. soft target
def test_soft_target_rectangle():
    poly = [100.0, 100.0, 300.0, 100.0, 300.0, 180.0, 100.0, 180.0]
    s = Sample(set_id="a", file_name="x.jpeg", polygons=[poly])
    one = soft_target([s], 1024)
    two = soft_target([s, s], 1024)
    assert one.dtype == np.uint8 and one.shape == (1024, 1024)
    assert np.array_equal(one, two)
    assert one.max() == 255 and one.min() == 0
    # the interior of the rectangle is fully covered at half resolution
    assert one[60:85, 60:140].min() == 255


def test_soft_target_thin_line_survives_downscale():
    # a 1-px-wide horizontal line at 2048
    poly = [100.0, 500.0, 900.0, 500.0, 900.0, 501.0, 100.0, 501.0]
    m = polygon_to_mask(poly)
    rows = np.unique(np.nonzero(m)[0])
    assert m.sum() > 0 and len(rows) == 1
    s = Sample(set_id="a", file_name="x.jpeg", polygons=[poly])
    t = soft_target([s], 1024)
    assert t.max() > 0
    assert t.max() < 255           # soft: the line covers a fraction of the 2x2 cell


# ----------------------------------------------------------------------------- 8. submission
def test_submission_round_trip_and_checks():
    masks = [_random_mask(s) for s in range(3)]
    counts = [encode_counts(m) for m in masks]
    assert any("\\" in c for c in counts)
    rows = rows_for_image("20160920230134Lh", masks)
    assert [r[0] for r in rows] == ["20160920230134Lh_1", "20160920230134Lh_2", "20160920230134Lh_3"]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub.csv"
        write_submission(rows, path)
        sub = verify_submission(path, rows, {"20160920230134Lh"})
        assert len(sub) == 3
        for (_, c), m in zip(rows, masks):
            assert np.array_equal(decode_counts(c), m)
        # duplicated filament_id
        dup = [rows[0], rows[0]]
        write_submission(dup, path)
        try:
            verify_submission(path, dup, {"20160920230134Lh"})
            raise AssertionError("duplicate filament_id not detected")
        except RuntimeError:
            pass
        # unknown stem
        write_submission(rows, path)
        try:
            verify_submission(path, rows, {"somethingelse"})
            raise AssertionError("unknown stem not detected")
        except RuntimeError:
            pass
        # a row with a forbidden character is refused before writing
        try:
            write_submission([("a_1", "1,2")], path)
            raise AssertionError("comma in counts not detected")
        except ValueError:
            pass


# ----------------------------------------------------------------------------- 9. weights
def test_save_load_weights_round_trip():
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        ref = model(x)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "w.pt"
        save_weights(model, path, encoder="resnet18", size=64, half=True)
        meta = json.loads((Path(td) / "w.pt.json").read_text(encoding="utf-8"))
        assert meta["encoder"] == "resnet18" and meta["size"] == 64 and meta["half"] is True
        sd = torch.load(path, map_location="cpu")
        assert all(v.dtype == torch.float16 for v in sd.values() if v.is_floating_point())
        loaded, meta2 = load_weights(path, "cpu")
        assert meta2 == meta
        with torch.no_grad():
            out = loaded(x)
    assert out.shape == ref.shape == (1, 1, 64, 64)
    assert torch.isfinite(out).all()
    assert (out - ref).abs().max().item() < 1e-2


# ----------------------------------------------------------------------------- 10. loss
def test_bce_dice_loss_soft_targets_and_fp16():
    loss_fn = BceDiceLoss(0.5)
    logits = torch.randn(2, 1, 32, 32)
    target = torch.full((2, 1, 32, 32), 0.5)
    loss = loss_fn(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # perfect prediction has a lower loss than a random one
    hard = (torch.rand(2, 1, 32, 32) > 0.5).float()
    good = loss_fn(hard * 20 - 10, hard)
    bad = loss_fn(torch.randn(2, 1, 32, 32), hard)
    assert good < bad
    # fp16 logits of a large map: the loss casts to fp32, so no overflow
    big = (torch.randn(1, 1, 512, 512) * 4).half()
    tgt = (torch.rand(1, 1, 512, 512) > 0.9).float()
    l2 = loss_fn(big, tgt)
    assert l2.dtype == torch.float32 and torch.isfinite(l2)


# ----------------------------------------------------------------------------- data tests
def test_data_rle_round_trip_real_masks():
    root = _data_root()
    if root is None:
        _skip("dataset not found"); return
    samples = load_samples(root)
    for s in samples[:3]:
        for m in instance_masks(s)[:3]:
            c = encode_counts(m)
            assert '"' not in c and "'" not in c
            assert np.array_equal(decode_counts(c), m)


def test_data_load_samples_counts():
    root = _data_root()
    if root is None:
        _skip("dataset not found"); return
    samples = load_samples(root)
    assert len(samples) == N_REAL_SETS
    assert sum(len(s.polygons) for s in samples) == N_REAL_POLYGONS
    assert all(len(s.polygons) >= 1 for s in samples)
    assert len({s.set_id for s in samples}) == N_REAL_SETS


def test_data_fast_pq_matches_reference():
    root = _data_root()
    if root is None:
        _skip("dataset not found"); return
    samples = load_samples(root)
    rng = random.Random(0)
    for s in rng.sample(samples, 10):
        gts = instance_masks(s)
        prob = semantic_mask(s).astype(np.float32)
        crops = gt_crops(gts)
        for link_px in (0, 6):
            ref = pq_stats(gts, prob_to_instances(prob, 0.5, 200, link_px))
            fast = pq_stats_labels(crops, prob_to_labels(prob, 0.5, 200, link_px))
            assert _same_stats(ref, fast), (s.set_id, link_px, ref, fast)


def test_data_split_by_file_no_shared_days():
    root = _data_root()
    if root is None:
        _skip("dataset not found"); return
    samples = load_samples(root)
    train, val = split_by_file(samples)
    assert len(train) + len(val) == len(samples)
    tf = {s.file_name for s in train}
    vf = {s.file_name for s in val}
    assert not (tf & vf)
    assert len({f[:8] for f in tf} & {f[:8] for f in vf}) == 0
    assert 0.10 <= len(vf) / len(tf | vf) <= 0.20


if __name__ == "__main__":
    import sys; failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print("ok  ", name)
            except Exception as e: failed += 1; print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
