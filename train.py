"""Train the U-Net, tune post-processing on a held-out split by local PQ, write submission.csv.

Works locally and on Kaggle (paths are auto-detected).
  Kaggle GPU, full run:          python train.py
  Local CPU smoke test:          python train.py --smoke
  Reproduce from saved weights:  python train.py --predict-only --weights weights/unet_r34_1024.pt \
                                                 --postprocess weights/postprocess.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from filament.data import (find_dataset_root, instance_masks, load_samples, read_gray, split_by_file,
                           test_image_dir, train_image_dir)
from filament.dataset import FilamentDataset
from filament.inference import predict_prob
from filament.metric import gt_crops, pq_from_stats, pq_stats_labels
from filament.model import BceDiceLoss, build_model, load_weights, save_weights
from filament.postprocess import DEFAULT_GRID, PostprocessParams, labels_to_instances, prob_to_labels
from filament.submission import rows_for_image, verify_submission, write_submission

SELECT_THRS = (0.3, 0.4, 0.5, 0.6)   # thresholds tried when ranking checkpoints


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="folder that contains MAGFiLO_1.0_Kaggle_2026")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--size", type=int, default=1024, help="network input resolution")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-tta8", action="store_true", help="use 4 flips instead of 8 symmetries at the end")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline end to end")
    ap.add_argument("--predict-only", action="store_true", help="skip training; needs --weights and --postprocess")
    ap.add_argument("--weights", default=None, help="state_dict saved by save_weights (with .json sidecar)")
    ap.add_argument("--postprocess", default=None, help="postprocess.json written by a previous run")
    a = ap.parse_args()
    if a.smoke:
        a.size, a.epochs, a.bs, a.no_pretrained = 256, 1, 2, True
    if a.predict_only and not (a.weights and a.postprocess):
        ap.error("--predict-only needs --weights and --postprocess")
    return a


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_report() -> dict:
    """Versions actually used (copied into requirements.txt after the final Kaggle run)."""
    from importlib.metadata import PackageNotFoundError, version
    rep = {"python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    for pkg in ("torch", "torchvision", "numpy", "opencv-python", "pycocotools",
                "segmentation-models-pytorch", "timm", "pandas"):
        try:
            rep[pkg] = version(pkg)
        except PackageNotFoundError:
            rep[pkg] = "not installed"
    return rep


class Validator:
    """Caches the validation images and GT crops once; scores a model under given post-processing.

    GT is scored per annotator set (170 sets for 106 files), which is exactly how the host's
    self-evaluation notebook groups the ground truth.
    """

    def __init__(self, val_samples, img_dir: Path, size: int, device: str):
        self.size, self.device = size, device
        self.files = sorted({s.file_name for s in val_samples})
        self.imgs = {f: read_gray(img_dir / f) for f in self.files}
        self.gt = [(s.file_name, gt_crops(instance_masks(s))) for s in val_samples]

    def probs(self, model, tta_full: bool = False) -> dict:
        model.eval()
        return {f: predict_prob(model, self.imgs[f], self.size, self.device, tta=True, tta_full=tta_full)
                for f in self.files}

    def score(self, probs: dict, p: PostprocessParams) -> dict:
        labels = {f: prob_to_labels(probs[f], p.thr, p.min_area, p.link_px) for f in self.files}
        return pq_from_stats([pq_stats_labels(gt, labels[f]) for f, gt in self.gt])

    def best_threshold_score(self, probs: dict) -> dict:
        """Rank checkpoints by the best of a few thresholds, not a fixed 0.5 (soft targets
        shift the calibration), with the default min_area / link_px."""
        best = None
        for thr in SELECT_THRS:
            r = self.score(probs, PostprocessParams(thr=thr))
            if best is None or r["PQ"] > best["PQ"]:
                best = {**r, "thr": thr}
        return best


def train_model(a, train_s, val: Validator, img_dir: Path, out: Path, device: str) -> tuple[torch.nn.Module, list]:
    t0 = time.time()
    ds = FilamentDataset(train_s, img_dir, a.size, augment=True)
    print(f"cached {len(ds)} train files in {time.time() - t0:.0f}s", flush=True)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=2 if device == "cuda" else 0,
                    drop_last=True, pin_memory=device == "cuda",
                    worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2**32))

    model = build_model(a.encoder, pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = BceDiceLoss()

    best_pq, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), torch.zeros((), device=device)
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach()
        msg = {"epoch": ep + 1, "loss": round(tot.item() / max(len(dl), 1), 4), "sec": round(time.time() - t0)}
        last = ep == a.epochs - 1
        if a.smoke or (ep + 1) % 5 == 0 or last or (ep >= a.epochs - 10 and (ep + 1) % 2 == 0):
            r = val.best_threshold_score(val.probs(model))
            msg.update({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
            if r["PQ"] > best_pq:
                best_pq = r["PQ"]
                save_weights(model, out / "best.pt", a.encoder, a.size)
        log.append(msg)
        print(msg, flush=True)
    save_weights(model, out / "last.pt", a.encoder, a.size)
    model, _ = load_weights(out / "best.pt", device)
    return model, log


def tune_postprocess(val: Validator, probs: dict) -> list[dict]:
    """Grid search over the post-processing knobs on the cached validation probabilities."""
    results = []
    for thr, ma, lp in itertools.product(DEFAULT_GRID["thr"], DEFAULT_GRID["min_area"], DEFAULT_GRID["link_px"]):
        p = PostprocessParams(thr, ma, lp)
        results.append({**p.to_dict(), **val.score(probs, p)})
    results.sort(key=lambda r: -r["PQ"])
    return results


def predict_test(model, params: PostprocessParams, test_dir: Path, size: int, device: str,
                 tta_full: bool, limit: int | None = None) -> tuple[list, list, list]:
    files = sorted(p.name for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))[:limit]
    rows, empty = [], []
    for f in files:
        prob = predict_prob(model, read_gray(test_dir / f), size, device, tta=True, tta_full=tta_full)
        labels = prob_to_labels(prob, params.thr, params.min_area, params.link_px)
        if labels.max() == 0:
            empty.append(f)
        rows.extend(rows_for_image(Path(f).stem, labels_to_instances(labels)))   # encoded one at a time
    return files, rows, empty


def main() -> None:
    a = parse_args()
    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(a.data) if a.data else (Path("/kaggle/input") if Path("/kaggle/input").exists()
                                             else Path(__file__).parent / "data")
    root = find_dataset_root(data_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = environment_report()
    print(f"device={device} root={root}\nenv={env}", flush=True)
    with (out / "environment.json").open("w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=1)

    samples = load_samples(root)
    train_s, val_s = split_by_file(samples, a.val_frac)
    if a.smoke:
        train_s, val_s = train_s[:6], val_s[:3]
    print(f"train sets={len(train_s)} ({len({s.file_name for s in train_s})} files)  "
          f"val sets={len(val_s)} ({len({s.file_name for s in val_s})} files)", flush=True)
    val = Validator(val_s, train_image_dir(root), a.size, device)

    if a.predict_only:
        model, meta = load_weights(Path(a.weights), device)
        a.size = meta["size"]
        val.size = a.size
        with open(a.postprocess, encoding="utf-8") as fh:
            params = PostprocessParams.from_dict(json.load(fh))
        probs = val.probs(model, tta_full=not a.no_tta8)
        r = val.score(probs, params)
        print("val PQ with the saved weights/params:", {k: round(v, 4) if isinstance(v, float) else v
                                                        for k, v in r.items()}, flush=True)
        log, grid = [], []
    else:
        model, log = train_model(a, train_s, val, train_image_dir(root), out, device)
        probs = val.probs(model, tta_full=not a.no_tta8)
        grid = tune_postprocess(val, probs)
        params = PostprocessParams.from_dict(grid[0])
        print("top post-processing settings (val PQ):", flush=True)
        for r in grid[:5]:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        with (out / "postprocess.json").open("w", encoding="utf-8") as fh:
            json.dump(params.to_dict(), fh)
        with (out / "report.json").open("w", encoding="utf-8") as fh:
            json.dump({"args": vars(a), "environment": env, "train_log": log, "postprocess": params.to_dict(),
                       "val_PQ": grid[0]["PQ"], "postprocess_grid": grid}, fh, indent=1)

    files, rows, empty = predict_test(model, params, test_image_dir(root), a.size, device,
                                      tta_full=not a.no_tta8, limit=3 if a.smoke else None)
    write_submission(rows, out / "submission.csv")
    verify_submission(out / "submission.csv", rows, {Path(f).stem for f in files})
    print(f"submission: {len(rows)} filaments over {len(files)} test images; "
          f"images with no prediction: {len(empty)} {empty[:5]}  -> {out / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
