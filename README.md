# SegmenThreads — Solar Filament Instance Segmentation

Entry for the Kaggle **Solar Filament Segmentation Challenge 2026** (IEEE BigData Cup 2026, slug
`filament-segmentation-2026`). The task is instance segmentation of dark filaments in 2048 x 2048
8-bit grayscale GONG H-alpha images; the metric is Panoptic Quality
(PQ = sum over TP of IoU / (TP + 0.5 FP + 0.5 FN), where a pair counts as TP only when IoU > 0.5).
This repository contains a single U-Net (ResNet34 encoder) trained at 1024 x 1024 on soft,
annotator-averaged targets, 8-fold test-time augmentation, bicubic upsampling back to 2048, and a
connected-components post-processing stage whose three knobs are tuned on a day-grouped validation
split with a local PQ implementation that reproduces the host's self-evaluation notebook. The code
is plain PyTorch plus `segmentation-models-pytorch`, runs end to end on a Kaggle T4 in about two
hours, and reproduces the submission from the saved weights without retraining.

## Results

| Quantity | Value | Note |
|---|---|---|
| Public leaderboard PQ | **TBD** | to be filled in after the first GPU run on Kaggle |
| Validation PQ (15 % of files, day-grouped, scored per annotator set) | **TBD** | to be filled in after the first GPU run; `train.py` prints it and stores it in `outputs/report.json` as `val_PQ` |
| Inter-annotator agreement PQ (reference) | 0.352 | one annotator scored against another on the same images; SQ 0.647, RQ 0.543, 60 files |

The inter-annotator number is a realistic ceiling for this metric rather than a target to beat:
the ground truth itself only agrees with a second human at PQ 0.35. The host has stated in the
competition announcements that any PQ above 0.35 is already valuable to them.

## Pipeline

```
2048x2048 grayscale JPEG
        |  cv2.INTER_AREA downscale
        v
1024x1024 image  ->  U-Net (ResNet34 encoder, ImageNet init, 1-channel input)  ->  sigmoid
        |  8 symmetries of the square (D4 test-time augmentation), averaged
        v
1024x1024 probability map
        |  bicubic upsample
        v
2048x2048 probability map
        |  threshold thr  ->  (optional dilation by link_px to decide connectivity)  ->  8-connected components
        |  drop components with area < min_area
        v
label map 0..K   ->  one COCO compressed-RLE row per instance  ->  submission.csv
```

1. **Input.** Images are read as single-channel grayscale (the Data page stresses that they must not
   be treated as RGB) and downscaled 2048 -> 1024 with `INTER_AREA`.
2. **Model.** `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)`.
   The library sums the RGB filters of the first convolution to build the 1-channel stem.
3. **Target.** One training item per *file*. The target is the mean, over that file's annotator
   sets, of the `INTER_AREA`-downscaled binary mask, kept **soft** (an area fraction per pixel, not
   binarised).
4. **Loss and optimiser.** BCE (which accepts soft targets) + soft Dice, weights 0.5 / 0.5; AdamW,
   learning rate 3e-4, weight decay 1e-4, OneCycle schedule, 40 epochs, batch size 4, mixed
   precision on GPU.
5. **Augmentation.** D4 geometric (rot90 and flips) plus photometric jitter (gain, offset, gamma) to
   cover the exposure differences between GONG sites.
6. **Inference.** Sigmoid output averaged over the 8 D4 symmetries, upsampled to 2048 with bicubic
   interpolation (`--no-tta8` falls back to the 4 flips).
7. **Post-processing.** Threshold, connected components (optionally computed on a dilated mask so
   that pieces up to `2*link_px` apart form one instance, then cut back to the undilated pixels),
   drop small components. The triple (`thr`, `min_area`, `link_px`) is grid-searched on the
   validation split.
8. **Output.** Each instance is encoded as COCO compressed RLE counts with `pycocotools` (Fortran
   order, size 2048 x 2048) and written as `<image_stem>_<k>,<counts>`. The CSV is read back with
   pandas defaults and the decoded masks are compared with the originals before the run finishes.

## Design decisions (measured)

All numbers below were measured on the 2026 Kaggle release of MAGFiLO in this project.

- **Data shape.** 707 training files, 180 test files, 1154 annotator sets (411 files annotated once,
  145 twice, 151 three times), 8199 filament polygons, 7.1 filaments per set on average (max 26).
  Filaments cover 0.37 % of the pixels.
- **Filaments are thin.** Width at the widest point (2 x max distance transform): median 15.6 px at 2048,
  5th percentile 8.0 px, i.e. 4 px at the 1024 network resolution. Ground-truth instance area
  percentiles: p1 = 209 px, p5 = 317 px, p50 = 1228 px (`scripts/measure_filaments.py`).
- **Why 1024 and soft targets.** Round-tripping the ground truth through the resize pipeline (no
  model) gives a PQ ceiling of 0.90 when masks are binarised at 1024, and 0.99 when the downscaled
  target is kept as an area fraction and the prediction is upsampled bicubically. Soft targets
  therefore cost almost nothing in attainable PQ while letting the network run at a quarter of the
  pixels.
- **Why the mean over annotators.** Up to three annotators mark the same image and they agree only
  moderately with each other (inter-annotator PQ 0.352). Averaging their masks has the same optimum
  as feeding the sets separately, with less gradient noise and without weighting multi-annotated
  files 2-3x.
- **Why `link_px = 0` is in the grid and expected to win.** Ground-truth filaments never touch, and
  any `link_px > 0` lowers the oracle PQ by merging neighbours. Linking is kept as an option in case
  the model fragments filaments; the grid decides.
- **Why `min_area` stops at 300.** The smallest 1 % of GT instances are about 209 px; values above
  roughly 300 px start removing real filaments. The grid is `thr in {0.3, 0.4, 0.5, 0.6}` x
  `min_area in {50, 100, 150, 200, 300}` x `link_px in {0, 3, 6}` (4 x 5 x 3 = 60 settings).
- **Checkpoint selection.** Validation PQ is evaluated at four thresholds (0.3, 0.4, 0.5, 0.6) and
  the best one ranks the checkpoint, because soft targets shift the calibration away from 0.5.
- **Metric fidelity.** `filament/metric.py` was checked line by line against the host's public
  self-evaluation notebook and gives identical TP / FP / FN / PQ on real annotator-vs-annotator data.
- **Validation split.** 15 % of files are held out, grouped by observation day (`YYYYMMDD` prefix),
  so that same-day images from different GONG sites never straddle the split; two near-identical
  same-minute frames (image correlation 0.97) leaked before this grouping was added. Scoring is per
  annotator set, exactly as the host groups the ground truth.

## Repository layout

| Path | Contents |
|---|---|
| `filament/data.py` | Loads the MAGFiLO COCO JSON (one `Sample` per annotator set), rasterises polygons with `pycocotools`, day-grouped train/val split, image reading. |
| `filament/dataset.py` | `FilamentDataset`: RAM-cached downscaled images, soft annotator-averaged targets, D4 + photometric augmentation, normalisation shared with inference. |
| `filament/model.py` | `build_model` (smp U-Net, ResNet34, 1-channel), `BceDiceLoss`, `save_weights` / `load_weights` (fp16 state_dict + JSON sidecar). |
| `filament/inference.py` | `predict_prob`: downscale, D4 test-time augmentation, bicubic upsample to 2048. |
| `filament/postprocess.py` | `PostprocessParams`, `DEFAULT_GRID`, `prob_to_labels` (threshold, optional linking, connected components, min-area filter), `labels_to_instances`. |
| `filament/metric.py` | Panoptic Quality for one class: per-image TP/FP/FN/IoU statistics, fast label-map variant, dataset-level aggregation plus SQ/RQ diagnostics. |
| `filament/rle.py` | COCO compressed RLE counts encode/decode for 2048 x 2048 masks (the host's convention). |
| `filament/submission.py` | Writes `submission.csv` without CSV quoting and verifies it the way the host reads it. |
| `train.py` | End-to-end CLI: train, select checkpoint, tune post-processing, predict the test set, write all outputs; `--smoke` and `--predict-only` modes. |
| `build_notebook.py` | Bundles `filament/` and `train.py` into `kaggle_baseline.ipynb` as `%%writefile` cells so the repository stays the single source of truth. |
| `kaggle_baseline.ipynb` | The generated Kaggle notebook (GPU + Internet On); `MODE = 'train'` or `'predict'`. |
| `scripts/` | Analysis helpers: `visualize.py` (prediction / GT overlays on the H-alpha disk), `annotator_agreement.py` (inter-annotator PQ), `measure_filaments.py` (width, area and count statistics quoted above). |
| `tests/` | `test_core.py`: unit tests for the RLE round trip, metric agreement, post-processing and the submission format. |
| `weights/` | Saved model weights `unet_r34_1024.pt` with its `.json` sidecar and the tuned `postprocess.json` (copied from `outputs/` after the final Kaggle run). |
| `docs/` | Supporting documentation for the technical report. |
| `kaggle_pages/` | Plain-text snapshots of the competition Overview / Data / Rules pages (fetched with `fetch_kaggle.py`). |
| `viz/` | Example overlay images produced by the visualisation script. |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, matplotlib, ruff, playwright
```

A CUDA build of PyTorch is recommended for training; the pinned CPU-only versions in
`requirements.txt` are enough for the smoke test, the unit tests and (slowly) for `--predict-only`.

**Data.** The dataset is *not* part of this repository and must not be redistributed: it is
released by the MAGFiLO authors under **CC BY-NC 4.0** and the competition rules restrict it to
non-commercial use. Accept the competition rules on Kaggle, download the archive from the Data page
and unpack it under `data/` so that the layout is

```
data/
  MAGFiLO_1.0_Kaggle_2026/
    train/
      train_images/*.jpeg
      MAGFiLO_1.0_Annotations_kaggle2026_train.json
    test/
      test_images/*.jpeg
```

`train.py` locates the annotation JSON by searching under `data/` (or `/kaggle/input` on Kaggle),
so one extra nesting level such as `data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/` also
works. `data/` is git-ignored.

## Run

```bash
# Full training run (GPU; about 2 h on a Kaggle T4). Writes outputs/ (see below).
python train.py

# CPU smoke test: 256 px, 1 epoch, 6 train files, 3 val files, 3 test images, no pretrained weights.
python train.py --smoke

# Inference only, from saved weights and tuned post-processing (no training).
python train.py --predict-only --weights weights/unet_r34_1024.pt --postprocess weights/postprocess.json

# Unit tests
python -m tests.test_core
# or
pytest

# Analysis scripts (each has a module docstring / --help)
python scripts/visualize.py
python scripts/annotator_agreement.py
python scripts/measure_filaments.py
```

Useful `train.py` options: `--data <dir>`, `--out <dir>` (default `outputs`), `--size` (default
1024), `--encoder` (default `resnet34`), `--epochs` (40), `--bs` (4), `--lr` (3e-4), `--val-frac`
(0.15), `--seed` (0), `--no-pretrained`, `--no-tta8`.

Outputs of a full run in `outputs/`:

| File | Contents |
|---|---|
| `best.pt` + `best.pt.json` | fp16 state_dict of the checkpoint with the best validation PQ (about 49 MB) and the sidecar needed to rebuild the network (encoder, input size, channels). |
| `last.pt` + `last.pt.json` | Weights after the final epoch. |
| `postprocess.json` | The tuned `thr` / `min_area` / `link_px`. |
| `report.json` | CLI arguments, environment, per-epoch training log, the full post-processing grid with validation PQ, and the selected `val_PQ`. |
| `environment.json` | Python, torch, numpy, opencv, pycocotools, segmentation-models-pytorch and timm versions plus the GPU name. |
| `submission.csv` | The competition file (`filament_id,segmentation_rle`). |

## Reproducing the submission from the saved weights

The files in `weights/` are the `best.pt`, `best.pt.json` and `postprocess.json` of the final Kaggle
run, renamed to `unet_r34_1024.pt`, `unet_r34_1024.pt.json` and `postprocess.json`. The `.json`
sidecar must sit next to the `.pt` file (the loader reads `<weights>.json`).

```bash
python train.py --predict-only --weights weights/unet_r34_1024.pt --postprocess weights/postprocess.json
```

This rebuilds the model from the sidecar, loads the fp16 weights, reports the validation PQ with
the saved parameters (a sanity check that the weights and split are the ones described here),
predicts all 180 test images with 8-fold TTA and writes `outputs/submission.csv`. No training is
performed. Inference is deterministic given the weights; on CPU it takes considerably longer than
on a GPU.

## Kaggle notebook

`kaggle_baseline.ipynb` is generated by `python build_notebook.py`. Each source file becomes a
`%%writefile` cell, followed by a call to `train.py`, so the notebook is self-contained and the
repository is the single source of truth; re-run `build_notebook.py` after editing any module.

- **Settings:** Accelerator = GPU (T4 or P100), Internet = On (needed for
  `pip install segmentation-models-pytorch pycocotools` and the ImageNet encoder weights).
- **`MODE = 'train'`** (first code cell): full pipeline, train -> tune post-processing ->
  `submission.csv`; about 2 h on a T4. Outputs land in `/kaggle/working/outputs/`.
- **`MODE = 'predict'`**: reproduce `submission.csv` from saved weights without training. Upload
  `best.pt`, `best.pt.json` and `postprocess.json` as a Kaggle dataset, attach it to the notebook and
  point `WEIGHTS_DIR` at it.

After the training run, copy `outputs/best.pt`, `outputs/best.pt.json` and `outputs/postprocess.json`
into `weights/` as described above, and replace the pins in `requirements.txt` with the versions
recorded in `outputs/environment.json`.

## Validation protocol

- `split_by_file` holds out about 15 % of the *files* (not annotator sets), choosing whole
  observation days (`YYYYMMDD` prefix of the file name) with a fixed seed until the target count is
  reached. All annotator sets of a file, and all same-day frames from any GONG site, land on the
  same side.
- Validation ground truth is scored **per annotator set**, exactly as the host's self-evaluation
  notebook groups it, so a file annotated three times contributes three sets.
- During training the model is evaluated every 5 epochs, every 2 epochs in the last 10, and at the
  end. Each evaluation computes PQ at thresholds 0.3 / 0.4 / 0.5 / 0.6 with the default `min_area`
  and `link_px` and keeps the checkpoint with the best PQ.
- After training, the best checkpoint's 8-fold TTA probabilities are cached once and the full
  60-point post-processing grid is scored; the best setting is written to `postprocess.json` and
  used for the test set.
- PQ is aggregated at the dataset level (sum of IoU over all TP divided by TP + 0.5 FP + 0.5 FN over
  all sets), which is the leaderboard definition. `SQ`, `RQ` and a per-image mean are reported as
  diagnostics only.

## Known limitations

- The network sees the Sun at 1024 x 1024. Soft targets and bicubic upsampling recover most of the
  lost detail (oracle PQ 0.99), but the finest barbs are below one network pixel.
- A single model, single seed, no ensembling beyond test-time augmentation.
- Instances are connected components of the thresholded map. Filaments that touch in the
  prediction are merged, and a filament split by a gap in the prediction becomes two instances;
  there is no learned instance separation.
- The validation PQ is an estimate of the leaderboard score: the split is 15 % of the training
  files, and the host applies additional, unspecified "verification strategies" for edge cases that
  the local metric does not model.
- Photometric augmentation ranges were set from the train + test disk statistics; images far
  outside that range (unusual exposure or seeing) are not specifically handled.

## Licence

- **Code**: MIT (see `LICENSE`), an OSI-approved licence that does not limit commercial use, as the
  competition rules require for shared and winning code.
- **Data**: the MAGFiLO images and annotations are distributed by the competition host under
  **CC BY-NC 4.0** and are not included in this repository. They may only be used for
  non-commercial purposes under the competition rules.
- **Pretrained weights**: the ResNet34 ImageNet initialisation is downloaded by
  `segmentation-models-pytorch` at training time; the fine-tuned weights in `weights/` are released
  under the MIT licence together with the code.

Please cite the dataset and the competition as the Data / Overview pages request:

- MAGFiLO: Manually Annotated GONG Filaments from H-Alpha Observations, *Scientific Data* (2024),
  doi:10.1038/s41597-024-03876-y.
- Azim Ahmadzadeh, Dustin J. Kempton, Qin Li, and Alexei A. Pevtsov. Solar Filament Segmentation
  Challenge 2026. https://www.kaggle.com/competitions/filament-segmentation-2026, 2026. Kaggle.

## Acknowledgements

- Competition organisers: Azim Ahmadzadeh (University of Missouri-St. Louis), Dustin J. Kempton
  (Georgia State University), Qin Li (New Jersey Institute of Technology) and Alexei A. Pevtsov
  (National Solar Observatory); sponsor: the NSF National Solar Observatory, operated by AURA.
- This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the
  National Solar Observatory, which is operated by the Association of Universities for Research in
  Astronomy (AURA), Inc. under a cooperative agreement with the National Science Foundation and with
  contribution from the National Oceanic and Atmospheric Administration. The GONG network of
  instruments is hosted by the Big Bear Solar Observatory, High Altitude Observatory, Learmonth
  Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofisica de Canarias, and Cerro
  Tololo Interamerican Observatory.
- Software: PyTorch, segmentation-models-pytorch and timm (U-Net and ResNet34 encoder),
  pycocotools (RLE and polygon rasterisation), OpenCV, NumPy, pandas.
- Metric definition: Kirillov et al., "Panoptic Segmentation", CVPR 2019
  (doi:10.1109/CVPR.2019.00963).
