# Run on the lab PC (RTX 5060 Ti) — deadline 2026-11-15 06:00 UTC (Taiwan 14:00)

Everything needed is in this repo except the competition data, which may not be redistributed
(CC BY-NC 4.0), so it is not on GitHub. Expected time on a 5060 Ti: 40–60 min for the full run.

## 1. Code and environment

```bash
git clone https://github.com/machencompeq/SegmenThreads.git
cd SegmenThreads
python -m venv .venv
.venv\Scripts\activate                      # Linux: source .venv/bin/activate
# RTX 50 series (Blackwell, sm_120) needs a CUDA 12.8 build of PyTorch; older builds fail with
# "no kernel image is available for execution on the device".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt             # if it tries to replace torch with the CPU pin, rerun the line above
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name.

## 2. Data (from Kaggle, on the lab PC)

Either:

- Browser: log in to kaggle.com (account ANDYCHEN1225), open
  https://www.kaggle.com/competitions/filament-segmentation-2026/data , click **Download All**,
  unzip into `data/`; or
- CLI: Kaggle → Settings → API → Create New Token (`kaggle.json` into `%USERPROFILE%\.kaggle\`), then
  ```bash
  pip install kaggle
  kaggle competitions download -c filament-segmentation-2026 -p data
  cd data && tar -xf filament-segmentation-2026.zip && cd ..
  ```

Expected layout (one extra folder level is fine):

```
data/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg
data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
```

## 3. Run

```bash
python train.py --smoke        # 1–2 min sanity check of the whole pipeline
python train.py                # full run: train 40 epochs, tune post-processing, predict test
```

Short on time? `python train.py --epochs 20` (about half the training time). If CUDA runs out of
memory (8 GB card), add `--bs 2`.

## 4. Submit and report back

1. Upload `outputs/submission.csv` at the competition page → **Submit Predictions**.
2. Send back: the public leaderboard PQ, `outputs/report.json` (has `val_PQ`), `outputs/postprocess.json`
   and the weights file in `outputs/`, so the README results table and the report can be filled in.
3. The Google form (report + code link) is due at the final deadline as well — check the competition
   Overview page.
