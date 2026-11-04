"""Bundle filament/ + train.py into one self-contained Kaggle notebook (kaggle_baseline.ipynb).

Each source file becomes a %%writefile cell, so the repo stays the single source of truth.
Re-run this after editing any module:  python build_notebook.py

The notebook has two modes, selected by the MODE variable in its first code cell:
  "train"   - full pipeline (train -> tune post-processing -> submission.csv); needs GPU + Internet.
  "predict" - reproduce submission.csv from saved weights attached as a Kaggle dataset, no training.
"""
import json
import uuid
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["filament/__init__.py", "filament/data.py", "filament/rle.py", "filament/metric.py",
         "filament/model.py", "filament/dataset.py", "filament/postprocess.py",
         "filament/inference.py", "filament/submission.py", "train.py"]
PINS = 'segmentation-models-pytorch==0.6.0 pycocotools==2.0.11'


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


cells = [
    md("# Solar filament segmentation — U-Net baseline\n\n"
       "Pipeline: grayscale H-alpha (2048²) → downscale to 1024 → U-Net(ResNet34) filament probability "
       "→ 8-way TTA → bicubic upsample → threshold → connected components = one instance per filament → COCO RLE.\n\n"
       "Training target = mean of the annotators' masks per image, kept soft (area fraction) when downscaled. "
       "Post-processing (threshold / min area / gap-linking) is tuned on a 15 % hold-out split grouped by "
       "observation day, with a local Panoptic Quality implementation that matches the host's self-evaluation notebook.\n\n"
       "**Settings needed:** Accelerator = GPU T4 x2 or P100, Internet = On (phone-verified account).  \n"
       "Full run ≈ 2 h on T4. Outputs land in `/kaggle/working/outputs/`: `submission.csv`, `best.pt` (+ `.json`), "
       "`postprocess.json`, `report.json`, `environment.json`."),
    code("MODE = 'train'   # 'train' = full pipeline; 'predict' = reproduce from saved weights (see last section)\n"
         "WEIGHTS_DIR = '/kaggle/input/segmenthreads-weights'   # only used when MODE == 'predict'"),
    code(f"!pip install -q {PINS}\n"
         "import torch, sys\n"
         "assert torch.cuda.is_available(), 'No GPU: Settings -> Accelerator -> GPU, then restart the session'\n"
         "print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))"),
    code("import os\nos.makedirs('filament', exist_ok=True)"),
]
for f in FILES:
    cells.append(code(f"%%writefile {f}\n" + (HERE / f).read_text(encoding="utf-8")))
cells += [
    md("## Run\n`-u` makes the progress lines appear as they are printed (the first ~5 min are spent caching "
       "the training set, then one line per epoch; validation PQ every 5 epochs and every 2 in the last 10)."),
    code("if MODE == 'train':\n"
         "    !python -u train.py --out /kaggle/working/outputs\n"
         "else:\n"
         "    !python -u train.py --predict-only --weights {WEIGHTS_DIR}/best.pt "
         "--postprocess {WEIGHTS_DIR}/postprocess.json --out /kaggle/working/outputs"),
    md("## Check the result\nSuccess looks like: 180 test images, few or no images without a prediction, "
       "validation PQ clearly above 0 (the two-annotator agreement reference is 0.35)."),
    code("import pandas as pd, json\n"
         "sub = pd.read_csv('/kaggle/working/outputs/submission.csv')\n"
         "n_img = sub.filament_id.str.split('_', n=1).str[0].nunique()\n"
         "print(f'rows={len(sub)}  images with >=1 filament={n_img}/180  (empty={180 - n_img})')\n"
         "env = json.load(open('/kaggle/working/outputs/environment.json'))\n"
         "print('environment:', env)\n"
         "if MODE == 'train':\n"
         "    rep = json.load(open('/kaggle/working/outputs/report.json'))\n"
         "    print('chosen post-processing:', rep['postprocess'], ' val PQ:', round(rep['val_PQ'], 4))\n"
         "    display(pd.DataFrame(rep['train_log']))\n"
         "    display(pd.DataFrame(rep['postprocess_grid']).head(10))"),
    md("## Reproducing without training\n"
       "After a training run, download `outputs/best.pt`, `outputs/best.pt.json` and `outputs/postprocess.json`, "
       "upload them as a private Kaggle dataset (e.g. `segmenthreads-weights`), attach it to this notebook, set "
       "`MODE = 'predict'` in the first cell and run all. The same files are committed to the public repository "
       "under `weights/` for the final deliverable."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "kaggle_baseline.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote kaggle_baseline.ipynb with", len(cells), "cells")
