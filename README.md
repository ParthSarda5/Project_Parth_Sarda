# Gravitational Lens Detection using CNNs
**Image and Video Processing with Deep Learning — IISER Pune 2026**  
**Student:** Parth Bhatt

---

## Project Overview

This project trains a ResNet-18 CNN to perform **binary classification** of simulated astronomical survey images: does the image contain a strong gravitational lens or not?

Strong gravitational lensing occurs when a foreground galaxy bends the light from a background source, producing arc-like or ring-shaped distortions. Detecting these lenses manually does not scale to modern surveys (Euclid, LSST), making CNN-based automation scientifically important.

**Task type:** Supervised binary image classification  
**Input:** 64×64 pixel grayscale image of a galaxy (`.npy` file)  
**Output:** `"Lens"` or `"Non-Lens"` with a probability score

---

## Directory Structure

```
project_parth_bhatt/
├── checkpoints/
│   └── final_weights.pth       ← trained model weights
├── data/
│   ├── lens_01.npy             ← 10 example lensed galaxy images
│   │   ...
│   ├── lens_10.npy
│   ├── nonlens_01.npy          ← 10 example non-lensed galaxy images
│   │   ...
│   └── nonlens_10.npy
├── config.py                   ← all hyperparameters and paths
├── dataset.py                  ← LensDataset class + get_dataloader()
├── generate_data.py            ← simulate full training dataset
├── model.py                    ← GravLensNet (ResNet-18 + binary head)
├── train.py                    ← train_model() + CLI entry point
├── predict.py                  ← predict_lenses() + CLI entry point
├── interface.py                ← standardised grading interface
└── README.md
```

---

## Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision scikit-learn matplotlib tqdm astropy lenstronomy
```

---

## How to Run

### Step 1 — Generate the training dataset

The training data is simulated using `lenstronomy`. Run this once and cache the output.

```bash
# Quick smoke test (~2 min, 200 images/class)
python generate_data.py --n 200

# Full dataset as used in the project (~30 min CPU, 10 000 images/class)
python generate_data.py
```

This creates:
- `data/images.npy` — full training array (N, 64, 64)
- `data/labels.npy` — binary labels (N,)
- `data/lens_01.npy` … `data/lens_10.npy` — 10 example lensed images
- `data/nonlens_01.npy` … `data/nonlens_10.npy` — 10 example non-lensed images

> **Note:** The 20 example `.npy` files already present in `data/` were generated with synthetic Gaussian blobs (no lenstronomy required) to satisfy the submission data requirement. Run `generate_data.py` to replace them with fully realistic lenstronomy-simulated examples.

### Step 2 — Train the model

```bash
python train.py
```

Saves the best checkpoint (by validation AUC) to `checkpoints/final_weights.pth`.  
All hyperparameters are controlled via `config.py`.

### Step 3 — Run inference

```bash
# On specific files
python predict.py data/lens_01.npy data/nonlens_03.npy

# On all example images
python predict.py
```

Output format:
```
Path                                Prob   Prediction
lens_01.npy                       0.9412  Lens
nonlens_03.npy                    0.0871  Non-Lens
```

---

## Interface (for grading)

The `interface.py` file exposes all components under standardised names:

```python
from interface import (
    TheModel,        # GravLensNet — ResNet-18 binary classifier
    the_trainer,     # train_model(model, num_epochs, train_loader, loss_fn, optimizer, ...)
    the_predictor,   # predict_lenses(list_of_image_paths) → list of dicts
    TheDataset,      # LensDataset(images, labels, transform)
    the_dataloader,  # get_dataloader() → (train_loader, val_loader, test_loader)
    the_batch_size,  # 64
    total_epochs,    # 30
)
```

### Re-running training programmatically

```python
import torch, torch.nn as nn
from torch.optim import Adam
from interface import TheModel, the_trainer, the_dataloader, the_batch_size, total_epochs

model        = TheModel(pretrained=True)
train_loader, val_loader, test_loader = the_dataloader()
optimizer    = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn      = nn.BCEWithLogitsLoss()

history = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer,
                      val_loader=val_loader)
```

### Running inference programmatically

```python
from interface import the_predictor

results = the_predictor(['data/lens_01.npy', 'data/nonlens_02.npy'])
for r in results:
    print(r['path'], r['prediction'], r['probability'])
```

`the_predictor` returns a list of dicts:
```python
{'path': 'data/lens_01.npy', 'prediction': 'Lens', 'probability': 0.9412}
```

---

## Configuration (`config.py`)

| Variable        | Default | Description                         |
|----------------|---------|-------------------------------------|
| `resize_x`      | 64      | Image width (pixels)                |
| `resize_y`      | 64      | Image height (pixels)               |
| `input_channels`| 3       | Input channels (greyscale → 3-ch)   |
| `batch_size`    | 64      | Training batch size                 |
| `epochs`        | 30      | Maximum training epochs             |
| `learning_rate` | 1e-4    | Adam learning rate                  |
| `patience`      | 8       | Early stopping patience             |
| `n_per_class`   | 10 000  | Images per class in full dataset    |
| `weights_path`  | `checkpoints/final_weights.pth` | Checkpoint path |

---

## Model Architecture

- **Backbone:** ResNet-18 (pretrained on ImageNet-1K)
- **Head:** `Linear(512 → 1)` for binary classification
- **Loss:** `BCEWithLogitsLoss` — numerically stable binary cross-entropy
- **Optimiser:** Adam with cosine annealing LR schedule and early stopping by validation AUC

**Preprocessing pipeline (identical at train and inference time):**
1. Log-stretch: `log1p(max(x, 0))` — compresses dynamic range
2. Per-image min-max normalisation to [0, 1]
3. Greyscale → 3-channel (repeated) for ImageNet-compatible backbone
4. Augmentation (train only): random horizontal/vertical flip + 180° rotation
5. Normalise: mean=0.5, std=0.5 → [−1, 1]

**Why 360° rotation augmentation?** Gravitational arcs have no preferred orientation in the sky — full rotation is physically motivated and significantly improves generalisation.

---

## Data

Training images are simulated using **lenstronomy** (Birrer & Amara, 2018), a ray-tracing package widely used in the astrophysics community.

- **Lensed images:** Singular isothermal sphere (SIS) lens + Sérsic source galaxy + Sérsic lens-galaxy light + realistic Poisson + Gaussian sky noise.
- **Unlensed images:** Sérsic galaxy with no deflection + same noise model.
- **Image size:** 64 × 64 pixels at 0.1 arcsec/pixel (6.4 arcsec field of view).

This simulation-based approach is standard practice in the field (Metcalf et al. 2019; Petrillo et al. 2019) because confirmed strong lenses are too rare (~1 per few thousand galaxies) to provide sufficient real training data.

---

## Expected Performance

| Metric   | Target (from proposal) |
|----------|------------------------|
| AUC-ROC  | ≥ 0.90                 |
| Accuracy | ≥ 85%                  |

Published CNN baselines on the Bologna Lens Challenge achieve AUC up to 0.96, so these targets are achievable within the 3-week timeline.
