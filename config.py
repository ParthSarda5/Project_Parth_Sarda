# config.py
# ─────────────────────────────────────────────────────────────────────────────
# All hyperparameters and dataset configuration for the gravitational lens
# classification project.  Every other file imports from here — do not
# hard-code these values elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

# ── Image dimensions ──────────────────────────────────────────────────────────
resize_x       = 64          # image width  (pixels)
resize_y       = 64          # image height (pixels)
input_channels = 3           # greyscale repeated to 3-ch for ImageNet backbone

# ── Training hyperparameters ──────────────────────────────────────────────────
batch_size     = 64
epochs         = 30
learning_rate  = 1e-4
weight_decay   = 1e-4
patience       = 8           # early-stopping: epochs without AUC improvement

# ── Data split fractions ──────────────────────────────────────────────────────
val_frac       = 0.15
test_frac      = 0.15

# ── Simulation parameters (used by generate_data.py) ─────────────────────────
n_per_class    = 10_000      # images per class in the full training dataset
pixel_scale    = 0.1         # arcsec / pixel
random_seed    = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
data_dir       = 'data'
checkpoint_dir = 'checkpoints'
weights_path   = 'checkpoints/final_weights.pth'

# ── Normalisation constants ───────────────────────────────────────────────────
norm_mean      = [0.5, 0.5, 0.5]
norm_std       = [0.5, 0.5, 0.5]

# ── Class labels ──────────────────────────────────────────────────────────────
classes        = {0: 'Non-Lens', 1: 'Lens'}
