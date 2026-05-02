# predict.py
# ─────────────────────────────────────────────────────────────────────────────
# Inference utilities for the gravitational lens classifier.
#
# Main function:
#   predict_lenses(list_of_image_paths)  →  list of (label_str, probability)
#
# Accepted file formats: .npy, .fits / .fit / .fts, .png, .jpg, .jpeg
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import config


# ── file loading ──────────────────────────────────────────────────────────────

def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    return arr[0] if arr.ndim == 3 else arr        # (H, W)


def _load_fits(path: str) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data[0] if data.ndim == 3 else data     # (H, W)


def _load_image_file(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')            # greyscale PIL
    return np.array(img, dtype=np.float32)


def load_raw_image(path: str) -> np.ndarray:
    """Load any supported file type and return a float32 (H, W) array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return _load_npy(path)
    elif ext in ('.fits', '.fit', '.fts'):
        return _load_fits(path)
    elif ext in ('.png', '.jpg', '.jpeg'):
        return _load_image_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── preprocessing ─────────────────────────────────────────────────────────────

_infer_tf = transforms.Compose([
    transforms.Normalize(config.norm_mean, config.norm_std),
])


def _preprocess(raw: np.ndarray) -> torch.Tensor:
    """
    Apply the same preprocessing pipeline used during training:
      log-stretch → per-image min-max norm → 3-channel tensor → normalise
    Returns: (1, 3, H, W) float32 tensor ready for model inference.
    """
    img = np.log1p(np.maximum(raw, 0.0))
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-8)
    img = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
    img = _infer_tf(img)
    return img.unsqueeze(0)                                             # (1,3,H,W)


# ── model loader ──────────────────────────────────────────────────────────────

_cached_model = None

def _get_model(device):
    global _cached_model
    if _cached_model is None:
        from model import GravLensNet
        m = GravLensNet(pretrained=False).to(device)
        if not os.path.exists(config.weights_path):
            raise FileNotFoundError(
                f"Weights not found at '{config.weights_path}'. "
                "Run train.py first, or place final_weights.pth in checkpoints/."
            )
        m.load_state_dict(torch.load(config.weights_path, map_location=device))
        m.eval()
        _cached_model = m
    return _cached_model


# ── main inference function ───────────────────────────────────────────────────

def predict_lenses(list_of_image_paths: list, device=None) -> list:
    """
    Run inference on a list of image file paths.

    Args:
        list_of_image_paths: list of str — paths to .npy, .fits, .png, or .jpg files.
        device: torch.device (optional) — defaults to CUDA if available, else CPU.

    Returns:
        list of dict, one per input path:
          {
            'path'       : str,
            'prediction' : 'Lens' | 'Non-Lens',
            'probability': float   (probability of being a lens, 0–1)
          }

    Example:
        results = predict_lenses(['data/img01.npy', 'data/img02.npy'])
        for r in results:
            print(r['path'], r['prediction'], f"{r['probability']:.3f}")
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = _get_model(device)
    results = []

    with torch.no_grad():
        for path in list_of_image_paths:
            try:
                raw   = load_raw_image(path)
                img_t = _preprocess(raw).to(device)     # (1, 3, H, W)
                prob  = torch.sigmoid(model(img_t)).item()
                label = config.classes[int(prob > 0.5)]
                results.append({
                    'path'       : path,
                    'prediction' : label,
                    'probability': prob,
                })
            except Exception as e:
                results.append({
                    'path'       : path,
                    'prediction' : 'ERROR',
                    'probability': -1.0,
                    'error'      : str(e),
                })

    return results


# ── command-line usage ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, glob
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(config.data_dir, '*.npy')))
    if not paths:
        print("Usage: python predict.py <image1.npy> [image2.npy ...]\n"
              "       (defaults to all .npy files in data/)")
        sys.exit(0)

    print(f"Running inference on {len(paths)} image(s)...\n")
    results = predict_lenses(paths)

    print(f"{'Path':<35}  {'Prob':>6}  Prediction")
    print('-' * 55)
    for r in results:
        name = os.path.basename(r['path'])
        if r['prediction'] == 'ERROR':
            print(f"{name:<35}  {'ERROR':>6}  {r.get('error','')}")
        else:
            print(f"{name:<35}  {r['probability']:>6.4f}  {r['prediction']}")
