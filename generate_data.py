# generate_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Simulates gravitational lens and non-lens galaxy images using lenstronomy.
#
# Generates TWO outputs:
#   1. Full training dataset:
#        data/images.npy   (2 * n_per_class, H, W) float32
#        data/labels.npy   (2 * n_per_class,)       int64
#
#   2. Ten example images per class for the data/ directory
#      (as required by the submission format):
#        data/lens_01.npy ... data/lens_10.npy       (label = 1)
#        data/nonlens_01.npy ... data/nonlens_10.npy (label = 0)
#
# Usage:
#   python generate_data.py                # full run (n_per_class from config)
#   python generate_data.py --n 500        # quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import numpy as np
from tqdm import tqdm

import config


# ── lenstronomy helpers ───────────────────────────────────────────────────────

def _make_data_cfg(num_pix, pixel_scale):
    from lenstronomy.Util import simulation_util as sim_util
    return sim_util.data_configure_simple(
        num_pix, pixel_scale, exposure_time=1000.0, background_rms=0.005)


def _make_psf(pixel_scale, fwhm):
    from lenstronomy.Data.psf import PSF
    return PSF(psf_type='GAUSSIAN', fwhm=fwhm, pixel_size=pixel_scale)


def _add_noise(image, rng, background_rms=0.005, exposure_time=1000.0):
    flux     = np.maximum(image * exposure_time, 0.0)
    poisson  = rng.poisson(flux).astype(np.float64) / exposure_time - np.maximum(image, 0.0)
    gaussian = rng.normal(0.0, background_rms, image.shape)
    return image + poisson + gaussian


# ── per-image simulators ──────────────────────────────────────────────────────

def simulate_lensed(rng, num_pix=64, pixel_scale=0.1) -> np.ndarray:
    """Return one (H, W) float32 strongly lensed galaxy image."""
    from lenstronomy.LensModel.lens_model   import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model      import ImageModel
    from lenstronomy.Data.imaging_data      import ImageData

    data_class = ImageData(**_make_data_cfg(num_pix, pixel_scale))
    psf_class  = _make_psf(pixel_scale, fwhm=rng.uniform(0.3, 0.6))

    lens_model   = LensModel(['SIS'])
    kwargs_lens  = [{'theta_E': rng.uniform(0.6, 1.4),
                     'center_x': 0.0, 'center_y': 0.0}]

    source_model  = LightModel(['SERSIC_ELLIPSE'])
    kwargs_source = [{'amp': rng.uniform(80, 250), 'R_sersic': rng.uniform(0.05, 0.20),
                      'n_sersic': rng.uniform(1.0, 4.0),
                      'e1': rng.uniform(-0.3, 0.3), 'e2': rng.uniform(-0.3, 0.3),
                      'center_x': rng.uniform(-0.25, 0.25),
                      'center_y': rng.uniform(-0.25, 0.25)}]

    lens_light_model  = LightModel(['SERSIC_ELLIPSE'])
    kwargs_lens_light = [{'amp': rng.uniform(30, 120), 'R_sersic': rng.uniform(0.2, 0.7),
                          'n_sersic': rng.uniform(3.0, 6.0),
                          'e1': rng.uniform(-0.2, 0.2), 'e2': rng.uniform(-0.2, 0.2),
                          'center_x': 0.0, 'center_y': 0.0}]

    image_model = ImageModel(data_class, psf_class,
                             lens_model_class=lens_model,
                             source_model_class=source_model,
                             lens_light_model_class=lens_light_model)
    image = image_model.image(kwargs_lens=kwargs_lens,
                              kwargs_source=kwargs_source,
                              kwargs_lens_light=kwargs_lens_light)
    return _add_noise(image, rng).astype(np.float32)


def simulate_unlensed(rng, num_pix=64, pixel_scale=0.1) -> np.ndarray:
    """Return one (H, W) float32 unlensed galaxy image."""
    from lenstronomy.LensModel.lens_model   import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model      import ImageModel
    from lenstronomy.Data.imaging_data      import ImageData

    data_class    = ImageData(**_make_data_cfg(num_pix, pixel_scale))
    psf_class     = _make_psf(pixel_scale, fwhm=rng.uniform(0.3, 0.6))
    lens_model    = LensModel([])
    galaxy_model  = LightModel(['SERSIC_ELLIPSE'])
    kwargs_source = [{'amp': rng.uniform(40, 180), 'R_sersic': rng.uniform(0.2, 1.0),
                      'n_sersic': rng.uniform(1.0, 5.5),
                      'e1': rng.uniform(-0.5, 0.5), 'e2': rng.uniform(-0.5, 0.5),
                      'center_x': rng.uniform(-0.4, 0.4),
                      'center_y': rng.uniform(-0.4, 0.4)}]
    image_model   = ImageModel(data_class, psf_class,
                               lens_model_class=lens_model,
                               source_model_class=galaxy_model)
    image = image_model.image(kwargs_lens=[], kwargs_source=kwargs_source)
    return _add_noise(image, rng).astype(np.float32)


# ── generation helpers ────────────────────────────────────────────────────────

def _safe_generate(fn, rng, n, num_pix, pixel_scale, desc):
    """Generate n images with retry logic for rare numerical failures."""
    imgs = []
    pbar = tqdm(total=n, desc=desc)
    while len(imgs) < n:
        for _ in range(10):
            try:
                img = fn(rng, num_pix, pixel_scale)
                if np.isfinite(img).all():
                    imgs.append(img)
                    pbar.update(1)
                    break
            except Exception:
                pass
    pbar.close()
    return imgs


# ── main ──────────────────────────────────────────────────────────────────────

def generate(n_per_class=None, num_pix=None, pixel_scale=None,
             seed=None, out_dir=None):
    n_per_class = n_per_class or config.n_per_class
    num_pix     = num_pix     or config.resize_x
    pixel_scale = pixel_scale or config.pixel_scale
    seed        = seed        if seed is not None else config.random_seed
    out_dir     = out_dir     or config.data_dir

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── Full training dataset ──────────────────────────────────────────────
    print(f"=== Generating full training dataset ({n_per_class} images/class) ===")
    lensed   = _safe_generate(simulate_lensed,   rng, n_per_class, num_pix, pixel_scale,
                              'Lensed')
    unlensed = _safe_generate(simulate_unlensed, rng, n_per_class, num_pix, pixel_scale,
                              'Unlensed')

    images = np.stack(lensed + unlensed)                    # (2N, H, W)
    labels = np.array([1]*len(lensed) + [0]*len(unlensed), dtype=np.int64)

    perm   = rng.permutation(len(labels))
    images, labels = images[perm], labels[perm]

    np.save(os.path.join(out_dir, 'images.npy'), images)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)
    print(f"Saved images.npy {images.shape}  labels.npy {labels.shape}\n")

    # ── 10 example images per class (submission requirement) ───────────────
    print("=== Saving 10 example images per class for data/ directory ===")
    ex_rng = np.random.default_rng(seed + 1)

    for i, img in enumerate(
            _safe_generate(simulate_lensed, ex_rng, 10, num_pix, pixel_scale,
                           'Example lensed'), start=1):
        path = os.path.join(out_dir, f'lens_{i:02d}.npy')
        np.save(path, img)

    for i, img in enumerate(
            _safe_generate(simulate_unlensed, ex_rng, 10, num_pix, pixel_scale,
                           'Example unlensed'), start=1):
        path = os.path.join(out_dir, f'nonlens_{i:02d}.npy')
        np.save(path, img)

    print("Done. Example images: lens_01.npy … lens_10.npy  "
          "nonlens_01.npy … nonlens_10.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate gravitational lensing dataset')
    parser.add_argument('--n',    type=int, default=None,
                        help=f'Images per class (default: {config.n_per_class})')
    parser.add_argument('--size', type=int, default=None,
                        help=f'Image size in pixels (default: {config.resize_x})')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out',  type=str, default=None, help='Output directory')
    args = parser.parse_args()
    generate(n_per_class=args.n, num_pix=args.size, seed=args.seed, out_dir=args.out)
