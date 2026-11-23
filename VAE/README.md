# VAE

Standalone spatial VAE for reconstructing the repository's prepared glyph images. It depends on neither `DiT-flow/` nor `ocr/`.

## Model

`GlyphVAE` is a convolutional VAE: 64×64 grayscale → 4×8×8 diagonal-Gaussian latent → 64×64 tanh image.

- Encoder: three stride-2 residual downsampling stages, then a 3×3 head that predicts mean and log-variance (2×4 = 8 channels).
- Decoder: mirrored 3×3 residual blocks with nearest-neighbour upsampling, ending in `Tanh`.
- **No skip connection** from encoder to decoder, so all reconstruction information must pass through the latent.
- Default `base_channels=64` gives **8,157,577 parameters**; `--base-channels 32` gives 2,049,225.

## Objective

```text
loss = MSE(reconstruction, image) + beta * KL(q(z|x) || N(0, I))
```

- MSE is averaged over batch, channels, and pixels; KL is averaged over batch and every scalar latent element. `beta` refers to this convention.
- `--beta` is the final KL weight; it ramps linearly from 0 to that value over `--kl-warmup-steps`.
- Training always draws `z = mu + exp(logvar/2) * eps`; validation reports both mean-decoded and sample-decoded reconstructions.

## Standard-glyph mixing

`--standard-ratio p` is the per-sample probability that a training item is a uniformly chosen printed glyph from `standards.npy` instead of a uniformly chosen training hand-writing image. It is an expected batch share, not a weighting or a count of classes.

- Training always uses only writers in the `train` split.
- The same `(seed, index)` mapping keeps the handwriting samples identical between a `p=0` and `p>0` run for indices still classified as handwriting; this makes the two runs comparable without one being a subset of the other.
- Printed glyphs are the same bank used by the prepared dataset. A nonzero run therefore tests "can the latent serve both handwriting and printed forms", **not** unseen-font generalization.

## Validation

Validation always uses writer-split `val` images (unseen authors) for handwriting, and separately the printed bank. `--val-samples 1024` evenly spaces the subset across all validation authors; `0` means the full split. Reported per split:

| Field | Meaning |
|---|---|
| `mse_mean` | MSE when decoding the posterior mean (deterministic) |
| `mse_sample` | MSE of a fixed-noise posterior draw |
| `mae_mean_255` | Mean absolute error in 0–255 grey levels |
| `kl` / `kl_per_image` | KL averaged per latent element / summed per image |
| `psnr_mean_db` | PSNR of the mean-decoded reconstruction |
| `samples` | Images evaluated |

`best.pt` is selected by `val_handwriting.mse_mean` in both runs, so the two runs are comparable on the same criterion. Validation uses a forked RNG and a fixed seed; it does not perturb training randomness.

## Run

```powershell
# Pure handwriting reconstruction
python -m VAE.train --data data/prepared --output runs/VAE/p0 \
  --standard-ratio 0 --steps 1000000 --batch-size 64 --workers 4 --device cuda

# Mixed with printed glyphs
python -m VAE.train --data data/prepared --output runs/VAE/p10 \
  --standard-ratio 0.1 --steps 1000000 --batch-size 64 --workers 4 --device cuda
```

Outputs: `run.json`, `metrics.jsonl`, `previews/<split>_<step>.png` (input / mean-decoded / sampled), `last.pt`, `best.pt`. SIGINT/SIGTERM finishes the current step and saves a checkpoint. Resume must repeat the saved CLI settings and the same device type:

```powershell
python -m VAE.train --data data/prepared --output runs/VAE/p0 --resume runs/VAE/p0/last.pt ...same flags...
```

## Tests

```powershell
python -m unittest discover -s VAE/tests -v
```

Eleven local tests cover reparameterisation gradients, analytic KL values and reductions, finite gradients, small-batch overfit, ratio determinism, split isolation, validation RNG restoration, and exact checkpoint resume. They use synthetic 16×16 images only.

## Known limitations

- The data fingerprint hashes `manifest.json` only; NPY contents are checked for dtype/shape, not re-hashed. Do not treat a replaced-but-same-shaped NPY as a new dataset.
- Validation metrics are image-weighted, not author-macro-averaged.
- `z` is unconstrained beyond the KL term; random-prior sampling quality is not evaluated here.
