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

## Results (2026-11-23, stopped at ~130k steps)

Full validation split (89,796 images, unseen writers); `IoU` and `ink MAE` use the same
ink threshold (grey < 96); `rate` is `KL` converted to bits per image.

| run | latent | β | rate | MAE | ink MAE | IoU | PSNR | std MAE | std IoU | retrieval top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p0` | 4×8×8 = 256 | 1e-4 | 1,265 bit | 2.18 | 9.35 | 0.598 | 32.76 | 14.33 | 0.795 | 21.9 % |
| `beta1e3` | 4×8×8 = 256 | 1e-3 | 730 bit | 2.30 | 10.10 | 0.576 | 32.38 | 12.95 | 0.876 | 13.9 % |
| `lat2` | 2×8×8 = 128 | 1e-4 | 761 bit | 3.28 | 14.91 | 0.459 | 29.14 | 17.84 | 0.735 | 38.4 % |
| `grid4` (deleted) | 4×4×4 = 64 | 1e-4 | 416 bit | 5.98 | 27.69 | 0.522 | 24.39 | 28.07 | 0.654 | 48.6 % |

`retrieval top-1` is leave-one-writer-out nearest neighbour among 256 characters × 24
validation writers (chance ≈ 0.4 %). `grid4`'s weights were removed at the user's request;
only `runs/VAE/grid4_retrieval.json` and its log survive.

Findings:

1. **Width matters at fixed rate.** At ~750 bit, the 256-dimensional `beta1e3` beats the
   128-dimensional `lat2` on every reconstruction metric (MAE 2.30 vs 3.28; ink MAE 10.1 vs
   14.9; IoU 0.576 vs 0.459). Squeezing bits with β is cheaper than removing dimensions.
2. **Narrower latents trade fidelity for character separability.** Retrieval rises
   21.9 % → 38.4 % → 48.6 % as the bottleneck shrinks 256 → 128 → 64, while IoU falls.
   A larger latent keeps appearance (writer, thickness) and that hurts cross-writer retrieval.
3. **No writer overfitting.** Training and validation MAE differ by ≈ 0.02 at equal steps,
   and 8.16 M parameters (≈ 16 MB fp16) cannot hold 800 k images.
4. **Printed glyphs: shape transfers, ink does not.** On the standard bank IoU is 0.74–0.88
   but ink MAE is 45–63 (handwriting: 9–10), i.e. reconstructions are structurally right and
   too light. This is out-of-distribution behaviour, not a memorisation measure.
5. **Plateau.** Between 122 k and 136 k steps, MAE moved < 0.01 per 7.5 k steps while KL stayed
   flat, so these β values had converged. No further gain expected from more steps at the
   same configuration.

Not measured: character-level generalisation. Both splits share all 3,755 classes, so these
results cannot distinguish "general glyph code" from "one stored template per character";
a character-holdout run would be needed.

## Tests

```powershell
python -m unittest discover -s VAE/tests -v
```

Eleven local tests cover reparameterisation gradients, analytic KL values and reductions, finite gradients, small-batch overfit, ratio determinism, split isolation, validation RNG restoration, and exact checkpoint resume. They use synthetic 16×16 images only.

## Known limitations

- The data fingerprint hashes `manifest.json` only; NPY contents are checked for dtype/shape, not re-hashed. Do not treat a replaced-but-same-shaped NPY as a new dataset.
- Validation metrics are image-weighted, not author-macro-averaged.
- `z` is unconstrained beyond the KL term; random-prior sampling quality is not evaluated here.
