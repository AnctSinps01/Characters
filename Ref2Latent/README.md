# Ref2Latent

Latent-space handwriting generator: `n` reference latents + one standard-glyph
latent -> the target glyph's latent mean. Not a diffusion/flow model: there is no
noise state, no timestep and no ODE; one forward pass produces the answer.

## Architecture

**One latent is one token.** A 4×8×8 latent is flattened to 256 numbers and
projected to the model width; the sequence is therefore only `n + 1` tokens.

```
n reference latents  [B, n, 4, 8, 8]  --Linear_ref-->  n tokens  (+ reference index)
1 standard latent    [B,    4, 8, 8]  --Linear_std-->  1 token
                     joint residual attention x L        (two QKV projections, one attention map)
                             |
        the standard token -> Linear -> predicted latent mean [B, 4, 8, 8]
```

- **Two QKV projections.** References and the standard glyph have their own
  `qkv` linear maps; keys and values are concatenated, so every token attends to
  both groups, but the two roles are projected differently.
- **Joint residual attention.** Each block is `x += Attn(LN(x))`, then a shared
  FFN over the concatenated sequence, added back to both groups.
- **Readout from the standard token.** The single token that entered from the
  standard glyph is mapped straight back to 4×8×8.

Default size: `dim=384, heads=6, layers=6, mlp_ratio=4` -> **13.6M parameters**,
5 tokens (4 references + 1 standard).

## Losses

```
loss = latent_weight * MSE(predicted_latent, target_latent)
     + image_weight  * MSE(decode(predicted_latent), target_image)   # optional
```

`image_weight` needs `--vae`; the decoder is frozen and only used for this term
and for the validation preview. Latent MSE is also the model-selection metric.

## Usage

```powershell
# 1. Encode every handwriting image and printed glyph once (posterior means, float16)
python -m Ref2Latent.prepare_latents --checkpoint runs/VAE/p0-b1e3/last.pt `
    --data data/prepared --output runs/latents/p0-b1e3

# 2. Train
python -m Ref2Latent.train --data data/prepared --latents runs/latents/p0-b1e3 `
    --vae runs/VAE/p0-b1e3/last.pt --output runs/Ref2Latent/base `
    --steps 200000 --batch-size 64 --references 4
```

`prepare_latents` writes one float16 NPY per writer plus `standards.npy` and a
manifest recording the VAE checkpoint. Re-encoding with another VAE means
another output directory; nothing else changes.

## Validation

Held-out writers only. Per validation step: latent MSE, decoded image MSE,
decoded image MAE, and `baseline_mae` = MAE of decoding the *true* target latent
(the VAE's own reconstruction floor). A useful model must beat a trivial copy of
the standard latent and approach `baseline_mae`.

## Tests

```powershell
python -m unittest discover -s Ref2Latent/tests -v
```

Nine tests: shapes and determinism, enforced reference count and latent shape,
separate projections, gradients reaching every parameter, block token lengths,
small-batch overfit, dataset shapes/deterministic references (never the target
row), split isolation, config validation.

## Known limitations

- Only the posterior mean is modelled; no sampling, so no per-instance diversity.
- The output ceiling is the VAE decoder's: fine strokes it dropped cannot return.
- References and the standard glyph are frozen pre-computed latents; the VAE is
  never fine-tuned jointly here.
