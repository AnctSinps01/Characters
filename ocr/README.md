# OCR

Standalone handwritten Chinese character recognition, separate from `DiT-flow/` and `twist/`.

## Model

`VisionTransformerOCR` uses 64x64 grayscale inputs by default. A strided convolution creates non-overlapping 8x8 patches (64 tokens). A learned CLS token is prepended; learned position vectors are added once to `[CLS, patches]` before the Transformer stack. Each of 12 blocks is pre-LN with residual self-attention followed by pre-LN residual MLP:

```text
x = x + SelfAttention(LN(x))
x = x + MLP(LN(x))
```

After the final LayerNorm, only the CLS token is sent to a two-layer MLP classifier. Default width 384, 6 heads, MLP ratio 4, 12 blocks, 3,755 output classes.

## Data and split

Uses the existing `data/prepared` v1 dataset. The training and validation sets are split by writer according to `manifest.json`; validation writers are not present in training. Both splits must contain all class IDs. Images are preloaded as uint8 into one contiguous CPU array, then converted and normalized per batch to `[-1, 1]` on the selected device.

## Train

Run from the repository root:

```powershell
python -m ocr.train --data data/prepared --output runs/ocr/vit-s8-12x384
```

Default budget is 1,000,000 optimizer steps, batch size 64, AdamW, 5k warmup then cosine decay, bf16 on CUDA, validation every 5k steps, and checkpoint save every 5k. Validation reports loss, top-1, and top-5 accuracy. `last.pt` can be resumed with `--resume runs/ocr/vit-s8-12x384/last.pt`.

For a small local architecture/gradient check, tests run with:

```powershell
$env:PYTHONPATH = "."
python -m unittest discover -s ocr/tests -v
```
