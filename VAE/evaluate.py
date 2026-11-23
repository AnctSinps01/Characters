"""Full-split reconstruction metrics for a trained VAE checkpoint.

    python -m VAE.evaluate --checkpoint runs/VAE/p0/last.pt \
        --data data/prepared --output runs/VAE/p0/full_eval.json

Reports handwriting (validation writers) and printed-standard reconstructions
separately. Distortion is measured with both plain pixel MAE and ink-restricted
MAE / IoU, because plain MAE is dominated by background and hides stroke errors.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GlyphStore, ReconstructionDataset
from .model import GlyphVAE, VAEConfig

INK_THRESHOLD = -0.25  # grey level below ~96 counts as ink for IoU/ink-MAE


def amp(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


@torch.no_grad()
def measure(model, loader, device, precision, seed=0):
    totals = {"n": 0.0, "mae": 0.0, "mse": 0.0, "ink_mae": 0.0, "ink_px": 0.0,
              "inter": 0.0, "union": 0.0, "kl_nat": 0.0}
    for raw in loader:
        x = raw["image"].to(device).float() / 127.5 - 1
        with amp(device, precision):
            mean, logvar = model.encode(x)
            y = model.decode(mean).float()
        n = len(x)
        ink = x < INK_THRESHOLD
        error = (y - x).abs()
        totals["n"] += n
        totals["mae"] += error.mean().item() * 127.5 * n
        totals["mse"] += (y - x).square().mean().item() * n
        totals["ink_mae"] += error[ink].sum().item() * 127.5 if ink.any() else 0.0
        totals["ink_px"] += ink.sum().item()
        pred = y < INK_THRESHOLD
        totals["inter"] += (pred & ink).sum().item()
        totals["union"] += (pred | ink).sum().item()
        kl = 0.5 * (mean.float().square() + logvar.float().exp() - 1 - logvar.float())
        totals["kl_nat"] += kl.flatten(1).sum(1).sum().item()
    n = max(totals["n"], 1.0)
    mse = totals["mse"] / n
    return {"samples": int(totals["n"]),
            "mae": totals["mae"] / n,
            "ink_mae": totals["ink_mae"] / max(totals["ink_px"], 1.0),
            "iou": totals["inter"] / max(totals["union"], 1.0),
            "mse": mse,
            "psnr_db": 10 * math.log10(4 / max(mse, 1e-12)),
            "kl_nat_per_image": totals["kl_nat"] / n,
            "rate_bits": totals["kl_nat"] / n / math.log(2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = GlyphStore(args.data)
    report = {"checkpoint": str(args.checkpoint), "step": checkpoint["step"],
              "latent_shape": list(VAEConfig(**checkpoint["model_config"]).latent_shape),
              "beta": checkpoint["training_config"]["beta"],
              "parameters": sum(p.numel() for p in model.parameters())}
    try:
        for split, source in (("val", "handwriting"), ("val", "standard")):
            dataset = ReconstructionDataset(store, split=split, source=source)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
            key = "handwriting" if source == "handwriting" else "standard"
            report[key] = measure(model, loader, device, args.precision)
            report[key]["split"] = split
    finally:
        store.close()
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
