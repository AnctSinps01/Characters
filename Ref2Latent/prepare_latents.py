"""Encode every handwriting image and printed glyph into VAE latent means.

    python -m Ref2Latent.prepare_latents --checkpoint runs/VAE/lat2/last.pt \
        --data data/prepared --output data/latents-lat2

Writes one float16 NPY per writer plus ``standards.npy`` and ``manifest.json``.
Only the posterior mean is stored: the downstream model regresses means.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from VAE.data import GlyphStore
from VAE.model import GlyphVAE, VAEConfig


@torch.no_grad()
def encode_arrays(model, images, device, batch_size=256):
    """images: uint8 [N, H, W] host array -> float16 latent means [N, C, h, w]."""
    out = []
    for start in range(0, len(images), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size]))
        x = chunk.to(device).float().div(127.5).sub(1).unsqueeze(1)
        mean, _ = model.encode(x)
        out.append(mean.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, *model.config.latent_shape), np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    wanted = set(args.splits.split(","))
    store = GlyphStore(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        classes = 0
        for writer in store.writers:
            if writer["split"] not in wanted:
                continue
            path = args.output / f"{writer['id']}.npy"
            if path.exists():
                continue
            latents = encode_arrays(model, store.writer_images(writer), device, args.batch_size)
            np.save(path, latents)
            classes += len(latents)
            print(f"{writer['id']} {writer['split']} {latents.shape} "
                  f"{time.time() - started:.0f}s", flush=True)
        standards_path = args.output / "standards.npy"
        if not standards_path.exists():
            np.save(standards_path, encode_arrays(model, store.standards, device,
                                                  args.batch_size))
        manifest = {
            "latent_shape": list(model.config.latent_shape),
            "image_size": int(store.image_size),
            "classes": len(store.characters),
            "vae_checkpoint": str(args.checkpoint),
            "vae_step": int(checkpoint["step"]),
            "vae_model_config": checkpoint["model_config"],
            "vae_state_sha256": hashlib.sha256(
                (args.checkpoint.parent / "last.pt").read_bytes()).hexdigest()
            if (args.checkpoint.parent / "last.pt").exists() else None,
            "splits": sorted(wanted),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
