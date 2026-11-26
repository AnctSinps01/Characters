"""Encode handwriting and standard glyphs into blocked-VAE codes for flow training.

Writes one flat array per split plus the bookkeeping needed to rebuild
(writer, character) pairs: the flow needs to fetch reference characters of the
same writer, which a flat array plus offsets cannot express on its own.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from VAE.data import GlyphStore
from VAE.model import GlyphVAE, VAEConfig


def encode(model, images, device, batch=512):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            chunk = torch.from_numpy(np.array(images[start:start + batch]).copy())
            x = chunk[:, None].float().to(device) / 127.5 - 1
            mean, _ = model.encode(x)
            outputs.append(mean.float().cpu().numpy().astype(np.float16))
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = GlyphStore(args.data)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        writers = [w for w in store.writers if w["split"] == split]
        latents, labels, owner = [], [], []
        offset = 0
        ranges = []
        for index, writer in enumerate(writers):
            codes = encode(model, store.writer_images(writer), device, args.batch)
            latents.append(codes)
            labels.append(np.asarray(writer["labels"], np.int32))
            owner.append(np.full(len(codes), index, np.int32))
            ranges.append({"id": writer["id"], "start": offset, "length": len(codes)})
            offset += len(codes)
            print(f"{split} {index + 1}/{len(writers)} {writer['id']} rows={len(codes)}", flush=True)
        np.save(out / f"{split}_latent.npy", np.concatenate(latents))
        np.save(out / f"{split}_label.npy", np.concatenate(labels))
        np.save(out / f"{split}_owner.npy", np.concatenate(owner))
        (out / f"{split}_writers.json").write_text(
            json.dumps({"characters": store.characters, "ranges": ranges}), encoding="utf-8")

    standards = encode(model, store.standards, device, args.batch)
    np.save(out / "standards_latent.npy", standards)
    (out / "manifest.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": asdict(VAEConfig(**checkpoint["model_config"])),
        "characters": store.characters,
        "image_size": store.image_size,
    }, indent=2), encoding="utf-8")
    store.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
