"""Leave-a-writer-out retrieval probe: does the latent encode the character?

For every validation writer and a fixed set of characters shared by all of them,
one sample per (writer, character) is encoded. Each sample then retrieves its
nearest neighbour among samples of *other* writers; a hit means the latent
places the same character closer than any other character.

    python -m VAE.probe_latent_retrieval --checkpoint runs/VAE/p0/last.pt \
        --data data/prepared --label p0

The probe is a property of the frozen encoder; no training is involved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import GlyphStore
from .model import GlyphVAE, VAEConfig


def shared_characters(writers, limit):
    """Characters written by every writer, truncated deterministically."""
    common = set(writers[0]["labels"])
    for writer in writers[1:]:
        common &= set(writer["labels"])
    return sorted(common)[:limit]


def encode_all(store, writers, characters, config, model, device):
    """Return [writers x characters x dims] mean vectors in float32."""
    per_writer = []
    for writer in writers:
        image_array = store.writer_images(writer)
        position = {c: i for i, c in enumerate(writer["labels"])}
        index = [position[c] for c in characters]
        pixels = np.stack([image_array[i] for i in index])[:, None].astype(np.float32)
        x = torch.from_numpy(pixels).to(device) / 127.5 - 1
        with torch.no_grad():
            mean, _ = model.encode(x)
        per_writer.append(mean.flatten(1).float().cpu().numpy())
    return np.stack(per_writer)


def retrieve(features, characters, *, metric="l2"):
    """Leave-one-writer-out top-1/top-5 accuracy for identical characters."""
    n_writers, n_chars, dims = features.shape
    flat = features.reshape(-1, dims)
    labels = np.tile(np.asarray(characters), n_writers)
    owners = np.repeat(np.arange(n_writers), n_chars)
    if metric == "cosine":
        norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-8)
        unit = flat / norms
        distance = 1.0 - (unit @ unit.T)
    else:
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b ; avoids an N*N*D temporary.
        square = (flat * flat).sum(1)
        distance = square[:, None] + square[None, :] - 2.0 * (flat @ flat.T)
        np.maximum(distance, 0.0, out=distance)
    # Exclude the whole query writer, so no same-handwriting shortcut remains.
    distance[owners[:, None] == owners[None, :]] = np.inf

    order = np.argsort(distance, axis=1)[:, :5]
    top1 = (labels[order[:, 0]] == labels).mean()
    top5 = (labels[order] == labels[:, None]).any(axis=1).mean()
    # Positive/negative distance margins on valid pairs only.
    same = labels[:, None] == labels[None, :]
    valid = np.isfinite(distance)
    pos = distance[same & valid]
    neg = distance[~same & valid]
    return {"top1": float(top1), "top5": float(top5),
            "same_char_distance": float(np.median(pos)),
            "other_char_distance": float(np.median(neg)),
            "separation": float(np.median(neg) / max(np.median(pos), 1e-8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--characters", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = GlyphStore(args.data)
    try:
        writers = [w for w in store.writers if w["split"] == "val"]
        characters = shared_characters(writers, args.characters)
        features = encode_all(store, writers, characters, model.config, model, device)
    finally:
        store.close()
    report = {"label": args.label or str(args.checkpoint), "step": checkpoint["step"],
              "latent_shape": list(checkpoint["model_config"].get("latent_shape", [])) or None,
              "writers": len(writers), "characters": len(characters),
              "dims": int(features.shape[-1])}
    for metric in ("l2", "cosine"):
        report[metric] = retrieve(features, characters, metric=metric)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
