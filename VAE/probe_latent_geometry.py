"""Geometry of the flattened latent: norms and directions.

    python -m VAE.probe_latent_geometry --checkpoint runs/VAE/p0/last.pt \
        --data data/prepared --output runs/VAE/p0/geometry

Panels: distribution of ||mu||; PCA of the raw latent coloured by writer; PCA of
unit-normalised latents (direction only) coloured by writer; and the same
directions with a few characters highlighted. PCA here is plain SVD of the
centred matrix — no t-SNE/UMAP is available in this environment.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import GlyphStore, ReconstructionDataset, spread_indices
from .model import GlyphVAE, VAEConfig


def encode(store, model, device, characters):
    dataset = ReconstructionDataset(store, split="val", source="handwriting")
    index = {c: [] for c in characters}
    writers = []
    per_writer = []
    position = 0
    cursor = 0
    for writer in [w for w in store.writers if w["split"] == "val"]:
        image_array = store.writer_images(writer)
        labels = writer["labels"]
        where = {c: i for i, c in enumerate(labels)}
        keys = [c for c in characters if c in where]
        if not keys:
            continue
        pixels = np.stack([image_array[where[c]] for c in keys])[:, None].astype(np.float32)
        x = torch.from_numpy(pixels).to(device) / 127.5 - 1
        with torch.no_grad():
            mean, _ = model.encode(x)
        flat = mean.flatten(1).float().cpu().numpy()
        per_writer.append(flat)
        writers.append(np.full(len(keys), position))
        for c in keys:
            index[c].append(cursor)
            cursor += 1
        position += 1
    return np.concatenate(per_writer), np.concatenate(writers), index


def pca(matrix, dims=2):
    centred = matrix - matrix.mean(0, keepdims=True)
    u, s, vh = np.linalg.svd(centred, full_matrices=False)
    coords = centred @ vh[:dims].T
    explained = (s[:dims] ** 2) / (s ** 2).sum()
    return coords, explained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--characters", type=int, default=256)
    parser.add_argument("--highlight", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlyphVAE(VAEConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    store = GlyphStore(args.data)
    try:
        writers = [w for w in store.writers if w["split"] == "val"]
        common = set(writers[0]["labels"])
        for w in writers[1:]:
            common &= set(w["labels"])
        characters = sorted(common)[:args.characters]
        features, owner, index = encode(store, model, device, characters)
        names = store.characters
    finally:
        store.close()

    norm = np.linalg.norm(features, axis=1)
    unit = features / np.clip(norm[:, None], 1e-8, None)

    raw_coords, raw_explained = pca(features)
    dir_coords, dir_explained = pca(unit)

    char_of = np.zeros(len(features), dtype=np.int64)
    for c, rows in index.items():
        char_of[rows] = c
    same_char = char_of[:, None] == char_of[None, :]
    same_writer = owner[:, None] == owner[None, :]
    np.fill_diagonal(same_char, False)
    np.fill_diagonal(same_writer, False)
    cosine = unit @ unit.T
    stats = {
        "checkpoint": str(args.checkpoint), "step": checkpoint["step"],
        "samples": int(len(features)), "dims": int(features.shape[1]),
        "writers": int(owner.max() + 1), "characters": len(characters),
        "norm_mean": float(norm.mean()), "norm_std": float(norm.std()),
        "norm_p05": float(np.quantile(norm, 0.05)), "norm_p95": float(np.quantile(norm, 0.95)),
        "raw_pc1_2_explained": raw_explained.round(4).tolist(),
        "dir_pc1_2_explained": dir_explained.round(4).tolist(),
        "cos_same_char": float(cosine[same_char].mean()),
        "cos_diff_char": float(cosine[~same_char].mean()),
        "cos_same_writer": float(cosine[same_writer].mean()),
        "cos_diff_writer": float(cosine[~same_writer].mean()),
    }
    stats["char_margin"] = stats["cos_same_char"] - stats["cos_diff_char"]
    stats["writer_margin"] = stats["cos_same_writer"] - stats["cos_diff_writer"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=140)
    axes[0, 0].hist(norm, bins=60, color="#4878a8")
    axes[0, 0].axvline(norm.mean(), color="crimson", lw=1.2, label=f"mean {norm.mean():.2f}")
    axes[0, 0].axvline(np.median(norm), color="darkorange", lw=1.2, ls="--",
                       label=f"median {np.median(norm):.2f}")
    axes[0, 0].set_title("||mu|| distribution (validation handwriting)")
    axes[0, 0].set_xlabel("latent L2 norm")
    axes[0, 0].set_ylabel("samples")
    axes[0, 0].legend(fontsize=8)

    scatter = axes[0, 1].scatter(raw_coords[:, 0], raw_coords[:, 1], c=owner, s=4,
                                 cmap="tab20", alpha=0.7)
    axes[0, 1].set_title(f"PCA of raw latent, coloured by writer "
                         f"(PC1+PC2 = {raw_explained.sum() * 100:.1f}% var)")
    fig.colorbar(scatter, ax=axes[0, 1], label="writer", fraction=0.046)

    scatter = axes[1, 0].scatter(dir_coords[:, 0], dir_coords[:, 1], c=owner, s=4,
                                 cmap="tab20", alpha=0.7)
    axes[1, 0].set_title(f"PCA of unit latent (direction only), by writer "
                         f"(PC1+PC2 = {dir_explained.sum() * 100:.1f}%)")
    fig.colorbar(scatter, ax=axes[1, 0], label="writer", fraction=0.046)

    shown = characters[:args.highlight]
    axes[1, 1].scatter(dir_coords[:, 0], dir_coords[:, 1], c="lightgrey", s=3, alpha=0.5)
    for i, c in enumerate(shown):
        rows = index[c]
        axes[1, 1].scatter(dir_coords[rows, 0], dir_coords[rows, 1], s=28, label=names[c])
    axes[1, 1].set_title("same directions, %d characters highlighted" % len(shown))
    axes[1, 1].legend(fontsize=8, markerscale=1.0)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle(f"{Path(args.checkpoint).parent.name} step {checkpoint['step']}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"))
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
