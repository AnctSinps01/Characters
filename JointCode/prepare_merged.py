"""Merge both latent spaces and the images into one contiguous file per split.

    python -m JointCode.prepare_merged --data data/prepared \
        --content-latents runs/latents/c1e3 --author-latents runs/latents/a1e3 \
        --output /dev/shm/jointcode

Layout per split (writer order, then row order, as in the prepared manifest):

    latent.npy  [N, 8, 8, 8] float16   first 4 channels content, last 4 author
    image.npy   [N, 64, 64]  uint8

One sequential file per split means a single reader instead of three memmaps
scattered over the network filesystem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Ref2Latent.data import LatentStore
from VAE.data import GlyphStore


def merge(store, content, author, writers):
    channels = content.shape[0] * 2
    side = content.shape[1]
    total = sum(len(w["labels"]) for w in writers)
    latents = np.empty((total, channels, side, side), np.float16)
    images = np.empty((total, store.image_size, store.image_size), np.uint8)
    cursor = 0
    for writer in writers:
        count = len(writer["labels"])
        latents[cursor:cursor + count, :content.shape[0]] = content.writer(writer)
        latents[cursor:cursor + count, content.shape[0]:] = author.writer(writer)
        images[cursor:cursor + count] = store.writer_images(writer)
        cursor += count
    return latents, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--content-latents", type=Path, required=True)
    parser.add_argument("--author-latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    store = GlyphStore(args.data)
    content = LatentStore(args.content_latents, cache_size=16)
    author = LatentStore(args.author_latents, cache_size=16)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"image_size": int(store.image_size),
                "latent_shape": [content.shape[0] * 2, *content.shape[1:]],
                "splits": {}}
    try:
        for split in args.splits.split(","):
            writers = [w for w in store.writers if w["split"] == split]
            latents, images = merge(store, content, author, writers)
            np.save(args.output / f"{split}_latent.npy", latents)
            np.save(args.output / f"{split}_image.npy", images)
            manifest["splits"][split] = {
                "samples": int(len(latents)),
                "writers": len(writers),
                "latent": f"{split}_latent.npy",
                "image": f"{split}_image.npy"}
            print(f"{split}: {latents.shape} latents, {images.shape} images", flush=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
    finally:
        store.close()
        content.close()
        author.close()


if __name__ == "__main__":
    main()
