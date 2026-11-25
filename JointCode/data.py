"""Pairs of (content code, author code) aligned by writer and character."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from Ref2Latent.data import LatentStore
from VAE.data import GlyphStore


class JointCodeDataset(Dataset):
    """Every (writer, character) pair with both codes and the original image.

    The two latent stores must share the dataset layout: same writers, same
    character order per writer.
    """
    def __init__(self, store: GlyphStore, content: LatentStore, author: LatentStore,
                 split="train"):
        if tuple(content.shape) != tuple(author.shape):
            raise ValueError("content and author codes must have the same shape")
        self.store = store
        self.content = content
        self.author = author
        self.writers = [w for w in store.writers if w["split"] == split]
        if not self.writers:
            raise ValueError(f"empty {split} split")
        self.pairs = [(wi, row, label) for wi, w in enumerate(self.writers)
                      for row, label in enumerate(w["labels"])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        wi, row, label = self.pairs[index]
        writer = self.writers[wi]
        return {
            "content": torch.from_numpy(self.content.writer(writer)[row].astype(np.float32)),
            "author": torch.from_numpy(self.author.writer(writer)[row].astype(np.float32)),
            "image": torch.from_numpy(self.store.writer_images(writer)[row].copy())[None].float(),
            "class_index": label,
            "writer_id": writer["id"],
        }
