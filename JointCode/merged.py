"""Dataset over merged latent and image files (one reader per split)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MergedDataset(Dataset):
    """Read a single contiguous latent file and a single image file."""

    def __init__(self, root, split="train"):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if split not in manifest["splits"]:
            raise ValueError(f"split {split} missing from {self.root}")
        entry = manifest["splits"][split]
        self.latent = np.load(self.root / entry["latent"], mmap_mode="r")
        self.image = np.load(self.root / entry["image"], mmap_mode="r")
        expected = tuple(manifest["latent_shape"])
        if self.latent.dtype != np.float16 or tuple(self.latent.shape[1:]) != expected:
            raise ValueError("merged latent file has the wrong dtype or shape")
        if self.image.dtype != np.uint8 or self.image.ndim != 3:
            raise ValueError("merged image file has the wrong dtype or shape")
        if len(self.latent) != len(self.image):
            raise ValueError("merged latent and image files differ in length")
        self.channels = expected[0] // 2

    def __len__(self):
        return len(self.latent)

    def close(self):
        """Release the memory maps (needed on Windows before deleting the files)."""
        for array in (self.latent, self.image):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __getitem__(self, index):
        latent = self.latent[index]
        return {
            "content": torch.from_numpy(np.asarray(latent[:self.channels], np.float32)),
            "author": torch.from_numpy(np.asarray(latent[self.channels:], np.float32)),
            "image": torch.from_numpy(np.asarray(self.image[index], np.uint8))[None].float(),
        }
