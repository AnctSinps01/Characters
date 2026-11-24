"""Pre-computed VAE latent means and the reference/target dataset on top of them."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from VAE.data import GlyphStore


class LatentStore:
    """Read per-writer latent means written by ``Ref2Latent.prepare_latents``."""

    def __init__(self, root, cache_size=8):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.shape = tuple(self.manifest["latent_shape"])
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._standards = None

    def _open(self, name, count):
        array = np.load(self.root / name, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float16 or array.shape != (count, *self.shape):
            raise ValueError(f"{name}: wrong latent dtype or shape")
        return array

    def writer(self, writer):
        key = writer["id"]
        if key not in self._cache:
            self._cache[key] = self._open(f"{key}.npy", len(writer["labels"]))
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old._mmap.close()
        return self._cache[key]

    @property
    def standards(self):
        if self._standards is None:
            self._standards = self._open("standards.npy", self.manifest["classes"])
        return self._standards

    def close(self):
        for array in self._cache.values():
            array._mmap.close()
        self._cache.clear()
        if self._standards is not None:
            self._standards._mmap.close()
            self._standards = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_cache=OrderedDict(), _standards=None)
        return state


class RefTargetDataset(Dataset):
    """All (writer, character) pairs of one split, with on-the-fly references.

    References are other characters of the same writer, chosen by a fixed
    per-item RNG so the same index always yields the same references.
    """
    def __init__(self, store: GlyphStore, latents: LatentStore, split="train",
                 references=4, seed=42):
        self.store = store
        self.latents = latents
        self.writers = [w for w in store.writers if w["split"] == split]
        if not self.writers:
            raise ValueError(f"empty {split} split")
        self.references = int(references)
        self.seed = seed
        if self.references < 1:
            raise ValueError("references must be positive")
        self.ends = np.cumsum([len(w["labels"]) for w in self.writers]).tolist()
        self.pairs = []
        # (writer index, row, class) flattened in writer order.
        for wi, writer in enumerate(self.writers):
            for row, label in enumerate(writer["labels"]):
                self.pairs.append((wi, row, label))
        if len(self.pairs) < 2:
            raise ValueError("split is too small")

    def __len__(self):
        return len(self.pairs)

    def reference_rows(self, writer_index, row, label):
        writer = self.writers[writer_index]
        rng = random.Random(f"{self.seed}:refs:{writer['id']}:{label}")
        candidates = [i for i in range(len(writer["labels"])) if i != row]
        if len(candidates) < self.references:
            raise ValueError("writer has too few characters for the requested references")
        return rng.sample(candidates, self.references)

    def __getitem__(self, index):
        writer_index, row, label = self.pairs[index]
        writer = self.writers[writer_index]
        rows = self.reference_rows(writer_index, row, label)
        reference = torch.from_numpy(
            self.latents.writer(writer)[rows].astype(np.float32))
        standard = torch.from_numpy(
            self.latents.standards[label].astype(np.float32))
        target = torch.from_numpy(
            self.latents.writer(writer)[row].astype(np.float32))
        image = self.store.writer_images(writer)[row]
        return {"reference": reference, "standard": standard, "target": target,
                "image": torch.from_numpy(image.copy())[None].float(),
                "class_index": label, "writer_id": writer["id"]}
