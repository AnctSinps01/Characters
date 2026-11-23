"""Read shared prepared data; no dependency on DiT-flow or OCR packages."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class GlyphStore:
    """Read-only NPY mappings, with a bounded per-process writer-file cache.

    Pixels stay uint8 until a batch is on the training device. Spawn workers
    reopen mappings, avoiding a separate multi-GB dataset copy per worker.
    """
    def __init__(self, root, cache_size=8):
        self.root = Path(root).resolve()
        raw = (self.root / "manifest.json").read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("version") != 1:
            raise ValueError("unsupported prepared-data version")
        self.image_size = self.manifest["image_size"]
        self.characters = self.manifest["characters"]
        self.writers = self.manifest["writers"]
        if not self.characters or not self.writers or cache_size < 1:
            raise ValueError("empty data or invalid cache size")
        seen = set()
        for w in self.writers:
            if w["id"] in seen or w["split"] not in ("train", "val"):
                raise ValueError("duplicate writer ID or unknown split")
            seen.add(w["id"])
            self._safe_path(w["file"])
            if not w["labels"] or any(type(c) is not int or not 0 <= c < len(self.characters)
                                      for c in w["labels"]):
                raise ValueError("invalid writer labels")
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._standards = None

    def _safe_path(self, relative):
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("NPY path escapes dataset directory")
        return path

    def _open(self, relative, count):
        data = np.load(self._safe_path(relative), mmap_mode="r", allow_pickle=False)
        if data.dtype != np.uint8 or data.shape != (count, self.image_size, self.image_size):
            raise ValueError(f"{relative}: wrong NPY dtype or shape")
        return data

    def writer_images(self, writer):
        key = writer["id"]
        if key not in self._cache:
            self._cache[key] = self._open(writer["file"], len(writer["labels"]))
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old._mmap.close()
        return self._cache[key]

    def close(self):
        """Release this process's mappings; returned image tensors are copies."""
        for array in self._cache.values():
            array._mmap.close()
        self._cache.clear()
        if self._standards is not None:
            self._standards._mmap.close()
            self._standards = None

    @property
    def standards(self):
        if self._standards is None:
            self._standards = self._open("standards.npy", len(self.characters))
        return self._standards

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_cache=OrderedDict(), _standards=None)
        return state


class ReconstructionDataset(Dataset):
    """Exact indexed validation views or stateless random training mixture.

    standard_ratio is the probability that a training item is a uniformly chosen
    printed class instead of a uniformly chosen handwriting image. This is a
    batch share in expectation, not a weight or a fraction of unique classes.
    The same index/seed always yields the same source image, even on resume.
    """
    def __init__(self, store: GlyphStore, split="train", *, standard_ratio=0.0,
                 length=None, seed=42, source="mixed"):
        if split not in ("train", "val") or source not in ("mixed", "handwriting", "standard"):
            raise ValueError("unknown split/source")
        if not np.isfinite(standard_ratio) or not 0 <= standard_ratio <= 1:
            raise ValueError("standard_ratio must be in [0, 1]")
        if source == "mixed" and (length is None or length < 1):
            raise ValueError("mixed training requires a positive length")
        self.store, self.seed, self.source = store, seed, source
        self.standard_ratio = standard_ratio
        self.writers = [w for w in store.writers if w["split"] == split]
        if not self.writers:
            raise ValueError(f"empty {split} writer split")
        self.ends = np.cumsum([len(w["labels"]) for w in self.writers]).tolist()
        self.handwriting_count = self.ends[-1]
        self.length = int(length) if source == "mixed" else (
            len(store.characters) if source == "standard" else self.handwriting_count)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.source == "mixed":
            # Independent RNGs keep the handwriting choice identical across
            # ratio=0/nonzero experiments for indices that remain handwritten.
            choice = random.Random(f"{self.seed}:source:{index}").random()
            printed = choice < self.standard_ratio
            rng = random.Random(f"{self.seed}:item:{index}")
            item = rng.randrange(len(self.store.characters) if printed else self.handwriting_count)
        else:
            printed, item = self.source == "standard", index
        if printed:
            image = self.store.standards[item]
            label, writer = item, "standard"
        else:
            wi = bisect_right(self.ends, item)
            row = item - (self.ends[wi - 1] if wi else 0)
            w = self.writers[wi]
            image = self.store.writer_images(w)[row]
            label, writer = w["labels"][row], w["id"]
        return {"image": torch.from_numpy(image.copy())[None],
                "is_standard": printed, "class_index": label, "writer_id": writer}


def spread_indices(length, maximum):
    """Fixed validation subset spread over all authors; maximum=0 means all."""
    if maximum <= 0 or maximum >= length:
        return range(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()
