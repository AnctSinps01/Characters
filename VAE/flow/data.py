"""Dataset that turns pre-encoded blocked-VAE codes into flow training pairs.

Each item carries the standard glyph's content channels, a random number of
author channels taken from *other* characters of the same writer, and the target
handwriting code.  Reference count varies per item, so the collate function pads
and returns the true count for masked pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_latent(cls, latent):
        data = torch.from_numpy(np.asarray(latent[:200_000], np.float32))
        return cls(mean=data.mean(dim=(0, 2, 3)), std=data.std(dim=(0, 2, 3)).clamp_min(1e-3))

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class LatentSet:
    """Memory-mapped codes for one split, with per-writer row indexes."""

    def __init__(self, root, split, stats=None):
        root = Path(root)
        self.latent = np.load(root / f"{split}_latent.npy", mmap_mode="r")
        self.label = np.load(root / f"{split}_label.npy", mmap_mode="r")
        self.owner = np.load(root / f"{split}_owner.npy", mmap_mode="r")
        info = json.loads((root / f"{split}_writers.json").read_text(encoding="utf-8"))
        self.characters = info["characters"]
        self.rows = [np.arange(r["start"], r["start"] + r["length"]) for r in info["ranges"]]
        self.standards = np.load(root / "standards_latent.npy", mmap_mode="r")
        self.stats = stats

    def __len__(self):
        return len(self.latent)

    def codes(self, rows):
        return torch.from_numpy(np.asarray(self.latent[rows], np.float32))


class FlowDataset(Dataset):
    def __init__(self, data: LatentSet, ref_min=1, ref_max=8, seed=0):
        self.data = data
        self.ref_min = ref_min
        self.ref_max = ref_max
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data
        owner = int(data.owner[index])
        character = int(data.label[index])
        target = torch.from_numpy(np.asarray(data.latent[index], np.float32))
        standard = torch.from_numpy(np.asarray(data.standards[character], np.float32))
        rows = data.rows[owner]
        count = int(torch.randint(self.ref_min, self.ref_max + 1, ()).item())
        choices = rows[np.random.randint(0, len(rows), size=count * 3)]
        choices = choices[np.asarray(data.label[choices]) != character][:count]
        if len(choices) == 0:
            choices = rows[:1]
        refs = torch.from_numpy(np.asarray(data.latent[choices], np.float32))
        return {"standard": standard[:2], "target": target[:3], "refs": refs[:, 2:3],
                "character": character, "owner": owner}


def collate(items):
    counts = torch.tensor([item["refs"].shape[0] for item in items])
    widest = int(counts.max())
    refs = torch.zeros(len(items), widest, 1, 8, 8)
    for row, item in enumerate(items):
        refs[row, :item["refs"].shape[0]] = item["refs"]
    return {"standard": torch.stack([i["standard"] for i in items]),
            "target": torch.stack([i["target"] for i in items]),
            "refs": refs, "counts": counts,
            "character": torch.tensor([i["character"] for i in items]),
            "owner": torch.tensor([i["owner"] for i in items])}
