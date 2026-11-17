"""Load the repository's prepared handwriting images for character OCR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    """Indexed view over a shared uint8 image buffer."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        if images.ndim != 3 or images.dtype != np.uint8:
            raise ValueError("images must be uint8 [count, height, width]")
        if labels.shape != (len(images),) or labels.dtype != np.int64:
            raise ValueError("labels must be int64 [count]")
        if indices.ndim != 1 or indices.dtype != np.int64:
            raise ValueError("indices must be int64 [count]")
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = int(self.indices[item])
        # Copy a single glyph so DataLoader workers never expose the immutable
        # preloaded array to PyTorch as writable storage.
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0)
        return image, int(self.labels[index])


def load_prepared_data(root: str | Path):
    """Return train/validation datasets, class names, and manifest fingerprint.

    The uint8 image buffer is allocated once and filled one writer file at a
    time, keeping peak memory close to the final ~3.5 GiB rather than holding
    both source arrays and a concatenated copy.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest.get("version") != 1:
        raise ValueError("unsupported prepared-data version")
    image_size = manifest.get("image_size")
    characters = manifest.get("characters")
    writers = manifest.get("writers")
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("manifest image_size must be a positive integer")
    if not isinstance(characters, list) or len(characters) < 2:
        raise ValueError("manifest must contain at least two characters")
    if not isinstance(writers, list) or not writers:
        raise ValueError("manifest must contain writers")

    total = sum(len(writer.get("labels", [])) for writer in writers)
    if total == 0:
        raise ValueError("prepared dataset contains no handwriting images")
    images = np.empty((total, image_size, image_size), dtype=np.uint8)
    labels = np.empty(total, dtype=np.int64)
    train_indices: list[int] = []
    val_indices: list[int] = []
    offset = 0
    class_count = len(characters)

    for writer in writers:
        writer_labels = writer.get("labels")
        if (not isinstance(writer_labels, list)
                or any(type(label) is not int or not 0 <= label < class_count
                       for label in writer_labels)):
            raise ValueError(f"writer {writer.get('id')} has invalid class labels")
        if len(set(writer_labels)) != len(writer_labels):
            raise ValueError(f"writer {writer.get('id')} has duplicate class labels")
        relative = Path(writer["file"])
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"writer file escapes prepared-data root: {relative}")
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        count = len(writer_labels)
        if source.dtype != np.uint8 or source.shape != (count, image_size, image_size):
            raise ValueError(f"{path}: expected uint8 {(count, image_size, image_size)}, "
                             f"got {source.dtype} {source.shape}")
        images[offset:offset + count] = source
        labels[offset:offset + count] = writer_labels
        target = train_indices if writer.get("split") == "train" else val_indices
        if writer.get("split") not in ("train", "val"):
            raise ValueError(f"writer {writer.get('id')} has unknown split")
        target.extend(range(offset, offset + count))
        offset += count
        del source

    train_labels = labels[np.asarray(train_indices, dtype=np.int64)]
    val_labels = labels[np.asarray(val_indices, dtype=np.int64)]
    train_coverage = np.unique(train_labels)
    val_coverage = np.unique(val_labels)
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("prepared data must have nonempty train and val writer splits")
    if len(train_coverage) != class_count or len(val_coverage) != class_count:
        raise ValueError("both writer splits must cover every OCR class")

    images.setflags(write=False)
    images_fingerprint = hashlib.sha256(raw_manifest).hexdigest()
    train = CharacterDataset(images, labels, np.asarray(train_indices, dtype=np.int64))
    val = CharacterDataset(images, labels, np.asarray(val_indices, dtype=np.int64))
    return train, val, characters, images_fingerprint
