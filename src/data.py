"""CASIA GNT parsing, glyph preprocessing, and writer-conditioned episodes."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import struct

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset


def iter_gnt(path: Path):
    """Read raw label bytes as GBK; numeric fields alone are little-endian."""
    with path.open("rb") as stream:
        file_size = path.stat().st_size
        while stream.tell() < file_size:
            offset = stream.tell()
            header = stream.read(10)
            if len(header) != 10:
                raise ValueError(f"{path}:{offset}: truncated GNT header")
            size, label, width, height = struct.unpack("<I2sHH", header)
            if not width or not height or size != 10 + width * height:
                raise ValueError(f"{path}:{offset}: invalid GNT dimensions/record size")
            if size > file_size - offset:
                raise ValueError(f"{path}:{offset}: truncated GNT bitmap")
            bitmap = stream.read(width * height)
            try:
                character = label.decode("gbk").rstrip("\x00")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}:{offset}: invalid GBK label {label.hex()}") from exc
            yield character, Image.frombytes("L", (width, height), bitmap)


def normalize_image(image: Image.Image, size: int, margin: int) -> np.ndarray:
    """Crop background, preserve aspect ratio, center on a white square."""
    if margin < 0 or size - 2 * margin < 1:
        raise ValueError("margin must leave at least one foreground pixel")
    image = image.convert("L")
    bbox = ImageChops.invert(image).getbbox()
    if bbox is None:
        raise ValueError("blank glyph")
    image = image.crop(bbox)
    scale = (size - 2 * margin) / max(image.size)
    shape = tuple(max(1, round(d * scale)) for d in image.size)
    image = image.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def render_character(character: str, font: ImageFont.FreeTypeFont,
                     size: int, margin: int) -> np.ndarray:
    left, top, right, bottom = font.getbbox(character)
    if right <= left or bottom <= top:
        raise ValueError(f"empty font glyph: {character!r}")
    canvas = Image.new("L", (right - left, bottom - top), 255)
    ImageDraw.Draw(canvas).text((-left, -top), character, font=font, fill=0)
    return normalize_image(canvas, size, margin)


def read_manifest(root: Path) -> tuple[dict, str]:
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != 1:
        raise ValueError("unsupported prepared-data version")
    writers = manifest["writers"]
    ids = [w["id"] for w in writers]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate writer IDs in manifest")
    if {w["split"] for w in writers} != {"train", "val"}:
        raise ValueError("manifest must contain train and val writers")
    return manifest, hashlib.sha256(raw).hexdigest()


class HandwritingEpisodes(Dataset):
    """Stateless sampling: an index always determines the same whole episode.

    A global training index is never reused, including after resume. This makes
    sampling independent of worker count and DataLoader prefetch progress.
    """

    def __init__(self, root: str | Path, split: str, *, length: int,
                 min_references: int, max_references: int, seed: int):
        self.root = Path(root)
        self.manifest, self.fingerprint = read_manifest(self.root)
        if not 1 <= min_references <= max_references:
            raise ValueError("expected 1 <= min_references <= max_references")
        if length < 1:
            raise ValueError("episode count must be positive")
        self.writers = [w for w in self.manifest["writers"] if w["split"] == split]
        if not self.writers:
            raise ValueError(f"no writers in {split}")
        for writer in self.writers:
            labels = writer["labels"]
            if len(set(labels)) != len(labels) or len(labels) <= max_references:
                raise ValueError(f"writer {writer['id']} needs distinct target + references")
        self.length = length
        self.min_references = min_references
        self.max_references = max_references
        self.seed = seed
        self._images: OrderedDict = OrderedDict()
        self._standards = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = OrderedDict()
        state["_standards"] = None
        return state

    def close(self):
        """Release this process's memory maps (also useful on Windows)."""
        self._images.clear()
        self._standards = None

    def __getitem__(self, index: int):
        rng = random.Random(f"{self.seed}:{index}")
        writer_index = rng.randrange(len(self.writers))
        writer = self.writers[writer_index]
        if writer_index not in self._images:
            self._images[writer_index] = np.load(self.root / writer["file"], mmap_mode="r")
            if len(self._images) > 8:
                self._images.popitem(last=False)
        self._images.move_to_end(writer_index)
        images = self._images[writer_index]
        if self._standards is None:
            self._standards = np.load(self.root / "standards.npy", mmap_mode="r")
        count = rng.randint(self.min_references, self.max_references)
        selected = rng.sample(range(len(writer["labels"])), count + 1)
        target, references = selected[0], selected[1:]
        labels = writer["labels"]

        def tensor(array):
            return torch.from_numpy(np.array(array, copy=True)).float().unsqueeze(0) / 127.5 - 1

        size = self.manifest["image_size"]
        ref_hand = torch.ones(self.max_references, 1, size, size)
        ref_std = torch.ones_like(ref_hand)
        for slot, sample in enumerate(references):
            ref_hand[slot] = tensor(images[sample])
            ref_std[slot] = tensor(self._standards[labels[sample]])
        mask = torch.arange(self.max_references) < count
        return {
            "target_standard": tensor(self._standards[labels[target]]),
            "target_handwriting": tensor(images[target]),
            "reference_handwriting": ref_hand,
            "reference_standard": ref_std,
            "reference_mask": mask,
        }
