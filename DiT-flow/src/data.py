"""CASIA GNT parsing, glyph preprocessing, and writer-conditioned episodes."""
from __future__ import annotations

import hashlib
import json
from multiprocessing import RawArray
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


class PreloadedGlyphs:
    """All prepared pixels, eagerly loaded as uint8 on CPU or one CUDA device.

    On CPU, RawArray's backing is inherited by fork workers and passed by handle
    to spawn workers. CUDA storage is used only from the training process.
    Prepared NPY files are closed before any episode is sampled.
    """

    def __init__(self, root: str | Path, *, device: str | torch.device = "cpu",
                 verbose: bool = False):
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("prepared glyphs can only be preloaded to cpu or cuda")
        self.manifest, self.fingerprint = read_manifest(self.root)
        self.image_size = self.manifest["image_size"]
        self.standard_count = len(self.manifest["characters"])
        self.writer_ranges = {}
        self.handwriting_count = 0
        for writer in self.manifest["writers"]:
            count = len(writer["labels"])
            if any(type(label) is not int or not 0 <= label < self.standard_count
                   for label in writer["labels"]):
                raise ValueError(f"writer {writer['id']} has invalid standard-glyph indices")
            self.writer_ranges[writer["id"]] = (self.handwriting_count, count)
            self.handwriting_count += count
        self.nbytes = (self.handwriting_count + self.standard_count) * self.image_size**2
        if not self.nbytes:
            raise ValueError("prepared dataset has no pixels")
        if verbose:
            print(f"Preloading all train/val glyphs: {self.handwriting_count:,} handwriting + "
                  f"{self.standard_count:,} standards, {self.nbytes / 1024**3:.3f} GiB uint8 "
                  + ("in shared CPU memory (one copy for all workers)."
                     if self.device.type == "cpu" else f"on {self.device}."), flush=True)
        if self.device.type == "cpu":
            self._buffer = RawArray("B", self.nbytes)
            self._attach_views()
            self._handwriting.setflags(write=True)
            self.standards.setflags(write=True)
        else:
            self._buffer = None
            pixels = torch.empty(
                self.handwriting_count + self.standard_count,
                self.image_size, self.image_size, dtype=torch.uint8, device=self.device)
            self._handwriting = pixels[:self.handwriting_count]
            self.standards = pixels[self.handwriting_count:]
        # Load one writer at a time to avoid a second full dataset-sized copy.
        self._load_into(self.root / "standards.npy", self.standards)
        writers = self.manifest["writers"]
        for index, writer in enumerate(writers):
            start, count = self.writer_ranges[writer["id"]]
            self._load_into(self.root / writer["file"], self._handwriting[start:start + count])
            if verbose and ((index + 1) % 20 == 0 or index + 1 == len(writers)):
                print(f"Preloaded {index + 1}/{len(writers)} writers.", flush=True)
        if self.device.type == "cpu":
            self._handwriting.setflags(write=False)
            self.standards.setflags(write=False)

    @staticmethod
    def _load_into(path: Path, destination: np.ndarray | torch.Tensor):
        images = np.load(path, allow_pickle=False)
        expected_shape = tuple(destination.shape)
        if images.dtype != np.uint8 or images.shape != expected_shape:
            raise ValueError(f"{path}: expected uint8 {destination.shape}, "
                             f"got {images.dtype} {images.shape}")
        if isinstance(destination, np.ndarray):
            np.copyto(destination, images)
        else:
            destination.copy_(torch.from_numpy(images))

    def _attach_views(self):
        pixels = np.frombuffer(self._buffer, dtype=np.uint8).reshape(-1, self.image_size, self.image_size)
        self._handwriting = pixels[:self.handwriting_count]
        self.standards = pixels[self.handwriting_count:]
        self._handwriting.setflags(write=False)
        self.standards.setflags(write=False)

    def __getstate__(self):
        if self.device.type != "cpu":
            raise RuntimeError("CUDA-preloaded glyphs require DataLoader workers=0")
        state = self.__dict__.copy()
        state.pop("_handwriting")
        state.pop("standards")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._attach_views()

    def writer_images(self, writer_id: str) -> np.ndarray | torch.Tensor:
        start, count = self.writer_ranges[writer_id]
        return self._handwriting[start:start + count]


class HandwritingEpisodes(Dataset):
    """Stateless sampling: an index always determines the same whole episode.

    A global training index is never reused, including after resume. This makes
    sampling independent of worker count and DataLoader prefetch progress.
    """

    def __init__(self, root: str | Path, split: str, *, length: int,
                 min_references: int, max_references: int, seed: int,
                 preloaded: PreloadedGlyphs | None = None):
        self.root = Path(root)
        self._preloaded = preloaded if preloaded is not None else PreloadedGlyphs(self.root)
        if self._preloaded.root != self.root.resolve():
            raise ValueError("preloaded glyphs belong to a different data directory")
        self.manifest = self._preloaded.manifest
        self.fingerprint = self._preloaded.fingerprint
        self.writer_indices = {
            writer["id"]: index for index, writer in enumerate(self.manifest["writers"])
        }
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

    def __len__(self):
        return self.length

    def close(self):
        """Release this dataset's reference; other datasets/workers may share it."""
        self._preloaded = None

    def __getitem__(self, index: int):
        rng = random.Random(f"{self.seed}:{index}")
        writer_index = rng.randrange(len(self.writers))
        writer = self.writers[writer_index]
        if self._preloaded is None:
            raise RuntimeError("dataset is closed")
        images = self._preloaded.writer_images(writer["id"])
        count = rng.randint(self.min_references, self.max_references)
        selected = rng.sample(range(len(writer["labels"])), count + 1)
        labels = writer["labels"]

        # Gather the whole episode at once instead of converting each glyph in
        # a Python loop. Advanced indexing creates only the small episode copy.
        hands = images[selected]
        standard = self._preloaded.standards[labels[selected[0]]]
        if isinstance(hands, np.ndarray):
            hands = torch.from_numpy(hands)
            standard = torch.from_numpy(standard.copy())
        hands = hands.float().unsqueeze(1) / 127.5 - 1
        standard = standard.float().unsqueeze(0) / 127.5 - 1

        size = self.manifest["image_size"]
        ref_hand = hands.new_ones(self.max_references, 1, size, size)
        ref_hand[:count] = hands[1:]
        mask = torch.arange(self.max_references, device=hands.device) < count
        return {
            "target_standard": standard,
            "target_handwriting": hands[0],
            "reference_handwriting": ref_hand,
            "reference_mask": mask,
            # A stable integer makes author-pair masks cheap in the training
            # loss while keeping string IDs out of device transfer/collation.
            "author_index": torch.tensor(
                self.writer_indices[writer["id"]], dtype=torch.long, device=hands.device
            ),
        }
