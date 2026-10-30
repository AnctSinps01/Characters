"""Prepared-dataset loading and writer-conditioned episodes."""
from __future__ import annotations

import hashlib
import json
from multiprocessing import RawArray
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


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
    """Prepared pixels for all or selected writers, loaded once as uint8.

    On CPU, RawArray's backing is inherited by fork workers and passed by handle
    to spawn workers. CUDA storage is used only from the training process.
    The standard-glyph bank is shared; prepared NPY files are closed before
    any episode is sampled.
    """

    def __init__(self, root: str | Path, *, device: str | torch.device = "cpu",
                 verbose: bool = False, writer_ids: tuple[str, ...] | None = None):
        self.root = Path(root).resolve()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("prepared glyphs can only be preloaded to cpu or cuda")
        self.manifest, self.fingerprint = read_manifest(self.root)
        self.image_size = self.manifest["image_size"]
        self.standard_count = len(self.manifest["characters"])
        writers = self.manifest["writers"]
        if writer_ids is not None:
            requested = set(writer_ids)
            unknown = requested - {writer["id"] for writer in writers}
            if not requested or unknown:
                raise ValueError(f"writer_ids must select known writers; unknown={sorted(unknown)}")
            writers = [writer for writer in writers if writer["id"] in requested]
        self.writer_ranges = {}
        self.handwriting_count = 0
        for writer in writers:
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
            scope = "all train/val" if writer_ids is None else "selected writer"
            print(f"Preloading {scope} glyphs: {self.handwriting_count:,} handwriting + "
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
                 preloaded: PreloadedGlyphs | None = None,
                 writer_id: str | None = None,
                 writer_ids: tuple[str, ...] | None = None,
                 writer_sample_indices: list[int] | None = None,
                 writer_sample_indices_by_id: dict[str, list[int]] | None = None):
        self.root = Path(root)
        self._preloaded = preloaded if preloaded is not None else PreloadedGlyphs(self.root)
        if self._preloaded.root != self.root.resolve():
            raise ValueError("preloaded glyphs belong to a different data directory")
        self.manifest = self._preloaded.manifest
        self.fingerprint = self._preloaded.fingerprint
        if not 1 <= min_references <= max_references:
            raise ValueError("expected 1 <= min_references <= max_references")
        if length < 1:
            raise ValueError("episode count must be positive")
        if writer_id is not None and writer_ids is not None:
            raise ValueError("writer_id and writer_ids are mutually exclusive")
        if writer_sample_indices is not None and writer_sample_indices_by_id is not None:
            raise ValueError("writer_sample_indices and writer_sample_indices_by_id are mutually exclusive")
        if writer_id is None and writer_ids is None:
            self.writers = [w for w in self.manifest["writers"] if w["split"] == split]
        elif writer_id is not None:
            # Fitting diagnostics deliberately evaluate the same writer and
            # character pool as training, with independently seeded episodes.
            self.writers = [w for w in self.manifest["writers"] if w["id"] == writer_id]
        else:
            requested = tuple(writer_ids)
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("writer_ids must be a non-empty sequence of distinct IDs")
            by_id = {writer["id"]: writer for writer in self.manifest["writers"]}
            unknown = [writer_id for writer_id in requested if writer_id not in by_id]
            if unknown:
                raise ValueError(f"writer_ids contains unknown IDs: {unknown}")
            # A selected train pool is intentionally reused for validation, just
            # like the existing single-author fitting mode.
            self.writers = [by_id[writer_id] for writer_id in requested]
        if not self.writers:
            raise ValueError(f"no writers for split={split}, writer_id={writer_id}")
        for writer in self.writers:
            if writer["id"] not in self._preloaded.writer_ranges:
                raise ValueError(f"writer {writer['id']} has not been preloaded")
            labels = writer["labels"]
            if len(set(labels)) != len(labels) or len(labels) <= max_references:
                raise ValueError(f"writer {writer['id']} needs distinct target + references")
        self.writer_sample_indices = None
        self.writer_sample_indices_by_id = None
        if writer_sample_indices is not None:
            if writer_id is None:
                raise ValueError("writer_sample_indices requires a single writer_id")
            indices = tuple(writer_sample_indices)
            if (len(indices) <= max_references or len(set(indices)) != len(indices)
                    or any(type(index) is not int or not 0 <= index < len(self.writers[0]["labels"])
                           for index in indices)):
                raise ValueError("writer subset needs distinct, valid target + reference indices")
            self.writer_sample_indices = indices
        elif writer_sample_indices_by_id is not None:
            if writer_id is not None:
                raise ValueError("writer_sample_indices_by_id requires writer_ids")
            selected = {}
            for writer in self.writers:
                if writer["id"] not in writer_sample_indices_by_id:
                    raise ValueError(f"missing character subset for writer {writer['id']}")
                indices = tuple(writer_sample_indices_by_id[writer["id"]])
                if (len(indices) <= max_references or len(set(indices)) != len(indices)
                        or any(type(index) is not int or not 0 <= index < len(writer["labels"])
                               for index in indices)):
                    raise ValueError(f"writer subset needs distinct, valid target + reference indices: "
                                     f"{writer['id']}")
                selected[writer["id"]] = indices
            self.writer_sample_indices_by_id = selected
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
        if self.writer_sample_indices is not None:
            pool = self.writer_sample_indices
        elif self.writer_sample_indices_by_id is not None:
            pool = self.writer_sample_indices_by_id[writer["id"]]
        else:
            pool = range(len(writer["labels"]))
        selected = rng.sample(pool, count + 1)
        labels = writer["labels"]

        # Gather the whole episode at once instead of converting each glyph in
        # a Python loop. Advanced indexing creates only the small episode copy.
        hands = images[selected]
        standards = self._preloaded.standards[[labels[index] for index in selected]]
        if isinstance(hands, np.ndarray):
            hands = torch.from_numpy(hands)
            standards = torch.from_numpy(standards)
        hands = hands.float().unsqueeze(1) / 127.5 - 1
        standards = standards.float().unsqueeze(1) / 127.5 - 1

        size = self.manifest["image_size"]
        ref_hand = hands.new_ones(self.max_references, 1, size, size)
        ref_hand[:count] = hands[1:]
        ref_standard = standards.new_ones(self.max_references, 1, size, size)
        ref_standard[:count] = standards[1:]
        mask = torch.arange(self.max_references, device=hands.device) < count
        return {
            "target_standard": standards[0],
            "target_handwriting": hands[0],
            "reference_handwriting": ref_hand,
            "reference_standard": ref_standard,
            "reference_mask": mask,
        }
