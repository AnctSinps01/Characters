"""选中字的标准图和全部书写者样本，一次载入内存。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from twist import ModelConfig, LossConfig, distance_fields
from twist.geometry import composed_grid, constrain_displacement, identity_grid, sample_image
from settings import TrainingConfig


def read_glyph(path: Path, index: int, expected_count: int, prepared_size: int,
               output_size: int) -> torch.Tensor:
    # mmap 只复制被选中的一张图；不把完整 writer/标准字数组读进内存。
    images = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if images.dtype != np.uint8 or images.shape != (expected_count, prepared_size, prepared_size):
            raise ValueError(f"{path}: expected uint8 {(expected_count, prepared_size, prepared_size)}, "
                             f"got {images.dtype} {images.shape}")
        pixels = np.array(images[index], copy=True)
    finally:
        del images
    if prepared_size != output_size:
        pixels = np.asarray(Image.fromarray(pixels).resize(
            (output_size, output_size), Image.Resampling.LANCZOS)).copy()
    return torch.from_numpy(pixels).float()[None, None] / 127.5 - 1


def synthetic_target(source: torch.Tensor, model: ModelConfig,
                     training: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """固定的温和可逆形变；真值坐标只用来诊断，不加入训练 loss。"""
    nodes = identity_grid(1, (model.control_grid_size, model.control_grid_size),
                          device=torch.device("cpu"))
    x, y = nodes[..., 0], nodes[..., 1]
    window = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    rng = torch.Generator().manual_seed(training.seed)
    coefficients = torch.randn(2, 3, generator=rng)
    basis = torch.stack((torch.ones_like(x), torch.cos(math.pi * x), torch.cos(math.pi * y)))
    raw = torch.einsum("cb,bnhw->nchw", coefficients, basis) * window[:, None]
    maximum = torch.linalg.vector_norm(raw, dim=1).amax().clamp_min(1e-8)
    raw = raw * (training.synthetic_displacement / maximum)
    field = constrain_displacement(raw, min(training.synthetic_lipschitz, model.lipschitz_limit))
    grid = composed_grid((field,), source)
    return sample_image(source, grid), grid, field


def load_glyphs(root: Path, model: ModelConfig, loss: LossConfig, training: TrainingConfig,
                writer_id: str | None = None) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}. "
                                "Use --data /path/to/Characters/data/prepared.")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") != 1:
        raise ValueError("expected prepared-data manifest version 1")
    characters = manifest["characters"]
    try:
        label = characters.index(training.character)
    except ValueError as exc:
        raise ValueError(f"Character {training.character!r} is not prepared; "
                         f"examples: {''.join(characters[:30])}") from exc
    size = manifest["image_size"]
    if model.in_channels != 1:
        raise ValueError("single-glyph training requires model.in_channels=1")
    source = read_glyph(root / "standards.npy", label, len(characters), size, model.image_size)
    metadata = {
        "character": training.character, "mode": training.mode, "data_root": str(root),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "prepared_image_size": size,
        "image_size": model.image_size,
    }
    writers = sorted([w for w in manifest["writers"] if label in w["labels"]], key=lambda w: w["id"])
    if not writers:
        raise ValueError(f"No writers contain {training.character!r}")
    if len({w["id"] for w in writers}) != len(writers):
        raise ValueError("duplicate writer IDs")
    handwriting = []
    for i, writer in enumerate(writers):
        if writer["split"] not in ("train", "val"):
            raise ValueError(f"invalid split for writer {writer['id']}")
        if writer["labels"].count(label) != 1:
            raise ValueError(f"duplicate character for writer {writer['id']}")
        handwriting.append(read_glyph(root / writer["file"], writer["labels"].index(label),
                                      len(writer["labels"]), size, model.image_size))
        if (i + 1) % 50 == 0 or i + 1 == len(writers):
            print(f"Loaded {training.character}: {i+1}/{len(writers)} writers", flush=True)
    handwriting = torch.cat(handwriting)
    all_writer_ids = [w["id"] for w in writers]
    metadata.update(loaded_writers=all_writer_ids,
                    handwriting_count=len(writers),
                    loaded_pixel_mib=(handwriting.numel() + source.numel()) * 4 / 1024**2)
    true_grid = true_field = None
    if training.mode == "synthetic":
        if writer_id is not None:
            raise ValueError("--writer is only valid with --mode handwriting")
        targets, true_grid, true_field = synthetic_target(source, model, training)
        train_indices, val_indices, target_ids = [0], [], ["synthetic"]
    else:
        targets, target_ids = handwriting, all_writer_ids
        if writer_id is not None:
            if writer_id not in all_writer_ids:
                raise ValueError(f"Writer {writer_id!r} does not contain {training.character!r}")
            train_indices, val_indices = [all_writer_ids.index(writer_id)], []
            metadata["overfit_writer"] = writer_id
        else:
            train_indices = [i for i, w in enumerate(writers) if w["split"] == "train"]
            val_indices = [i for i, w in enumerate(writers) if w["split"] == "val"]
    if not train_indices:
        raise ValueError(f"No train writers contain {training.character!r}")
    # 一次准备，训练与续训都使用缓存；每张 B 必须也有墨迹。
    if not bool((targets <= loss.distance_threshold).flatten(1).any(dim=1).all()):
        raise ValueError("some targets have no foreground at distance_threshold")
    metadata.update(train_count=len(train_indices), val_count=len(val_indices),
                    train_ids=[target_ids[i] for i in train_indices],
                    val_ids=[target_ids[i] for i in val_indices])
    return {"source": source, "targets": targets, "handwriting": handwriting,
            "train_indices": torch.tensor(train_indices), "val_indices": torch.tensor(val_indices, dtype=torch.long),
            "target_ids": target_ids,
            "distance": distance_fields(source, threshold=loss.distance_threshold),
            "true_grid": true_grid, "true_field": true_field, "metadata": metadata}
