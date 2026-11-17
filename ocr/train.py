"""Train a CLS-token ViT OCR model on ``data/prepared``.

Run from the repository root:

    python -m ocr.train --data data/prepared --output runs/ocr/vit-s8-12x384
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_prepared_data
from .model import ViTConfig, VisionTransformerOCR


@dataclass(frozen=True, slots=True)
class TrainConfig:
    total_steps: int = 1_000_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5_000
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    workers: int = 4
    validate_every: int = 5_000
    val_batches: int = 256
    save_every: int = 5_000
    log_every: int = 100
    precision: str = "bf16"
    seed: int = 42


def learning_rate(config: TrainConfig, step: int) -> float:
    if step <= config.warmup_steps:
        factor = step / max(config.warmup_steps, 1)
    else:
        progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        factor = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
    return config.learning_rate * factor


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(model: VisionTransformerOCR, loader: DataLoader, device: torch.device,
             precision: str, max_batches: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct1 = correct5 = samples = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            if index >= max_batches:
                break
            images = images.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            batch = labels.shape[0]
            loss_sum += loss.item() * batch
            correct1 += (logits.argmax(dim=1) == labels).sum().item()
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
            samples += batch
    model.train(was_training)
    if samples == 0:
        raise RuntimeError("validation loader produced no samples")
    return {"loss": loss_sum / samples,
            "top1": correct1 / samples,
            "top5": correct5 / samples,
            "samples": samples}


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    train_data, val_data, characters, data_fingerprint = load_prepared_data(args.data)
    model_config = ViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=len(characters),
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        workers=args.workers,
        validate_every=args.validate_every,
        val_batches=args.val_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        precision=args.precision,
        seed=args.seed,
    )
    if train_config.total_steps <= 0 or train_config.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if not 0 <= train_config.warmup_steps < train_config.total_steps:
        raise ValueError("warmup-steps must be in [0, steps)")
    if train_config.workers < 0 or train_config.val_batches < 1:
        raise ValueError("workers must be nonnegative and val-batches positive")
    if train_config.precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "last.pt"
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("data_fingerprint") != data_fingerprint:
            raise ValueError("resume checkpoint was trained on different prepared data")
        if resume.get("model_config") != asdict(model_config):
            raise ValueError("resume checkpoint model config does not match")
    elif any(output.iterdir()):
        raise ValueError("output is nonempty; use --resume or choose a new output directory")

    model = VisionTransformerOCR(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        betas=(0.9, 0.999), weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and train_config.precision == "fp16")
    start_step = 0
    best_top1 = 0.0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume.get("scaler", {}))
        start_step = int(resume["step"])
        best_top1 = float(resume.get("best_top1", 0.0))
        if resume.get("train_config") != asdict(train_config):
            raise ValueError("resume checkpoint train config does not match")

    common_loader = dict(
        batch_size=train_config.batch_size,
        num_workers=train_config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=train_config.workers > 0,
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **common_loader)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **common_loader)
    train_iterator = iter(train_loader)
    criterion = nn.CrossEntropyLoss()
    metrics_path = output / "metrics.jsonl"
    if not resume:
        (output / "characters.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")
        (output / "run.json").write_text(json.dumps({
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "classes": len(characters),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "data_fingerprint": data_fingerprint,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"device={device}; classes={len(characters)}; image={model_config.image_size}; "
          f"patch={model_config.patch_size}; tokens={model_config.token_count}; "
          f"depth={model_config.depth}; width={model_config.hidden_size}; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"train={len(train_data):,}; val={len(val_data):,}; "
          f"step={start_step}->{train_config.total_steps}", flush=True)

    model.train()
    running_loss = running_correct = running_samples = 0
    started = time.monotonic()
    for step in range(start_step + 1, train_config.total_steps + 1):
        try:
            images, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, labels = next(train_iterator)
        images = images.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
        labels = labels.to(device, non_blocking=True)
        lr = learning_rate(train_config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_config.precision):
            logits = model(images)
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite OCR loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite OCR gradient norm at step {step}")
        scaler.step(optimizer)
        scaler.update()

        count = labels.numel()
        running_loss += loss.item() * count
        running_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        running_samples += count
        should_validate = step % train_config.validate_every == 0 or step == train_config.total_steps
        should_save = step % train_config.save_every == 0 or should_validate
        if step % train_config.log_every == 0 or should_validate:
            record = {
                "step": step,
                "train_loss": running_loss / running_samples,
                "train_top1": running_correct / running_samples,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - started,
            }
            running_loss = running_correct = running_samples = 0
            if should_validate:
                validation = evaluate(model, val_loader, device, train_config.precision,
                                      train_config.val_batches)
                record.update({f"val_{key}": value for key, value in validation.items()})
                if validation["top1"] > best_top1:
                    best_top1 = validation["top1"]
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)

        if should_save:
            state = {
                "step": step,
                "best_top1": best_top1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "characters": characters,
                "data_fingerprint": data_fingerprint,
            }
            save_checkpoint(latest_path, state)
            if should_validate and record["val_top1"] >= best_top1:
                save_checkpoint(output / "best.pt", state)

    print(f"completed {train_config.total_steps} steps; best val top1={best_top1:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/ocr/vit-s8-12x384"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=5_000)
    parser.add_argument("--val-batches", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)


if __name__ == "__main__":
    main()
