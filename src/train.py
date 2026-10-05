"""Single-GPU Glyph2Hand training with deterministic episodes and resumable state."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from . import Glyph2Hand, ModelConfig, build_flow_matching_batch, flow_matching_loss, sample_euler
from .config import TrainingConfig
from .data import HandwritingEpisodes, read_manifest


def learning_rate(step: int, config: TrainingConfig) -> float:
    """One-based optimizer step, including linear warmup and cosine decay."""
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * factor


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batch_loss(model, batch):
    flow = build_flow_matching_batch(batch["target_standard"], batch["target_handwriting"], model.config.sigma)
    prediction = model(flow.state, flow.times, batch["target_standard"],
                       batch["reference_handwriting"], batch["reference_standard"],
                       batch["reference_mask"])
    # Accumulate the MSE in float32 even under autocast.
    return flow_matching_loss(prediction.float(), flow.target_velocity.float())


def optimizer_update(model, optimizer, scaler, batches, device, precision, config):
    """Retry FP16 overflow at a lower scale using the same episodes and noise."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(16):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for raw in batches:
            batch = move_batch(raw, device)
            with autocast_context(device, precision):
                loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss; last checkpoint preserved")
            total_loss += loss.detach().item() / config.accumulation_steps
            scaler.scale(loss / config.accumulation_steps).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                   error_if_nonfinite=not scaler.is_enabled())
        if not torch.isfinite(grad_norm):
            if not any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                raise FloatingPointError("gradient norm overflow")
            # unscale_ recorded overflow; GradScaler skips this optimizer update.
            scaler.step(optimizer)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        return total_loss, grad_norm.item()
    raise FloatingPointError("FP16 gradients remain non-finite after 16 scale reductions")


def save_preview(batch, generated, path: Path):
    count = min(4, generated.shape[0])
    references = batch["reference_handwriting"].shape[1]
    size = generated.shape[-1]
    cell = max(64, size)
    gap, header = 4, 22
    canvas = Image.new("RGB", ((references + 3) * (cell + gap), count * (cell + gap) + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate([f"Ref {i + 1}" for i in range(references)] + ["Standard", "Truth", "Generated"]):
        draw.text((column * (cell + gap), 3), label, fill="black")
    for row in range(count):
        items = list(batch["reference_handwriting"][row]) + [
            batch["target_standard"][row], batch["target_handwriting"][row], generated[row]]
        for column, tensor in enumerate(items):
            array = ((tensor.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
            tile = Image.fromarray(array).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * (cell + gap), header + row * (cell + gap)))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, loader, device, precision, config, preview_path):
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Fixed validation times/noise; validation must not perturb training RNG.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 10_000)
        total, samples = 0.0, 0
        for index, raw in enumerate(loader):
            batch = move_batch(raw, device)
            with autocast_context(device, precision):
                loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total += loss.item() * len(batch["target_standard"])
            samples += len(batch["target_standard"])
            if index == 0:
                preview = {key: value[:4] for key, value in batch.items()}
        # Preview is generated after all losses so sample settings do not change
        # the validation RNG stream used to compare checkpoints.
        with autocast_context(device, precision):
            generated = sample_euler(model, preview["target_standard"],
                                     preview["reference_handwriting"], preview["reference_standard"],
                                     steps=config.sample_steps, reference_mask=preview["reference_mask"])
        if not torch.isfinite(generated).all():
            raise FloatingPointError("non-finite generated image")
        save_preview(preview, generated, preview_path)
    return total / samples


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resolve_device(device_name: str) -> torch.device:
    """Select an explicit CUDA index before capability checks or allocations."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this trainer supports cpu or a single cuda device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; install a CUDA PyTorch build on the GPU server")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def is_unstarted_run(output: Path, settings: dict, fingerprint: str) -> bool:
    """Allow retry of matching startup metadata, never overwrite training results."""
    metadata = output / "run.json"
    try:
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            return False
        if previous.get("settings") != settings or previous.get("data_fingerprint") != fingerprint:
            return False
        for entry in output.iterdir():
            if entry.is_symlink():
                return False
            if entry == metadata:
                continue
            if entry.name == "previews" and entry.is_dir() and not any(entry.iterdir()):
                continue
            return False
    except (OSError, ValueError):
        return False
    return True


def run(args):
    device = resolve_device(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    if checkpoint is not None:
        settings = checkpoint["settings"]
        if set(settings) != {"model", "training"}:
            raise ValueError("checkpoint settings must contain model and training sections")
        model_config = ModelConfig(**settings["model"])
        config = TrainingConfig(**settings["training"])
    else:
        model_config = ModelConfig()
        config = TrainingConfig()
    settings = {"model": asdict(model_config), "training": asdict(config)}
    precision = config.precision if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; set precision='fp16' in src/config.py for a new run")
    manifest, fingerprint = read_manifest(args.data)
    if model_config.in_channels != 1 or model_config.image_size != manifest["image_size"]:
        raise ValueError("model image size/channels must match prepared grayscale data")
    if checkpoint and checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("prepared dataset differs from checkpoint; refusing to resume")
    completed = checkpoint["step"] if checkpoint else 0
    stop_step = min(config.steps, args.stop_after or config.steps)
    if not completed < stop_step:
        raise ValueError("stop-after/total steps must exceed the saved step")
    workers = config.workers if args.workers is None else args.workers
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        if not is_unstarted_run(args.output, settings, fingerprint):
            raise ValueError("output is nonempty; use --resume or choose a new output directory")
    if (args.resume and args.output.exists() and any(args.output.iterdir())
            and not (args.output / "run.json").exists()):
        raise ValueError("nonempty resume output has no run.json; choose a new directory")
    if checkpoint and (args.output / "run.json").exists():
        previous = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        if previous["settings"] != settings or previous["data_fingerprint"] != fingerprint:
            raise ValueError("output belongs to another run; choose a new output directory")
        latest = args.output / "last.pt"
        if latest.exists():
            latest_step = torch.load(latest, map_location="cpu", weights_only=True)["step"]
            if latest_step > completed:
                raise ValueError("checkpoint is older than output/last.pt; resume into a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "previews").mkdir(exist_ok=True)
    (args.output / "run.json").write_text(json.dumps({
        "settings": settings, "data_fingerprint": fingerprint,
        "device": str(device), "precision": precision, "torch": str(torch.__version__),
    }, indent=2), encoding="utf-8")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = Glyph2Hand(model_config).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    best = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best = checkpoint["best_val_loss"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    del checkpoint
    episodes_per_step = config.batch_size * config.accumulation_steps
    common = dict(min_references=config.min_references, max_references=config.max_references)
    training = HandwritingEpisodes(args.data, "train", length=config.steps * episodes_per_step,
                                   seed=config.seed, **common)
    validation = HandwritingEpisodes(args.data, "val", length=config.val_batches * config.batch_size,
                                     seed=config.seed + 1, **common)
    # Private generators prevent iterator construction from advancing model RNG.
    loader = DataLoader(training, batch_size=config.batch_size,
                        sampler=range(completed * episodes_per_step, stop_step * episodes_per_step),
                        num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(validation, batch_size=config.batch_size, num_workers=0,
                            generator=torch.Generator().manual_seed(config.seed + 1))
    iterator = iter(loader)
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("Stop requested; finishing this optimizer step, then saving last.pt.", flush=True)

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def save(name):
        atomic_save({
            "version": 1, "settings": settings, "data_fingerprint": fingerprint,
            "step": completed, "best_val_loss": best,
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, args.output / name)

    print(f"Device={device}, precision={precision}, parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"effective batch={episodes_per_step}; start step={completed}", flush=True)
    start = time.monotonic()
    try:
        for step in range(completed + 1, stop_step + 1):
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batches = [next(iterator) for _ in range(config.accumulation_steps)]
            total_loss, grad_norm = optimizer_update(
                model, optimizer, scaler, batches, device, precision, config)
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1 - config.ema_decay)
            completed = step
            metrics = {"step": step, "train_loss": total_loss, "lr": lr,
                       "grad_norm": grad_norm, "elapsed_seconds": time.monotonic() - start}
            final = step == stop_step or interrupted
            improved = False
            if not interrupted and (step % config.validate_every == 0 or final):
                val_loss = evaluate(ema, val_loader, device, precision, config,
                                    args.output / "previews" / f"step_{step:07d}.png")
                metrics["val_loss_ema"] = val_loss
                improved = val_loss < best
                best = min(best, val_loss)
            if step % config.log_every == 0 or "val_loss_ema" in metrics or final:
                if device.type == "cuda":
                    metrics["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
                line = json.dumps(metrics)
                print(line, flush=True)
                with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            if improved:
                save("best.pt")
            if step % config.save_every == 0 or final or interrupted:
                save("last.pt")
            if interrupted:
                break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        training.close()
        validation.close()
    print(f"Saved step {completed}: {args.output / 'last.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--workers", type=int, help="override loader workers, also allowed on resume")
    parser.add_argument("--stop-after", type=int, help="stop at this absolute optimizer step without changing LR schedule")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    run(args)


if __name__ == "__main__":
    main()
