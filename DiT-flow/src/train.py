"""Single-GPU Glyph2Hand training with deterministic episodes and resumable state."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

# Also support ``python DiT-flow/src/train.py`` from the repository root.
# Normal package execution (``python -m src.train``) keeps its existing path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from . import (
    Glyph2Hand,
    ModelConfig,
    author_style_loss,
    build_flow_matching_batch,
    flow_matching_loss,
    sample_euler,
)
from .config import TrainingConfig
from .data import HandwritingEpisodes, PreloadedGlyphs, read_manifest
from .diagnostics import GradientSnapshot, json_number


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


def metrics_json(metrics: dict) -> str:
    """Serialize one compact, human-readable metrics record.

    Unitless floats remain JSON numbers so analysis tools can read them directly.
    Eight decimal places are used at ordinary scales; very large or very small
    values use scientific notation to avoid unwieldy log lines.
    """
    fields = []
    for key, value in metrics.items():
        encoded_key = json.dumps(key)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be finite")
            magnitude = abs(value)
            encoded_value = (
                f"{value:.8e}"
                if magnitude >= 1e8 or 0 < magnitude < 1e-7
                else f"{value:.8f}"
            )
        else:
            encoded_value = json.dumps(value)
        fields.append(f"{encoded_key}: {encoded_value}")
    return "{" + ", ".join(fields) + "}"


def batch_loss(model, batch):
    flow = build_flow_matching_batch(batch["target_handwriting"])
    style = model.encode_style(batch["reference_handwriting"], batch["reference_mask"])
    content = model.encode_content(batch["target_standard"])
    prediction = model.predict_velocity(
        flow.state, flow.times, content, style, batch["reference_mask"]
    )
    # Accumulate the MSE in float32 even under autocast.
    loss = flow_matching_loss(prediction.float(), flow.target_velocity.float())
    style_weight = getattr(model, "author_style_loss_weight", 0.0)
    if style_weight:
        if "author_index" not in batch:
            raise ValueError("author_style_loss_weight requires author_index in each batch")
        grouped = style.reshape(
            style.shape[0], batch["reference_handwriting"].shape[1],
            model.config.description_tokens, model.config.hidden_size,
        )
        loss = loss + style_weight * author_style_loss(
            grouped, batch["author_index"], reference_mask=batch["reference_mask"]
        )
    return loss


def optimizer_update(model, optimizer, scaler, raw, device, precision, config,
                     *, on_diagnostic=None, diagnose=False):
    """Retry FP16 overflow at a lower scale using the same episodes and noise."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(16):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch(raw, device)
        with autocast_context(device, precision):
            loss = batch_loss(model, batch)
        if not torch.isfinite(loss):
            if on_diagnostic is not None:
                on_diagnostic({"event": "nonfinite_loss", "attempt": attempt + 1,
                               "loss": json_number(loss.detach().item())})
            raise FloatingPointError("non-finite training loss; last checkpoint preserved")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = GradientSnapshot.capture(model)
        finite = bool(torch.isfinite(gradients.total))
        if on_diagnostic is not None and (diagnose or not finite or gradients.used_float64):
            report = gradients.report(config.gradient_clip)
            report.update({"attempt": attempt + 1, "amp_scale": scaler.get_scale()})
            on_diagnostic(report)
        if not finite:
            if not scaler.is_enabled():
                bad = gradients.report(config.gradient_clip)["nonfinite_parameters"]
                raise FloatingPointError(
                    f"non-finite gradients in {', '.join(bad)}; last checkpoint preserved"
                )
            # unscale_ recorded overflow; GradScaler skips this optimizer update.
            scaler.step(optimizer)
            scaler.update()
            continue
        gradients.clip(config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach().item(), gradients.total.item()
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
    was_training = model.training
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Fixed validation times/noise; validation must not perturb training RNG.
    try:
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
            # Preview sampling must not change the loss RNG stream.
            with autocast_context(device, precision):
                generated = sample_euler(
                    model, preview["target_standard"], preview["reference_handwriting"],
                    steps=config.sample_steps, reference_mask=preview["reference_mask"]
                )
            if not torch.isfinite(generated).all():
                raise FloatingPointError("non-finite generated image")
            save_preview(preview, generated, preview_path)
        return total / samples
    finally:
        model.train(was_training)


def atomic_save(payload, path):
    """Retry checkpoint I/O without replacing the last complete checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            # A Python stream exposes OS errors more clearly than PyTorch's
            # filename-based C++ writer (which can report only iostream error).
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            return
        except Exception as error:
            # Failed archives are unusable and may consume the remaining space.
            # Never remove the previous checkpoint to make room for a new one.
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(f"Could not remove incomplete checkpoint {temporary}: {cleanup_error}",
                      file=sys.stderr, flush=True)
            # Do not misreport unrelated serialization/model errors as disk I/O.
            writer_error = isinstance(error, RuntimeError) and any(
                marker in str(error).lower() for marker in (
                    "iostream error", "pytorchstreamwriter failed", "unexpected pos",
                )
            )
            if not isinstance(error, OSError) and not writer_error:
                raise
            try:
                free = shutil.disk_usage(path.parent).free / 1024**3
                storage = f"Filesystem free space after cleanup: {free:.3f} GiB. "
            except OSError as usage_error:
                storage = f"Could not query filesystem free space: {usage_error}. "
            message = (
                f"Checkpoint save failed ({attempt}/{attempts}) for {path}: "
                f"{type(error).__name__}: {error}. "
                "The previous checkpoint has not been replaced. "
                f"{storage}Check disk space, user/platform quota, and mount health; "
                "an atomic save needs space for an additional full checkpoint."
            )
            if attempt == attempts:
                raise RuntimeError(message) from error
            delay = 2 ** (attempt - 1)
            print(f"{message} Retrying the same checkpoint in {delay} s.",
                  file=sys.stderr, flush=True)
            time.sleep(delay)


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


def is_unstarted_run(output: Path, settings: dict, fingerprint: str,
                     *, data_selection: dict | None = None) -> bool:
    """Allow retry of matching startup metadata, never overwrite training results."""
    metadata = output / "run.json"
    try:
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            return False
        if comparable_settings(previous.get("settings")) != comparable_settings(settings) \
                or previous.get("data_fingerprint") != fingerprint \
                or previous.get("data_selection") != data_selection:
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


def comparable_settings(settings):
    """Remove runtime-only options from saved settings comparisons."""
    if not isinstance(settings, dict):
        return settings
    settings = copy.deepcopy(settings)
    training = settings.get("training")
    if isinstance(training, dict):
        training.pop("preload_to_gpu", None)
    return settings


def select_data(manifest: dict, config: TrainingConfig, *, single_writer: bool,
                writer_id: str | None, checkpoint: dict | None) -> dict | None:
    """Choose fitting data once; a resume must keep the saved selection."""
    saved = checkpoint.get("data_selection") if checkpoint is not None else None
    if checkpoint is not None:
        if not single_writer:
            if saved is not None:
                raise ValueError("single-author checkpoint requires the train-single entry point")
            return None
        if not isinstance(saved, dict) or saved.get("mode") != "single_writer_fit":
            raise ValueError("train-single requires a single-author checkpoint to resume")
        if writer_id is not None and writer_id != saved.get("writer_id"):
            raise ValueError("writer-id differs from the saved single-author selection")
        writer_id = saved.get("writer_id")
        if not isinstance(writer_id, str):
            raise ValueError("single-author checkpoint has no valid writer ID")
    elif not single_writer:
        return None

    eligible = [writer for writer in manifest["writers"]
                if writer["split"] == "train"
                and len(writer["labels"]) > config.max_references
                and len(set(writer["labels"])) == len(writer["labels"])]
    if writer_id is None:
        if not eligible:
            raise ValueError("no training writer has distinct target + max_references characters")
        # A stable tie-breaker keeps automatic selection independent of file order.
        writer = min(eligible, key=lambda item: (-len(item["labels"]), item["id"]))
    else:
        matches = [writer for writer in eligible if writer["id"] == writer_id]
        if not matches:
            raise ValueError(f"writer {writer_id!r} must be a training writer with distinct "
                             "target + max_references characters")
        writer = matches[0]
    selection = {"mode": "single_writer_fit", "writer_id": writer["id"],
                 "character_count": len(writer["labels"]),
                 "evaluation": "same_writer_seen_characters"}
    if checkpoint is not None and selection != saved:
        raise ValueError("single-author data selection differs from checkpoint")
    return selection


def run(args, *, single_writer: bool = False):
    device = resolve_device(args.device)
    local_config = TrainingConfig()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("version") in (1, 2):
        raise ValueError(
            "old checkpoint architecture is incompatible with content/style CA and pure-noise flow; "
            "start a new run without --resume in a new output directory"
        )
    if checkpoint is not None and checkpoint.get("version") != 3:
        raise ValueError("unsupported checkpoint version")
    if checkpoint is not None:
        settings = checkpoint["settings"]
        if set(settings) != {"model", "training"}:
            raise ValueError("checkpoint settings must contain model and training sections")
        model_config = ModelConfig(**settings["model"])
        saved_training = dict(settings["training"])
        saved_training.pop("preload_to_gpu", None)
        config = replace(TrainingConfig(**saved_training),
                         preload_to_gpu=local_config.preload_to_gpu)
    else:
        model_config = ModelConfig()
        config = local_config
    saved_training = asdict(config)
    saved_training.pop("preload_to_gpu")
    settings = {"model": asdict(model_config), "training": saved_training}
    precision = config.precision if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; set precision='fp16' in src/config.py for a new run")
    manifest, fingerprint = read_manifest(args.data)
    if model_config.in_channels != 1 or model_config.image_size != manifest["image_size"]:
        raise ValueError("model image size/channels must match prepared grayscale data")
    if checkpoint and checkpoint["data_fingerprint"] != fingerprint:
        raise ValueError("prepared dataset differs from checkpoint; refusing to resume")
    data_selection = select_data(
        manifest, config, single_writer=single_writer,
        writer_id=getattr(args, "writer_id", None), checkpoint=checkpoint,
    )
    completed = checkpoint["step"] if checkpoint else 0
    stop_step = min(config.steps, args.stop_after or config.steps)
    if not completed < stop_step:
        raise ValueError("stop-after/total steps must exceed the saved step")
    workers = config.workers if args.workers is None else args.workers
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if config.preload_to_gpu:
        if device.type != "cuda":
            raise ValueError("preload_to_gpu=True requires CUDA training")
        if workers:
            print(f"preload_to_gpu=True: using workers=0 instead of {workers}; "
                  "CUDA-resident data must be sampled in the training process.", flush=True)
            workers = 0
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        if not is_unstarted_run(args.output, settings, fingerprint, data_selection=data_selection):
            raise ValueError("output is nonempty; use --resume or choose a new output directory")
    if (args.resume and args.output.exists() and any(args.output.iterdir())
            and not (args.output / "run.json").exists()):
        raise ValueError("nonempty resume output has no run.json; choose a new directory")
    if checkpoint and (args.output / "run.json").exists():
        previous = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        if comparable_settings(previous["settings"]) != comparable_settings(settings) \
                or previous["data_fingerprint"] != fingerprint \
                or previous.get("data_selection") != data_selection:
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
        "preload_to_gpu": config.preload_to_gpu,
        "data_selection": data_selection,
    }, indent=2), encoding="utf-8")
    if data_selection is not None:
        print(f"Single-author fitting: writer={data_selection['writer_id']}, "
              f"characters={data_selection['character_count']}; "
              "evaluation uses the same writer and training character pool.", flush=True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = Glyph2Hand(model_config).to(device).train()
    # Keep this runtime-only: it is restored from TrainingConfig and should not
    # become a model parameter or alter the checkpoint state-dict format.
    model.author_style_loss_weight = config.author_style_loss_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    best = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best = checkpoint["best_val_loss"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    del checkpoint
    episodes_per_step = config.batch_size
    data_device = device if config.preload_to_gpu else torch.device("cpu")
    selected_writer = data_selection["writer_id"] if data_selection is not None else None
    preloaded = PreloadedGlyphs(args.data, device=data_device, verbose=True,
                               writer_ids=(selected_writer,) if selected_writer is not None else None)
    if preloaded.fingerprint != fingerprint:
        raise ValueError("prepared manifest changed during startup")
    common = dict(min_references=config.min_references, max_references=config.max_references,
                  preloaded=preloaded, writer_id=selected_writer)
    training = HandwritingEpisodes(args.data, "train", length=config.steps * episodes_per_step,
                                   seed=config.seed, **common)
    validation = HandwritingEpisodes(args.data, "val", length=config.val_batches * config.batch_size,
                                     seed=config.seed + 1, **common)
    # Private generators prevent iterator construction from advancing model RNG.
    loader = DataLoader(training, batch_size=config.batch_size,
                        sampler=range(completed * episodes_per_step, stop_step * episodes_per_step),
                        num_workers=workers,
                        pin_memory=device.type == "cuda" and not config.preload_to_gpu,
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
            "version": 3, "settings": settings, "data_fingerprint": fingerprint,
            "data_selection": data_selection,
            "step": completed, "best_val_loss": best,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, args.output / name)

    print(f"Device={device}, precision={precision}, parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"batch={episodes_per_step}; start step={completed}", flush=True)
    start_step = completed
    start = time.monotonic()
    try:
        for step in range(completed + 1, stop_step + 1):
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = next(iterator)

            def record_diagnostic(report):
                record = {"step": step, "precision": precision, **report}
                with (args.output / "gradient_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if report["event"] != "gradients" or report.get("norm_float64_fallback"):
                    print(f"Gradient diagnostic at step {step}: {report['event']}; "
                          f"see {args.output / 'gradient_diagnostics.jsonl'}", flush=True)

            total_loss, grad_norm = optimizer_update(
                model, optimizer, scaler, batch, device, precision, config,
                on_diagnostic=record_diagnostic,
                diagnose=(step == start_step + 1 or step % config.log_every == 0 or step == stop_step))
            completed = step
            metrics = {"step": step, "train_loss": total_loss, "lr": lr,
                       "grad_norm": grad_norm,
                       "grad_clip_scale": min(1.0, config.gradient_clip / (grad_norm + 1e-6)),
                       "elapsed": f"{time.monotonic() - start:.8f} s"}
            final = step == stop_step or interrupted
            improved = False
            if not interrupted and (step % config.validate_every == 0 or final):
                val_loss = evaluate(model, val_loader, device, precision, config,
                                    args.output / "previews" / f"step_{step:07d}.png")
                metrics["val_loss"] = val_loss
                improved = val_loss < best
                best = min(best, val_loss)
            if step % config.log_every == 0 or "val_loss" in metrics or final:
                if device.type == "cuda":
                    allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    metrics["peak_allocated"] = f"{allocated_gb:.8f} GB"
                line = metrics_json(metrics)
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


def main(*, single_writer: bool = False):
    description = "Fit one automatically selected training writer." if single_writer else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--workers", type=int, help="override loader workers, also allowed on resume")
    parser.add_argument("--stop-after", type=int, help="stop at this absolute optimizer step without changing LR schedule")
    if single_writer:
        parser.add_argument("--writer-id", help="training writer ID; defaults to the writer with most characters")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    run(args, single_writer=single_writer)


if __name__ == "__main__":
    main()
