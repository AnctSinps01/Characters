"""Full-dataset registration between different writers' same-character glyphs."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from twist import LossConfig, ModelConfig, Twist, registration_loss
from twist.geometry import identity_grid, lipschitz_bound
from twist.losses import soft_weighted_mse, source_distance_loss
from twist.images import NORMALIZATION
from twist.preview import save_preview, strip, tile
from pairs import PreparedPairs, ShuffledPairSampler
from settings import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="At most this many updates during this invocation")
    parser.add_argument("--cpu-threads", type=int, default=1)
    for config_class in (TrainingConfig, ModelConfig, LossConfig):
        group = parser.add_argument_group(config_class.__name__)
        defaults = config_class()
        for field in fields(config_class):
            value = getattr(defaults, field.name)
            name = "--lr" if field.name == "learning_rate" else "--" + field.name.replace("_", "-")
            kwargs = {"dest": field.name, "default": None}
            if isinstance(value, bool):
                kwargs["action"] = argparse.BooleanOptionalAction
            else:
                kwargs["type"] = type(value)
            group.add_argument(name, **kwargs)
    return parser.parse_args()


def configured(config_class, args, saved=None, mutable=()):
    values = dict(saved) if saved is not None else asdict(config_class())
    for field in fields(config_class):
        supplied = getattr(args, field.name)
        if supplied is not None:
            if saved is not None and field.name not in mutable and supplied != values[field.name]:
                raise ValueError(f"--{field.name.replace('_', '-')} differs from checkpoint; start a new run")
            values[field.name] = supplied
    return config_class(**values)


def atomic_save(state, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preview_indices(data, validation_pairs, count):
    selected, used_labels = [], set()
    for i, pair in enumerate(validation_pairs):
        label = int(data.labels[int(pair[0])])
        if label not in used_labels:
            selected.append(i)
            used_labels.add(label)
            if len(selected) == count:
                break
    if len(selected) < min(count, len(validation_pairs)):
        selected.extend(i for i in range(len(validation_pairs)) if i not in selected)
    return torch.tensor(selected[:count], dtype=torch.long)


def train(args):
    if args.cpu_threads < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError("cpu-threads and stop-after must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("choose an available cpu or cuda device")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint and (checkpoint.get("kind") != "twist-all-pair" or checkpoint.get("version") != 1):
        raise ValueError("expected twist-all-pair checkpoint version 1")
    if checkpoint and checkpoint.get("normalization") != NORMALIZATION:
        raise ValueError("Checkpoint uses a different normalization; start a new run in a new --output directory")
    model_config = configured(ModelConfig, args, checkpoint["model_config"] if checkpoint else None)
    loss_config = configured(LossConfig, args, checkpoint["loss_config"] if checkpoint else None)
    training = configured(TrainingConfig, args, checkpoint["training_config"] if checkpoint else None,
                          mutable=("steps", "batch_size", "log_every", "validate_every", "save_every", "precision",
                                   "preload_to_gpu", "distance_cache_size", "animation_frames", "preview_samples"))
    if model_config.in_channels != 1:
        raise ValueError("prepared handwriting requires in_channels=1")
    supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision = ("bf16" if supports_bf16 else "fp32") if training.precision == "auto" else training.precision
    if precision == "bf16" and not supports_bf16:
        raise ValueError("bf16 is unavailable; use --precision fp32")
    if training.preload_to_gpu and device.type != "cuda":
        raise ValueError("preload_to_gpu requires CUDA")
    start_step = checkpoint["step"] if checkpoint else 0
    if training.steps < start_step:
        raise ValueError(f"--steps is cumulative and must be >= resumed step {start_step}")
    end_step = min(training.steps, start_step + args.stop_after) if args.stop_after is not None else training.steps
    output = (args.output or (args.resume.parent if args.resume else ROOT / "runs" / "twist" / "train-all-pair" / "first")).resolve()
    populated = output.exists() and any(output.iterdir())
    if populated and checkpoint is None:
        raise ValueError(f"Output is nonempty: {output}; use --resume or a new --output")
    if populated and checkpoint:
        if output != args.resume.resolve().parent:
            raise ValueError("resume into the checkpoint directory or a new empty --output")
        metrics_path = output / "metrics.jsonl"
        if metrics_path.exists():
            recorded = [json.loads(line)["step"] for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            if recorded and max(recorded) > start_step:
                raise ValueError("Logs contain later steps than this checkpoint; use a new --output for this resume")
    torch.manual_seed(training.seed)
    data = PreparedPairs(args.data, image_size=model_config.image_size, threshold=loss_config.distance_threshold,
                         storage_device=device if training.preload_to_gpu else "cpu", distance_cache_size=training.distance_cache_size)
    if checkpoint and data.fingerprint != checkpoint["data_fingerprint"]:
        raise ValueError("Prepared manifest differs from checkpoint")
    if checkpoint and data.pairing_fingerprint != checkpoint["pairing_fingerprint"]:
        raise ValueError("Eligible same-character pairs differ from checkpoint; start a new run")
    validation_pairs = checkpoint["validation_pairs"] if checkpoint else data.fixed_validation(
        training.val_batches * training.batch_size, training.seed ^ 0x6A09E667)
    previews = preview_indices(data, validation_pairs, training.preview_samples)
    sampler = ShuffledPairSampler(data, training.seed, checkpoint["sampler"] if checkpoint else None)
    model = Twist(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_cpu"])
        rng_cuda = checkpoint.get("rng_cuda", [])
        if device.type == "cuda" and len(rng_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng_cuda)
    def precision_context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "excluded_images.json", data.invalid_images)
    preview_descriptions = [data.describe(validation_pairs[i]) for i in previews.tolist()]
    dump_json(output / "preview_pairs.json", preview_descriptions)
    inputs = []
    for i in previews.tolist():
        a, b, _ = data.batch(validation_pairs[i:i+1], device)
        inputs.extend((tile(a, f"A: {data.describe(validation_pairs[i])['source_writer']}"),
                       tile(b, f"B: {data.describe(validation_pairs[i])['target_writer']}")))
    strip(inputs).save(output / "inputs.png")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    run_info = {"model": asdict(model_config), "loss": asdict(loss_config), "training": asdict(training),
                "data_root": str(data.root), "data_fingerprint": data.fingerprint, "handwriting_count": data.count,
                "normalization": NORMALIZATION, "constant_images": data.constant_skipped,
                "pairing_fingerprint": data.pairing_fingerprint,
                "train_sources": len(data.records["train"]), "val_sources": len(data.records["val"]),
                "skipped_sources": data.skipped, "validation_pairs": len(validation_pairs),
                "data_loading_seconds": data.load_timings,
                "pixel_gib": data.pixel_bytes / 1024**3, "parameters": parameters, "device": str(device),
                "actual_precision": precision, "torch_version": str(torch.__version__),
                "resume_from": str(args.resume.resolve()) if args.resume else None}
    dump_json(output / "run.json", run_info)
    print(f"Device={device}; precision={precision}; parameters={parameters:,}; batch={training.batch_size}; "
          f"fixed validation pairs={len(validation_pairs):,}; output={output}", flush=True)
    best_total = checkpoint["best_total"] if checkpoint and (output / "best.pt").exists() else math.inf
    step, latest_validation = start_step, None
    started = time.monotonic()
    loss_names = ("total", "mse", "distance", "deformation", "bending", "displacement")

    def append_metrics(record):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def save_checkpoint(name):
        # 不打包全量字图或距离缓存，检查点大小主要由模型/优化器决定。
        atomic_save({"kind": "twist-all-pair", "version": 1, "step": step, "best_total": best_total,
                     "model_config": asdict(model_config), "loss_config": asdict(loss_config),
                     "training_config": asdict(training), "data_fingerprint": data.fingerprint,
                     "normalization": NORMALIZATION, "pairing_fingerprint": data.pairing_fingerprint,
                     "validation_pairs": validation_pairs, "sampler": sampler.state_dict(),
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "validation": latest_validation, "baseline": baseline}, output / name)

    @torch.no_grad()
    def evaluate(pairs, identity=False):
        values = {name: 0.0 for name in (("mse", "distance") if identity else loss_names)}
        bounds = [0.0] * model_config.num_stages
        for chunk in pairs.split(training.batch_size):
            a, b, d = data.batch(chunk, device)
            if identity:
                grid = identity_grid(len(chunk), a.shape[-2:], device=device)
                losses = {"mse": soft_weighted_mse(a, b, loss_config.epsilon),
                          "distance": source_distance_loss(grid, b, d, loss_config.epsilon)}
            else:
                with precision_context():
                    result = model(a, b)
                    breakdown = registration_loss(result, b, d, loss_config)
                losses = {name: getattr(breakdown, name) for name in loss_names}
                bounds = [max(previous, float(lipschitz_bound(u).max())) for previous, u in zip(bounds, result.displacements)]
            for name in values:
                values[name] += float(losses[name]) * len(chunk)
        values = {name: value / len(pairs) for name, value in values.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite evaluated loss; inspect the last saved step")
        values["count"] = len(pairs)
        if not identity:
            values["stage_lipschitz_max"] = bounds
        return values

    baseline = checkpoint["baseline"] if checkpoint and "baseline" in checkpoint else evaluate(validation_pairs, identity=True)
    run_info["validation_identity_baseline"] = baseline
    dump_json(output / "run.json", run_info)
    @torch.no_grad()
    def report_validation(preview=True, append=True):
        nonlocal best_total, latest_validation
        model.eval()
        latest_validation = evaluate(validation_pairs)
        latest_validation["evaluated_step"] = step
        latest_validation["mse_vs_identity"] = latest_validation["mse"] / max(baseline["mse"], loss_config.epsilon)
        if append:
            append_metrics({"step": step, "split": "val", "epoch": sampler.epoch_progress,
                            "elapsed_seconds": time.monotonic() - started, **latest_validation})
        print(f"step={step:7d} val={latest_validation['total']:.6f} mse={latest_validation['mse']:.6f} "
              f"distance={latest_validation['distance']:.6f} def={latest_validation['deformation']:.6f} "
              f"mse/baseline={latest_validation['mse_vs_identity']:.3f}", flush=True)
        if latest_validation["total"] < best_total:
            best_total = latest_validation["total"]
            save_checkpoint("best.pt")
        if preview:
            for position, i in enumerate(previews.tolist()):
                a, b, _ = data.batch(validation_pairs[i:i+1], device)
                with precision_context():
                    result = model(a, b)
                char = data.describe(validation_pairs[i])["character"]
                save_preview(result, b, output / "previews", step, training.animation_frames,
                             split=f"val_{position:02d}_u{ord(char):04x}")
        model.train()

    @torch.no_grad()
    def report_train(pairs, grad_norm):
        model.eval()
        metrics = evaluate(pairs)
        append_metrics({"step": step, "split": "train_batch", "epoch": sampler.epoch_progress,
                        "elapsed_seconds": time.monotonic() - started, "grad_norm_before_clip": grad_norm, **metrics})
        print(f"step={step:7d} epoch={sampler.epoch_progress:.3f} train={metrics['total']:.6f} "
              f"mse={metrics['mse']:.6f} distance={metrics['distance']:.6f} def={metrics['deformation']:.6f} "
              f"grad={grad_norm:.3f}", flush=True)
        model.train()

    report_validation(append=not populated)
    save_checkpoint("last.pt")
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing this update and saving.", flush=True)
    old_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        while step < end_step and not stop_requested:
            pairs = sampler.next(training.batch_size)
            a, b, d = data.batch(pairs, device)
            optimizer.zero_grad(set_to_none=True)
            with precision_context():
                result = model(a, b)
                loss = registration_loss(result, b, d, loss_config)
            if not bool(torch.isfinite(loss.total)):
                raise FloatingPointError(f"Nonfinite training loss at attempted step {step+1}")
            loss.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % training.log_every == 0 or step == start_step + 1 or step == end_step:
                report_train(pairs, float(norm))
            if step % training.validate_every == 0 or step == end_step:
                report_validation()
            if step % training.save_every == 0 or step == end_step:
                save_checkpoint("last.pt")
        if stop_requested:
            if latest_validation["evaluated_step"] != step:
                report_validation()
            save_checkpoint("last.pt")
    finally:
        signal.signal(signal.SIGINT, old_handler)
    status = "interrupted" if stop_requested else ("completed" if step == training.steps else "stopped_early")
    dump_json(output / "summary.json", {"status": status, "step": step, "epoch": sampler.epoch_progress,
              "best_total": best_total, "validation": latest_validation, "baseline_identity": baseline})
    print(f"{status}: step {step}; best validation total={best_total:.6f}; output={output}", flush=True)


if __name__ == "__main__":
    train(parse_args())
