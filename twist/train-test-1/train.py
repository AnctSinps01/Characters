"""Train one character using all its prepared samples; run from any directory."""
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
from twist.preview import save_preview
from glyph_pair import load_glyphs
from settings import TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "prepared")
    parser.add_argument("--output", type=Path, help="Default: runs/twist/train-test-1/first")
    parser.add_argument("--writer", help="Overfit only this writer; handwriting mode only")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--resume", type=Path, help="Restore configs, cached glyphs, model and optimizer")
    parser.add_argument("--cpu-threads", type=int, default=1)
    aliases = {"character": "--char", "learning_rate": "--lr"}
    for config_class in (TrainingConfig, ModelConfig, LossConfig):
        group = parser.add_argument_group(config_class.__name__)
        defaults = config_class()
        for field in fields(config_class):
            name = aliases.get(field.name, "--" + field.name.replace("_", "-"))
            value = getattr(defaults, field.name)
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


def atomic_save(state, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(args):
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this test supports cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA PyTorch or pass --device cpu")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("unsupported single-glyph checkpoint version")
    model_config = configured(ModelConfig, args, checkpoint["model_config"] if checkpoint else None)
    loss_config = configured(LossConfig, args, checkpoint["loss_config"] if checkpoint else None)
    training = configured(TrainingConfig, args, checkpoint["training_config"] if checkpoint else None,
                          mutable=("steps", "batch_size", "log_every", "preview_every", "precision", "animation_frames"))
    if training.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a CUDA device with BF16 support; use --precision fp32")
    torch.manual_seed(training.seed)
    start_step = checkpoint["step"] if checkpoint else 0
    if training.steps < start_step:
        raise ValueError(f"--steps is a cumulative target and must be >= resumed step {start_step}")
    output = (args.output or (args.resume.parent if args.resume else ROOT / "runs" / "twist" / "train-test-1" / "first")).resolve()
    populated = output.exists() and any(output.iterdir())
    if populated and checkpoint is None:
        raise ValueError(f"Output is nonempty: {output}. Use --resume or a new --output.")
    if populated and checkpoint is not None:
        if output != args.resume.resolve().parent:
            raise ValueError("resume into the checkpoint directory or a new empty --output")
        metrics_path = output / "metrics.jsonl"
        if metrics_path.exists():
            recorded_steps = [json.loads(line)["step"] for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            if recorded_steps and max(recorded_steps) > start_step:
                raise ValueError("Metrics contain later steps than this checkpoint; resume into a new --output.")
    glyphs = checkpoint["glyphs"] if checkpoint else load_glyphs(args.data, model_config, loss_config, training, args.writer)
    if checkpoint and args.writer is not None and args.writer != glyphs["metadata"].get("overfit_writer"):
        raise ValueError("--writer differs from the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    atomic_save(glyphs, output / "glyphs.pt")
    source, targets, distance = (glyphs[name].to(device) for name in ("source", "targets", "distance"))
    train_indices, val_indices = glyphs["train_indices"], glyphs["val_indices"]
    true_grid = glyphs["true_grid"].to(device) if glyphs["true_grid"] is not None else None
    batch_size = min(training.batch_size, len(train_indices))
    model = Twist(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_cpu"])
        # 抽样使用 CPU RNG；同数量 CUDA 设备时也恢复设备 RNG。
        rng_cuda = checkpoint.get("rng_cuda", [])
        if device.type == "cuda" and len(rng_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng_cuda)
    precision_context = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if training.precision == "bf16" else nullcontext()
    parameters = sum(p.numel() for p in model.parameters())
    def batch(indices):
        target = targets[indices.to(device)]
        count = len(indices)
        return source.expand(count, -1, -1, -1), target, distance.expand(count, -1, -1, -1)

    @torch.no_grad()
    def identity_baseline(indices):
        if not len(indices):
            return None
        values = {"mse": 0.0, "distance": 0.0}
        for chunk in indices.split(training.batch_size):
            a, b, d = batch(chunk)
            grid = identity_grid(len(chunk), source.shape[-2:], device=device)
            values["mse"] += float(soft_weighted_mse(a, b, loss_config.epsilon)) * len(chunk)
            values["distance"] += float(source_distance_loss(grid, b, d, loss_config.epsilon)) * len(chunk)
        return {key: value / len(indices) for key, value in values.items()}

    baseline = {"train": identity_baseline(train_indices), "val": identity_baseline(val_indices)}
    dump_json(output / "run.json", {"model": asdict(model_config), "loss": asdict(loss_config),
              "training": asdict(training), "glyphs": glyphs["metadata"], "parameters": parameters,
              "baseline_identity": baseline, "device": str(device), "torch_version": str(torch.__version__),
              "resume_from": str(args.resume.resolve()) if args.resume else None})
    print(f"Character: {training.character}; mode: {training.mode}; loaded handwriting: "
          f"{glyphs['metadata']['handwriting_count']}; train={len(train_indices)} val={len(val_indices)}; "
          f"pixels={glyphs['metadata']['loaded_pixel_mib']:.2f} MiB", flush=True)
    print(f"Device: {device}; precision: {training.precision}; parameters: {parameters:,}; batch={batch_size}", flush=True)
    print(f"Identity baseline: {baseline}; output: {output}", flush=True)
    best_total = checkpoint["best_total"] if checkpoint and (output / "best.pt").exists() else math.inf
    step = start_step
    started = time.monotonic()
    latest = {}

    def save_checkpoint(name):
        atomic_save({"version": 1, "step": step, "best_total": best_total,
                     "model_config": asdict(model_config), "loss_config": asdict(loss_config),
                     "training_config": asdict(training), "glyphs": glyphs,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "metrics": latest}, output / name)

    @torch.no_grad()
    def evaluate(indices, split, preview=False):
        if not len(indices):
            return None
        values = {name: 0.0 for name in ("total", "mse", "distance", "deformation", "bending", "displacement")}
        bounds = [0.0] * model_config.num_stages
        grid_squared = 0.0
        for offset, chunk in enumerate(indices.split(training.batch_size)):
            a, b, d = batch(chunk)
            with precision_context():
                result = model(a, b)
                loss = registration_loss(result, b, d, loss_config)
            for name in values:
                values[name] += float(getattr(loss, name)) * len(chunk)
            bounds = [max(previous, float(lipschitz_bound(u).max()))
                      for previous, u in zip(bounds, result.displacements)]
            if true_grid is not None:
                grid_squared += float((result.grid - true_grid).square().mean()) * len(chunk)
            if preview and offset == 0:
                # 固定预览该 split 的第一个 writer，便于跨训练步对照。
                single_a, single_b, _ = batch(chunk[:1])
                with precision_context():
                    single = model(single_a, single_b)
                save_preview(single, output / "previews", step, training.animation_frames, split)
        values = {name: value / len(indices) for name, value in values.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Nonfinite evaluation loss; inspect the previous saved step")
        values["count"] = len(indices)
        values["stage_lipschitz_max"] = bounds
        reference = baseline[split]["mse"]
        values["mse_vs_identity"] = values["mse"] / max(reference, loss_config.epsilon)
        if true_grid is not None:
            # 背景坐标不由墨迹唯一确定；此项仅供诊断，不参与 loss。
            values["grid_rmse"] = math.sqrt(grid_squared / len(indices))
        return values

    @torch.no_grad()
    def report(preview=False, grad_norm=None, append=True, indices=None):
        nonlocal best_total, latest
        model.eval()
        train_metrics = evaluate(train_indices if preview else indices, "train", preview)
        val_metrics = evaluate(val_indices, "val", preview) if preview else None
        latest = {"step": step, "elapsed_seconds": time.monotonic() - started,
                  "scope": "full" if preview else "batch", "train": train_metrics, "val": val_metrics}
        if grad_norm is not None:
            latest["grad_norm_before_clip"] = grad_norm
        if append:
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(latest, ensure_ascii=False, allow_nan=False) + "\n")
        val_text = f" val={val_metrics['total']:.6f}" if val_metrics is not None else ""
        print(f"step={step:5d} [{latest['scope']}] train={train_metrics['total']:.6f} "
              f"mse={train_metrics['mse']:.6f} distance={train_metrics['distance']:.6f} "
              f"def={train_metrics['deformation']:.6f}{val_text} "
              f"Lip={[round(v, 3) for v in train_metrics['stage_lipschitz_max']]}", flush=True)
        if preview:
            score = (val_metrics or train_metrics)["total"]
            if score < best_total:
                best_total = score
                save_checkpoint("best.pt")
            save_checkpoint("last.pt")
        model.train()

    report(preview=True, append=not populated)
    # SIGINT 只设置停止标记，等当前更新完成再保存，避免保存到一半的优化器状态。
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing this step and saving.", flush=True)
    old_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        while step < training.steps and not stop_requested:
            selected = train_indices[torch.randint(len(train_indices), (batch_size,))]
            a, b, d = batch(selected)
            optimizer.zero_grad(set_to_none=True)
            with precision_context():
                result = model(a, b)
                loss = registration_loss(result, b, d, loss_config)
            if not bool(torch.isfinite(loss.total)):
                raise FloatingPointError(f"Nonfinite training loss at attempted step {step + 1}")
            loss.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            make_preview = step % training.preview_every == 0 or step == training.steps
            if make_preview or step % training.log_every == 0 or step == start_step + 1:
                # 评估更新后的权重，确保日志、best.pt 和预览对应同一状态。
                report(preview=make_preview, grad_norm=float(norm), indices=selected)
        if stop_requested:
            report(preview=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    status = "interrupted" if stop_requested else "completed"
    dump_json(output / "summary.json", {"status": status, "step": step, "best_total": best_total,
              "baseline_identity": baseline, "final": latest})
    print(f"{status}: step {step}; best total {best_total:.6f}; checkpoints and previews in {output}", flush=True)


if __name__ == "__main__":
    train(parse_args())
