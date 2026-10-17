"""Measure the effective Content/Style CA residuals on real validation data.

Run from the ``DiT-flow`` directory, for example::

    python inspect_ca_contribution.py \
        ../runs/DiT-full/best.pt \
        --data ../data/prepared \
        --device cuda \
        --batches 4

The script reads the model and training settings saved in the checkpoint,
recreates the validation episode stream, and measures each block's raw CA
output plus the residual that is actually added after LayerScale.  The
reported ``contribution_to_input`` is the RMS of that residual divided by the
RMS of the main-stream features entering the corresponding CA.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.data import HandwritingEpisodes, PreloadedGlyphs
from src.flow import build_flow_matching_batch
from src.model import Glyph2Hand


@dataclass
class Accumulator:
    """Sum-of-squares accumulator for one branch in one transformer block."""

    raw_square_sum: float = 0.0
    contribution_square_sum: float = 0.0
    input_square_sum: float = 0.0
    element_count: int = 0
    batches: int = 0
    ratio_sum: float = 0.0

    def add(self, raw: torch.Tensor, contribution: torch.Tensor, inputs: torch.Tensor) -> None:
        raw = raw.detach().float()
        contribution = contribution.detach().float()
        inputs = inputs.detach().float()
        if raw.shape != contribution.shape:
            raise RuntimeError(
                f"CA output and residual shapes differ: {raw.shape} vs {contribution.shape}"
            )
        if raw.shape != inputs.shape:
            raise RuntimeError(
                f"CA output and branch-input shapes differ: {raw.shape} vs {inputs.shape}"
            )
        self.raw_square_sum += float(raw.square().sum().item())
        self.contribution_square_sum += float(contribution.square().sum().item())
        self.input_square_sum += float(inputs.square().sum().item())
        self.element_count += raw.numel()
        self.batches += 1
        input_rms = torch.sqrt(inputs.square().mean()).item()
        contribution_rms = torch.sqrt(contribution.square().mean()).item()
        self.ratio_sum += contribution_rms / max(input_rms, 1e-12)

    def report(self) -> dict[str, float | int]:
        if not self.element_count:
            raise RuntimeError("no CA observations were captured")
        raw_rms = (self.raw_square_sum / self.element_count) ** 0.5
        contribution_rms = (self.contribution_square_sum / self.element_count) ** 0.5
        input_rms = (self.input_square_sum / self.element_count) ** 0.5
        return {
            "raw_output_rms": raw_rms,
            "contribution_rms": contribution_rms,
            "input_rms": input_rms,
            "raw_to_input": raw_rms / max(input_rms, 1e-12),
            "contribution_to_input": contribution_rms / max(input_rms, 1e-12),
            "mean_batch_contribution_to_input": self.ratio_sum / self.batches,
            "batches": self.batches,
            "elements": self.element_count,
        }


class BlockCapture:
    """Capture the input and output of both CA branches in one block."""

    def __init__(self, block: torch.nn.Module) -> None:
        self.block = block
        self.content_inputs: torch.Tensor | None = None
        self.style_inputs: torch.Tensor | None = None
        self.content = Accumulator()
        self.style = Accumulator()

    @staticmethod
    def _branch_input(module: torch.nn.Module, args: tuple[Any, ...]) -> torch.Tensor:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError(f"expected tensor input for {module.__class__.__name__}")
        return args[0]

    def content_norm_pre_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...]
    ) -> None:
        self.content_inputs = self._branch_input(module, args).detach()

    def style_norm_pre_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...]
    ) -> None:
        self.style_inputs = self._branch_input(module, args).detach()

    @staticmethod
    def _attention_output(output: Any) -> torch.Tensor:
        # MultiheadAttention returns (attention_output, attention_weights).
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("attention hook did not receive a tensor output")
        return value

    def content_attention_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], output: Any
    ) -> None:
        if self.content_inputs is None:
            raise RuntimeError("content CA input was not captured before its output")
        raw = self._attention_output(output)
        scale = self.block.content_scale.reshape(1, 1, -1)
        self.content.add(raw, raw * scale, self.content_inputs)

    def style_attention_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], output: Any
    ) -> None:
        if self.style_inputs is None:
            raise RuntimeError("style CA input was not captured before its output")
        raw = self._attention_output(output)
        scale = self.block.style_scale.reshape(1, 1, -1)
        self.style.add(raw, raw * scale, self.style_inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint file such as best.pt")
    parser.add_argument(
        "--data", type=Path, required=True,
        help="prepared dataset directory containing manifest.json",
    )
    parser.add_argument(
        "--device", default="auto",
        help="torch device (default: auto; use cuda, cuda:0, cpu, ...)",
    )
    parser.add_argument(
        "--batches", type=int, default=4,
        help="number of validation batches to measure (default: 4)",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="override the saved batch size; default uses the checkpoint setting",
    )
    parser.add_argument(
        "--json-output", type=Path,
        help="optional path for a machine-readable JSON report",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch 2.x releases without weights_only.
        return torch.load(path, map_location="cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_validation_loader(
    data_root: Path,
    training: TrainingConfig,
    selection: dict[str, Any] | None,
    batch_size: int,
    batches: int,
) -> DataLoader:
    writer_id = selection.get("writer_id") if selection is not None else None
    subset_indices = selection.get("subset_indices") if selection is not None else None
    preloaded = PreloadedGlyphs(data_root, device="cpu", writer_ids=(writer_id,) if writer_id else None)
    dataset = HandwritingEpisodes(
        data_root,
        "val",
        length=batches * batch_size,
        min_references=training.min_references,
        max_references=training.max_references,
        seed=training.seed + 1,
        preloaded=preloaded,
        writer_id=writer_id,
        writer_sample_indices=subset_indices,
    )
    # Keep the validation stream deterministic and avoid worker processes in a
    # diagnostic tool that only needs a few batches.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(training.seed + 1),
    )


def inspect_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
    batches: int,
    batch_size_override: int | None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    settings = checkpoint.get("settings")
    if not isinstance(settings, dict) or "model" not in settings or "training" not in settings:
        raise ValueError(f"{checkpoint_path}: checkpoint has no saved model/training settings")
    model_config = settings["model"]
    training_config = TrainingConfig(**settings["training"])
    batch_size = batch_size_override or training_config.batch_size
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if batches < 1:
        raise ValueError("batches must be positive")

    model = Glyph2Hand(ModelConfig(**model_config))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    captures: list[BlockCapture] = []
    handles = []
    for block in model.generator.blocks:
        capture = BlockCapture(block)
        captures.append(capture)
        handles.extend([
            block.content_query_norm.register_forward_pre_hook(capture.content_norm_pre_hook),
            block.style_query_norm.register_forward_pre_hook(capture.style_norm_pre_hook),
            block.content_attention.register_forward_hook(capture.content_attention_hook),
            block.style_attention.register_forward_hook(capture.style_attention_hook),
        ])

    selection = checkpoint.get("data_selection")
    if selection is not None and not isinstance(selection, dict):
        raise ValueError(f"{checkpoint_path}: invalid data_selection in checkpoint")
    loader = make_validation_loader(data_root, training_config, selection, batch_size, batches)
    samples = 0
    try:
        with torch.no_grad():
            for raw in loader:
                batch = move_batch(raw, device)
                flow = build_flow_matching_batch(batch["target_handwriting"])
                style = model.encode_style(batch["reference_handwriting"], batch["reference_mask"])
                content = model.encode_content(batch["target_standard"])
                model.predict_velocity(
                    flow.state,
                    flow.times,
                    content,
                    style,
                    batch["reference_mask"],
                )
                samples += batch["target_handwriting"].shape[0]
    finally:
        for handle in handles:
            handle.remove()
        loader.dataset.close()

    return {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "device": str(device),
        "data": str(data_root),
        "batches": batches,
        "batch_size": batch_size,
        "samples": samples,
        "model": model_config,
        "blocks": [
            {
                "block": index,
                "content_ca": capture.content.report(),
                "style_ca": capture.style.report(),
            }
            for index, capture in enumerate(captures)
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== {report['checkpoint']} (step={report['step']}) ===")
    print(
        f"data={report['data']}; batches={report['batches']}; "
        f"batch_size={report['batch_size']}; samples={report['samples']}"
    )
    print("RMS: raw CA output / actual scaled residual / CA input / residual-to-input")
    print("block | Content CA                                      | Style CA")
    for item in report["blocks"]:
        content = item["content_ca"]
        style = item["style_ca"]
        print(
            f"{item['block']:5d} | "
            f"{content['raw_output_rms']:.6e} / "
            f"{content['contribution_rms']:.6e} / "
            f"{content['input_rms']:.6e} / "
            f"{content['contribution_to_input']:.6e} | "
            f"{style['raw_output_rms']:.6e} / "
            f"{style['contribution_rms']:.6e} / "
            f"{style['input_rms']:.6e} / "
            f"{style['contribution_to_input']:.6e}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    report = inspect_checkpoint(
        args.checkpoint,
        args.data,
        device,
        args.batches,
        args.batch_size,
    )
    print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.json_output}")


if __name__ == "__main__":
    main()




