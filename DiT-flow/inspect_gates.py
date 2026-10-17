"""Inspect DiT gate and LayerScale magnitudes from one or more checkpoints.

Run from the ``DiT-flow`` directory, for example::

    python inspect_gates.py runs/DiT-full/best.pt runs/DiT-full/last.pt

The script does not load the dataset.  It reports the learned static
Content/Style CA LayerScale parameters and the time-dependent SA/FFN gates at
fixed times, so checkpoints from different training steps can be compared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src import Glyph2Hand, ModelConfig


DEFAULT_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="checkpoint files such as best.pt and last.pt",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device (default: auto; use cuda, cuda:0, cpu, ...)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for a machine-readable JSON report",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Keep the utility usable on older PyTorch 2.x releases that do not
        # expose the weights_only keyword yet.  These checkpoints are local
        # files produced by this project, so the fallback is intentional.
        return torch.load(path, map_location=device)


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    absolute = value.abs()
    return {
        "mean_abs": float(absolute.mean().item()),
        "rms": float(torch.linalg.vector_norm(value).item() / value.numel() ** 0.5),
        "max_abs": float(absolute.max().item()),
    }


def inspect_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, device)
    settings = checkpoint.get("settings")
    if not isinstance(settings, dict) or "model" not in settings:
        raise ValueError(f"{path}: checkpoint has no saved model settings")

    model_config = ModelConfig(**settings["model"])
    model = Glyph2Hand(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    static_scales: list[dict[str, Any]] = []
    dynamic_gates: list[dict[str, Any]] = []
    with torch.no_grad():
        times = torch.tensor(DEFAULT_TIMES, dtype=torch.float32, device=device)
        time_embeddings = model.generator.time_embed(times)
        for index, block in enumerate(model.generator.blocks):
            static_scales.append({
                "block": index,
                "content_scale": tensor_stats(block.content_scale),
                "style_scale": tensor_stats(block.style_scale),
            })

            modulation = block.modulation(time_embeddings)
            _, _, gate_sa, _, _, gate_mlp = modulation.chunk(6, dim=1)
            for time_index, time in enumerate(DEFAULT_TIMES):
                dynamic_gates.append({
                    "block": index,
                    "time": time,
                    "gate_sa": tensor_stats(gate_sa[time_index]),
                    "gate_mlp": tensor_stats(gate_mlp[time_index]),
                })

    return {
        "checkpoint": str(path),
        "step": checkpoint.get("step"),
        "device": str(device),
        "model": settings["model"],
        "static_scales": static_scales,
        "dynamic_gates": dynamic_gates,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\n=== {report['checkpoint']} (step={report['step']}) ===")
    print("Static CA LayerScale: mean_abs / rms / max_abs")
    print("block | content_scale                 | style_scale")
    for item in report["static_scales"]:
        content = item["content_scale"]
        style = item["style_scale"]
        print(
            f"{item['block']:5d} | "
            f"{content['mean_abs']:.6e} / {content['rms']:.6e} / {content['max_abs']:.6e} | "
            f"{style['mean_abs']:.6e} / {style['rms']:.6e} / {style['max_abs']:.6e}"
        )

    print("\nTime-dependent gates: mean_abs / rms / max_abs")
    print("block | time | gate_sa                      | gate_mlp")
    for item in report["dynamic_gates"]:
        sa = item["gate_sa"]
        mlp = item["gate_mlp"]
        print(
            f"{item['block']:5d} | {item['time']:.2f} | "
            f"{sa['mean_abs']:.6e} / {sa['rms']:.6e} / {sa['max_abs']:.6e} | "
            f"{mlp['mean_abs']:.6e} / {mlp['rms']:.6e} / {mlp['max_abs']:.6e}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    reports = [inspect_checkpoint(path, device) for path in args.checkpoints]
    for report in reports:
        print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.json_output}")


if __name__ == "__main__":
    main()
