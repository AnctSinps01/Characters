"""Run from the repository root: python -m twist.count_parameters."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json

from .config import ModelConfig
from .model import Twist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ModelConfig()
    for field in fields(ModelConfig):
        default = getattr(defaults, field.name)
        name = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(name, type=type(default), default=default)
    config = ModelConfig(**vars(parser.parse_args()))
    model = Twist(config)
    per_stage = sum(parameter.numel() for parameter in model.stages[0].parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    print(f"One stage: {per_stage:,} ({per_stage / 1e6:.3f}M)")
    print(f"Unique parameter sets: {len(model.stages)}; executed stages: {config.num_stages}")
    print(f"Total: {total:,} ({total / 1e6:.3f}M)")
    print(f"Trainable: {trainable:,}")


if __name__ == "__main__":
    main()
