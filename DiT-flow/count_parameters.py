"""Print the parameter count of the Glyph2Hand model.

Run from the ``DiT-flow/`` directory with::

    python count_parameters.py

Any ``ModelConfig`` field can be overridden using its kebab-case command-line
name, for example ``--hidden-size 256 --generator-depth 12``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any

from src import Glyph2Hand, ModelConfig


def _parse_value(raw: str, field_type: Any) -> Any:
    """Convert a command-line string to the type used by ``ModelConfig``."""

    # ``from __future__ import annotations`` makes dataclass types strings in
    # some Python versions, so infer the scalar type from the default where
    # possible instead of relying only on ``field.type``.
    if field_type is bool:
        return raw.lower() in {"1", "true", "yes", "on"}
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def parse_config() -> ModelConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    config_fields = fields(ModelConfig)
    defaults = ModelConfig()

    for field in config_fields:
        default = getattr(defaults, field.name)
        # All current config fields are scalar values.  Keeping the parser
        # generated from the dataclass means new scalar fields are picked up
        # automatically when config.py changes.
        value_type = type(default)
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            default=default,
            type=lambda raw, t=value_type: _parse_value(raw, t),
            help=f"{field.name} (default: {default})",
        )

    return ModelConfig(**vars(parser.parse_args()))


def format_count(value: int) -> str:
    """Format a parameter count with separators and a compact suffix."""

    if value >= 1_000_000:
        return f"{value:,} ({value / 1_000_000:.3f}M)"
    if value >= 1_000:
        return f"{value:,} ({value / 1_000:.3f}K)"
    return f"{value:,}"


def main() -> None:
    config = parse_config()
    model = Glyph2Hand(config)

    reference_parameters = sum(
        parameter.numel() for parameter in model.reference_encoder.parameters()
    )
    generator_parameters = sum(
        parameter.numel() for parameter in model.generator.parameters()
    )
    content_parameters = sum(
        parameter.numel() for parameter in model.content_encoder.parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"Configuration: {config}")
    print(f"Reference encoder: {format_count(reference_parameters)}")
    print(f"Content encoder:   {format_count(content_parameters)}")
    print(f"Generator:         {format_count(generator_parameters)}")
    print(f"Total:              {format_count(total_parameters)}")
    print(f"Trainable:          {format_count(trainable_parameters)}")


if __name__ == "__main__":
    main()
