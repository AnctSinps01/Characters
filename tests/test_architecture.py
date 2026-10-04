from __future__ import annotations

import unittest

import torch

from src import (
    Glyph2Hand,
    ModelConfig,
    build_flow_matching_batch,
    flow_matching_loss,
    sample_euler,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=4,
        hidden_size=32,
        num_heads=4,
        mlp_ratio=2.0,
        local_depth=1,
        set_pool_depth=2,
        generator_depth=2,
        description_tokens=3,
        style_tokens=4,
        time_frequency_size=32,
        sigma=0.1,
    )


class ArchitectureTests(unittest.TestCase):
    def test_set_pool_depth_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(set_pool_depth=0)

    def test_reference_order_does_not_change_style(self) -> None:
        torch.manual_seed(7)
        model = Glyph2Hand(tiny_config()).eval()
        self.assertEqual(len(model.reference_encoder.set_pool_blocks), 2)
        handwriting = torch.randn(2, 3, 1, 16, 16)
        standard = torch.randn(2, 3, 1, 16, 16)
        mask = torch.tensor([[True, True, False], [True, False, True]])
        permutation = torch.tensor([2, 0, 1])

        original = model.encode_style(handwriting, standard, mask)
        reordered = model.encode_style(
            handwriting[:, permutation], standard[:, permutation], mask[:, permutation]
        )

        self.assertEqual(original.shape, (2, 4, 32))
        torch.testing.assert_close(original, reordered, atol=1e-6, rtol=1e-6)

    def test_end_to_end_velocity_and_flow_loss(self) -> None:
        torch.manual_seed(11)
        model = Glyph2Hand(tiny_config())
        reference_handwriting = torch.randn(2, 2, 1, 16, 16)
        reference_standard = torch.randn(2, 2, 1, 16, 16)
        target_standard = torch.randn(2, 1, 16, 16).clamp(-1, 1)
        target_handwriting = torch.randn(2, 1, 16, 16).clamp(-1, 1)
        flow_batch = build_flow_matching_batch(
            target_standard, target_handwriting, model.config.sigma
        )

        predicted = model(
            flow_batch.state,
            flow_batch.times,
            target_standard,
            reference_handwriting,
            reference_standard,
        )
        loss = flow_matching_loss(predicted, flow_batch.target_velocity)
        loss.backward()

        self.assertEqual(predicted.shape, target_handwriting.shape)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(model.generator.final.projection.weight.grad)

    def test_euler_sampler_returns_bounded_images(self) -> None:
        torch.manual_seed(13)
        model = Glyph2Hand(tiny_config()).eval()
        references = torch.randn(1, 1, 1, 16, 16)
        standards = torch.randn(1, 1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)

        result = sample_euler(
            model,
            target,
            references,
            standards,
            steps=2,
            noise=torch.ones_like(target),
        )

        self.assertEqual(result.shape, target.shape)
        self.assertGreaterEqual(result.min().item(), -1)
        self.assertLessEqual(result.max().item(), 1)


if __name__ == "__main__":
    unittest.main()
