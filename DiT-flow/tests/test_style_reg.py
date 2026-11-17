"""Tests for distribution losses on actual trunk style-memory tokens."""
import unittest

import torch

from src.style_reg import participation_ratio, style_terms


class ParticipationRatioTest(unittest.TestCase):
    def test_single_direction_is_one(self):
        direction = torch.tensor([1.0, 0.0, 0.0])
        scales = torch.tensor([0.5, 1.0, 2.0, 4.0]).unsqueeze(1)
        self.assertAlmostEqual(participation_ratio(scales * direction).item(), 1.0, places=4)

    def test_isotropic_cloud_uses_multiple_directions(self):
        torch.manual_seed(0)
        values = torch.randn(8, 64)
        value = participation_ratio(values).item()
        self.assertGreater(value, 5.0)
        self.assertLessEqual(value, 8.0)


class StyleTermsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        batch, tokens, width = 16, 4, 16
        values = torch.randn(batch, tokens, width)
        values -= values.mean(dim=0, keepdim=True)
        self.style = [values, values * 2]
        self.null = [torch.zeros_like(values), torch.zeros_like(values)]
        self.mask = torch.ones(batch, tokens, dtype=torch.bool)

    def test_loss_uses_each_block_memory_tensor(self):
        terms = style_terms(self.style, self.null, self.mask, token_norm_target=1.0)
        self.assertLess(terms["null_norm"].item(), 1e-8)
        self.assertLess(terms["mean_norm"].item(), 1e-6)
        self.assertGreater(terms["participation_ratio"].item(), 8.0)
        self.assertGreater(terms["mean_token_norm"].item(), 1.0)

    def test_target_is_mean_l2_norm_not_squared_norm(self):
        unit = torch.zeros(8, 2, 4)
        unit[..., 0] = 1.0
        terms = style_terms([unit], [torch.zeros_like(unit)],
                            torch.ones(8, 2, dtype=torch.bool), token_norm_target=1.0)
        self.assertAlmostEqual(terms["mean_token_norm"].item(), 1.0, places=6)
        self.assertLess(terms["token_norm_loss"].item(), 1e-8)

    def test_rank_detects_collapsed_cloud(self):
        direction = torch.randn(16)
        scales = torch.linspace(0.5, 1.5, 16).view(16, 1, 1)
        collapsed = scales * direction.view(1, 1, 16).expand(16, 4, 16)
        terms = style_terms([collapsed], [torch.zeros_like(collapsed)],
                            torch.ones(16, 4, dtype=torch.bool))
        self.assertLess(terms["participation_ratio"].item(), 1.01)
        self.assertGreater(terms["rank_loss"].item(), 0.9)

    def test_masked_tokens_do_not_contribute_to_norms(self):
        style = torch.ones(4, 3, 8)
        null = torch.zeros_like(style)
        mask = torch.tensor([[1, 0, 0]] * 4, dtype=torch.bool)
        terms = style_terms([style], [null], mask, token_norm_target=8**0.5)
        self.assertAlmostEqual(terms["mean_token_norm"].item(), 8**0.5, places=5)
        self.assertLess(terms["token_norm_loss"].item(), 1e-8)

    def test_sequence_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            style_terms([torch.zeros(2, 4, 8)], [], torch.ones(2, 4, dtype=torch.bool))

    def test_mask_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            style_terms([torch.zeros(2, 4, 8)], [torch.zeros(2, 4, 8)],
                        torch.ones(2, 3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
