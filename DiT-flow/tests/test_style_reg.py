"""Tests for the style-token distribution terms."""
import unittest

import torch

from src.style_reg import participation_ratio, pooled_tokens, style_terms


class PooledTokensTest(unittest.TestCase):
    def test_pooling_averages_tokens(self):
        style = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        pooled = pooled_tokens(style)
        self.assertEqual(pooled.shape, (1, 2))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 3.0]])))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            pooled_tokens(torch.zeros(4, 8))


class ParticipationRatioTest(unittest.TestCase):
    def test_single_direction_is_one(self):
        direction = torch.tensor([1.0, 0.0, 0.0])
        scales = torch.tensor([0.5, 1.0, 2.0, 4.0]).unsqueeze(1)
        values = scales * direction
        self.assertAlmostEqual(participation_ratio(values).item(), 1.0, places=4)

    def test_isotropic_is_the_batch_size(self):
        torch.manual_seed(0)
        values = torch.randn(8, 64)
        value = participation_ratio(values).item()
        self.assertGreater(value, 5.0)
        self.assertLessEqual(value, 8.0)


class StyleTermsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        values = torch.randn(16, 384)
        self.centred = values - values.mean(dim=0, keepdim=True)
        # Every sample repeats one vector over its tokens, so pooling returns it.
        self.style = self.centred.unsqueeze(1).repeat(1, 4, 1)
        self.null = torch.zeros(16, 4, 384)

    def test_null_and_mean_vanish_for_centred_cloud(self):
        terms = style_terms(self.style, self.null)
        self.assertLess(terms["null_norm"].item(), 1e-8)
        self.assertLess(terms["mean_norm"].item(), 1e-6)

    def test_rank_loss_is_small_at_full_rank(self):
        terms = style_terms(self.style, self.null)
        self.assertGreater(terms["participation_ratio"].item(), 8.0)
        self.assertLess(terms["rank_loss"].item(), 0.3)

    def test_collapsed_cloud_has_rank_loss_near_one(self):
        direction = torch.randn(384)
        scales = torch.linspace(0.5, 1.5, 16).unsqueeze(1)
        collapsed = (scales * direction).unsqueeze(1).repeat(1, 4, 1)
        terms = style_terms(collapsed, self.null)
        self.assertLess(terms["participation_ratio"].item(), 1.01)
        self.assertGreater(terms["rank_loss"].item(), 0.9)

    def test_energy_loss_is_zero_at_target(self):
        unit = torch.nn.functional.normalize(torch.randn(16, 384), dim=-1) * 8.0
        style = unit.unsqueeze(1).repeat(1, 4, 1)
        terms = style_terms(style, self.null, energy_target=64.0)
        self.assertLess(terms["energy_loss"].item(), 1e-6)

    def test_null_of_wrong_size_is_rejected(self):
        with self.assertRaises(ValueError):
            style_terms(self.style, torch.zeros(8, 4, 384))


if __name__ == "__main__":
    unittest.main()
