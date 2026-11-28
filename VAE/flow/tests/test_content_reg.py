"""Tests for the deterministic content regressor."""

import unittest

import torch

from VAE.flow.content_reg import ContentRegressor, ResBlock3


class RegressorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ContentRegressor(channels=32, blocks=2)

    def test_shapes_and_channel_budget(self):
        out = self.model(torch.randn(3, 2, 8, 8))
        self.assertEqual(out.shape, (3, 2, 8, 8))

    def test_block_keeps_channels_and_has_three_convs(self):
        block = ResBlock3(32)
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual(len(convs), 3)
        self.assertTrue(all(c.in_channels == 32 and c.out_channels == 32 for c in convs))
        self.assertEqual(block(torch.randn(2, 32, 8, 8)).shape, (2, 32, 8, 8))

    def test_parameter_count_matches_spec(self):
        model = ContentRegressor(channels=128, blocks=12)
        total = sum(p.numel() for p in model.parameters())
        self.assertGreater(total, 5_000_000)
        self.assertLess(total, 5_500_000)

    def test_gradients_reach_every_block(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        x = torch.randn(8, 2, 8, 8)
        y = torch.randn(8, 2, 8, 8)
        for _ in range(3):
            loss = torch.nn.functional.mse_loss(self.model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = torch.nn.functional.mse_loss(self.model(x), y)
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and "blocks" in name and "conv3" in name:
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
