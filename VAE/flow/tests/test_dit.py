"""Tests for the DiT-style flow model."""

import unittest

import torch

from VAE.flow.dit import DiTConfig, DiTFlow, flow_loss, sample


class DiTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = DiTConfig(depth=2, d_model=32, heads=4, ffn=64, time_dim=32)
        self.model = DiTFlow(self.config)
        self.refs = torch.randn(2, 3, 1, 8, 8)
        self.standard = torch.randn(2, 4, 8, 8)
        self.x_t = torch.randn(2, 4, 8, 8)
        self.t = torch.rand(2)
        self.counts = torch.tensor([2, 3])

    def test_shapes(self):
        self.assertEqual(self.config.target_channels, 4)
        self.assertEqual(self.config.spatial_tokens, 64)
        out = self.model(self.refs, self.standard, self.x_t, self.t, self.counts)
        self.assertEqual(out.shape, (2, 4, 8, 8))
        self.assertEqual(sample(self.model, self.refs, self.standard, steps=3,
                                counts=self.counts).shape, (2, 4, 8, 8))

    def test_initial_velocity_is_zero(self):
        out = self.model(self.refs, self.standard, self.x_t, self.t, self.counts)
        self.assertLess(out.abs().max().item(), 1e-6)

    def test_reference_count_is_arbitrary(self):
        for count in (1, 5):
            refs = torch.randn(2, count, 1, 8, 8)
            counts = torch.full((2,), count)
            out = self.model(refs, self.standard, self.x_t, self.t, counts)
            self.assertEqual(out.shape, (2, 4, 8, 8))

    def test_padded_references_do_not_change_the_output(self):
        """Padding must be masked, otherwise extra empty tokens would leak in."""
        counts = torch.tensor([2, 3])
        out_a = self.model(self.refs, self.standard, self.x_t, self.t, counts)
        wider = torch.cat([self.refs, torch.randn(2, 2, 1, 8, 8) * 10], dim=1)
        out_b = self.model(wider, self.standard, self.x_t, self.t, counts)
        self.assertLess((out_a - out_b).abs().max().item(), 1e-5)

    def test_gradients_reach_the_blocks(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for _ in range(3):
            self.model.zero_grad()
            loss, _ = flow_loss(self.model, self.refs, self.standard, self.x_t, self.counts)
            loss.backward()
            optimizer.step()
        self.model.zero_grad()
        loss, _ = flow_loss(self.model, self.refs, self.standard, self.x_t, self.counts)
        loss.backward()
        grad = self.model.blocks[0].modulation[1].weight.grad
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_two_qkv_sets_are_separate(self):
        block = self.model.blocks[0]
        self.assertIsNot(block.spatial_qkv, block.ref_qkv)
        self.assertEqual(block.spatial_qkv.in_features, self.config.d_model)
        self.assertEqual(block.ref_qkv.in_features, self.config.d_model)


if __name__ == "__main__":
    unittest.main()
