"""Tests for the rectified-flow model over blocked-VAE codes."""

import unittest

import torch

from VAE.flow.model import FlowConfig, FlowModel, flow_loss, sample, timestep_embedding


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = FlowConfig(latent_size=8, blocks=2, time_dim=32, ref_hidden=8)
        self.model = FlowModel(self.config)
        self.refs = torch.randn(3, 4, 1, 8, 8)
        self.standard = torch.randn(3, 2, 8, 8)
        self.x_t = torch.randn(3, 3, 8, 8)
        self.t = torch.rand(3)

    def test_shapes_and_channel_budget(self):
        self.assertEqual(self.config.target_channels, 3)
        self.assertEqual(self.config.input_channels, 6)
        out = self.model(self.refs, self.standard, self.x_t, self.t)
        self.assertEqual(out.shape, (3, 3, 8, 8))
        loss, _ = flow_loss(self.model, self.refs, self.standard, self.x_t)
        self.assertEqual(loss.dim(), 0)

    def test_zero_init_blocks_are_identity(self):
        """Every block and the head start as no-ops, so the initial velocity is zero."""
        out = self.model(self.refs, self.standard, self.x_t, self.t)
        self.assertLess(out.abs().max().item(), 1e-6)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.01)
        self.assertGreater(self.model(self.refs, self.standard, self.x_t, self.t).abs().max().item(), 1e-4)

    def test_reference_count_is_arbitrary(self):
        for count in (1, 3, 9):
            refs = torch.randn(2, count, 1, 8, 8)
            out = self.model(refs, torch.randn(2, 2, 8, 8), torch.randn(2, 3, 8, 8), torch.rand(2))
            self.assertEqual(out.shape, (2, 3, 8, 8))
        empty = self.model(torch.zeros(2, 0, 1, 8, 8), torch.randn(2, 2, 8, 8),
                           torch.randn(2, 3, 8, 8), torch.rand(2))
        self.assertEqual(empty.shape, (2, 3, 8, 8))

    def test_reference_pooling_is_order_invariant(self):
        permuted = self.refs[:, [2, 0, 3, 1]]
        a = self.model(self.refs, self.standard, self.x_t, self.t)
        b = self.model(permuted, self.standard, self.x_t, self.t)
        self.assertLess((a - b).abs().max().item(), 1e-6)

    def test_sample_returns_target_shape(self):
        out = sample(self.model, self.refs, self.standard, steps=4)
        self.assertEqual(out.shape, (3, 3, 8, 8))

    def test_time_embedding_is_bounded_and_distinct(self):
        embedding = timestep_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
        self.assertEqual(embedding.shape, (3, 16))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertGreater((embedding[0] - embedding[1]).abs().max().item(), 1e-3)

    def test_trunk_keeps_six_channels(self):
        model = FlowModel(FlowConfig(blocks=3))
        for block in model.blocks:
            self.assertEqual(block.conv.in_channels, 6)
            self.assertEqual(block.conv.out_channels, 6)
        self.assertEqual(model.head.in_channels, 6)
        self.assertEqual(model.head.out_channels, 3)


if __name__ == "__main__":
    unittest.main()
