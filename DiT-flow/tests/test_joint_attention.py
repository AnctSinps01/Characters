from __future__ import annotations

import math
import unittest

import torch

from src import Glyph2Hand, ModelConfig
from src.layers import JointAttentionBlock
from src.reference_encoder import ReferenceEncoder
from src.train import batch_loss


def tiny_config(**overrides):
    values = dict(image_size=16, patch_size=4, hidden_size=32, num_heads=4,
                  local_depth=2, content_depth=1, generator_depth=1,
                  description_tokens=3, time_frequency_size=32)
    return ModelConfig(**(values | overrides))


class JointAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_attention_matches_one_softmax_over_both_streams(self):
        torch.manual_seed(21)
        block = JointAttentionBlock(32, 4, 2.0)
        with torch.no_grad():
            block.handwriting.attention_scale.fill_(0.2)
            block.standard.attention_scale.fill_(0.2)
        handwriting, standard = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
        queries, keys, values = [
            torch.cat(parts, dim=2)
            for parts in zip(block.handwriting.project(handwriting),
                             block.standard.project(standard))
        ]
        weights = (queries @ keys.transpose(-2, -1) / math.sqrt(8)).softmax(dim=-1)
        attended = (weights @ values).transpose(1, 2).reshape(2, 10, 32)
        expected = (
            block.handwriting.update(handwriting, attended[:, :5]),
            block.standard.update(standard, attended[:, 5:]),
        )
        actual = block(handwriting, standard)
        for result, oracle in zip(actual, expected):
            torch.testing.assert_close(result, oracle, atol=1e-6, rtol=1e-6)

    def test_attention_exchanges_information_in_both_directions(self):
        torch.manual_seed(22)
        block = JointAttentionBlock(32, 4, 2.0)
        handwriting = torch.randn(2, 5, 32, requires_grad=True)
        standard = torch.randn(2, 5, 32, requires_grad=True)
        hand_output, standard_output = block(handwriting, standard)
        to_standard, = torch.autograd.grad(hand_output.square().mean(), standard,
                                           retain_graph=True)
        to_hand, = torch.autograd.grad(standard_output.square().mean(), handwriting)
        for gradient in (to_standard, to_hand):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_flow_gradients_reach_both_patch_and_qkv_projections(self):
        torch.manual_seed(23)
        model = Glyph2Hand(tiny_config())
        # The production output head is zero-initialized. Open it to inspect
        # encoder gradients after that first-step initialization barrier.
        torch.nn.init.normal_(model.generator.final.projection.weight, std=0.02)
        references = torch.randn(2, 2, 1, 16, 16)
        loss = batch_loss(model, {
            'target_standard': torch.randn(2, 1, 16, 16),
            'target_handwriting': torch.randn(2, 1, 16, 16),
            'reference_handwriting': references,
            'reference_standard': torch.randn_like(references),
            'reference_mask': torch.tensor([[True, True], [True, False]]),
        })
        loss.backward()
        encoder = model.reference_encoder
        projections = [encoder.handwriting_patch_embed.projection,
                       encoder.standard_patch_embed.projection]
        for block in encoder.local_blocks:
            projections.extend((block.handwriting.qkv, block.standard.qkv))
        for projection in projections:
            gradient = projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            if gradient.shape[0] == 96:
                for part in gradient.chunk(3):
                    self.assertGreater(part.abs().sum().item(), 0)

    def test_reference_pairs_are_independent_with_or_without_joint_layers(self):
        torch.manual_seed(24)
        handwriting = torch.randn(2, 3, 1, 16, 16)
        standard = torch.randn_like(handwriting)
        for depth in (0, 2):
            with self.subTest(local_depth=depth):
                encoder = ReferenceEncoder(tiny_config(local_depth=depth))
                baseline = encoder(handwriting, standard).reshape(2, 3, 3, 32)
                changed_hand, changed_standard = handwriting.clone(), standard.clone()
                changed_hand[0, 1] = torch.randn_like(changed_hand[0, 1])
                changed_standard[0, 1] = torch.randn_like(changed_standard[0, 1])
                changed = encoder(changed_hand, changed_standard).reshape(2, 3, 3, 32)
                torch.testing.assert_close(changed[1], baseline[1], rtol=0, atol=0)
                torch.testing.assert_close(changed[0, [0, 2]], baseline[0, [0, 2]], rtol=0, atol=0)
                self.assertGreater((changed[0, 1] - baseline[0, 1]).abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
