from __future__ import annotations

from dataclasses import fields
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src import (
    Glyph2Hand, ModelConfig, TrainingConfig, author_style_loss,
    build_flow_matching_batch, flow_matching_loss, sample_euler,
)
from src.layers import AdaLNZeroDiTBlock, CrossAttentionBlock, PatchEmbed2D
from src.train import batch_loss


def tiny_config() -> ModelConfig:
    return ModelConfig(
        image_size=16, patch_size=4, hidden_size=32, num_heads=4,
        mlp_ratio=2.0, local_depth=1, content_depth=2, generator_depth=2,
        description_tokens=3, time_frequency_size=32,
    )


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_independent_content_depth_single_reference_pool_and_removed_config(self):
        model = Glyph2Hand(tiny_config())
        self.assertEqual(len(model.reference_encoder.local_blocks), 1)
        self.assertEqual(len(model.content_encoder.local_blocks), 2)
        self.assertEqual(sum(isinstance(m, CrossAttentionBlock)
                             for m in model.reference_encoder.modules()), 1)
        self.assertFalse(any(isinstance(m, CrossAttentionBlock)
                             for m in model.content_encoder.modules()))
        self.assertFalse(set(model.reference_encoder.parameters())
                         & set(model.content_encoder.parameters()))
        self.assertEqual(model.reference_encoder.patch_embed.projection.in_channels, 1)
        self.assertFalse({"style_tokens", "set_pool_depth", "sigma"}
                         & {f.name for f in fields(ModelConfig)})
        self.assertFalse({"accumulation_steps", "ema_decay"}
                         & {f.name for f in fields(TrainingConfig)})

    def test_positions_added_once_to_each_patch_branch_and_not_description_queries(self):
        model = Glyph2Hand(tiny_config())
        image = torch.randn(2, 1, 16, 16)
        for module in (model.reference_encoder.patch_embed,
                       model.content_encoder.patch_embed, model.generator.patch_embed):
            raw = module.projection(image).flatten(2).transpose(1, 2)
            torch.testing.assert_close(module(image), raw + module.positions)
        seen = []
        handle = model.reference_encoder.local_pool.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            model.encode_style(image[:, None])
        finally:
            handle.remove()
        expected = model.reference_encoder.description_queries[None].expand(2, -1, -1)
        torch.testing.assert_close(seen[0], expected)
        self.assertEqual(sum(isinstance(m, PatchEmbed2D)
                             for m in model.reference_encoder.modules()), 1)

    def test_block_execution_order(self):
        block = AdaLNZeroDiTBlock(32, 4, 2.0)
        observed, handles = [], []
        for name, module in (("SA", block.self_attention),
                             ("Content CA", block.content_attention),
                             ("Style CA", block.style_attention), ("FFN", block.mlp)):
            handles.append(module.register_forward_hook(
                lambda module, inputs, output, name=name: observed.append(name)
            ))
        try:
            block(torch.randn(2, 16, 32), torch.randn(2, 16, 32),
                  torch.randn(2, 9, 32), torch.randn(2, 32))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed, ["SA", "Content CA", "Style CA", "FFN"])

    def test_reference_permutation_and_padding_do_not_change_conditioned_velocity(self):
        torch.manual_seed(7)
        model = Glyph2Hand(tiny_config()).eval()
        torch.nn.init.normal_(model.generator.final.projection.weight, std=0.02)
        refs = torch.randn(2, 3, 1, 16, 16)
        standard, state = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
        times = torch.tensor([0.2, 0.8])
        mask = torch.tensor([[True, True, False], [True, False, True]])
        permutation = torch.tensor([2, 0, 1])
        original = model.encode_style(refs, mask)
        reordered = model.encode_style(refs[:, permutation], mask[:, permutation])
        self.assertEqual(original.shape, (2, 9, 32))
        torch.testing.assert_close(
            reordered.reshape(2, 3, 3, 32),
            original.reshape(2, 3, 3, 32)[:, permutation], atol=1e-6, rtol=1e-6,
        )
        expected = model(state, times, standard, refs, mask)
        actual = model(state, times, standard, refs[:, permutation], mask[:, permutation])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        changed = refs.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 20
        torch.testing.assert_close(model(state, times, standard, changed, mask), expected)
        self.assertFalse(torch.allclose(model(state, times, standard + 3, refs, mask), expected))
        changed = refs + 3
        self.assertFalse(torch.allclose(model(state, times, standard, changed, mask), expected))
        with self.assertRaises(ValueError):
            model.encode_style(refs, torch.zeros_like(mask))

    def test_author_loss_exact_pair_means_and_reference_order_invariance(self):
        torch.manual_seed(3)
        tokens = torch.randn(4, 3, 2, 5, requires_grad=True)
        authors = torch.tensor([0, 0, 1, 2])
        mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.bool)
        summary = torch.stack([tokens[i, mask[i]].mean(0) for i in range(4)])
        normalized = F.normalize(summary, dim=-1, eps=1e-8)
        same, different = [], []
        for i in range(4):
            for j in range(4):
                if i != j:
                    cosine = (normalized[i] * normalized[j]).sum(-1)
                    (same if authors[i] == authors[j] else different).append(
                        1 - cosine if authors[i] == authors[j] else cosine.square()
                    )
        expected = torch.stack(same).mean() + torch.stack(different).mean()
        actual = author_style_loss(tokens, authors, reference_mask=mask)
        torch.testing.assert_close(actual, expected)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(
            author_style_loss(tokens[:, permutation], authors, reference_mask=mask[:, permutation]), actual
        )
        actual.backward()
        torch.testing.assert_close(tokens.grad[~mask], torch.zeros_like(tokens.grad[~mask]))
        self.assertGreater(tokens.grad[mask].abs().sum().item(), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reduced = author_style_loss(tokens.detach(), authors, reference_mask=mask)
        self.assertEqual(reduced.dtype, torch.float32)
        torch.testing.assert_close(reduced, expected.detach())

    def test_author_loss_orthogonal_and_opposite_directions_and_missing_pairs(self):
        basis = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])[:, None, None, :].expand(-1, 2, 3, -1)
        self.assertEqual(author_style_loss(basis, torch.tensor([0, 0, 1])).item(), 0.)
        opposite = torch.tensor([[1., 0.], [-1., 0.]])[:, None, None, :]
        self.assertEqual(author_style_loss(opposite, torch.tensor([0, 1])).item(), 1.)
        single = torch.randn(1, 2, 3, 8, requires_grad=True)
        zero = author_style_loss(single, torch.tensor([4]))
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertIsNotNone(single.grad)
        self.assertTrue(torch.isfinite(author_style_loss(basis, torch.tensor([0, 0, 0]))))

    def test_flow_uses_only_pure_noise_initial_state(self):
        target, noise = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
        times = torch.tensor([0., 1.])
        flow = build_flow_matching_batch(target, noise=noise, times=times)
        torch.testing.assert_close(flow.initial_state, noise)
        torch.testing.assert_close(flow.state[0], noise[0])
        torch.testing.assert_close(flow.state[1], target[1])
        torch.testing.assert_close(flow.target_velocity, target - noise)

    def test_training_loss_contains_both_terms_and_gradients_reach_both_encoders(self):
        torch.manual_seed(11)
        model = Glyph2Hand(tiny_config())
        torch.nn.init.normal_(model.generator.final.projection.weight, std=0.02)
        model.author_style_loss_weight = 0.05
        batch = {
            "target_standard": torch.randn(3, 1, 16, 16),
            "target_handwriting": torch.randn(3, 1, 16, 16),
            "reference_handwriting": torch.randn(3, 2, 1, 16, 16),
            "reference_mask": torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool),
            "author_index": torch.tensor([0, 0, 1]),
        }
        flow = build_flow_matching_batch(batch["target_handwriting"])
        style = model.encode_style(batch["reference_handwriting"], batch["reference_mask"])
        content = model.encode_content(batch["target_standard"])
        velocity = model.predict_velocity(flow.state, flow.times, content, style, batch["reference_mask"])
        mse = flow_matching_loss(velocity, flow.target_velocity)
        author = author_style_loss(style.reshape(3, 2, 3, 32), batch["author_index"],
                                   reference_mask=batch["reference_mask"])
        with patch("src.train.build_flow_matching_batch", return_value=flow):
            loss = batch_loss(model, batch)
        torch.testing.assert_close(loss, mse + 0.05 * author)
        loss.backward()
        for module in (model.reference_encoder, model.content_encoder, model.generator):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                   if p.grad is not None), 0)

    def test_sampler_starts_at_noise_and_encodes_conditions_once(self):
        model = Glyph2Hand(tiny_config()).eval()
        refs = torch.randn(1, 2, 1, 16, 16)
        standard = torch.randn(1, 1, 16, 16)
        noise = torch.randn_like(standard)
        calls = []
        with patch.object(model, "encode_content", wraps=model.encode_content) as content, \
             patch.object(model, "encode_style", wraps=model.encode_style) as style:
            handle = model.generator.register_forward_pre_hook(
                lambda module, inputs: calls.append((inputs[0].clone(), inputs[1].clone()))
            )
            try:
                result = sample_euler(model, standard, refs, steps=3, noise=noise)
            finally:
                handle.remove()
            self.assertEqual(content.call_count, 1)
            self.assertEqual(style.call_count, 1)
        torch.testing.assert_close(result, noise.clamp(-1, 1))
        torch.testing.assert_close(calls[0][0], noise)
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(torch.stack([t for _, t in calls]).flatten(),
                                   torch.tensor([0., 1/3, 2/3]))


if __name__ == "__main__":
    unittest.main()
