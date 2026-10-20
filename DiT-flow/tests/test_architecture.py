from __future__ import annotations

from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from src import (
    Glyph2Hand, ModelConfig, TrainingConfig,
    build_flow_matching_batch, flow_matching_loss, sample_euler,
)
from src.layers import (
    AdaLNZeroDiTBlock, CrossAttentionBlock, JointAttentionBlock, PatchEmbed2D, SelfAttentionBlock,
)
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
        reference = model.reference_encoder
        self.assertEqual(reference.handwriting_patch_embed.projection.in_channels, 1)
        self.assertEqual(reference.standard_patch_embed.projection.in_channels, 1)
        self.assertIsInstance(reference.local_blocks[0], JointAttentionBlock)
        self.assertFalse(any(isinstance(m, SelfAttentionBlock) for m in reference.modules()))
        self.assertFalse(set(reference.handwriting_patch_embed.parameters())
                         & set(reference.standard_patch_embed.parameters()))
        self.assertFalse(set(reference.local_blocks[0].handwriting.parameters())
                         & set(reference.local_blocks[0].standard.parameters()))
        self.assertFalse({"style_tokens", "set_pool_depth", "sigma"}
                         & {f.name for f in fields(ModelConfig)})
        self.assertFalse({"accumulation_steps", "ema_decay"}
                         & {f.name for f in fields(TrainingConfig)})

    def test_input_positions_and_positioned_difference_but_no_query_positions(self):
        model = Glyph2Hand(tiny_config())
        reference = model.reference_encoder
        image = torch.randn(2, 1, 16, 16)
        standard = torch.randn_like(image)
        for module in (reference.handwriting_patch_embed, reference.standard_patch_embed,
                       model.content_encoder.patch_embed, model.generator.patch_embed):
            raw = module.projection(image).flatten(2).transpose(1, 2)
            torch.testing.assert_close(module(image), raw + module.positions)
        inputs, outputs, pool_inputs = [], [], []
        handles = [
            reference.local_blocks[0].register_forward_pre_hook(
                lambda module, args: inputs.append(tuple(t.detach().clone() for t in args))
            ),
            reference.local_blocks[-1].register_forward_hook(
                lambda module, args, result: outputs.append(tuple(t.detach().clone() for t in result))
            ),
            reference.local_pool.register_forward_pre_hook(
                lambda module, args: pool_inputs.append(tuple(t.detach().clone() for t in args))
            ),
        ]
        try:
            model.encode_style(image[:, None], standard[:, None])
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(inputs[0][0], reference.handwriting_patch_embed(image)
                                   + reference.type_embeddings[0])
        torch.testing.assert_close(inputs[0][1], reference.standard_patch_embed(standard)
                                   + reference.type_embeddings[1])
        expected_queries = reference.description_queries[None].expand(2, -1, -1)
        expected_memory = outputs[0][0] - outputs[0][1] + reference.handwriting_patch_embed.positions
        torch.testing.assert_close(pool_inputs[0][0], expected_queries)
        torch.testing.assert_close(pool_inputs[0][1], expected_memory)
        self.assertEqual(sum(isinstance(m, PatchEmbed2D) for m in reference.modules()), 2)

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

    def test_static_layer_scales_are_small_and_learnable(self):
        encoder_block = SelfAttentionBlock(32, 4, 2.0)
        pool_block = CrossAttentionBlock(32, 4, 2.0)
        generator_block = AdaLNZeroDiTBlock(32, 4, 2.0)
        joint_block = JointAttentionBlock(32, 4, 2.0)
        for parameter in (
            joint_block.handwriting.attention_scale,
            joint_block.handwriting.mlp_scale,
            joint_block.standard.attention_scale,
            joint_block.standard.mlp_scale,
            encoder_block.attention_scale,
            encoder_block.mlp_scale,
            pool_block.attention_scale,
            pool_block.mlp_scale,
            generator_block.content_scale,
            generator_block.style_scale,
        ):
            self.assertTrue(parameter.requires_grad)
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, 1e-4), rtol=0, atol=0
            )
        modulation = generator_block.modulation[-1]
        self.assertTrue(torch.equal(modulation.weight, torch.zeros_like(modulation.weight)))
        self.assertTrue(torch.equal(modulation.bias, torch.zeros_like(modulation.bias)))

    def test_reference_permutation_and_padding_do_not_change_conditioned_velocity(self):
        torch.manual_seed(7)
        model = Glyph2Hand(tiny_config()).eval()
        torch.nn.init.normal_(model.generator.final.projection.weight, std=0.02)
        refs = torch.randn(2, 3, 1, 16, 16)
        ref_standard = torch.randn_like(refs)
        with torch.no_grad():
            model.reference_encoder.local_pool.attention_scale.fill_(0.1)
            for block in model.generator.blocks:
                block.content_scale.fill_(0.1)
                block.style_scale.fill_(0.1)
        standard, state = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
        times = torch.tensor([0.2, 0.8])
        mask = torch.tensor([[True, True, False], [True, False, True]])
        permutation = torch.tensor([2, 0, 1])
        original = model.encode_style(refs, ref_standard, mask)
        reordered = model.encode_style(refs[:, permutation], ref_standard[:, permutation], mask[:, permutation])
        self.assertEqual(original.shape, (2, 9, 32))
        torch.testing.assert_close(
            reordered.reshape(2, 3, 3, 32),
            original.reshape(2, 3, 3, 32)[:, permutation], atol=1e-6, rtol=1e-6,
        )
        expected = model(state, times, standard, refs, ref_standard, mask)
        actual = model(state, times, standard, refs[:, permutation], ref_standard[:, permutation], mask[:, permutation])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        changed = refs.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 20
        torch.testing.assert_close(model(state, times, standard, changed, ref_standard, mask), expected)
        self.assertFalse(torch.allclose(model(state, times, standard + 3, refs, ref_standard, mask), expected))
        changed = refs + 3
        self.assertFalse(torch.allclose(model(state, times, standard, changed, ref_standard, mask), expected))
        changed_standard = ref_standard.clone()
        changed_standard[~mask] = torch.randn_like(changed_standard[~mask]) * 20
        torch.testing.assert_close(model(state, times, standard, refs, changed_standard, mask), expected)
        changed_standard = ref_standard + 3
        self.assertFalse(torch.allclose(
            model(state, times, standard, refs, changed_standard, mask), expected
        ))
        with self.assertRaises(ValueError):
            model.encode_style(refs, ref_standard, torch.zeros_like(mask))

    def test_flow_uses_only_pure_noise_initial_state(self):
        target, noise = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
        times = torch.tensor([0., 1.])
        flow = build_flow_matching_batch(target, noise=noise, times=times)
        torch.testing.assert_close(flow.initial_state, noise)
        torch.testing.assert_close(flow.state[0], noise[0])
        torch.testing.assert_close(flow.state[1], target[1])
        torch.testing.assert_close(flow.target_velocity, target - noise)

    def test_training_loss_is_flow_mse_and_gradients_reach_both_encoders(self):
        torch.manual_seed(11)
        model = Glyph2Hand(tiny_config())
        torch.nn.init.normal_(model.generator.final.projection.weight, std=0.02)
        batch = {
            "target_standard": torch.randn(3, 1, 16, 16),
            "target_handwriting": torch.randn(3, 1, 16, 16),
            "reference_handwriting": torch.randn(3, 2, 1, 16, 16),
            "reference_standard": torch.randn(3, 2, 1, 16, 16),
            "reference_mask": torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool),
        }
        flow = build_flow_matching_batch(batch["target_handwriting"])
        style = model.encode_style(
            batch["reference_handwriting"], batch["reference_standard"], batch["reference_mask"]
        )
        content = model.encode_content(batch["target_standard"])
        velocity = model.predict_velocity(flow.state, flow.times, content, style, batch["reference_mask"])
        mse = flow_matching_loss(velocity, flow.target_velocity)
        with patch("src.train.build_flow_matching_batch", return_value=flow):
            loss = batch_loss(model, batch)
        torch.testing.assert_close(loss, mse)
        loss.backward()
        for module in (model.reference_encoder, model.content_encoder, model.generator):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                   if p.grad is not None), 0)

    def test_sampler_starts_at_noise_and_encodes_conditions_once(self):
        model = Glyph2Hand(tiny_config()).eval()
        refs = torch.randn(1, 2, 1, 16, 16)
        ref_standard = torch.randn_like(refs)
        standard = torch.randn(1, 1, 16, 16)
        noise = torch.randn_like(standard)
        calls = []
        with patch.object(model, "encode_content", wraps=model.encode_content) as content, \
             patch.object(model, "encode_style", wraps=model.encode_style) as style:
            handle = model.generator.register_forward_pre_hook(
                lambda module, inputs: calls.append((inputs[0].clone(), inputs[1].clone()))
            )
            try:
                result = sample_euler(model, standard, refs, ref_standard, steps=3, noise=noise)
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
