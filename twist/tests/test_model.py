from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from twist import ModelConfig, RegistrationStage, Twist, distance_fields, registration_loss
from twist.geometry import composed_grid, lipschitz_bound, sample_image
from twist.layers import DualStreamBlock


def tiny_config(**overrides) -> ModelConfig:
    return replace(ModelConfig(image_size=16, patch_size=4, token_dim=32, num_heads=4,
                               blocks_per_stage=1, num_stages=2), **overrides)


def images() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.ones(2, 1, 16, 16)
    target = torch.ones_like(source)
    source[:, :, 2:14, 4:6] = -1
    target[:, :, 2:14, 8:10] = -1
    return source, target


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        torch.set_num_threads(1)

    def test_cross_attention_reads_A_and_updates_B_query_tokens(self) -> None:
        block = DualStreamBlock(tiny_config()).eval()
        source = torch.randn(2, 16, 32)
        target = torch.randn_like(source)
        a_norm = block.source_self_norm(source)
        a = source + block.source_self_attention(a_norm, a_norm, a_norm, need_weights=False)[0]
        a = a + block.source_ffn(block.source_ffn_norm(a))
        b_norm = block.target_self_norm(target)
        b = target + block.target_self_attention(b_norm, b_norm, b_norm, need_weights=False)[0]
        captured = []
        hook = block.cross_attention.register_forward_pre_hook(
            lambda module, args: captured.append(tuple(value.detach() for value in args[:3])))
        try:
            updated_a, updated_b = block(source, target)
        finally:
            hook.remove()
        query, key, value = captured[0]
        torch.testing.assert_close(query, block.cross_query_norm(b))
        torch.testing.assert_close(key, block.cross_memory_norm(a))
        torch.testing.assert_close(value, key)
        torch.testing.assert_close(updated_a, a)
        self.assertFalse(torch.equal(updated_b, b))

    def test_standalone_stage_outputs_constrained_two_component_grid(self) -> None:
        config = tiny_config()
        field = RegistrationStage(config)(*images())
        self.assertEqual(field.shape, (2, 2, config.control_grid_size, config.control_grid_size))
        self.assertTrue(bool((lipschitz_bound(field) <= config.lipschitz_limit).all()))

    def test_each_stage_sees_accumulated_warp_and_original_B(self) -> None:
        model = Twist(tiny_config(num_stages=3))
        source, target = images()
        captured = []
        hooks = [stage.register_forward_pre_hook(
            lambda module, args: captured.append((args[0], args[1]))) for stage in model.stages]
        try:
            result = model(source, target)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertIs(captured[0][0], source)
        for index in range(1, 3):
            torch.testing.assert_close(captured[index][0], result.stage_images[index - 1])
            self.assertGreater((captured[index][0] - source).abs().max().item(), 1e-5)
        self.assertTrue(all(item[1] is target for item in captured))
        torch.testing.assert_close(result.image, sample_image(source, composed_grid(result.displacements, source)))
        # Resampling an already interpolated previous image would blur differently.
        recursive = sample_image(result.stage_images[-2], composed_grid([result.displacements[-1]], source))
        self.assertGreater((recursive - result.image).abs().max().item(), 1e-5)

    def test_loss_reaches_every_parameter_and_stage_without_nonfinite_gradients(self) -> None:
        model = Twist(tiny_config())
        source, target = images()
        losses = registration_loss(model(source, target), target, distance_fields(source))
        losses.total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for stage in model.stages:
            self.assertGreater(stage.displacement_head.weight.grad.abs().sum().item(), 0)
            self.assertGreater(stage.blocks[0].cross_attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_cached_trajectory_queries_do_not_rerun_neural_stages(self) -> None:
        model = Twist(tiny_config()).eval()
        source, target = images()
        result = model(source, target)
        hooks = [stage.register_forward_pre_hook(
            lambda module, args: self.fail("render must not rerun a neural stage")) for stage in model.stages]
        try:
            torch.testing.assert_close(result.render(0).image, source, rtol=0, atol=1e-6)
            torch.testing.assert_close(result.render(1).image, result.image)
            first = result.render(0.37)
            result.render(0.8)
            torch.testing.assert_close(first.image, result.render(0.37).image, rtol=0, atol=0)
            self.assertGreater((first.image - source).abs().max().item(), 1e-5)
        finally:
            for hook in hooks:
                hook.remove()

    def test_shared_stages_reuse_weights_and_independent_stages_scale_count(self) -> None:
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        one = Twist(tiny_config(num_stages=1))
        independent = Twist(tiny_config(num_stages=3))
        shared = Twist(tiny_config(num_stages=3, share_stage_weights=True))
        self.assertEqual(count(independent), 3 * count(one))
        self.assertEqual(count(shared), count(one))
        self.assertEqual(len(shared(*images()).displacements), 3)

    def test_cpu_autocast_keeps_sampling_fields_and_losses_float32(self) -> None:
        model = Twist(tiny_config())
        source, target = images()
        # Some CPUs run oneDNN BF16 forward but cannot run its BF16 backward.
        # Use PyTorch's native CPU kernels to test the model's AMP behavior.
        with torch.backends.mkldnn.flags(enabled=False):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = model(source, target)
                losses = registration_loss(result, target, distance_fields(source))
            self.assertEqual(result.grid.dtype, torch.float32)
            self.assertEqual(result.image.dtype, torch.float32)
            self.assertEqual(result.displacements[0].dtype, torch.float32)
            losses.total.backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_invalid_config_and_input_shapes_are_rejected(self) -> None:
        for change in ({"lipschitz_limit": 1.0}, {"image_size": 15}, {"token_dim": 30},
                       {"num_stages": 0}, {"blocks_per_stage": 0}):
            with self.assertRaises(ValueError):
                tiny_config(**change)
        source, target = images()
        with self.assertRaises(ValueError):
            Twist(tiny_config())(source, target[:, :, :, :-1])


if __name__ == "__main__":
    unittest.main()
