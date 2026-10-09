from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from twist import ModelConfig, RegistrationStage, Twist, distance_fields, registration_loss
from twist.geometry import jacobian_determinant, sample_image, velocity_grid
from twist.layers import DualStreamBlock


def tiny_config(**overrides) -> ModelConfig:
    return replace(ModelConfig(image_size=16, patch_size=4, token_dim=32, num_heads=4,
                               num_blocks=2, squaring_steps=5), **overrides)


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

    def test_network_outputs_one_unclipped_velocity_with_zero_boundary(self):
        config = tiny_config()
        stage = RegistrationStage(config)
        with torch.no_grad():
            stage.velocity_head.weight.zero_()
            stage.velocity_head.bias.fill_(3.0)
        field = stage(*images())
        self.assertEqual(field.shape, (2, 2, config.image_size, config.image_size))
        torch.testing.assert_close(field[:, :, 1:-1, 1:-1], torch.full_like(field[:, :, 1:-1, 1:-1], 3.0))
        for edge in (field[:, :, 0], field[:, :, -1], field[:, :, :, 0], field[:, :, :, -1]):
            self.assertEqual(int(torch.count_nonzero(edge)), 0)

    def test_256_dim_token_decodes_64_xy_vectors_and_a_64_pixel_canvas(self):
        config = ModelConfig(num_blocks=1)
        stage = RegistrationStage(config)
        self.assertEqual(stage.velocity_head.in_features, 256)
        self.assertEqual(stage.velocity_head.out_features, 64 * 2)
        self.assertEqual(config.patch_size, 8)
        decoded = []
        hook = stage.velocity_head.register_forward_hook(
            lambda module, args, output: decoded.append(output))
        try:
            field = stage(torch.zeros(1, 1, 64, 64), torch.ones(1, 1, 64, 64))
        finally:
            hook.remove()
        self.assertEqual(decoded[0].shape, (1, 64, 128))
        self.assertEqual(field.shape, (1, 2, 64, 64))

    def test_default_pixel_head_starts_near_identity_on_shifted_strokes(self):
        source = torch.ones(1, 1, 64, 64)
        source[:, :, 10:54, 27:32] = -1
        source[:, :, 28:33, 12:52] = -1
        target = torch.roll(source, 8, dims=3)
        with torch.no_grad():
            for seed in (0, 8, 42):
                torch.manual_seed(seed)
                result = Twist()(source, target)
                rms_pixels = float(result.velocity.square().mean().sqrt()) * 63
                self.assertLess(rms_pixels, 0.25)
                for grid in (result.grid, result.reverse_grid):
                    self.assertGreater(float(jacobian_determinant(grid).min()), 0)

    def test_patch_vectors_land_on_their_own_pixels_and_xy_components(self):
        config = tiny_config(num_blocks=1)
        stage = RegistrationStage(config)
        batch, patch, grid = 2, config.patch_size, config.patch_grid_size
        vectors = torch.arange(batch * grid**2 * patch**2 * 2, dtype=torch.float32).reshape(
            batch, grid**2, patch**2 * 2)
        hook = stage.velocity_head.register_forward_hook(
            lambda module, args, output: vectors)
        try:
            field = stage(*images())
        finally:
            hook.remove()
        expected = torch.zeros(batch, 2, config.image_size, config.image_size)
        for y in range(1, config.image_size - 1):
            for x in range(1, config.image_size - 1):
                token_index = (y // patch) * grid + x // patch
                vector_index = 2 * ((y % patch) * patch + x % patch)
                expected[:, :, y, x] = vectors[:, token_index, vector_index:vector_index+2]
        torch.testing.assert_close(field, expected, rtol=0, atol=0)
        self.assertNotEqual(float(field[0, 0, 5, 5]), float(field[0, 0, 5, 6]))

    def test_each_pixel_gradient_reaches_only_its_own_decoded_vector(self):
        config = tiny_config(num_blocks=1)
        stage = RegistrationStage(config)
        decoded = []
        def capture(module, args, output):
            output.retain_grad()
            decoded.append(output)
        hook = stage.velocity_head.register_forward_hook(capture)
        try:
            field = stage(*images())
        finally:
            hook.remove()
        y, x, component = 7, 9, 1
        field[0, component, y, x].backward()
        token_index = (y // config.patch_size) * config.patch_grid_size + x // config.patch_size
        vector_index = 2 * ((y % config.patch_size) * config.patch_size + x % config.patch_size) + component
        self.assertEqual(float(decoded[0].grad[0, token_index, vector_index]), 1)
        self.assertEqual(int(torch.count_nonzero(decoded[0].grad)), 1)

    def test_single_network_embeds_A_B_once_and_decodes_after_all_blocks(self):
        model = Twist(tiny_config(num_blocks=3))
        source, target = images()
        embeddings, decoded, crossed = [], [], []
        hooks = [model.network.patch_embedding.register_forward_pre_hook(
                    lambda module, args: embeddings.append(args[0])),
                 model.network.velocity_head.register_forward_hook(
                    lambda module, args, output: decoded.append(output))]
        hooks += [block.cross_attention.register_forward_hook(
                    lambda module, args, output: crossed.append(output)) for block in model.network.blocks]
        try:
            result = model(source, target)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(embeddings), 2)
        self.assertIs(embeddings[0], source)
        self.assertIs(embeddings[1], target)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(crossed), 3)
        torch.testing.assert_close(result.image, sample_image(source, result.grid))
        torch.testing.assert_close(result.reverse_image, sample_image(target, result.reverse_grid))
        torch.testing.assert_close(result.grid, velocity_grid(result.velocity, source, squaring_steps=5))

    def test_bidirectional_loss_reaches_every_block_and_parameter(self):
        model = Twist(tiny_config())
        source, target = images()
        result = model(source, target)
        losses = registration_loss(result, distance_fields(source), distance_fields(target))
        reverse_gradient = torch.autograd.grad(losses.reverse_similarity, result.velocity, retain_graph=True)[0]
        self.assertGreater(float(reverse_gradient.abs().sum()), 0)
        losses.total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for block in model.network.blocks:
            self.assertGreater(float(block.cross_attention.in_proj_weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.network.velocity_head.weight.grad.abs().sum()), 0)

    def test_cached_render_has_exact_endpoints_and_the_bidirectional_weighted_middle(self):
        model = Twist(tiny_config()).eval()
        source, target = images()
        result = model(source, target)
        hook = model.network.register_forward_pre_hook(
            lambda module, args: self.fail("render must not rerun the network"))
        try:
            torch.testing.assert_close(result.render(0).image, source, rtol=0, atol=0)
            torch.testing.assert_close(result.render(1).image, target, rtol=0, atol=0)
            torch.testing.assert_close(result.render(1).source_image, result.image)
            torch.testing.assert_close(result.render(0).target_image, result.reverse_image)
            middle = result.render(0.37)
            expected_a = sample_image(source, velocity_grid(result.velocity, source, 0.37, squaring_steps=5))
            expected_b = sample_image(target, velocity_grid(result.velocity, target, -0.63, squaring_steps=5))
            torch.testing.assert_close(middle.image, 0.63 * expected_a + 0.37 * expected_b)
            torch.testing.assert_close(middle.image, result.render(0.37).image, rtol=0, atol=0)
        finally:
            hook.remove()

    def test_batched_render_times_and_reverse_symmetry(self):
        model = Twist(tiny_config())
        source, target = images()
        result = model(source, target)
        times = torch.tensor([0.2, 0.8])
        batch = result.render(times)
        for index, time in enumerate(times):
            single = replace(result, source=source[index:index+1], target=target[index:index+1],
                             velocity=result.velocity[index:index+1], grid=result.grid[index:index+1])
            torch.testing.assert_close(batch.image[index:index+1], single.render(time).image)
        reversed_result = replace(result, source=target, target=source, velocity=-result.velocity)
        torch.testing.assert_close(batch.image, reversed_result.render(1 - times).image, atol=2e-6, rtol=1e-5)

    def test_scalar_endpoint_render_reuses_cached_maps_and_preserves_gradients(self):
        source, target = images()
        result = Twist(tiny_config())(source, target)
        result.velocity.retain_grad()
        with patch("twist.model.velocity_grid", side_effect=AssertionError("endpoint reintegration")), \
                patch("twist.model.sample_image", side_effect=AssertionError("endpoint resampling")):
            start, end = result.render(0), result.render(1)
        self.assertIs(start.reverse_grid, result.reverse_grid)
        self.assertIs(start.target_image, result.reverse_image)
        self.assertIs(end.grid, result.grid)
        self.assertIs(end.source_image, result.image)
        (start.target_image.square().mean() + end.source_image.square().mean()).backward()
        self.assertGreater(float(result.velocity.grad.abs().sum()), 0)

    def test_scalar_preview_times_stay_scalars_and_tensor_times_keep_time_gradients(self):
        source, target = images()
        result = Twist(tiny_config())(source, target)
        with patch("twist.model.velocity_grid", wraps=velocity_grid) as integrate:
            result.render(0.37)
        self.assertEqual(integrate.call_count, 2)
        self.assertEqual(integrate.call_args_list[0].args[2], 0.37)
        self.assertEqual(integrate.call_args_list[1].args[2], -0.63)
        time = torch.tensor(0.37, requires_grad=True)
        result.render(time).image.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(time.grad)))
        self.assertGreater(float(time.grad.abs()), 0)

    def test_block_depth_controls_parameter_count(self):
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        one = Twist(tiny_config(num_blocks=1))
        three = Twist(tiny_config(num_blocks=3))
        block_count = sum(parameter.numel() for parameter in DualStreamBlock(tiny_config()).parameters())
        self.assertEqual(count(three) - count(one), 2 * block_count)

    def test_cpu_autocast_keeps_velocity_sampling_and_losses_float32(self):
        model = Twist(tiny_config())
        source, target = images()
        with torch.backends.mkldnn.flags(enabled=False):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = model(source, target)
                losses = registration_loss(result, distance_fields(source), distance_fields(target))
            for tensor in (result.velocity, result.grid, result.image, result.reverse_grid,
                           result.reverse_image, result.render(0.5).image, losses.total):
                self.assertEqual(tensor.dtype, torch.float32)
            losses.total.backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_invalid_config_inputs_and_render_times(self):
        for change in ({"squaring_steps": -1}, {"squaring_steps": True}, {"image_size": 15},
                       {"token_dim": 30}, {"num_blocks": 0}):
            with self.assertRaises(ValueError):
                tiny_config(**change)
        source, target = images()
        with self.assertRaises(ValueError):
            Twist(tiny_config())(source, target[:, :, :, :-1])
        result = Twist(tiny_config())(source, target)
        for time in (-0.01, 1.01, float("nan"), torch.zeros(2, 1)):
            with self.assertRaises(ValueError):
                result.render(time)


if __name__ == "__main__":
    unittest.main()
