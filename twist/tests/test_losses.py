from __future__ import annotations

from dataclasses import fields, replace
import unittest

import torch

from twist import LossConfig, ModelConfig, Twist, distance_fields, registration_loss
from twist.geometry import identity_grid, sample_image
from twist.losses import (foreground_weights, velocity_smoothness, gradient_weighted_mse,
                         source_distance_loss)


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def test_gradient_mse_falls_back_to_plain_mse_for_constant_targets(self) -> None:
        target = torch.ones(1, 1, 8, 8)
        white = torch.ones_like(target)
        self.assertAlmostEqual(gradient_weighted_mse(-white, white).item(), 4)
        gray = torch.full_like(target, 0.8)
        prediction = torch.zeros_like(gray)
        self.assertAlmostEqual(gradient_weighted_mse(prediction, gray).item(), 0.64, places=6)

    def test_gradient_mse_emphasizes_edges_but_penalizes_extra_and_interior_ink(self) -> None:
        target = torch.ones(1, 1, 32, 32)
        target[:, :, 8:24, 8:24] = -1
        at_edge, extra, interior = (target.clone() for _ in range(3))
        at_edge[:, :, 7, 16] -= 0.5
        extra[:, :, 2, 2] -= 0.5
        interior[:, :, 16, 16] += 0.5
        edge_loss = gradient_weighted_mse(at_edge, target)
        extra_loss = gradient_weighted_mse(extra, target)
        interior_loss = gradient_weighted_mse(interior, target)
        self.assertGreater(edge_loss.item(), 10 * extra_loss.item())
        self.assertGreater(extra_loss.item(), 0)
        self.assertGreater(interior_loss.item(), 0)
        torch.testing.assert_close(extra_loss, interior_loss)
        torch.testing.assert_close(gradient_weighted_mse(at_edge, target, uniform_ratio=1),
                                   (at_edge - target).square().mean())

    def test_gradient_weight_normalization_retains_error_scale_per_image(self) -> None:
        for size in (16, 32):
            target = torch.ones(2, 1, size, size)
            target[:, :, size//4:3*size//4, size//4:3*size//4] = -1
            target[1] = 0.3 * target[1] + 0.2
            prediction = target + 0.2
            self.assertAlmostEqual(gradient_weighted_mse(prediction, target).item(), 0.04, places=6)

    def test_gradient_weights_are_detached_and_prediction_gradients_are_finite(self) -> None:
        target = torch.ones(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = -1
        target.requires_grad_()
        prediction = (torch.randn_like(target) * 0.2).requires_grad_()
        loss = gradient_weighted_mse(prediction, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertGreater(float(prediction.grad.abs().sum()), 0)
        torch.testing.assert_close(target.grad, -prediction.grad)

    def test_distance_field_uses_normalized_xy_units_and_no_preparation_gradients(self) -> None:
        source = torch.ones(1, 1, 5, 9, requires_grad=True)
        with torch.no_grad():
            source[:, :, 2, 4] = -1
        distance = distance_fields(source)
        self.assertFalse(distance.requires_grad)
        self.assertEqual(distance[0, 0, 2, 4].item(), 0)
        self.assertAlmostEqual(distance[0, 0, 2, 5].item(), 1 / 8)
        self.assertAlmostEqual(distance[0, 0, 3, 4].item(), 1 / 4)
        with self.assertRaises(ValueError):
            distance_fields(torch.ones_like(source))

    def test_distance_loss_has_coordinate_gradient_where_image_mse_has_none(self) -> None:
        source = torch.ones(1, 1, 17, 17)
        target = torch.ones_like(source)
        source[:, :, 2:15, 3] = -1
        target[:, :, 2:15, 11] = -1
        grid = identity_grid(1, (17, 17), device=torch.device("cpu")).clone().requires_grad_()
        distance_loss = source_distance_loss(grid, target, distance_fields(source))
        distance_gradient = torch.autograd.grad(distance_loss, grid, retain_graph=True)[0]
        image_loss = gradient_weighted_mse(sample_image(source, grid), target)
        image_gradient = torch.autograd.grad(image_loss, grid)[0]
        self.assertGreater(distance_gradient[0, 8, 11, 0].item(), 0)
        self.assertEqual(image_gradient[0, 8, 11].abs().sum().item(), 0)

    def test_viscous_energy_respects_normalized_spacing_and_allows_constant_flow(self):
        for size in (5, 9, 17):
            nodes = identity_grid(1, (size, size), device=torch.device("cpu"))
            x, y = nodes[..., 0], nodes[..., 1]
            field = torch.stack((0.3*x + 0.2*y, -0.2*x + 0.1), dim=1)
            self.assertAlmostEqual(float(velocity_smoothness(field)), 0.17, places=6)
            torch.testing.assert_close(velocity_smoothness(field), velocity_smoothness(-field))
            self.assertEqual(float(velocity_smoothness(torch.ones_like(field))), 0)

    def test_distance_uses_detached_soft_ink_and_the_exact_mass_plus_epsilon(self):
        target = torch.ones(1, 1, 5, 5)
        target[:, :, 2, 2] = -1
        target[:, :, 2, 3] = 0
        target.requires_grad_()
        weights = foreground_weights(target)
        self.assertFalse(weights.requires_grad)
        self.assertEqual(float(weights[0, 0, 2, 2]), 1)
        self.assertEqual(float(weights[0, 0, 2, 3]), 0.5)
        distance = torch.linspace(0, 1, 25).reshape(1, 1, 5, 5).requires_grad_()
        grid = identity_grid(1, (5, 5), device=target.device).clone().requires_grad_()
        loss = source_distance_loss(grid, target, distance, epsilon=0.2)
        expected = (distance.detach()[0, 0, 2, 2].square()
                    + 0.5*distance.detach()[0, 0, 2, 3].square()) / (1.5 + 0.2)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(distance.grad)
        self.assertTrue(bool(torch.isfinite(grid.grad).all()))
        for epsilon in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                source_distance_loss(grid, target, distance, epsilon)

    def test_exactly_three_losses_sum_directions_and_preserve_swap_symmetry(self):
        source = torch.ones(1, 1, 16, 16)
        target = torch.ones_like(source)
        source[:, :, 3:13, 4:6] = -1
        target[:, :, 3:13, 8:10] = -1
        model = Twist(ModelConfig(image_size=16, patch_size=4, token_dim=16, num_heads=4,
                                  num_blocks=2, squaring_steps=5))
        result = model(source, target)
        distance_a, distance_b = distance_fields(source), distance_fields(target)
        config = LossConfig(similarity_weight=2, distance_weight=3, smoothness_weight=0.2)
        loss = registration_loss(result, distance_a, distance_b, config)
        forward = gradient_weighted_mse(result.image, target, config.mse_uniform_ratio)
        reverse = gradient_weighted_mse(result.reverse_image, source, config.mse_uniform_ratio)
        torch.testing.assert_close(loss.similarity, forward + reverse)
        torch.testing.assert_close(loss.distance, loss.forward_distance + loss.reverse_distance)
        torch.testing.assert_close(loss.forward_distance,
                                   source_distance_loss(result.grid, target, distance_a))
        torch.testing.assert_close(loss.reverse_distance,
                                   source_distance_loss(result.reverse_grid, source, distance_b))
        torch.testing.assert_close(loss.smoothness, velocity_smoothness(result.velocity))
        torch.testing.assert_close(loss.total, 2*loss.similarity + 3*loss.distance + 0.2*loss.smoothness)
        default = registration_loss(result, distance_a, distance_b)
        torch.testing.assert_close(default.total, default.similarity + default.distance + 1e-3*default.smoothness)
        self.assertEqual({field.name for field in fields(LossConfig) if field.name.endswith("_weight")},
                         {"similarity_weight", "distance_weight", "smoothness_weight"})
        self.assertFalse(hasattr(default, "folding"))
        swapped = replace(result, source=target, target=source, velocity=-result.velocity,
                          image=result.reverse_image, grid=result.reverse_grid,
                          reverse_image=result.image, reverse_grid=result.grid)
        swapped_loss = registration_loss(swapped, distance_b, distance_a, config)
        for name in ("total", "similarity", "distance", "smoothness"):
            torch.testing.assert_close(getattr(loss, name), getattr(swapped_loss, name))
        with self.assertRaises(TypeError):
            registration_loss(result, distance_a)

    def test_both_distance_directions_supply_coordinate_gradients_for_separated_strokes(self):
        source = torch.ones(1, 1, 17, 17)
        target = torch.ones_like(source)
        source[:, :, 2:15, 3] = -1
        target[:, :, 2:15, 11] = -1
        model = Twist(ModelConfig(image_size=17, patch_size=1, token_dim=8, num_heads=2,
                                  num_blocks=1, head_init_std=0))
        result = model(source, target)
        loss = registration_loss(result, distance_fields(source), distance_fields(target))
        forward_gradient = torch.autograd.grad(loss.forward_distance, result.grid, retain_graph=True)[0]
        reverse_gradient = torch.autograd.grad(loss.reverse_distance, result.reverse_grid, retain_graph=True)[0]
        self.assertGreater(float(forward_gradient[0, 8, 11, 0]), 0)
        self.assertLess(float(reverse_gradient[0, 8, 3, 0]), 0)
        forward_image_gradient = torch.autograd.grad(loss.forward_similarity, result.grid, retain_graph=True)[0]
        reverse_image_gradient = torch.autograd.grad(loss.reverse_similarity, result.reverse_grid)[0]
        self.assertEqual(float(forward_image_gradient[0, 8, 11].abs().sum()), 0)
        self.assertEqual(float(reverse_image_gradient[0, 8, 3].abs().sum()), 0)

    def test_invalid_loss_weights(self):
        for change in ({"smoothness_weight": -1}, {"distance_weight": float("nan")},
                       {"mse_uniform_ratio": 1.1}, {"epsilon": 0},
                       {"similarity_weight": 0, "distance_weight": 0, "smoothness_weight": 0}):
            with self.assertRaises(ValueError):
                LossConfig(**change)
        with self.assertRaises(TypeError):
            LossConfig(folding_weight=1)


if __name__ == "__main__":
    unittest.main()
