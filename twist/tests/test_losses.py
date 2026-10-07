from __future__ import annotations

import unittest

import torch

from twist import LossConfig, ModelConfig, Twist, distance_fields, registration_loss
from twist.geometry import identity_grid, sample_image
from twist.losses import (bending_energy, deformation_components, gradient_weighted_mse,
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

    def test_bending_allows_affine_fields_and_includes_mixed_derivatives(self) -> None:
        grid = identity_grid(1, (7, 7), device=torch.device("cpu"))
        x, y = grid[..., 0], grid[..., 1]
        affine = torch.stack((0.3 * x + 0.2 * y, -0.2 * x + 0.1), dim=1)
        self.assertLess(bending_energy(affine).item(), 1e-8)
        mixed = torch.stack((x * y, torch.zeros_like(x)), dim=1)
        self.assertAlmostEqual(bending_energy(mixed).item(), 2, places=5)

    def test_bending_grid_spacing_makes_quadratic_energy_resolution_consistent(self) -> None:
        for size in (5, 9, 17):
            grid = identity_grid(1, (size, size), device=torch.device("cpu"))
            x = grid[..., 0]
            field = torch.stack((x.square(), torch.zeros_like(x)), dim=1)
            self.assertAlmostEqual(bending_energy(field).item(), 4)

    def test_three_configured_loss_proportions_and_inner_displacement_ratio(self) -> None:
        source = torch.ones(1, 1, 16, 16)
        target = torch.ones_like(source)
        source[:, :, 3:13, 4:6] = -1
        target[:, :, 3:13, 8:10] = -1
        model = Twist(ModelConfig(image_size=16, patch_size=4, token_dim=16, num_heads=4,
                                  blocks_per_stage=1, num_stages=1))
        result = model(source, target)
        config = LossConfig(mse_weight=2, distance_weight=3, deformation_weight=0.2,
                            displacement_ratio=0.05)
        loss = registration_loss(result, target, distance_fields(source), config)
        torch.testing.assert_close(loss.deformation, loss.bending + 0.05 * loss.displacement)
        torch.testing.assert_close(loss.total, 2 * loss.mse + 3 * loss.distance + 0.2 * loss.deformation)
        bending, movement = deformation_components(result.displacements)
        torch.testing.assert_close(loss.bending, bending)
        torch.testing.assert_close(loss.displacement, movement)


if __name__ == "__main__":
    unittest.main()
