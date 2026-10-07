from __future__ import annotations

import unittest

import torch

from twist.geometry import (compose_displacements, constrain_displacement,
                            identity_grid, lipschitz_bound, sample_displacement,
                            sample_image)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_boundary_projection_and_entire_bilinear_field_bound(self) -> None:
        raw = torch.randn(3, 2, 7, 9, requires_grad=True)
        field = constrain_displacement(raw, 0.75)
        for edge in (field[:, :, 0], field[:, :, -1], field[:, :, :, 0], field[:, :, :, -1]):
            self.assertEqual(torch.count_nonzero(edge).item(), 0)
        self.assertTrue(bool((lipschitz_bound(field) <= 0.75).all()))

        # Check the continuous interpolant, including points inside control cells.
        points = torch.rand(3, 8, 11, 2)
        other = torch.rand_like(points)
        mapped = points + sample_displacement(field, points)
        mapped_other = other + sample_displacement(field, other)
        separation = torch.linalg.vector_norm(mapped - mapped_other, dim=-1)
        initial = torch.linalg.vector_norm(points - other, dim=-1)
        self.assertTrue(bool((separation + 1e-6 >= 0.25 * initial).all()))
        self.assertTrue(bool(((mapped >= -1e-6) & (mapped <= 1 + 1e-6)).all()))
        field.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_zero_displacement_has_finite_backward(self) -> None:
        raw = torch.zeros(1, 2, 5, 5, requires_grad=True)
        constrained = constrain_displacement(raw, 0.8)
        constrained.sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertEqual(torch.count_nonzero(constrained).item(), 0)

    def test_backward_composition_is_ordered_and_not_displacement_addition(self) -> None:
        fields = [constrain_displacement(torch.randn(1, 2, 6, 6), 0.6) for _ in range(2)]
        coordinates = identity_grid(1, (13, 17), device=torch.device("cpu"))
        inner = coordinates + sample_displacement(fields[1], coordinates)
        expected = inner + sample_displacement(fields[0], inner)
        actual = compose_displacements(fields, coordinates)
        torch.testing.assert_close(actual, expected)
        added = coordinates + sum(sample_displacement(field, coordinates) for field in fields)
        self.assertGreater((actual - added).abs().max().item(), 1e-4)

    def test_time_zero_and_fixed_boundary_through_multiple_maps(self) -> None:
        fields = [constrain_displacement(torch.randn(2, 2, 6, 6), 0.7) for _ in range(3)]
        identity = identity_grid(2, (15, 19), device=torch.device("cpu"))
        torch.testing.assert_close(compose_displacements(fields, identity, 0), identity)
        grid = compose_displacements(fields, identity, torch.tensor([0.25, 0.9]))
        for actual, expected in ((grid[:, 0], identity[:, 0]), (grid[:, -1], identity[:, -1]),
                                 (grid[:, :, 0], identity[:, :, 0]), (grid[:, :, -1], identity[:, :, -1])):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_compression_of_sampling_grid_can_thicken_a_stroke(self) -> None:
        nodes = identity_grid(1, (17, 17), device=torch.device("cpu"))
        x, y = nodes[..., 0], nodes[..., 1]
        window = 16 * x * (1 - x) * y * (1 - y)
        raw = torch.stack((-0.3 * (x - 0.5) * window, torch.zeros_like(x)), dim=1)
        field = constrain_displacement(raw, 0.8)
        source = torch.ones(1, 1, 65, 65)
        source[:, :, 12:53, 30:35] = -1
        coordinates = identity_grid(1, (65, 65), device=torch.device("cpu"))
        warped = sample_image(source, compose_displacements([field], coordinates))
        original_width = ((1 - source[0, 0, 32]) / 2).sum()
        new_width = ((1 - warped[0, 0, 32]) / 2).sum()
        self.assertGreater(new_width.item(), original_width.item() * 1.1)

    def test_invalid_time_and_control_shapes_are_rejected(self) -> None:
        coordinates = identity_grid(2, (5, 5), device=torch.device("cpu"))
        for time in (-0.01, 1.1, float("nan"), torch.ones(2, 1)):
            with self.assertRaises(ValueError):
                compose_displacements([], coordinates, time)
        with self.assertRaises(ValueError):
            constrain_displacement(torch.zeros(1, 2, 2, 2), 0.8)


if __name__ == "__main__":
    unittest.main()
