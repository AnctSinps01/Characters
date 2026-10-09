from __future__ import annotations

import unittest

import torch

from twist.geometry import (compose_grids, identity_grid, jacobian_determinant,
                            sample_displacement, sample_image, scaling_and_squaring,
                            velocity_grid, zero_boundary)


def logistic_velocity(batch=1, size=65, rate=0.8):
    nodes = identity_grid(batch, (size, size), device=torch.device("cpu"), dtype=torch.float64)
    x, y = nodes[..., 0], nodes[..., 1]
    return zero_boundary(torch.stack((rate * x * (1 - x) * torch.sin(torch.pi * y),
                                      torch.zeros_like(x)), dim=1))


def logistic_flow(time, size=65, rate=0.8):
    nodes = identity_grid(1, (size, size), device=torch.device("cpu"), dtype=torch.float64).clone()
    x, y = nodes[..., 0].clone(), nodes[..., 1]
    factor = torch.exp(rate * torch.sin(torch.pi * y) * time)
    nodes[..., 0] = x * factor / (1 - x + x * factor)
    return nodes


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_boundary_projection_preserves_unrestricted_interior_and_gradients(self):
        raw = torch.full((2, 2, 7, 9), 3.0, requires_grad=True)
        field = zero_boundary(raw)
        torch.testing.assert_close(field[:, :, 1:-1, 1:-1], raw[:, :, 1:-1, 1:-1], rtol=0, atol=0)
        for edge in (field[:, :, 0], field[:, :, -1], field[:, :, :, 0], field[:, :, :, -1]):
            self.assertEqual(int(torch.count_nonzero(edge)), 0)
        field.sum().backward()
        torch.testing.assert_close(raw.grad[:, :, 1:-1, 1:-1], torch.ones_like(raw.grad[:, :, 1:-1, 1:-1]))
        self.assertEqual(int(torch.count_nonzero(raw.grad[:, :, 0])), 0)

    def test_squaring_exponent_controls_initial_step_and_self_composition(self):
        field = logistic_velocity(size=17)
        identity = identity_grid(1, (17, 17), device=field.device, dtype=field.dtype)
        initial = sample_displacement(field, identity)
        torch.testing.assert_close(scaling_and_squaring(field, time=0.6, squaring_steps=0),
                                   (0.6 * initial).permute(0, 3, 1, 2))
        small_map = identity + 0.3 * initial
        expected = compose_grids(small_map, small_map) - identity
        torch.testing.assert_close(scaling_and_squaring(field, time=0.6, squaring_steps=1),
                                   expected.permute(0, 3, 1, 2))

    def test_exponential_matches_analytic_nonlinear_flow_and_converges(self):
        velocity = logistic_velocity()
        reference = torch.zeros(1, 1, 65, 65, dtype=torch.float64)
        expected = logistic_flow(1)
        errors = [(velocity_grid(velocity, reference, squaring_steps=n) - expected).abs().max().item()
                  for n in (0, 4, 8)]
        self.assertGreater(errors[0], 10 * errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 3e-4)

    def test_time_scales_velocity_instead_of_final_displacement(self):
        velocity = logistic_velocity()
        reference = torch.zeros(1, 1, 65, 65, dtype=torch.float64)
        identity = identity_grid(1, (65, 65), device=reference.device, dtype=reference.dtype)
        half = velocity_grid(velocity, reference, 0.5, squaring_steps=8)
        endpoint = velocity_grid(velocity, reference, squaring_steps=8)
        torch.testing.assert_close(half, logistic_flow(0.5), rtol=0, atol=3e-4)
        self.assertGreater(float((half - (identity + 0.5 * (endpoint - identity))).abs().max()), 1e-3)

    def test_inverse_and_group_composition_are_accurate_with_discretization(self):
        velocity = logistic_velocity()
        reference = torch.zeros(1, 1, 65, 65, dtype=torch.float64)
        identity = identity_grid(1, (65, 65), device=reference.device, dtype=reference.dtype)
        forward = velocity_grid(velocity, reference, squaring_steps=9)
        reverse = velocity_grid(velocity, reference, -1, squaring_steps=9)
        torch.testing.assert_close(compose_grids(forward, reverse), identity, rtol=0, atol=4e-4)
        first = velocity_grid(velocity, reference, 0.35, squaring_steps=8)
        second = velocity_grid(velocity, reference, 0.25, squaring_steps=8)
        whole = velocity_grid(velocity, reference, 0.6, squaring_steps=8)
        torch.testing.assert_close(compose_grids(first, second), whole, rtol=0, atol=3e-4)
        self.assertGreater(float(jacobian_determinant(forward).min()), 0)

    def test_batched_signed_times_and_fixed_boundary(self):
        velocity = logistic_velocity(batch=2, size=33)
        reference = torch.zeros(2, 1, 33, 33, dtype=torch.float64)
        times = torch.tensor([0.25, -0.9], dtype=torch.float64)
        actual = velocity_grid(velocity, reference, times)
        for index, time in enumerate(times):
            torch.testing.assert_close(actual[index:index+1],
                                       velocity_grid(velocity[index:index+1], reference[index:index+1], time))
        identity = identity_grid(2, (33, 33), device=reference.device, dtype=reference.dtype)
        torch.testing.assert_close(velocity_grid(velocity, reference, 0), identity, rtol=0, atol=0)
        for edge in (0, -1):
            torch.testing.assert_close(actual[:, edge], identity[:, edge], rtol=0, atol=0)
            torch.testing.assert_close(actual[:, :, edge], identity[:, :, edge], rtol=0, atol=0)

    def test_zero_velocity_and_time_have_finite_backward(self):
        velocity = torch.zeros(1, 2, 5, 5, requires_grad=True)
        displacement = scaling_and_squaring(velocity, time=0.4)
        displacement.sum().backward()
        self.assertTrue(bool(torch.isfinite(velocity.grad).all()))
        self.assertEqual(int(torch.count_nonzero(displacement)), 0)

    def test_compression_can_thicken_strokes(self):
        velocity = -logistic_velocity(rate=0.8).float()
        source = torch.ones(1, 1, 65, 65)
        source[:, :, 12:53, 30:35] = -1
        # Centered contraction, zero on every boundary.
        nodes = identity_grid(1, (65, 65), device=source.device)
        x, y = nodes[..., 0], nodes[..., 1]
        velocity[:, 0] = -0.8 * (x - 0.5) * 16 * x * (1-x) * y * (1-y)
        warped = sample_image(source, velocity_grid(zero_boundary(velocity), source))
        self.assertGreater(float(((1 - warped[0, 0, 32]) / 2).sum()),
                           1.1 * float(((1 - source[0, 0, 32]) / 2).sum()))

    def test_integration_requires_pixel_resolution_and_scales_pixels_directly(self):
        reference = torch.zeros(1, 1, 16, 16)
        coarse = torch.zeros(1, 2, 4, 4)
        with self.assertRaisesRegex(ValueError, "same spatial resolution"):
            velocity_grid(coarse, reference)
        velocity = zero_boundary(torch.randn(1, 2, 16, 16))
        torch.testing.assert_close(scaling_and_squaring(velocity, time=0.3, squaring_steps=0),
                                   velocity * 0.3, rtol=0, atol=0)
        boundary_only = torch.ones(1, 2, 2, 2, requires_grad=True)
        displacement = scaling_and_squaring(boundary_only)
        self.assertEqual(displacement.shape, boundary_only.shape)
        self.assertEqual(int(torch.count_nonzero(displacement)), 0)
        displacement.sum().backward()
        self.assertTrue(bool(torch.isfinite(boundary_only.grad).all()))

    def test_invalid_times_shapes_and_squaring_exponents(self):
        field = logistic_velocity(batch=2, size=5)
        for time in (-1.01, 1.1, float("nan"), torch.ones(2, 1)):
            with self.assertRaises(ValueError):
                scaling_and_squaring(field, time=time)
        for exponent in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                scaling_and_squaring(field, squaring_steps=exponent)
        with self.assertRaises(ValueError):
            zero_boundary(torch.zeros(1, 2, 1, 2))


if __name__ == "__main__":
    unittest.main()
