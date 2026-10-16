from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import patch

import torch

from src.config import TrainingConfig
from src.diagnostics import GradientSnapshot
from src.train import make_grad_scaler, optimizer_update


class GradientDiagnosticTests(unittest.TestCase):
    def test_matches_standard_clipping_on_finite_gradients(self):
        torch.manual_seed(14)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        model(torch.randn(5, 3)).square().mean().backward()
        expected = copy.deepcopy(model)
        for actual, reference in zip(model.parameters(), expected.parameters()):
            reference.grad = actual.grad.clone()
        expected_norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 0.1)
        snapshot = GradientSnapshot.capture(model)
        report = snapshot.report(0.1)
        self.assertFalse(snapshot.used_float64)
        torch.testing.assert_close(snapshot.total, expected_norm)
        self.assertEqual(report["event"], "gradients")
        self.assertEqual(report["nonfinite_parameter_count"], 0)
        self.assertGreaterEqual(report["top_parameters"][0]["norm"],
                                report["top_parameters"][-1]["norm"])
        json.dumps(report, allow_nan=False)
        snapshot.clip(0.1)
        for actual, reference in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual.grad, reference.grad, atol=0, rtol=0)

    def test_finite_large_gradients_survive_norm_overflow(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[1e20, -1e20]])
        self.assertFalse(torch.isfinite(torch.linalg.vector_norm(model.weight.grad)))
        snapshot = GradientSnapshot.capture(model)
        self.assertTrue(snapshot.used_float64)
        self.assertTrue(torch.isfinite(snapshot.total))
        report = snapshot.report(1.0)
        self.assertEqual(report["nonfinite_parameter_count"], 0)
        self.assertTrue(report["norm_float64_fallback"])
        self.assertAlmostEqual(report["grad_norm"] / 1e20, math.sqrt(2), places=6)
        snapshot.clip(1.0)
        torch.testing.assert_close(model.weight.grad,
                                   torch.tensor([[1 / math.sqrt(2), -1 / math.sqrt(2)]]))

    def test_nonfinite_gradients_are_reported_without_mutation(self):
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.tensor([[float("inf"), float("nan")]])
        model.bias.grad = torch.ones_like(model.bias)
        snapshot = GradientSnapshot.capture(model)
        report = snapshot.report(1.0)
        self.assertEqual(report["event"], "nonfinite_gradients")
        self.assertEqual(report["nonfinite_parameters"], ["weight"])
        self.assertEqual(report["grad_norm"], "nan")
        json.dumps(report, allow_nan=False)
        with self.assertRaises(FloatingPointError):
            snapshot.clip(1.0)
        self.assertTrue(torch.isinf(model.weight.grad[0, 0]))
        self.assertTrue(torch.isnan(model.weight.grad[0, 1]))
        torch.testing.assert_close(model.bias.grad, torch.ones_like(model.bias))

    def test_nonfinite_backward_stops_before_optimizer_update(self):
        class InfiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return torch.full_like(gradient, float("inf"))

        model = torch.nn.Linear(1, 1)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = make_grad_scaler(torch.device("cpu"), enabled=False)
        reports = []
        with patch("src.train.batch_loss", side_effect=lambda model, batch:
                   InfiniteBackward.apply(model.weight).sum()):
            with self.assertRaisesRegex(FloatingPointError, "non-finite gradients in weight"):
                optimizer_update(model, optimizer, scaler, {}, torch.device("cpu"),
                                 "fp32", TrainingConfig(), on_diagnostic=reports.append)
        self.assertFalse(optimizer.state)
        self.assertEqual(reports[0]["event"], "nonfinite_gradients")
        self.assertTrue(torch.isinf(model.weight.grad).all())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_nonfinite_loss_is_logged_without_backward(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = make_grad_scaler(torch.device("cpu"), enabled=False)
        reports = []
        with patch("src.train.batch_loss", return_value=torch.tensor(float("inf"))):
            with self.assertRaisesRegex(FloatingPointError, "non-finite training loss"):
                optimizer_update(model, optimizer, scaler, {}, torch.device("cpu"),
                                 "fp32", TrainingConfig(), on_diagnostic=reports.append)
        self.assertEqual(reports[0]["event"], "nonfinite_loss")
        self.assertEqual(reports[0]["loss"], "inf")
        self.assertFalse(optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
