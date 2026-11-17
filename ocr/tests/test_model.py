"""Architecture and gradient tests for the OCR ViT."""
import unittest

import torch
from torch import nn

from ocr.model import TransformerBlock, ViTConfig, VisionTransformerOCR


class VisionTransformerOCRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = ViTConfig(
            image_size=16, patch_size=4, num_classes=7,
            hidden_size=32, depth=2, num_heads=4, mlp_ratio=2.0,
        )
        self.model = VisionTransformerOCR(self.config)

    def test_sequence_has_cls_and_patches_with_one_position_addition(self):
        images = torch.randn(3, 1, 16, 16)
        tokens = self.model.tokenize(images)
        patches = self.model.patch_embed(images)
        expected = torch.cat((self.model.cls_token.expand(3, -1, -1), patches), dim=1)
        expected = expected + self.model.position_embedding
        self.assertEqual(tokens.shape, (3, 17, 32))
        self.assertTrue(torch.allclose(tokens, expected))

    def test_only_cls_features_enter_classifier(self):
        seen = []
        handle = self.model.head[0].register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].shape))
        logits = self.model(torch.randn(5, 1, 16, 16))
        handle.remove()
        self.assertEqual(logits.shape, (5, 7))
        self.assertEqual(seen, [torch.Size((5, 32))])

    def test_block_has_two_residual_branches(self):
        class ZeroAttention(nn.Module):
            def forward(self, query, _key, _value, need_weights=False):
                return torch.zeros_like(query), None

        block = TransformerBlock(self.config)
        block.attention = ZeroAttention()
        block.mlp = nn.Sequential(nn.Linear(32, 32))
        nn.init.zeros_(block.mlp[0].weight)
        nn.init.zeros_(block.mlp[0].bias)
        x = torch.randn(2, 17, 32)
        self.assertTrue(torch.equal(block(x), x))

    def test_forward_and_backward_have_finite_nonzero_gradients(self):
        images = torch.randn(8, 1, 16, 16)
        labels = torch.randint(0, 7, (8,))
        loss = nn.functional.cross_entropy(self.model(images), labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name in ("cls_token", "position_embedding"):
            grad = getattr(self.model, name).grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.norm().item(), 0, name)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_small_batch_can_be_overfit(self):
        torch.manual_seed(11)
        config = ViTConfig(image_size=16, patch_size=4, num_classes=4,
                           hidden_size=32, depth=2, num_heads=4, mlp_ratio=2.0)
        model = VisionTransformerOCR(config)
        images = torch.randn(12, 1, 16, 16)
        labels = torch.arange(12) % 4
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
        with torch.no_grad():
            initial = nn.functional.cross_entropy(model(images), labels).item()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            final = nn.functional.cross_entropy(model(images), labels).item()
        self.assertLess(final, initial * 0.1, f"initial={initial:.4f}, final={final:.4f}")

    def test_invalid_image_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 1, 12, 16))


if __name__ == "__main__":
    unittest.main()
