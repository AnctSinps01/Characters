"""Intensity preprocessing must preserve faint ink and reject empty shapes."""
import unittest

import torch

from twist.images import ConstantImageError, normalize_minmax


class ImageNormalizationTests(unittest.TestCase):
    def test_minmax_retains_gray_coverage_and_is_per_image(self):
        values = torch.tensor([[[[200, 225], [250, 225]]],
                               [[[160, 180], [200, 180]]]], dtype=torch.uint8)
        expected = torch.tensor([[[[-1., 0.], [1., 0.]]]]).expand(2, -1, -1, -1)
        actual = normalize_minmax(values)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_constant_images_report_their_batch_positions(self):
        values = torch.tensor([[[[200, 255], [200, 255]]],
                               [[[255, 255], [255, 255]]],
                               [[[0, 0], [0, 0]]]], dtype=torch.uint8)
        with self.assertRaises(ConstantImageError) as caught:
            normalize_minmax(values)
        self.assertEqual(caught.exception.indices, [1, 2])


if __name__ == "__main__":
    unittest.main()
