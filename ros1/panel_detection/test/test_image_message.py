#!/usr/bin/env python3
"""Regression tests for cv_bridge-independent debug image publishing."""

import unittest

import numpy as np

from panel_detection.image_message import bgr8_image_message


class FakeImage:
    pass


class Bgr8ImageMessageTest(unittest.TestCase):
    def test_packs_bgr_image_with_ros_layout(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        header = object()

        message = bgr8_image_message(image, FakeImage, header)

        self.assertIs(message.header, header)
        self.assertEqual(message.height, 4)
        self.assertEqual(message.width, 5)
        self.assertEqual(message.encoding, "bgr8")
        self.assertEqual(message.is_bigendian, 0)
        self.assertEqual(message.step, 15)
        self.assertEqual(message.data, image.tobytes())


if __name__ == "__main__":
    unittest.main()
