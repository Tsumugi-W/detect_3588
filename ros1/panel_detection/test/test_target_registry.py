#!/usr/bin/env python3
"""Regression tests for ROS2-compatible panel-ID matching."""

import unittest

import cv2
import numpy as np

from panel_detection.target_registry import (
    FrameDetection,
    TargetRegistry,
)


def _detection(class_name, x1, color_bgr):
    x2 = x1 + 30
    y1, y2 = 20, 50
    detection = FrameDetection(
        class_name=class_name,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        bbox=(x1, y1, x2, y2),
        confidence=0.9,
    )
    return detection, (x1, y1, x2, y2, color_bgr)


def _image_with_detections(draws):
    max_x = max((x2 for _, _, x2, _, _ in draws), default=220)
    image = np.zeros(
        (80, max(220, max_x + 20), 3), dtype=np.uint8
    )
    for x1, y1, x2, y2, color_bgr in draws:
        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, -1)
    return image


class Ros2PanelLayoutParityTest(unittest.TestCase):
    def test_known_panel_window_0_to_3_keeps_absolute_ids(self):
        """Guard the reversed 0..6 main-row layout used by ROS1."""

        green, green_draw = _detection(
            "button", 20, (0, 255, 0)
        )
        red_1, red_1_draw = _detection(
            "button", 70, (0, 0, 255)
        )
        black_knob, black_knob_draw = _detection(
            "knob", 120, (0, 0, 0)
        )
        red_knob, red_knob_draw = _detection("knob", 170, (0, 0, 255))
        image = _image_with_detections(
            [green_draw, red_1_draw, black_knob_draw, red_knob_draw]
        )

        matched = TargetRegistry().identify(
            [green, red_1, black_knob, red_knob], image
        )

        self.assertEqual(
            [target_id for target_id, _ in matched],
            [0, 1, 2, 3],
        )


class CameraIndependentRegistryTest(unittest.TestCase):
    def test_legacy_subset_does_not_hide_other_visible_controls(self):
        green, g = _detection("button", 20, (0, 255, 0))
        red, r = _detection("button", 70, (0, 0, 255))
        image = _image_with_detections([g, r])
        expected = TargetRegistry().identify([green, red], image)
        with self.assertWarns(DeprecationWarning):
            old_config = TargetRegistry(allowed_ids=[5, 6])
        actual = old_config.identify([green, red], image)
        self.assertEqual([i for i, _ in actual], [i for i, _ in expected])
        self.assertEqual(len(actual), 2)


if __name__ == "__main__":
    unittest.main()
