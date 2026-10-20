#!/usr/bin/env python3
"""Regression tests for ROS-independent knob angle estimation."""

import unittest

import cv2
import numpy as np

from panel_detection.knob_angle import (
    DiscreteAngleStabilizer,
    estimate_knob_angle,
)


def _marked_knob(first, second, brightness=255):
    image = np.full((120, 120, 3), 30, dtype=np.uint8)
    marker = (brightness, brightness, brightness)
    cv2.line(image, first, second, marker, 8, cv2.LINE_AA)
    return image


class KnobAngleTest(unittest.TestCase):
    def test_vertical_marker_is_zero_degrees(self):
        angle = estimate_knob_angle(
            _marked_knob((60, 20), (60, 100)),
            pointer_color="white_line",
        )
        self.assertIsNotNone(angle)
        # An unoriented vertical axis may be represented as either 0 or 180
        # before the ROS2-compatible stabilizer maps it to discrete 0.
        normalized = float(angle) % 180.0
        vertical_difference = min(normalized, 180.0 - normalized)
        self.assertAlmostEqual(vertical_difference, 0.0, delta=2.0)
        self.assertEqual(DiscreteAngleStabilizer().update(4, angle), 0.0)

    def test_horizontal_marker_is_ninety_degrees(self):
        angle = estimate_knob_angle(
            _marked_knob((20, 60), (100, 60)),
            pointer_color="white_line",
        )
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, delta=2.0)

    def test_marker_survives_lower_camera_exposure(self):
        angle = estimate_knob_angle(
            _marked_knob((60, 20), (60, 100), brightness=160),
            pointer_color="white_line",
        )
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 0.0, delta=2.0)

    def test_discrete_change_requires_confirmed_frames(self):
        stabilizer = DiscreteAngleStabilizer(
            switch_margin=8.0, confirm_frames=2
        )
        self.assertEqual(stabilizer.update(4, 5.0), 0.0)
        self.assertEqual(stabilizer.update(4, 70.0), 0.0)
        self.assertEqual(stabilizer.update(4, 70.0), 90.0)

    def test_red_handle_mode_detects_downward_handle(self):
        image = np.full((160, 160, 3), 25, dtype=np.uint8)
        cv2.circle(image, (80, 70), 32, (0, 0, 210), -1)
        cv2.rectangle(image, (70, 70), (90, 145), (0, 0, 210), -1)
        angle = estimate_knob_angle(
            image,
            pointer_color="red_handle",
            angle_range=(180.0, 270.0),
        )
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, delta=3.0)
        self.assertEqual(DiscreteAngleStabilizer().update(4, angle), 0.0)

    def test_red_handle_unconstrained_mode_is_available(self):
        image = np.full((160, 160, 3), 25, dtype=np.uint8)
        cv2.circle(image, (80, 80), 32, (0, 0, 210), -1)
        handle = np.asarray([[80, 80], [145, 95], [141, 111]], np.int32)
        cv2.fillConvexPoly(image, handle, (0, 0, 210))
        angle = estimate_knob_angle(
            image,
            pointer_color="red_handle",
            angle_range=None,
        )
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 112.0, delta=3.0)


if __name__ == "__main__":
    unittest.main()
