#!/usr/bin/env python3
"""Regression tests for ROS2-compatible panel-ID matching."""

import unittest

import cv2
import numpy as np

from panel_detection.target_registry import (
    FrameDetection,
    TargetRegistry,
    _match_individual_allowed,
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


class RestrictedTargetRegistryTest(unittest.TestCase):
    def test_single_red_right_button_is_id5(self):
        self.assertEqual(
            _match_individual_allowed([("button", "red")], [5, 6]),
            [5],
        )

    def test_single_green_right_button_is_id6(self):
        self.assertEqual(
            _match_individual_allowed([("button", "green")], [5, 6]),
            [6],
        )

    def test_colour_identity_does_not_depend_on_image_order(self):
        observed = [("button", "green"), ("button", "red")]
        self.assertEqual(
            _match_individual_allowed(observed, [5, 6]),
            [6, 5],
        )

    def test_unrelated_detection_is_not_relabelled(self):
        observed = [
            ("knob", "red"),
            ("button", "red"),
            ("button", "green"),
        ]
        self.assertEqual(
            _match_individual_allowed(observed, [5, 6]),
            [-1, 5, 6],
        )

    def test_unknown_colours_are_rejected(self):
        observed = [("button", "unknown"), ("button", "unknown")]
        self.assertEqual(
            _match_individual_allowed(observed, [5, 6]),
            [-1, -1],
        )

    def test_single_unknown_button_is_rejected_as_ambiguous(self):
        self.assertEqual(
            _match_individual_allowed([("button", "unknown")], [5, 6]),
            [-1],
        )

    def test_matching_depends_on_panel_order_not_image_order(self):
        """Document why the wrist view needs a panel axis.

        ``_match_individual_allowed`` consumes the observations in the order it
        is given.  A restricted wrist view must therefore hand them over in
        physical panel order; sorting by raw image x (the knob-free cold start)
        produces the reversed case below and loses ID5 entirely.
        """

        # True panel order: ID5 red first, then ID6 whose colour was unreadable.
        self.assertEqual(
            _match_individual_allowed(
                [("button", "red"), ("button", "unknown")], [5, 6]
            ),
            [5, 6],
        )
        # Mirrored camera image: the unusable colour now comes first.
        self.assertEqual(
            _match_individual_allowed(
                [("button", "unknown"), ("button", "red")], [5, 6]
            ),
            [-1, 5],
        )

    def test_allowed_ids_are_validated(self):
        with self.assertRaises(ValueError):
            TargetRegistry(allowed_ids=[5, 5])
        with self.assertRaises(ValueError):
            TargetRegistry(allowed_ids=[8])

    def test_zero_is_a_valid_main_row_id(self):
        TargetRegistry(allowed_ids=[0])


if __name__ == "__main__":
    unittest.main()
