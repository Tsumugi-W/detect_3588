"""Unit tests for the tag-relative board-card localizer."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from panel_detection.board_card_localizer import (
    BoardCardLocalizer,
    locate_card_strip,
)


class _Marker:
    tag_id = 7
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )


def _image(
    colour=(20, 20, 20),
    x0=250,
    x1=330,
    y0=200,
    y1=600,
    valid_bottom=799,
):
    image = np.full((800, 560, 3), 225, dtype=np.uint8)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, -1)
    valid = np.ones((800, 560), dtype=bool)
    valid[valid_bottom + 1 :] = False
    return image, valid


def _context(image, valid):
    identity = np.eye(3, dtype=np.float64)
    return {
        "roi": image,
        "valid": valid,
        "roi_to_image": identity,
        "image_to_roi": identity,
        "transform": identity,
        "tag": _Marker(),
    }


class BoardCardStripTest(unittest.TestCase):
    def test_black_strip_geometry(self):
        image, valid = _image()
        result = locate_card_strip(image, valid)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["top_px"][0], 290.0, delta=2.0)
        self.assertAlmostEqual(result["top_px"][1], 200.0, delta=2.0)
        self.assertAlmostEqual(result["bottom_px"][1], 600.0, delta=2.0)
        self.assertAlmostEqual(result["width_px"], 80.0, delta=2.0)
        self.assertFalse(result["clipped"])
        self.assertGreater(result["confidence"], 0.6)

    def test_blue_strip_geometry(self):
        image, valid = _image(colour=(120, 60, 20), x0=260, x1=320)
        result = locate_card_strip(image, valid)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["top_px"][0], 290.0, delta=2.0)
        self.assertAlmostEqual(result["width_px"], 60.0, delta=2.0)
        self.assertFalse(result["clipped"])

    def test_clipped_strip(self):
        image, valid = _image(y1=399, valid_bottom=399)
        result = locate_card_strip(image, valid)
        self.assertIsNotNone(result)
        self.assertTrue(result["clipped"])
        self.assertLess(result["visible_length_px"], 220.0)


class BoardCardLocalizerTest(unittest.TestCase):
    def test_nominal_length_keeps_centre_stable(self):
        localizer = BoardCardLocalizer({"min_confidence": 0.1})
        marker = _Marker()

        full, full_valid = _image(y1=600)
        context = _context(full, full_valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            first = localizer.observe(full, marker=marker)
            self.assertIsNotNone(first)
            self.assertFalse(first["clipped"])
            self.assertAlmostEqual(first["center_px"][1], 400.0, delta=2.0)

            clipped, clipped_valid = _image(y1=399, valid_bottom=399)
            context = _context(clipped, clipped_valid)
            second = localizer.observe(clipped, marker=marker)

        self.assertIsNotNone(second)
        self.assertTrue(second["clipped"])
        self.assertTrue(second["length_estimated"])
        self.assertAlmostEqual(second["center_px"][1], 400.0, delta=2.0)
        self.assertAlmostEqual(
            second["center_px"][1], first["center_px"][1], delta=2.0
        )
        self.assertEqual(second["insertion_state"], "pulled")

    def test_depth_plane_position(self):
        image, valid = _image()
        depth = np.full((800, 560), 500, dtype=np.uint16)
        intrinsics = {
            "K": [500.0, 0.0, 280.0, 0.0, 500.0, 400.0, 0.0, 0.0, 1.0]
        }
        localizer = BoardCardLocalizer({"min_confidence": 0.1})
        context = _context(image, valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            result = localizer.observe(
                image, depth=depth, camera_info=intrinsics, marker=_Marker()
            )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["position"])
        self.assertAlmostEqual(result["position"][2], 0.5, delta=0.01)
        self.assertAlmostEqual(result["plane_rms_m"], 0.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
