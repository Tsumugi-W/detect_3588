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
    # Large enough for the panel-plane sampler; the ROI is mocked in these
    # tests, so only the depth-region geometry uses the tag side length.
    corners = np.array(
        [[100.0, 100.0], [180.0, 100.0], [180.0, 180.0], [100.0, 180.0]],
        dtype=np.float32,
    )
    center_xy = (140.0, 140.0)


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
    # Same tag->ROI mapping as the production ROI: 80 px per tag side,
    # x offset +240 px and y offset -120 px (tag corner coordinates).
    transform = np.array(
        [[80.0, 0.0, 240.0], [0.0, 80.0, -120.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        "roi": image,
        "valid": valid,
        "roi_to_image": identity,
        "image_to_roi": identity,
        "transform": transform,
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

    def test_blue_strip_is_not_card(self):
        image, valid = _image(colour=(120, 60, 20), x0=260, x1=320)
        result = locate_card_strip(image, valid)
        self.assertIsNone(result)

    def test_clipped_strip(self):
        image, valid = _image(y1=399, valid_bottom=399)
        result = locate_card_strip(image, valid)
        self.assertIsNotNone(result)
        self.assertTrue(result["clipped"])
        self.assertLess(result["visible_length_px"], 220.0)


class BoardCardLocalizerTest(unittest.TestCase):
    def test_inserted_uses_fixed_card_length(self):
        localizer = BoardCardLocalizer({"min_confidence": 0.1})
        marker = _Marker()

        full, full_valid = _image(y1=600)
        context = _context(full, full_valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            first = None
            for _ in range(3):
                first = localizer.observe(full, marker=marker)
            self.assertIsNotNone(first)
            self.assertEqual(first["insertion_state"], "inserted")
            self.assertFalse(first["clipped"])
            self.assertAlmostEqual(first["center_px"][1], 485.2, delta=2.0)

            # Fully inserted card whose bottom is clipped by the image edge:
            # still inserted, and the canonical centre stays fixed.
            clipped, clipped_valid = _image(y1=520, valid_bottom=520)
            context = _context(clipped, clipped_valid)
            second = localizer.observe(clipped, marker=marker)

        self.assertIsNotNone(second)
        self.assertTrue(second["clipped"])
        self.assertFalse(second["length_estimated"])
        self.assertAlmostEqual(second["center_px"][1], 485.2, delta=2.0)
        self.assertAlmostEqual(
            second["center_px"][1], first["center_px"][1], delta=2.0
        )
        self.assertEqual(second["insertion_state"], "inserted")

    def test_empty_slot_requires_confirmation(self):
        image, valid = _image(y0=700, y1=701)
        context = _context(image, valid)
        localizer = BoardCardLocalizer({"min_confidence": 0.1})
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            states = []
            for _ in range(5):
                observation = localizer.observe(image, marker=_Marker())
                states.append(observation["insertion_state"])
        self.assertEqual(states[:4], ["unknown"] * 4)
        self.assertEqual(states[4], "empty")
        self.assertFalse(observation["card_present"])
        self.assertIsNone(observation["position"])
        self.assertIsNotNone(observation["slot_px"])
        self.assertTrue(observation["stable"])

    def test_short_dark_blob_is_empty(self):
        image, valid = _image(y0=650, y1=750)
        context = _context(image, valid)
        localizer = BoardCardLocalizer({"min_confidence": 0.1})
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            for _ in range(5):
                observation = localizer.observe(image, marker=_Marker())
        self.assertEqual(observation["insertion_state"], "empty")

    def test_pose_lock_ignores_edge_jitter(self):
        localizer = BoardCardLocalizer(
            {
                "min_confidence": 0.1,
                "inserted_confirm_frames": 1,
                "pose_lock_frames": 4,
            }
        )
        marker = _Marker()
        top_a, valid = _image(y0=200, y1=600)
        top_b, _ = _image(y0=204, y1=600)
        context = _context(top_a, valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            observations = []
            for i in range(8):
                image = top_a if i % 2 == 0 else top_b
                context = _context(image, valid)
                observations.append(localizer.observe(image, marker=marker))
        self.assertTrue(observations[-1]["pose_locked"])
        self.assertAlmostEqual(
            observations[-1]["anchor_px"][1],
            observations[-2]["anchor_px"][1],
            delta=0.5,
        )

    def test_panel_plane_offset(self):
        image, valid = _image()
        depth = np.full((800, 560), 500, dtype=np.uint16)
        intrinsics = {
            "K": [500.0, 0.0, 280.0, 0.0, 500.0, 400.0, 0.0, 0.0, 1.0]
        }
        localizer = BoardCardLocalizer(
            {"min_confidence": 0.1, "panel_depth_offset_m": 0.01}
        )
        context = _context(image, valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            result = None
            for _ in range(5):
                result = localizer.observe(
                    image, depth=depth, camera_info=intrinsics, marker=_Marker()
                )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["position"][2], 0.49, delta=0.01)
        self.assertEqual(result["depth_source"], "panel_plane")

    def test_stillness_locks_final_position(self):
        image, valid = _image()
        depth = np.full((800, 560), 500, dtype=np.uint16)
        intrinsics = {
            "K": [500.0, 0.0, 280.0, 0.0, 500.0, 400.0, 0.0, 0.0, 1.0]
        }
        localizer = BoardCardLocalizer(
            {"min_confidence": 0.1, "still_frames": 3}
        )
        context = _context(image, valid)
        with patch(
            "panel_detection.board_card_localizer.rectify_board_card_roi",
            side_effect=lambda *args, **kwargs: context,
        ):
            result = None
            for _ in range(6):
                result = localizer.observe(
                    image, depth=depth, camera_info=intrinsics, marker=_Marker()
                )
        self.assertTrue(result["position_stable"])
        self.assertAlmostEqual(result["position"][2], 0.5, delta=0.01)

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
            result = None
            for _ in range(3):
                result = localizer.observe(
                    image, depth=depth, camera_info=intrinsics, marker=_Marker()
                )
        self.assertIsNotNone(result)
        self.assertEqual(result["insertion_state"], "inserted")
        self.assertIsNotNone(result["position"])
        self.assertAlmostEqual(result["position"][2], 0.5, delta=0.01)
        self.assertAlmostEqual(result["plane_rms_m"], 0.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
