#!/usr/bin/env python3
"""Regression tests for the ROS2-compatible eye-in-hand filter."""

import unittest

from panel_detection.position_stabilizer import PositionStabilizer


class PositionStabilizerTest(unittest.TestCase):
    def test_camera_motion_discards_old_stable_position(self):
        stabilizer = PositionStabilizer(
            still_time=0.2,
            pixel_thresh=2.0,
            window_size=5,
            ema_alpha=0.25,
            depth_std_thresh=0.01,
        )
        stabilizer.update([(3, (10, 10), (0.1, 0.0, 0.5))], 0.0)
        stabilizer.update([(3, (10, 10), (0.1, 0.0, 0.5))], 0.1)
        stabilizer.update([(3, (10, 10), (0.1, 0.0, 0.5))], 0.3)

        moved = stabilizer.update(
            [(3, (30, 10), (0.2, 0.0, 0.6))], 0.4
        )

        self.assertEqual(moved[3], [0.2, 0.0, 0.6])


if __name__ == "__main__":
    unittest.main()
