import cv2
import numpy as np

from panel_detection.depth_utils import get_masked_robust_depth
from panel_detection.nut_localizer import localize_nut


def _hex_points(center, radius, angle_deg=30.0):
    angles = np.deg2rad(np.arange(6) * 60.0 + angle_deg)
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])
    return np.round(pts).astype(np.int32)


def test_localize_nut_prefers_outer_hex_center():
    image = np.full((160, 160, 3), 35, dtype=np.uint8)
    true_center = (80, 80)
    cv2.fillConvexPoly(image, _hex_points(true_center, 46), (170, 170, 168))
    cv2.circle(image, true_center, 18, (95, 95, 95), -1)
    cv2.circle(image, true_center, 9, (55, 55, 55), -1)

    # Simulate a YOLO box biased toward the central screw instead of the hex.
    bbox = (45, 45, 125, 125)
    loc = localize_nut(image, bbox)

    assert loc is not None
    assert loc.confidence >= 0.45
    assert np.linalg.norm(np.array(loc.center) - np.array(true_center)) <= 3.0
    assert loc.depth_mask[true_center[1], true_center[0]] == 0


def test_masked_depth_ignores_center_screw_depth():
    depth = np.full((80, 80), 1000, dtype=np.uint16)
    depth[35:45, 35:45] = 700

    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 24, 255, -1)
    cv2.circle(mask, (40, 40), 10, 0, -1)

    assert get_masked_robust_depth(depth, mask, depth_scale=0.001) == 1.0
