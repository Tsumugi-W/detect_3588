import cv2
import numpy as np

from panel_detection.knob_angle import estimate_valve_angle


def _draw_valve_like_cross(angle_deg):
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    center = np.array([80.0, 80.0])
    red = (0, 0, 220)
    radius_outer = 58
    radius_inner = 18

    cv2.circle(image, tuple(center.astype(int)), radius_outer, red, 10)
    cv2.circle(image, tuple(center.astype(int)), radius_inner, (20, 20, 20), -1)

    for base in (angle_deg, angle_deg + 45, angle_deg + 90, angle_deg + 135):
        rad = np.radians(base)
        direction = np.array([np.cos(rad), np.sin(rad)])
        p1 = center + direction * 18
        p2 = center + direction * 58
        cv2.line(image, tuple(np.round(p1).astype(int)),
                 tuple(np.round(p2).astype(int)), red, 9, cv2.LINE_AA)
        p3 = center - direction * 18
        p4 = center - direction * 58
        cv2.line(image, tuple(np.round(p3).astype(int)),
                 tuple(np.round(p4).astype(int)), red, 9, cv2.LINE_AA)

    return image


def test_valve_angle_uses_red_cross_horizontal_angle():
    roi = _draw_valve_like_cross(17.0)

    angle = estimate_valve_angle(roi)

    assert angle is not None
    assert abs(angle - 17.0) < 4.0


def test_valve_angle_wraps_to_octagon_symmetry_range():
    roi = _draw_valve_like_cross(58.0)

    angle = estimate_valve_angle(roi)

    assert angle is not None
    assert abs(angle - 13.0) < 4.0
