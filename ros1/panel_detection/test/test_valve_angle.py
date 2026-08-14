import cv2
import numpy as np

from panel_detection.knob_angle import estimate_valve_angle, estimate_valve_angle_candidates


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


def test_valve_angle_prefers_spokes_over_outer_octagon_edges():
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    center = np.array([90.0, 90.0])
    red = (0, 0, 220)

    # Strong outer octagon edges are deliberately rotated away from the spoke
    # angle.  The valve angle should still follow the red cross/spoke geometry.
    outer_pts = []
    for base in np.arange(8) * 45.0 + 30.0:
        rad = np.radians(base)
        outer_pts.append(center + np.array([np.cos(rad), np.sin(rad)]) * 70.0)
    cv2.polylines(image, [np.round(outer_pts).astype(np.int32)],
                  isClosed=True, color=red, thickness=10, lineType=cv2.LINE_AA)

    for base in (0.0, 45.0, 90.0, 135.0):
        rad = np.radians(base)
        direction = np.array([np.cos(rad), np.sin(rad)])
        p1 = center + direction * 18
        p2 = center + direction * 58
        p3 = center - direction * 18
        p4 = center - direction * 58
        cv2.line(image, tuple(np.round(p1).astype(int)),
                 tuple(np.round(p2).astype(int)), red, 8, cv2.LINE_AA)
        cv2.line(image, tuple(np.round(p3).astype(int)),
                 tuple(np.round(p4).astype(int)), red, 8, cv2.LINE_AA)

    angle = estimate_valve_angle(image)

    assert angle is not None
    assert min(abs(angle), abs(angle - 45.0)) < 4.0


def test_valve_angle_candidates_include_secondary_spoke_mode():
    image = _draw_valve_like_cross(17.0)
    cv2.line(image, (20, 90), (140, 90), (0, 0, 220), 14, cv2.LINE_AA)

    candidates = estimate_valve_angle_candidates(image, max_candidates=5)

    assert any(abs(angle - 17.0) < 4.0 for angle in candidates)
