import cv2
import numpy as np

from panel_detection.concentric_bolt_marker import (
    detect_concentric_bolt_markers,
    estimate_concentric_bolt_poses,
)


class _Intrinsics:
    fx = 600.0
    fy = 600.0
    cx = 160.0
    cy = 120.0


def _draw_marker(image, center=(160, 120), radius=34, gap_angle=35.0):
    center = tuple(int(value) for value in center)
    cv2.circle(image, center, radius, (235, 235, 235), -1)
    cv2.circle(image, center, int(radius * 0.78), (25, 25, 25), -1)
    cv2.circle(image, center, int(radius * 0.61), (235, 235, 235), -1)
    ring_radius = int(radius * 0.40)
    gap_half_width = 22.0
    cv2.ellipse(
        image, center, (ring_radius, ring_radius), 0.0,
        gap_angle + gap_half_width,
        gap_angle + 360.0 - gap_half_width,
        (25, 25, 25), max(3, int(radius * 0.15)))
    cv2.circle(image, center, max(2, int(radius * 0.10)), (25, 25, 25), -1)


def _angle_error(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_detects_center_and_orientation_gap_without_yolo():
    image = np.full((240, 320, 3), 45, dtype=np.uint8)
    _draw_marker(image)

    detections = detect_concentric_bolt_markers(image)

    assert len(detections) == 1
    marker = detections[0]
    assert np.linalg.norm(np.asarray(marker.center) - (160.0, 120.0)) < 0.5
    assert marker.gap_angle_deg is not None
    assert _angle_error(marker.gap_angle_deg, 35.0) < 8.0
    assert marker.confidence > 0.75


def test_plain_circle_with_center_dot_is_rejected():
    image = np.full((240, 320, 3), 45, dtype=np.uint8)
    cv2.circle(image, (160, 120), 30, (235, 235, 235), -1)
    cv2.circle(image, (160, 120), 4, (25, 25, 25), -1)

    assert detect_concentric_bolt_markers(image) == []


def test_shared_plane_fills_a_marker_with_no_depth():
    image = np.full((240, 320, 3), 45, dtype=np.uint8)
    _draw_marker(image, center=(110, 120), gap_angle=0.0)
    _draw_marker(image, center=(210, 120), gap_angle=180.0)
    markers = detect_concentric_bolt_markers(image)
    assert len(markers) == 2

    depth = np.zeros((240, 320), dtype=np.uint16)
    depth[100:141, 90:131] = 500
    poses = estimate_concentric_bolt_poses(
        markers, depth, _Intrinsics(), normal=[0.0, 0.0, -1.0])

    assert all(pose.point_3d is not None for pose in poses)
    assert all(abs(float(pose.point_3d[2]) - 0.5) < 1e-6 for pose in poses)
    assert all(pose.depth_source == 'single_depth_plane' for pose in poses)
