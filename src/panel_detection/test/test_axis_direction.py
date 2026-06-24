import numpy as np

from panel_detection.camera.base import CameraIntrinsics
from panel_detection.depth_utils import (
    estimate_fastener_group_axis_direction,
    estimate_fastener_line_constrained_axis,
    estimate_fastener_patch_axis_direction,
    estimate_mounting_plane_axis_direction,
    estimate_object_axis_direction,
    estimate_valve_wheel_axis_direction,
)


def _intrinsics():
    return CameraIntrinsics(
        width=120,
        height=120,
        fx=100.0,
        fy=100.0,
        cx=60.0,
        cy=60.0,
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_valve_axis_ignores_hollow_center_depth_drop():
    depth = np.full((120, 120), 1000, dtype=np.uint16)
    depth[46:74, 46:74] = 1600

    result = estimate_object_axis_direction(
        depth,
        _intrinsics(),
        bbox=(30, 30, 90, 90),
        object_class='valve',
        depth_scale=0.001,
        sample_stride=1,
        min_points=20,
    )

    assert result is not None
    normal, centroid, point_count = result
    assert point_count >= 20
    assert normal[2] < -0.95
    assert abs(centroid[2] - 1.0) < 0.02


def test_valve_wheel_axis_uses_near_annulus_not_hollow_center():
    depth = np.full((120, 120), 1800, dtype=np.uint16)
    yy, xx = np.mgrid[0:120, 0:120]
    cx, cy = 60, 60
    radius = np.hypot(xx - cx, yy - cy)
    ring = (radius >= 18) & (radius <= 30)
    center_opening = radius < 14
    depth[ring] = 1000
    depth[center_opening] = 1650

    result = estimate_valve_wheel_axis_direction(
        depth,
        _intrinsics(),
        bbox=(30, 30, 90, 90),
        depth_scale=0.001,
        sample_stride=1,
        min_points=20,
    )

    assert result is not None
    normal, centroid, point_count = result
    assert point_count >= 20
    assert normal[2] < -0.95
    assert abs(centroid[2] - 1.0) < 0.02


def test_bolt_axis_estimates_front_face_normal():
    depth = np.full((100, 100), 900, dtype=np.uint16)

    result = estimate_object_axis_direction(
        depth,
        _intrinsics(),
        bbox=(35, 35, 75, 75),
        object_class='bolt',
        depth_scale=0.001,
        sample_stride=1,
        min_points=20,
    )

    assert result is not None
    normal, _, _ = result
    assert normal[2] < -0.95


def test_mounting_plane_axis_isolated_between_two_planes():
    intrin = _intrinsics()
    yy, xx = np.mgrid[0:120, 0:120]
    depth_m = np.where(
        xx < 60,
        0.85 + (xx - 30) * 0.002,
        1.05 - (xx - 90) * 0.003,
    )
    depth = np.round(depth_m / 0.001).astype(np.uint16)

    left_bbox = (24, 48, 38, 62)
    right_bbox = (82, 48, 96, 62)
    all_bboxes = [left_bbox, right_bbox]

    left = estimate_mounting_plane_axis_direction(
        depth, intrin, left_bbox, all_bboxes=all_bboxes,
        depth_scale=0.001, sample_stride=1, min_points=30)
    right = estimate_mounting_plane_axis_direction(
        depth, intrin, right_bbox, all_bboxes=all_bboxes,
        depth_scale=0.001, sample_stride=1, min_points=30)

    assert left is not None
    assert right is not None
    left_normal, _, _ = left
    right_normal, _, _ = right
    assert left_normal[0] > 0.1
    assert right_normal[0] < -0.1


def test_fastener_group_axis_uses_nearby_coplanar_points():
    candidates = [
        {'center_xy': (10, 10), 'point_3d': [0.0, 0.0, 1.0]},
        {'center_xy': (30, 10), 'point_3d': [0.1, 0.0, 1.03]},
        {'center_xy': (10, 30), 'point_3d': [0.0, 0.1, 1.0]},
        {'center_xy': (95, 95), 'point_3d': [1.0, 1.0, 0.5]},
    ]

    result = estimate_fastener_group_axis_direction(
        (12, 12), candidates, max_distance_px=40, min_points=3)

    assert result is not None
    normal, _, count = result
    assert count == 3
    assert normal[0] > 0.2
    assert normal[2] < -0.9


def test_fastener_group_axis_requires_three_points():
    candidates = [
        {'center_xy': (10, 10), 'point_3d': [0.0, 0.0, 1.0]},
        {'center_xy': (30, 10), 'point_3d': [0.1, 0.0, 1.03]},
    ]

    assert estimate_fastener_group_axis_direction(
        (12, 12), candidates, max_distance_px=40, min_points=3) is None


def test_fastener_patch_axis_uses_depth_connected_mounting_patch():
    intrin = _intrinsics()
    yy, xx = np.mgrid[0:120, 0:120]
    depth_m = 1.0 + (xx - 60) * 0.0005 + (yy - 60) * 0.0002
    depth = np.round(depth_m / 0.001).astype(np.uint16)
    bbox = (50, 50, 70, 70)
    depth[50:70, 50:70] = 930
    depth[:, :22] = 1500

    result = estimate_fastener_patch_axis_direction(
        depth, intrin, bbox, all_bboxes=[bbox], depth_scale=0.001,
        sample_stride=1, min_points=30)

    assert result is not None
    normal, _, count = result
    assert count >= 30
    assert normal[2] < -0.95


def test_fastener_patch_axis_rejects_low_z_patch():
    intrin = _intrinsics()
    yy, xx = np.mgrid[0:120, 0:120]
    depth_m = 0.65 + (xx - 60) * 0.025 + yy * 0.0
    depth_m = np.clip(depth_m, 0.25, 2.0)
    depth = np.round(depth_m / 0.001).astype(np.uint16)
    bbox = (50, 50, 70, 70)

    assert estimate_fastener_patch_axis_direction(
        depth, intrin, bbox, all_bboxes=[bbox], depth_scale=0.001,
        sample_stride=1, min_points=30, min_abs_z=0.45) is None


def test_fastener_line_constraint_works_with_two_points():
    candidates = [
        {'center_xy': (10, 10), 'point_3d': [0.0, 0.0, 1.0]},
        {'center_xy': (30, 10), 'point_3d': [0.1, 0.0, 1.03]},
    ]

    result = estimate_fastener_line_constrained_axis(
        (12, 12), candidates, base_normal=[1.0, 0.0, -1.0],
        max_distance_px=40)

    assert result is not None
    normal, _, count = result
    assert count == 2
    line = np.array([0.1, 0.0, 0.03], dtype=np.float64)
    line /= np.linalg.norm(line)
    assert abs(float(np.dot(normal, line))) < 1e-6
    assert normal[2] < 0
