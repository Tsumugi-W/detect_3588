import cv2
import numpy as np

from panel_detection.apriltag_reference import (
    _detect_aruco_markers,
    _get_apriltag_dictionary,
    _normalize_aruco_result,
    estimate_apriltag_pnp_normal,
    estimate_apriltag_pnp_pose,
)


class _Intrinsics:
    fx = 600.0
    fy = 600.0
    cx = 320.0
    cy = 240.0
    coeffs = [0.0] * 5


def test_aruco_detector_api_is_compatible_with_installed_opencv():
    dictionary = _get_apriltag_dictionary('DICT_APRILTAG_36h11')
    assert dictionary is not None

    gray = np.zeros((120, 120), dtype=np.uint8)
    corners, ids, rejected = _detect_aruco_markers(gray, dictionary)

    assert corners is not None
    assert ids is None
    assert rejected is not None


def test_aruco_detector_finds_generated_marker_when_supported():
    aruco = cv2.aruco
    dictionary = _get_apriltag_dictionary('DICT_APRILTAG_36h11')
    if hasattr(aruco, 'generateImageMarker'):
        marker = aruco.generateImageMarker(dictionary, 0, 100)
    else:
        marker = np.zeros((100, 100), dtype=np.uint8)
        aruco.drawMarker(dictionary, 0, 100, marker, 1)

    gray = np.full((160, 160), 255, dtype=np.uint8)
    gray[30:130, 30:130] = marker
    corners, ids, _ = _detect_aruco_markers(gray, dictionary)

    assert len(corners) == 1
    assert ids.shape == (1, 1)
    assert ids.reshape(-1).tolist() == [0]


def test_aruco_result_normalizes_flat_marker_ids():
    corners = [np.zeros((1, 4, 2), dtype=np.float32)]
    normalized_corners, ids, rejected = _normalize_aruco_result(
        (corners, np.array([7], dtype=np.int32), []))

    assert normalized_corners is corners
    assert ids.shape == (1, 1)
    assert ids.tolist() == [[7]]
    assert rejected == []


def test_pnp_normal_does_not_require_the_tag_size():
    corners = np.array([
        [260.0, 300.0],
        [380.0, 300.0],
        [380.0, 180.0],
        [260.0, 180.0],
    ], dtype=np.float32)

    result = estimate_apriltag_pnp_normal(corners, _Intrinsics())

    assert result is not None
    normal, reprojection_error = result
    assert np.allclose(normal, [0.0, 0.0, -1.0], atol=1e-6)
    assert reprojection_error < 1e-5


def test_pnp_pose_provides_right_handed_tag_axes():
    corners = np.array([
        [260.0, 300.0],
        [380.0, 300.0],
        [380.0, 180.0],
        [260.0, 180.0],
    ], dtype=np.float32)

    pose = estimate_apriltag_pnp_pose(corners, _Intrinsics())

    assert pose is not None
    assert np.allclose(np.linalg.norm(pose['x_axis']), 1.0)
    assert np.allclose(np.linalg.norm(pose['y_axis']), 1.0)
    assert np.allclose(np.linalg.norm(pose['normal']), 1.0)
    assert abs(float(np.dot(pose['x_axis'], pose['normal']))) < 1e-6
    assert np.allclose(
        np.cross(pose['x_axis'], pose['y_axis']), pose['normal'], atol=1e-6)
