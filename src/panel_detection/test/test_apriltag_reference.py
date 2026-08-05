import cv2
import numpy as np

from panel_detection.apriltag_reference import (
    _detect_aruco_markers,
    _get_apriltag_dictionary,
)


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
    assert ids.reshape(-1).tolist() == [0]
