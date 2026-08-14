import cv2
import numpy as np

from panel_detection.panel_apriltag import (
    PanelAprilTagTracker,
    PanelTagMarker,
    detect_panel_tags,
    forced_class_for_tag,
    reclassify_buttons_without_tags,
)
from panel_detection.target_registry import FrameDetection


def _detection(class_name, center_x, center_y=110.0, size=40.0):
    half = size / 2.0
    return FrameDetection(
        class_name=class_name,
        center_x=center_x,
        center_y=center_y,
        bbox=(center_x - half, center_y - half,
              center_x + half, center_y + half),
        confidence=0.9,
    )


def _marker(tag_id, center_x, center_y=45.0, size=20.0):
    half = size / 2.0
    corners = np.array([
        [center_x - half, center_y - half],
        [center_x + half, center_y - half],
        [center_x + half, center_y + half],
        [center_x - half, center_y + half],
    ], dtype=np.float32)
    return PanelTagMarker(
        tag_id=tag_id,
        corners=corners,
        center_xy=(center_x, center_y),
        area=size * size,
    )


def test_tag_class_mapping_matches_panel_definition():
    assert [forced_class_for_tag(tag_id) for tag_id in range(7)] == [
        'button', 'button', 'button', 'knob', 'knob', 'button', 'button',
    ]
    assert forced_class_for_tag(7) == 'light'
    assert forced_class_for_tag(39) == 'light'
    assert forced_class_for_tag(-1) is None
    assert forced_class_for_tag(40) is None


def test_buttons_become_lights_when_current_frame_has_no_tags():
    detections = [
        _detection('button', 50.0),
        _detection('door_button', 100.0),
        _detection('knob', 150.0),
        _detection('light', 200.0),
    ]

    changed = reclassify_buttons_without_tags(detections, [])

    assert changed == [detections[0]]
    assert [item.class_name for item in detections] == [
        'light', 'door_button', 'knob', 'light']


def test_buttons_keep_yolo_class_when_any_current_tag_is_visible():
    detections = [_detection('button', 50.0)]

    changed = reclassify_buttons_without_tags(
        detections, [_marker(7, 50.0)])

    assert changed == []
    assert detections[0].class_name == 'button'


def test_tags_above_components_are_associated_one_to_one():
    detections = [
        _detection('light', 50.0),
        _detection('button', 150.0),
        _detection('knob', 250.0),
    ]
    assignments = PanelAprilTagTracker().update(
        detections, [_marker(0, 50.0), _marker(3, 150.0), _marker(12, 250.0)])

    assert set(assignments) == {0, 1, 2}
    assert [(item.target_id, item.forced_class) for item in assignments.values()] == [
        (1, 'button'), (4, 'knob'), (13, 'light'),
    ]
    assert all(item.source == 'apriltag_36h11'
               for item in assignments.values())


def test_tag_below_component_or_far_to_the_side_is_rejected():
    detections = [_detection('button', 60.0)]
    tracker = PanelAprilTagTracker()

    assert tracker.update(detections, [_marker(0, 60.0, center_y=140.0)]) == {}
    assert tracker.update(detections, [_marker(0, 260.0)]) == {}


def test_track_bridges_short_tag_gap_but_expires_from_last_real_tag_frame():
    tracker = PanelAprilTagTracker({'stale_frames': 2, 'track_distance_px': 50.0})
    first = tracker.update([_detection('light', 80.0)], [_marker(3, 80.0)])
    second = tracker.update([_detection('light', 82.0)], [])
    third = tracker.update([_detection('button', 84.0)], [])
    fourth = tracker.update([_detection('knob', 86.0)], [])

    assert first[0].source == 'apriltag_36h11'
    assert second[0].source == 'apriltag_track'
    assert third[0].source == 'apriltag_track'
    assert fourth == {}


def test_only_decoded_ids_00_to_39_are_returned_from_image():
    dictionary_id = getattr(cv2.aruco, 'DICT_APRILTAG_36h11')
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    def render(tag_id):
        if hasattr(cv2.aruco, 'generateImageMarker'):
            return cv2.aruco.generateImageMarker(dictionary, tag_id, 100)
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, tag_id, 100, image, 1)
        return image

    image = np.full((180, 300), 255, dtype=np.uint8)
    image[40:140, 20:120] = render(7)
    image[40:140, 180:280] = render(40)
    markers = detect_panel_tags(image, {'min_area_px': 100.0})

    assert [marker.tag_id for marker in markers] == [7]
