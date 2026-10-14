import cv2
import numpy as np

from panel_detection.panel_apriltag import (
    PanelAprilTagTracker,
    PanelTagMarker,
    detect_panel_tags,
    forced_class_for_tag,
    reclassify_buttons_without_tags,
    redraw_reclassified_detection,
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
    assert [forced_class_for_tag(tag_id, {'scene': 'test_site'}) for tag_id in range(7)] == [
        'button', 'button', 'button', 'knob', 'knob', 'button', 'button',
    ]
    assert forced_class_for_tag(7, {'scene': 'test_site'}) == 'light'
    assert forced_class_for_tag(39, {'scene': 'test_site'}) == 'light'
    assert forced_class_for_tag(-1, {'scene': 'test_site'}) is None
    assert forced_class_for_tag(40, {'scene': 'test_site'}) is None


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


def test_button_keeps_class_only_when_its_own_tag_is_directly_assigned():
    detections = [_detection('button', 50.0)]
    assignments = PanelAprilTagTracker({'scene': 'test_site'}).update(
        detections, [_marker(0, 50.0)])

    changed = reclassify_buttons_without_tags(detections, assignments)

    assert changed == []
    assert detections[0].class_name == 'button'


def test_unassigned_button_becomes_light_even_when_another_tag_is_visible():
    detections = [_detection('button', 50.0), _detection('button', 150.0)]
    assignments = PanelAprilTagTracker({'scene': 'test_site'}).update(
        detections, [_marker(0, 50.0)])

    changed = reclassify_buttons_without_tags(detections, assignments)

    assert changed == [detections[1]]
    assert [item.class_name for item in detections] == ['button', 'light']


def test_tracked_tag_does_not_keep_button_class_without_current_marker():
    tracker = PanelAprilTagTracker({'scene': 'test_site', 'stale_frames': 3})
    first_detection = _detection('button', 50.0)
    tracker.update([first_detection], [_marker(0, 50.0)])
    current_detection = _detection('button', 52.0)
    assignments = tracker.update([current_detection], [])

    changed = reclassify_buttons_without_tags(
        [current_detection], assignments)

    assert assignments[0].source == 'apriltag_track'
    assert changed == [current_detection]
    assert current_detection.class_name == 'light'


def test_canvas_redraw_uses_only_reclassified_light_label(monkeypatch):
    raw_image = np.full((120, 180, 3), 127, dtype=np.uint8)
    canvas = raw_image.copy()
    detection = _detection('light', 80.0, center_y=70.0, size=40.0)
    labels = []
    original_put_text = cv2.putText

    def record_put_text(image, text, *args, **kwargs):
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, 'putText', record_put_text)
    redraw_reclassified_detection(
        canvas, raw_image, detection, 'button', (255, 255, 0))

    assert labels == ['light 0.90']
    assert np.any(canvas != raw_image)


def test_tags_above_components_are_associated_one_to_one():
    detections = [
        _detection('light', 50.0),
        _detection('button', 150.0),
        _detection('knob', 250.0),
    ]
    assignments = PanelAprilTagTracker({'scene': 'test_site'}).update(
        detections, [_marker(0, 50.0), _marker(3, 150.0), _marker(12, 250.0)])

    assert set(assignments) == {0, 1, 2}
    assert [(item.target_id, item.forced_class) for item in assignments.values()] == [
        (1, 'button'), (4, 'knob'), (13, 'light'),
    ]
    assert all(item.source == 'apriltag_36h11'
               for item in assignments.values())


def test_tag_below_component_or_far_to_the_side_is_rejected():
    detections = [_detection('button', 60.0)]
    tracker = PanelAprilTagTracker({'scene': 'test_site'})

    assert tracker.update(detections, [_marker(0, 60.0, center_y=140.0)]) == {}
    assert tracker.update(detections, [_marker(0, 260.0)]) == {}


def test_track_bridges_short_tag_gap_but_expires_from_last_real_tag_frame():
    tracker = PanelAprilTagTracker({'scene': 'test_site', 'stale_frames': 2, 'track_distance_px': 50.0})
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


def test_validation_tag07_assigns_board_identity_without_class_correction():
    from panel_detection.panel_apriltag import apply_panel_tag_rules, draw_panel_tag_assignments
    detection = _detection('board_card', 100.0)
    assignments, changed = apply_panel_tag_rules(
        [detection], [_marker(7, 100.0)], PanelAprilTagTracker())
    assert detection.class_name == 'board_card'
    assert changed == []
    assert assignments[0].tag_id == 7
    assert assignments[0].target_id == 8
    canvas = np.zeros((180, 240, 3), dtype=np.uint8)
    draw_panel_tag_assignments(canvas, [_marker(7, 100.0)], assignments, [detection])
    assert canvas.any()


def test_board_card_keeps_yolo_class_without_tags_and_after_track_expiry():
    from panel_detection.panel_apriltag import apply_panel_tag_rules
    tracker = PanelAprilTagTracker({'stale_frames': 1})
    for markers, expected_assignment in [([], False), ([_marker(7, 100.0)], True),
                                          ([], True), ([], False)]:
        detection = _detection('board_card', 100.0)
        assignments, changed = apply_panel_tag_rules([detection], markers, tracker)
        assert detection.class_name == 'board_card'
        assert changed == []
        assert bool(assignments) == expected_assignment


def test_board_tag_cannot_claim_other_classes_or_cached_control():
    from panel_detection.panel_apriltag import apply_panel_tag_rules
    for class_name in ['button', 'knob', 'light', 'door_button']:
        tracker = PanelAprilTagTracker()
        tracker.update([_detection('board_card', 100.0)], [_marker(7, 100.0)])
        detection = _detection(class_name, 100.0)
        assignments, _ = apply_panel_tag_rules([detection], [_marker(7, 100.0)], tracker)
        assert assignments == {}
        assert detection.class_name != 'board_card'
        assert tracker.update([_detection(class_name, 100.0)], []) == {}


def test_other_tags_cannot_reclassify_board_cards_directly_or_via_tracking():
    from panel_detection.panel_apriltag import apply_panel_tag_rules
    for scene, tag_id in [('validation_site', 0), ('validation_site', 2), ('test_site', 7)]:
        tracker = PanelAprilTagTracker({'scene': scene})
        tracker.update([_detection('light', 100.0)], [_marker(tag_id, 100.0)])
        for markers in [[_marker(tag_id, 100.0)], []]:
            detection = _detection('board_card', 100.0)
            assignments, changed = apply_panel_tag_rules([detection], markers, tracker)
            assert assignments == {}
            assert changed == []
            assert detection.class_name == 'board_card'


def test_board_tag_matches_board_when_nearer_control_is_present():
    from panel_detection.panel_apriltag import apply_panel_tag_rules
    detections = [_detection('knob', 100.0), _detection('board_card', 105.0)]
    assignments, changed = apply_panel_tag_rules(
        detections, [_marker(7, 100.0)], PanelAprilTagTracker())
    assert set(assignments) == {1}
    assert [d.class_name for d in detections] == ['knob', 'board_card']
    assert changed == []
