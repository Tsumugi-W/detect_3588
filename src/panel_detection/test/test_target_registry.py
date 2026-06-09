import cv2
import numpy as np

from panel_detection.target_registry import (
    FrameDetection,
    PersistentPanelAxis,
    TargetRegistry,
)


def _detection(class_name, x1, color_bgr):
    x2 = x1 + 30
    y1, y2 = 20, 50
    return FrameDetection(
        class_name=class_name,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        bbox=(x1, y1, x2, y2),
        confidence=0.9,
    ), (x1, y1, x2, y2, color_bgr)


def _image_with_detections(draws):
    image = np.zeros((80, 220, 3), dtype=np.uint8)
    for x1, y1, x2, y2, color_bgr in draws:
        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, -1)
    return image


def test_single_red_knob_anchors_right_side_buttons_when_other_knob_missing():
    knob, knob_draw = _detection('knob', 20, (0, 0, 255))
    red_button, red_draw = _detection('button', 70, (0, 0, 255))
    green_button, green_draw = _detection('button', 120, (0, 255, 0))
    image = _image_with_detections([knob_draw, red_draw, green_draw])

    matched = TargetRegistry().identify([knob, red_button, green_button], image)

    assert [target_id for target_id, _ in matched] == [4, 6, 7]


def test_single_black_knob_anchors_left_side_buttons_when_other_knob_missing():
    green_button, green_draw = _detection('button', 20, (0, 255, 0))
    red_button_a, red_draw_a = _detection('button', 70, (0, 0, 255))
    red_button_b, red_draw_b = _detection('button', 120, (0, 0, 255))
    knob, knob_draw = _detection('knob', 160, (0, 0, 0))
    image = _image_with_detections([
        green_draw, red_draw_a, red_draw_b, knob_draw,
    ])

    matched = TargetRegistry().identify([
        green_button, red_button_a, red_button_b, knob,
    ], image)

    assert [target_id for target_id, _ in matched] == [1, 2, 3, 5]


def test_persistent_axis_relocates_parallel_line_after_camera_translation():
    axis = PersistentPanelAxis()
    axis.update((40.0, 40.0), (1.0, 0.0))
    row_button, _ = _detection('button', 40, (0, 0, 255))
    row_button.center_y = 90.0
    row_button.bbox = (row_button.bbox[0], 75.0, row_button.bbox[2], 105.0)
    off_row_button, _ = _detection('button', 90, (0, 0, 255))
    off_row_button.center_y = 170.0
    off_row_button.bbox = (
        off_row_button.bbox[0], 155.0, off_row_button.bbox[2], 185.0)

    result = axis.select([row_button, off_row_button])

    assert result is not None
    selected, origin, vector = result
    assert selected == [row_button]
    assert origin[1] == row_button.center_y
    assert vector == (1.0, 0.0)


def test_persistent_axis_converts_button_like_light_on_cached_row():
    axis = PersistentPanelAxis()
    axis.update((40.0, 40.0), (1.0, 0.0))
    light, _ = _detection('light', 40, (0, 255, 0))
    light.center_y = 88.0
    light.bbox = (light.bbox[0], 73.0, light.bbox[2], 103.0)

    result = axis.select([light])

    assert result is not None
    selected, _, _ = result
    assert selected == [light]
    assert light.class_name == 'button'
