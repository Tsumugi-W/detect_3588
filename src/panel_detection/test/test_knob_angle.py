import builtins

import cv2
import numpy as np

from panel_detection.knob_angle import (
    _try_color_handle, estimate_knob_angle, observe_knob, KnobDiscreteAngleStabilizer,
)


def test_radial_scan_does_not_import_scipy(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('scipy'):
            raise AssertionError('knob angle estimation must not import scipy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    roi = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.line(roi, (40, 40), (40, 10), (255, 255, 255), 5)
    circle_mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(circle_mask, (40, 40), 34, 255, -1)

    angle, contour, binary = _try_color_handle(
        roi, circle_mask, np.pi * 34 ** 2, 0.01, 0.25, 40, 40)

    assert angle is not None
    assert contour is None
    assert binary is None


def test_axis_aligned_four_vertex_stripes_are_measured():
    for vertical in [True, False]:
        roi = np.zeros((120, 120, 3), np.uint8)
        bounds = ((55, 35), (65, 90)) if vertical else ((35, 55), (90, 65))
        cv2.rectangle(roi, *bounds, (235, 235, 235), -1)
        angle, debug = estimate_knob_angle(roi, pointer_color='white_line', debug=True)
        assert debug['method'] == 'white_line_axis'
        distance = min(angle % 180, 180-angle % 180)
        assert distance < 3 if vertical else abs(distance-90) < 3


def test_metal_rim_does_not_override_vertical_stripe():
    roi = np.zeros((140, 140, 3), np.uint8)
    cv2.ellipse(roi, (70, 70), (55, 55), 0, 220, 320, (255, 255, 255), 6)
    cv2.rectangle(roi, (65, 45), (76, 105), (230, 230, 230), -1)
    angle, debug = estimate_knob_angle(roi, pointer_color='white_line', debug=True)
    assert debug['method'] == 'white_line_axis'
    assert min(angle % 180, 180-angle % 180) < 5


def test_untagged_knob_still_has_visual_angle_without_invented_id():
    from types import SimpleNamespace
    image = np.zeros((120, 120, 3), np.uint8)
    cv2.rectangle(image, (55, 35), (65, 90), (235, 235, 235), -1)
    detection = SimpleNamespace(bbox=(0, 0, 120, 120), confidence=.9)
    observation = observe_knob(image, detection, None, 1, KnobDiscreteAngleStabilizer())
    assert observation['id'] is None
    assert observation['angle_valid'] and observation['angle'] == 0


def test_real_video_vertical_and_horizontal_knobs():
    from pathlib import Path
    from types import SimpleNamespace
    for name, expected in [('knob_vertical', 0), ('knob_horizontal', 90)]:
        image = cv2.imread(str(Path(__file__).parent/'fixtures'/'board'/(name+'.png')))
        assert image is not None
        h, w = image.shape[:2]
        detection = SimpleNamespace(bbox=(0, 0, w, h), confidence=.95)
        result = observe_knob(image, detection, 3, 1, KnobDiscreteAngleStabilizer())
        assert result['angle'] == expected and result['angle_valid']
        assert result['angle_source'] == 'white_line_axis'
