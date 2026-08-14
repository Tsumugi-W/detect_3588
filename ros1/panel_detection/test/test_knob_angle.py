import builtins

import cv2
import numpy as np

from panel_detection.knob_angle import _try_color_handle


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
