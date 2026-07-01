import math

import cv2
import numpy as np

from panel_detection.pipe_axis import (
    axial_angle_delta_deg,
    estimate_pipe_axis_from_image,
    weighted_axial_mean,
)


def test_weighted_axial_mean_handles_180_degree_ambiguity():
    mean = weighted_axial_mean(
        [math.radians(2.0), math.radians(-178.0)],
        [1.0, 1.0],
    )

    assert abs(math.degrees(mean) - 2.0) < 1e-6


def test_axial_angle_delta_uses_axis_not_arrow_direction():
    assert axial_angle_delta_deg(5.0, -175.0) == 0.0
    assert axial_angle_delta_deg(10.0, 40.0) == 30.0


def test_estimate_pipe_axis_from_synthetic_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(image, (30, 125), (290, 105), (180, 180, 180), 18)
    cv2.line(image, (30, 125), (290, 105), (230, 230, 230), 3)

    result = estimate_pipe_axis_from_image(
        image,
        leak_point=(160, 115),
        roi_half_size=(130, 70),
        min_line_length=50,
        max_line_distance=45,
        angle_prior_deg=0.0,
        angle_tolerance_deg=25.0,
    )

    assert result is not None
    assert axial_angle_delta_deg(result.angle_deg, -4.4) < 3.0
    assert len(result.segments) > 0
