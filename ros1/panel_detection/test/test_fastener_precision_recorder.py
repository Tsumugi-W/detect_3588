import numpy as np

from panel_detection.fastener_precision_recorder import (
    PrecisionAccumulator,
    build_tag_frame,
    camera_point_to_tag,
)


def _reference():
    return {
        'tag_id': 1,
        'source': 'apriltag_pnp',
        'origin': [1.0, 2.0, 3.0],
        'x_axis': [0.0, 1.0, 0.0],
        'y_axis': [1.0, 0.0, 0.0],
        'normal': [0.0, 0.0, -1.0],
        'reprojection_error_px': 0.2,
        'rms_error_m': 0.001,
    }


def _target(target_id, point):
    return {
        'class': 'bolt',
        'group_id': 1,
        'id': target_id,
        'slot': 'top_left' if target_id == 1 else 'top_right',
        'registered': True,
        'position': {'x': point[0], 'y': point[1], 'z': point[2]},
        'confidence': 0.9,
    }


def test_camera_point_is_expressed_in_tag_coordinates():
    origin, basis = build_tag_frame(_reference())
    expected = np.array([0.1, 0.2, 0.3])
    camera_point = origin + basis.T @ expected

    transformed = camera_point_to_tag(camera_point, _reference())

    assert np.allclose(transformed, expected)


def test_accumulator_reports_tag_relative_and_pair_distance_precision():
    accumulator = PrecisionAccumulator(max_tag_distance_m=0.30)
    for frame_index, jitter_m in enumerate([-0.001, 0.0, 0.001]):
        reference = _reference()
        origin, basis = build_tag_frame(reference)
        first = origin + basis.T @ np.array([0.10 + jitter_m, 0.20, 0.03])
        second = origin + basis.T @ np.array([0.20 + jitter_m, 0.20, 0.03])
        payload = {
            'stamp': float(frame_index),
            'targets': [_target(1, first), _target(2, second)],
        }
        geometry = {'stamp': float(frame_index), 'axis_reference': reference}
        record = accumulator.add_frame(payload, geometry)
        assert len(record['targets']) == 2

    summary = accumulator.summary()

    assert summary['frames_usable'] == 3
    first = summary['targets']['slot:top_left']['accepted']
    assert first['samples'] == 3
    assert np.allclose(
        first['median_tag_position_m'],
        [0.10, 0.20, 0.03], atol=1e-9)
    assert 0.8 < first['rms_3d_mm'] < 0.9
    pair = summary['pair_distances']['slot:top_left|slot:top_right']
    assert pair['samples'] == 3
    assert abs(pair['median_distance_mm'] - 100.0) < 1e-9
    assert pair['std_mm'] < 1e-9


def test_accumulator_skips_reference_without_in_plane_axes():
    accumulator = PrecisionAccumulator()
    geometry = {
        'stamp': 1.0,
        'axis_reference': {
            'origin': [0.0, 0.0, 1.0],
            'normal': [0.0, 0.0, -1.0],
        },
    }

    assert accumulator.add_frame({'stamp': 1.0, 'targets': []}, geometry) is None
    assert accumulator.summary()['frames_without_tag_frame'] == 1


def test_accumulator_flags_implausible_tag_depth_without_losing_pair_metric():
    accumulator = PrecisionAccumulator(max_tag_distance_m=0.20)
    reference = _reference()
    payload = {
        'stamp': 1.0,
        'targets': [
            _target(1, [1.0, 2.0, 2.5]),
            _target(2, [1.1, 2.0, 2.5]),
        ],
    }

    record = accumulator.add_frame(
        payload, {'stamp': 1.0, 'axis_reference': reference})
    summary = accumulator.summary()

    assert record['tag_geometry_valid'] is False
    assert summary['frames_tag_geometry_outlier'] == 1
    assert summary['targets']['slot:top_left']['accepted'] is None
    assert summary['pair_distances'][
        'slot:top_left|slot:top_right']['samples'] == 1
