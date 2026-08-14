import numpy as np

from panel_detection.fastener_axis_stabilizer import (
    FastenerAxisMeasurement,
    FastenerAxisStabilizer,
)


def _measurement(det_idx, center, axis):
    x, y = center
    return FastenerAxisMeasurement(
        det_idx=det_idx,
        center_xy=center,
        bbox=(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
        axis_direction=axis,
    )


def test_stabilizer_canonicalizes_axis_sign():
    stabilizer = FastenerAxisStabilizer()
    first = stabilizer.update([
        _measurement(0, (100.0, 100.0), (0.1, 0.0, -0.995)),
    ], frame_index=1)[0]
    second = stabilizer.update([
        _measurement(3, (101.0, 100.0), (-0.1, 0.0, 0.995)),
    ], frame_index=2)[3]

    assert first[2] < 0.0
    assert second[2] < 0.0
    assert np.dot(first, second) > 0.999


def test_stabilizer_holds_single_large_jump():
    stabilizer = FastenerAxisStabilizer(max_jump_deg=10.0, confirm_frames=3)
    initial = stabilizer.update([
        _measurement(0, (100.0, 100.0), (0.0, 0.0, -1.0)),
    ], frame_index=1)[0]
    jumped = stabilizer.update([
        _measurement(1, (101.0, 100.0), (0.8, 0.0, -0.6)),
    ], frame_index=2)[1]

    assert np.allclose(jumped, initial)


def test_stabilizer_accepts_confirmed_direction_change():
    stabilizer = FastenerAxisStabilizer(
        max_jump_deg=10.0, confirm_frames=3, max_output_step_deg=3.0)
    stabilizer.update([
        _measurement(0, (100.0, 100.0), (0.0, 0.0, -1.0)),
    ], frame_index=1)
    new_axis = np.array((0.8, 0.0, -0.6))

    for frame_index in (2, 3):
        held = stabilizer.update([
            _measurement(frame_index, (100.0, 100.0), new_axis),
        ], frame_index=frame_index)[frame_index]
        assert np.allclose(held, (0.0, 0.0, -1.0))

    accepted = stabilizer.update([
        _measurement(4, (100.0, 100.0), new_axis),
    ], frame_index=4)[4]
    accepted_angle = np.degrees(np.arccos(np.clip(
        np.dot(accepted, (0.0, 0.0, -1.0)), -1.0, 1.0)))
    assert 0.0 < accepted_angle <= 3.1

    previous = accepted
    for frame_index in range(5, 30):
        current = stabilizer.update([
            _measurement(frame_index, (100.0, 100.0), new_axis),
        ], frame_index=frame_index)[frame_index]
        step = np.degrees(np.arccos(np.clip(
            np.dot(previous, current), -1.0, 1.0)))
        assert step <= 3.1
        previous = current
    assert np.dot(previous, new_axis) > 0.999


def test_stabilizer_tracks_two_fasteners_independently():
    stabilizer = FastenerAxisStabilizer(match_distance_px=30.0)
    stabilizer.update([
        _measurement(0, (100.0, 100.0), (-0.2, 0.0, -0.98)),
        _measurement(1, (300.0, 100.0), (0.2, 0.0, -0.98)),
    ], frame_index=1)
    result = stabilizer.update([
        _measurement(8, (302.0, 101.0), (0.21, 0.0, -0.978)),
        _measurement(9, (99.0, 101.0), (-0.21, 0.0, -0.978)),
    ], frame_index=2)

    assert result[8][0] > 0.0
    assert result[9][0] < 0.0
