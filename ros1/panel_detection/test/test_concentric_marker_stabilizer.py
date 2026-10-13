import numpy as np

from panel_detection.concentric_marker_stabilizer import (
    ConcentricMarkerStabilizer,
)


def test_normal_holds_single_large_jump_and_recovers():
    stabilizer = ConcentricMarkerStabilizer(normal_max_jump_deg=12.0)
    first, status = stabilizer.update_reference((0.0, 0.0, -1.0), 0.0)
    assert status == 'init'
    held, status = stabilizer.update_reference((0.8, 0.0, -0.6), 0.1)
    assert status == 'held_jump'
    assert np.allclose(held, first)
    recovered, status = stabilizer.update_reference((0.05, 0.0, -0.999), 0.2)
    assert status == 'accepted'
    assert recovered[2] < -0.99


def test_position_rejects_short_dt_depth_spike():
    stabilizer = ConcentricMarkerStabilizer(
        position_ema_alpha=0.5,
        position_jump_thresh_m=0.03,
        position_max_velocity_mps=1.0,
    )
    first = stabilizer.update_marker(
        (1, 1), 0.0, 1, (0.0, 0.0, 0.5), angle_deg=10.0)
    assert first['position_status'] == 'init'
    held = stabilizer.update_marker(
        (1, 1), 0.033, 2, (0.0, 0.0, 0.57), angle_deg=10.5)
    assert held['position_status'] == 'held_jump'
    assert np.allclose(held['position'], first['position'])
    accepted = stabilizer.update_marker(
        (1, 1), 0.066, 3, (0.0, 0.0, 0.502), angle_deg=11.0)
    assert accepted['position_status'] == 'accepted'
    assert accepted['position'][2] < 0.51


def test_angle_wrap_and_missing_direction_are_smoothed():
    stabilizer = ConcentricMarkerStabilizer(
        angle_ema_alpha=0.5, angle_max_jump_deg=20.0,
        direction_ema_alpha=0.5)
    stabilizer.update_marker(
        (1, 2), 0.0, 1, (0.0, 0.0, 0.5), angle_deg=359.0,
        direction=(1.0, 0.0, 0.0))
    wrapped = stabilizer.update_marker(
        (1, 2), 0.1, 2, (0.0, 0.0, 0.5), angle_deg=1.0)
    assert wrapped['angle_status'] == 'accepted'
    assert wrapped['angle_deg'] < 1.0 or wrapped['angle_deg'] > 359.0
    assert wrapped['direction'] is not None
    assert wrapped['direction_status'] == 'missing'
