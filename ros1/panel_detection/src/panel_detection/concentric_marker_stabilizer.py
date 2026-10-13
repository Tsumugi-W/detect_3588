"""Temporal gating and smoothing for concentric bolt-marker poses."""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def _unit(value, *, align_to=None, directed=False):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm < 1e-9:
        return None
    array = array / norm
    if align_to is not None and not directed and float(np.dot(array, align_to)) < 0.0:
        array = -array
    return array


def _angle_deg(first, second, *, absolute=True):
    if first is None or second is None:
        return None
    dot = float(np.dot(first, second))
    if absolute:
        dot = abs(dot)
    return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))


def _ema(previous, current, alpha, *, align_to=None, directed=False):
    current = _unit(current, align_to=align_to, directed=directed)
    if current is None:
        return previous
    if previous is None:
        return current
    if align_to is None and not directed and float(np.dot(current, previous)) < 0.0:
        current = -current
    blended = (1.0 - alpha) * previous + alpha * current
    return _unit(blended, align_to=previous, directed=directed)


def _cyclic_delta(current, previous):
    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


@dataclass
class _MarkerState:
    position: np.ndarray
    last_raw_position: np.ndarray
    direction: Optional[np.ndarray]
    angle_deg: Optional[float]
    last_stamp: float
    last_seen_frame: int


class ConcentricMarkerStabilizer:
    """Reject one-frame jumps and smooth marker poses by stable target key.

    The reference normal is shared by all markers. Positions and notch
    directions are maintained independently per ``(group_id, target_id)``.
    """

    def __init__(self, enabled=True, position_ema_alpha=0.35,
                 position_jump_thresh_m=0.03,
                 position_max_velocity_mps=1.0,
                 normal_ema_alpha=0.25, normal_max_jump_deg=12.0,
                 angle_ema_alpha=0.30, angle_max_jump_deg=30.0,
                 direction_ema_alpha=0.30, direction_max_jump_deg=20.0,
                 max_dt=0.5, stale_frames=90):
        self.enabled = bool(enabled)
        self.position_ema_alpha = float(np.clip(position_ema_alpha, 0.0, 1.0))
        self.position_jump_thresh_m = float(max(0.0, position_jump_thresh_m))
        self.position_max_velocity_mps = float(max(0.0, position_max_velocity_mps))
        self.normal_ema_alpha = float(np.clip(normal_ema_alpha, 0.0, 1.0))
        self.normal_max_jump_deg = float(max(0.0, normal_max_jump_deg))
        self.angle_ema_alpha = float(np.clip(angle_ema_alpha, 0.0, 1.0))
        self.angle_max_jump_deg = float(max(0.0, angle_max_jump_deg))
        self.direction_ema_alpha = float(np.clip(direction_ema_alpha, 0.0, 1.0))
        self.direction_max_jump_deg = float(max(0.0, direction_max_jump_deg))
        self.max_dt = float(max(0.0, max_dt))
        self.stale_frames = int(max(1, stale_frames))
        self._normal = None
        self._normal_stamp = None
        self._states = {}

    @property
    def normal(self):
        return None if self._normal is None else self._normal.copy()

    def reset(self):
        self._normal = None
        self._normal_stamp = None
        self._states.clear()

    def update_reference(self, normal, stamp):
        """Return a filtered normal and a short diagnostic status string."""
        incoming = _unit(normal)
        if incoming is None:
            return self.normal, 'invalid'
        stamp = float(stamp)
        if not self.enabled or self._normal is None:
            self._normal = incoming
            self._normal_stamp = stamp
            return self.normal, 'init'

        dt = stamp - float(self._normal_stamp)
        if dt <= 0.0:
            self._normal_stamp = stamp
            return self.normal, 'held_time'
        if dt > self.max_dt:
            # A slow detector or a dropped queue can make the bag timestamp
            # jump by more than max_dt. Do not turn that scheduling gap into a
            # one-frame 60--100 degree pose jump; hold the last normal until a
            # nearby observation is available again.
            incoming = _unit(incoming, align_to=self._normal)
            jump = _angle_deg(self._normal, incoming)
            self._normal_stamp = stamp
            if jump is not None and jump > self.normal_max_jump_deg:
                return self.normal, 'held_gap_jump'
            self._normal = _ema(
                self._normal, incoming, self.normal_ema_alpha,
                align_to=self._normal)
            return self.normal, 'accepted_gap'
        incoming = _unit(incoming, align_to=self._normal)
        jump = _angle_deg(self._normal, incoming)
        self._normal_stamp = stamp
        if jump is not None and jump > self.normal_max_jump_deg:
            return self.normal, 'held_jump'
        self._normal = _ema(
            self._normal, incoming, self.normal_ema_alpha,
            align_to=self._normal)
        return self.normal, 'accepted'

    def update_marker(self, key, stamp, frame_index, position,
                      angle_deg=None, direction=None):
        """Filter one marker and return JSON-ready values plus statuses."""
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        stamp = float(stamp)
        frame_index = int(frame_index)
        state = self._states.get(key)
        if not self.enabled or state is None:
            state = _MarkerState(
                position=point.copy(),
                last_raw_position=point.copy(),
                direction=_unit(direction, directed=True),
                angle_deg=None if angle_deg is None else float(angle_deg) % 360.0,
                last_stamp=stamp,
                last_seen_frame=frame_index,
            )
            self._states[key] = state
            return self._result(state, 'init', 'init')

        dt = stamp - state.last_stamp
        gap = dt <= 0.0 or dt > self.max_dt
        if gap:
            state.position = point.copy()
            state.last_raw_position = point.copy()
            position_status = 'reset_gap'
        else:
            raw_jump = float(np.linalg.norm(point - state.last_raw_position))
            allowed_jump = max(
                self.position_jump_thresh_m,
                self.position_max_velocity_mps * max(0.0, dt),
            )
            if raw_jump > allowed_jump:
                position_status = 'held_jump'
            else:
                alpha = self.position_ema_alpha
                state.position = (1.0 - alpha) * state.position + alpha * point
                state.last_raw_position = point.copy()
                position_status = 'accepted'

        direction_status = 'missing'
        incoming_direction = _unit(direction, directed=True)
        if incoming_direction is not None:
            if state.direction is None or gap:
                state.direction = incoming_direction
                direction_status = 'reset_gap' if gap else 'init'
            else:
                jump = _angle_deg(state.direction, incoming_direction, absolute=False)
                if jump is not None and jump > self.direction_max_jump_deg:
                    direction_status = 'held_jump'
                else:
                    state.direction = _ema(
                        state.direction, incoming_direction,
                        self.direction_ema_alpha, directed=True)
                    direction_status = 'accepted'

        angle_status = 'missing'
        if angle_deg is not None and np.isfinite(float(angle_deg)):
            incoming_angle = float(angle_deg) % 360.0
            if state.angle_deg is None or gap:
                state.angle_deg = incoming_angle
                angle_status = 'reset_gap' if gap else 'init'
            else:
                delta = _cyclic_delta(incoming_angle, state.angle_deg)
                if abs(delta) > self.angle_max_jump_deg:
                    angle_status = 'held_jump'
                else:
                    state.angle_deg = (state.angle_deg
                                       + self.angle_ema_alpha * delta) % 360.0
                    angle_status = 'accepted'

        state.last_stamp = stamp
        state.last_seen_frame = frame_index
        return self._result(state, position_status, direction_status,
                            angle_status=angle_status)

    def prune(self, frame_index):
        frame_index = int(frame_index)
        self._states = {
            key: state for key, state in self._states.items()
            if frame_index - state.last_seen_frame <= self.stale_frames
        }

    @staticmethod
    def _result(state, position_status, direction_status,
                angle_status='missing'):
        return {
            'position': state.position.copy(),
            'direction': (None if state.direction is None
                          else state.direction.copy()),
            'angle_deg': state.angle_deg,
            'position_status': position_status,
            'direction_status': direction_status,
            'angle_status': angle_status,
        }
