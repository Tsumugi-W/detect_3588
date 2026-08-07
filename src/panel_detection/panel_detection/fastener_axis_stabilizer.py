from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FastenerAxisMeasurement:
    det_idx: int
    center_xy: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    axis_direction: Sequence[float]


@dataclass
class _AxisTrack:
    center_xy: np.ndarray
    size_px: float
    stable_axis: np.ndarray
    last_seen_frame: int
    pending_axis: np.ndarray | None = None
    pending_count: int = 0


def _normalize_camera_facing(axis) -> np.ndarray | None:
    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return None
    value = value / norm
    if value[2] > 0.0:
        value = -value
    return value


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


class FastenerAxisStabilizer:
    """Stabilize bolt/nut normals using short-lived image-space tracks."""

    def __init__(self, enabled=True, ema_alpha=0.18, max_jump_deg=12.0,
                 confirm_frames=5, match_distance_px=60.0,
                 match_size_ratio=1.2, stale_frames=60):
        self.enabled = bool(enabled)
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.max_jump_deg = float(max_jump_deg)
        self.confirm_frames = max(1, int(confirm_frames))
        self.match_distance_px = float(match_distance_px)
        self.match_size_ratio = float(match_size_ratio)
        self.stale_frames = max(1, int(stale_frames))
        self._tracks: Dict[int, _AxisTrack] = {}
        self._next_track_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, measurements: Iterable[FastenerAxisMeasurement],
               frame_index: int) -> Dict[int, np.ndarray]:
        measurements = list(measurements)
        self._prune(frame_index)

        valid = []
        for measurement in measurements:
            axis = _normalize_camera_facing(measurement.axis_direction)
            if axis is not None:
                valid.append((measurement, axis))

        if not self.enabled:
            return {measurement.det_idx: axis for measurement, axis in valid}
        if not valid:
            return {}

        matches = self._associate(valid)
        result: Dict[int, np.ndarray] = {}
        matched_measurements = set()

        for track_id, measurement_index in matches:
            measurement, raw_axis = valid[measurement_index]
            track = self._tracks[track_id]
            self._update_track(track, measurement, raw_axis, frame_index)
            result[measurement.det_idx] = track.stable_axis.copy()
            matched_measurements.add(measurement_index)

        for measurement_index, (measurement, raw_axis) in enumerate(valid):
            if measurement_index in matched_measurements:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _AxisTrack(
                center_xy=np.asarray(measurement.center_xy, dtype=np.float64),
                size_px=self._measurement_size(measurement),
                stable_axis=raw_axis.copy(),
                last_seen_frame=frame_index,
            )
            result[measurement.det_idx] = raw_axis.copy()

        return result

    def _associate(self, valid):
        candidates = []
        for track_id, track in self._tracks.items():
            for measurement_index, (measurement, _) in enumerate(valid):
                center = np.asarray(measurement.center_xy, dtype=np.float64)
                distance = float(np.linalg.norm(center - track.center_xy))
                size_px = self._measurement_size(measurement)
                gate = max(
                    self.match_distance_px,
                    self.match_size_ratio * max(track.size_px, size_px),
                )
                if distance <= gate:
                    candidates.append((distance, track_id, measurement_index))

        matches = []
        used_tracks = set()
        used_measurements = set()
        for _, track_id, measurement_index in sorted(candidates):
            if track_id in used_tracks or measurement_index in used_measurements:
                continue
            matches.append((track_id, measurement_index))
            used_tracks.add(track_id)
            used_measurements.add(measurement_index)
        return matches

    def _update_track(self, track, measurement, raw_axis, frame_index):
        angle = _angle_deg(track.stable_axis, raw_axis)
        if angle <= self.max_jump_deg:
            blended = ((1.0 - self.ema_alpha) * track.stable_axis
                       + self.ema_alpha * raw_axis)
            normalized = _normalize_camera_facing(blended)
            if normalized is not None:
                track.stable_axis = normalized
            track.pending_axis = None
            track.pending_count = 0
        else:
            pending_matches = (
                track.pending_axis is not None
                and _angle_deg(track.pending_axis, raw_axis) <= self.max_jump_deg
            )
            track.pending_axis = raw_axis.copy()
            track.pending_count = track.pending_count + 1 if pending_matches else 1
            if track.pending_count >= self.confirm_frames:
                track.stable_axis = raw_axis.copy()
                track.pending_axis = None
                track.pending_count = 0

        center = np.asarray(measurement.center_xy, dtype=np.float64)
        track.center_xy = 0.7 * track.center_xy + 0.3 * center
        track.size_px = 0.7 * track.size_px + 0.3 * self._measurement_size(measurement)
        track.last_seen_frame = frame_index

    def _prune(self, frame_index):
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame <= self.stale_frames
        }

    @staticmethod
    def _measurement_size(measurement):
        x1, y1, x2, y2 = [float(value) for value in measurement.bbox]
        return max(1.0, x2 - x1, y2 - y1)
