"""AprilTag-based classification and numbering for panel controls."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .apriltag_reference import _detect_aruco_markers, _get_apriltag_dictionary
from .target_registry import FrameDetection


BUTTON_TAG_IDS = frozenset((0, 1, 2, 5, 6))
KNOB_TAG_IDS = frozenset((3, 4))
PANEL_TAG_ID_MIN = 0
PANEL_TAG_ID_MAX = 39
ELIGIBLE_CLASSES = frozenset(('button', 'knob', 'light'))


def reclassify_buttons_without_tags(detections, markers):
    """Treat YOLO buttons as lights when no tag is decoded this frame."""
    if markers:
        return []
    changed = []
    for detection in detections:
        if detection.class_name == 'button':
            detection.class_name = 'light'
            changed.append(detection)
    return changed


def redraw_reclassified_detection(canvas, raw_image, detection,
                                   old_class, color):
    """Remove the detector's old label and draw the final class on Canvas."""
    x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
    height, width = canvas.shape[:2]
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    font_scale = 2.0 / 3.0
    thickness = 1
    old_label = f'{old_class} {detection.confidence:.2f}'
    new_label = f'{detection.class_name} {detection.confidence:.2f}'
    old_size = cv2.getTextSize(old_label, 0, font_scale, thickness)[0]
    new_size = cv2.getTextSize(new_label, 0, font_scale, thickness)[0]
    restore_top = max(0, y1 - max(old_size[1], new_size[1]) - 7)
    restore_right = min(width, x1 + max(old_size[0], new_size[0]) + 5)
    if restore_top < y1 + 2 and x1 < restore_right:
        canvas[restore_top:min(height, y1 + 2), x1:restore_right] = \
            raw_image[restore_top:min(height, y1 + 2), x1:restore_right]

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    text_width, text_height = new_size
    text_y = max(text_height + 3, y1 - 2)
    cv2.rectangle(
        canvas,
        (x1, max(0, text_y - text_height - 3)),
        (min(width - 1, x1 + text_width + 4), min(height - 1, text_y + 2)),
        (0, 0, 0), -1)
    cv2.putText(canvas, new_label, (x1 + 2, text_y), 0, font_scale,
                color, thickness, cv2.LINE_AA)


def forced_class_for_tag(tag_id: int) -> Optional[str]:
    """Return the authoritative component class encoded by a panel tag."""
    tag_id = int(tag_id)
    if tag_id < PANEL_TAG_ID_MIN or tag_id > PANEL_TAG_ID_MAX:
        return None
    if tag_id in BUTTON_TAG_IDS:
        return 'button'
    if tag_id in KNOB_TAG_IDS:
        return 'knob'
    return 'light'


@dataclass(frozen=True)
class PanelTagMarker:
    tag_id: int
    corners: np.ndarray
    center_xy: Tuple[float, float]
    area: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        points = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        return (
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        )


@dataclass(frozen=True)
class PanelTagAssignment:
    detection_index: int
    tag_id: int
    target_id: int
    forced_class: str
    source: str
    marker: Optional[PanelTagMarker] = None


@dataclass
class _TagTrack:
    tag_id: int
    center_xy: np.ndarray
    size_wh: np.ndarray
    velocity_xy: np.ndarray
    last_frame: int
    last_tag_frame: int


def detect_panel_tags(color_image: np.ndarray, cfg=None) -> List[PanelTagMarker]:
    """Detect decoded tag36h11 markers reserved for panel control IDs."""
    cfg = cfg or {}
    if color_image is None or color_image.size == 0 or not hasattr(cv2, 'aruco'):
        return []
    dictionary = _get_apriltag_dictionary(
        cfg.get('dictionary', 'DICT_APRILTAG_36h11'))
    if dictionary is None:
        return []

    if color_image.ndim == 2:
        gray = color_image
    else:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, _ = _detect_aruco_markers(gray, dictionary)
    except (AttributeError, TypeError, ValueError, cv2.error):
        # Tag support is auxiliary and must never terminate the detection node
        # when OpenCV's optional aruco API differs between deployments.
        return []
    if ids is None:
        return []

    min_id = int(cfg.get('min_id', PANEL_TAG_ID_MIN))
    max_id = int(cfg.get('max_id', PANEL_TAG_ID_MAX))
    min_area = float(cfg.get('min_area_px', 64.0))
    marker_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    markers_by_id: Dict[int, PanelTagMarker] = {}
    for index, raw_corners in enumerate(corners):
        if index >= marker_ids.size:
            break
        tag_id = int(marker_ids[index])
        if tag_id < min_id or tag_id > max_id:
            continue
        points = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        area = abs(float(cv2.contourArea(points)))
        if area < min_area:
            continue
        marker = PanelTagMarker(
            tag_id=tag_id,
            corners=points,
            center_xy=tuple(float(value) for value in np.mean(points, axis=0)),
            area=area,
        )
        previous = markers_by_id.get(tag_id)
        if previous is None or marker.area > previous.area:
            markers_by_id[tag_id] = marker
    return sorted(markers_by_id.values(), key=lambda marker: marker.tag_id)


class PanelAprilTagTracker:
    """Associate tags above controls and bridge short tag detection gaps."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.max_horizontal_ratio = float(cfg.get('max_horizontal_ratio', 1.35))
        self.max_vertical_ratio = float(cfg.get('max_vertical_ratio', 3.0))
        self.overlap_tolerance_ratio = float(
            cfg.get('overlap_tolerance_ratio', 0.35))
        self.min_horizontal_gate_px = float(
            cfg.get('min_horizontal_gate_px', 24.0))
        self.min_vertical_gate_px = float(cfg.get('min_vertical_gate_px', 30.0))
        self.track_distance_px = float(cfg.get('track_distance_px', 90.0))
        self.track_size_ratio = float(cfg.get('track_size_ratio', 1.3))
        self.stale_frames = max(0, int(cfg.get('stale_frames', 8)))
        self._frame_index = 0
        self._tracks: Dict[int, _TagTrack] = {}

    def update(self, detections: List[FrameDetection],
               markers: Iterable[PanelTagMarker]) -> Dict[int, PanelTagAssignment]:
        self._frame_index += 1
        markers = list(markers)
        assignments: Dict[int, PanelTagAssignment] = {}

        eligible = [
            index for index, detection in enumerate(detections)
            if detection.class_name in ELIGIBLE_CLASSES
        ]
        candidates = []
        for marker in markers:
            if forced_class_for_tag(marker.tag_id) is None:
                continue
            for detection_index in eligible:
                score = self._direct_score(marker, detections[detection_index])
                if score is not None:
                    candidates.append((score, marker.tag_id, detection_index, marker))

        used_tags = set()
        used_detections = set()
        for _, tag_id, detection_index, marker in sorted(candidates):
            if tag_id in used_tags or detection_index in used_detections:
                continue
            assignment = self._make_assignment(
                detection_index, tag_id, 'apriltag_36h11', marker)
            assignments[detection_index] = assignment
            used_tags.add(tag_id)
            used_detections.add(detection_index)
            self._update_track(tag_id, detections[detection_index], tag_seen=True)

        track_candidates = []
        for tag_id, track in self._tracks.items():
            if tag_id in used_tags:
                continue
            tag_age = self._frame_index - track.last_tag_frame
            if tag_age > self.stale_frames:
                continue
            for detection_index in eligible:
                if detection_index in used_detections:
                    continue
                score = self._track_score(track, detections[detection_index])
                if score is not None:
                    track_candidates.append((score, tag_id, detection_index))

        for _, tag_id, detection_index in sorted(track_candidates):
            if tag_id in used_tags or detection_index in used_detections:
                continue
            assignments[detection_index] = self._make_assignment(
                detection_index, tag_id, 'apriltag_track', None)
            used_tags.add(tag_id)
            used_detections.add(detection_index)
            self._update_track(tag_id, detections[detection_index], tag_seen=False)

        self._tracks = {
            tag_id: track for tag_id, track in self._tracks.items()
            if self._frame_index - track.last_tag_frame <= self.stale_frames
        }
        return assignments

    def _make_assignment(self, detection_index, tag_id, source, marker):
        return PanelTagAssignment(
            detection_index=detection_index,
            tag_id=int(tag_id),
            target_id=int(tag_id) + 1,
            forced_class=forced_class_for_tag(tag_id),
            source=source,
            marker=marker,
        )

    def _direct_score(self, marker: PanelTagMarker,
                      detection: FrameDetection) -> Optional[float]:
        x1, y1, x2, y2 = [float(value) for value in detection.bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        mx1, my1, mx2, my2 = marker.bounds
        marker_width = max(1.0, mx2 - mx1)
        marker_height = max(1.0, my2 - my1)
        dx = abs(float(marker.center_xy[0]) - detection.center_x)
        horizontal_gate = max(
            self.min_horizontal_gate_px,
            width * self.max_horizontal_ratio,
            marker_width * 2.5,
        )
        if dx > horizontal_gate:
            return None

        # The tag must be above the component center. A small overlap with the
        # component bbox is allowed for perspective and loose YOLO boxes.
        if float(marker.center_xy[1]) >= detection.center_y:
            return None
        vertical_gap = y1 - my2
        overlap_tolerance = height * self.overlap_tolerance_ratio
        vertical_gate = max(
            self.min_vertical_gate_px,
            height * self.max_vertical_ratio,
            marker_height * 6.0,
        )
        if vertical_gap < -overlap_tolerance or vertical_gap > vertical_gate:
            return None
        return (
            dx / horizontal_gate +
            max(0.0, vertical_gap) / vertical_gate +
            0.05 * abs(marker_width - marker_height) / max(marker_width, marker_height)
        )

    def _track_score(self, track: _TagTrack,
                     detection: FrameDetection) -> Optional[float]:
        center = np.array(
            [detection.center_x, detection.center_y], dtype=np.float64)
        size = self._detection_size(detection)
        delta_frames = max(1, self._frame_index - track.last_frame)
        predicted = track.center_xy + track.velocity_xy * delta_frames
        distance = float(np.linalg.norm(center - predicted))
        size_scale = max(float(np.max(track.size_wh)), float(np.max(size)), 1.0)
        distance_gate = max(self.track_distance_px, size_scale * 0.9)
        if distance > distance_gate:
            return None
        size_error = float(np.max(np.abs(size - track.size_wh) /
                                  np.maximum(track.size_wh, 1.0)))
        if size_error > self.track_size_ratio:
            return None
        return distance / distance_gate + 0.25 * size_error

    def _update_track(self, tag_id: int, detection: FrameDetection,
                      tag_seen: bool) -> None:
        center = np.array(
            [detection.center_x, detection.center_y], dtype=np.float64)
        size = self._detection_size(detection)
        previous = self._tracks.get(tag_id)
        if previous is None:
            velocity = np.zeros(2, dtype=np.float64)
            last_tag_frame = self._frame_index
        else:
            delta_frames = max(1, self._frame_index - previous.last_frame)
            measured_velocity = (center - previous.center_xy) / delta_frames
            velocity = 0.5 * previous.velocity_xy + 0.5 * measured_velocity
            last_tag_frame = (
                self._frame_index if tag_seen else previous.last_tag_frame)
        self._tracks[tag_id] = _TagTrack(
            tag_id=tag_id,
            center_xy=center,
            size_wh=size,
            velocity_xy=velocity,
            last_frame=self._frame_index,
            last_tag_frame=last_tag_frame,
        )

    @staticmethod
    def _detection_size(detection: FrameDetection) -> np.ndarray:
        x1, y1, x2, y2 = [float(value) for value in detection.bbox]
        return np.array([max(1.0, x2 - x1), max(1.0, y2 - y1)],
                        dtype=np.float64)


def draw_panel_tag_assignments(canvas: np.ndarray,
                               markers: Iterable[PanelTagMarker],
                               assignments: Dict[int, PanelTagAssignment],
                               detections: List[FrameDetection]) -> None:
    """Draw decoded tags and their authoritative component assignments."""
    marker_to_detection = {
        assignment.tag_id: detection_index
        for detection_index, assignment in assignments.items()
        if assignment.marker is not None
    }
    for marker in markers:
        points = np.round(marker.corners).astype(np.int32).reshape(4, 2)
        assigned = marker.tag_id in marker_to_detection
        color = (0, 255, 255) if assigned else (0, 140, 255)
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        tx, ty = [int(round(value)) for value in marker.center_xy]
        cv2.putText(canvas, f'T{marker.tag_id:02d}', (tx - 14, max(14, ty - 8)),
                    0, 0.45, color, 1, cv2.LINE_AA)

    class_colors = {
        'button': (0, 255, 0),
        'knob': (0, 200, 255),
        'light': (255, 255, 0),
    }
    for detection_index, assignment in assignments.items():
        if detection_index >= len(detections):
            continue
        detection = detections[detection_index]
        x1, y1, x2, _ = [int(round(value)) for value in detection.bbox]
        color = class_colors[assignment.forced_class]
        label = (f'#{assignment.target_id} T{assignment.tag_id:02d} '
                 f'{assignment.forced_class}')
        (text_width, text_height), _ = cv2.getTextSize(label, 0, 0.48, 1)
        label_y = max(text_height + 3, y1 - 3)
        cv2.rectangle(canvas, (x1, label_y - text_height - 3),
                      (min(canvas.shape[1] - 1, x1 + text_width + 4), label_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (x1 + 2, label_y), 0, 0.48,
                    color, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, int(round(detection.bbox[3]))),
                      color, 2, cv2.LINE_AA)
        if assignment.marker is not None:
            marker_center = tuple(
                int(round(value)) for value in assignment.marker.center_xy)
            component_center = (
                int(round(detection.center_x)), int(round(detection.center_y)))
            cv2.line(canvas, marker_center, component_center, color, 1,
                     cv2.LINE_AA)
