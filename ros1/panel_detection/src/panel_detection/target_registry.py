"""Panel target absolute-ID matching used by the ROS1 detector.

The fixed main-row panel layout is:

    0 button/green, 1 button/red,
    2 knob/black, 3 knob/red,
    4 button/red, 5 button/red, 6 button/green.

The matcher accepts a partial view.  It first matches the visible
``(class, colour)`` sequence against a contiguous window of the full layout.
After one complete view has been seen, recent image positions are also used to
disambiguate narrow views.  ``PersistentPanelAxis`` keeps the panel-row
direction when one or both knobs leave the image.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Detection-based lower-row ID matching is disabled: the model reads the
# new buttons inconsistently and panel lights pollute green/white.  IDs
# 8/9/10 use the ArUco prior + refinement path instead.
LOWER_ROW_DETECTION_ENABLED = True


@dataclass
class RegisteredTarget:
    """Cached image-space reference for one absolute target ID."""

    target_id: int
    class_name: str
    center_x: float
    center_y: float
    bbox_w: float
    bbox_h: float


@dataclass
class FrameDetection:
    """Detection representation shared by the row selector and ID matcher."""

    class_name: str
    center_x: float
    center_y: float
    bbox: Tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """Keep the detector's existing ``Detection.center`` interface."""

        return (self.center_x, self.center_y)


def _is_button_like_detection(det: FrameDetection) -> bool:
    """Reject elongated indicator lights when recovering a missed button."""

    width = max(1.0, det.bbox[2] - det.bbox[0])
    height = max(1.0, det.bbox[3] - det.bbox[1])
    aspect = width / height
    return 0.55 <= aspect <= 1.8


def _normalize_axis(axis: Tuple[float, float]) -> Optional[np.ndarray]:
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis_array))
    if norm < 1e-6:
        return None
    axis_array = axis_array / norm
    if axis_array[0] < 0.0:
        axis_array = -axis_array
    return axis_array


class PersistentPanelAxis:
    """Cache the panel-row direction and relocate its parallel image line."""

    def __init__(
        self,
        dist_ratio: float = 0.85,
        min_dist: float = 45.0,
        max_dist: float = 110.0,
        min_proj_margin: float = 450.0,
    ):
        self.dist_ratio = float(dist_ratio)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)
        self.min_proj_margin = float(min_proj_margin)
        self._axis: Optional[np.ndarray] = None

    @property
    def has_axis(self) -> bool:
        return self._axis is not None

    def reset(self) -> None:
        self._axis = None

    def update(
        self,
        origin: Tuple[float, float],
        vector: Tuple[float, float],
    ) -> None:
        # The origin is deliberately not cached: after camera translation the
        # row moves in the image, while its direction is normally still valid.
        del origin
        axis = _normalize_axis(vector)
        if axis is not None:
            self._axis = axis

    def select(
        self,
        detections: List[FrameDetection],
    ) -> Optional[
        Tuple[
            List[FrameDetection],
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        if self._axis is None:
            return None

        candidates = [
            det
            for det in detections
            if det.class_name in ("button", "knob")
            or (
                det.class_name == "light"
                and _is_button_like_detection(det)
            )
        ]
        if not candidates:
            return None

        centers = np.asarray(
            [(det.center_x, det.center_y) for det in candidates],
            dtype=np.float64,
        )
        axis = self._axis
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

        panel_sizes = [
            max(1.0, det.bbox[3] - det.bbox[1])
            for det in candidates
            if det.class_name in ("button", "knob")
        ]
        median_height = float(np.median(panel_sizes)) if panel_sizes else 80.0
        distance_threshold = float(
            np.clip(
                median_height * self.dist_ratio,
                self.min_dist,
                self.max_dist,
            )
        )

        # Relocate the row by clustering candidates on the cached axis normal.
        normal_coordinates = centers.dot(normal)
        best_index = max(
            range(len(candidates)),
            key=lambda index: int(
                np.sum(
                    np.abs(
                        normal_coordinates - normal_coordinates[index]
                    )
                    <= distance_threshold
                )
            ),
        )
        best_normal_coordinate = normal_coordinates[best_index]
        row_mask = (
            np.abs(normal_coordinates - best_normal_coordinate)
            <= distance_threshold
        )
        row_centers = centers[row_mask]
        origin = np.mean(row_centers, axis=0)

        projections = [
            float(
                np.dot(
                    np.asarray(
                        [det.center_x, det.center_y], dtype=np.float64
                    )
                    - origin,
                    axis,
                )
            )
            for det in candidates
        ]
        projection_min = min(projections) - self.min_proj_margin
        projection_max = max(projections) + self.min_proj_margin

        selected = []
        for index, det in enumerate(candidates):
            if not row_mask[index]:
                continue
            point = (
                np.asarray([det.center_x, det.center_y], dtype=np.float64)
                - origin
            )
            normal_distance = abs(float(np.dot(point, normal)))
            projection = float(np.dot(point, axis))
            if (
                normal_distance <= distance_threshold
                and projection_min <= projection <= projection_max
            ):
                if det.class_name == "light":
                    det.class_name = "button"
                selected.append(det)

        if not selected:
            return None
        return (
            selected,
            (float(origin[0]), float(origin[1])),
            (float(axis[0]), float(axis[1])),
        )


# First (main) row, left to right: IDs 0..6.
ROW1_IDS = list(range(0, 7))
ROW1_LAYOUT = [
    ("button", "green"),  # ID 0 (old ID 7)
    ("button", "red"),    # ID 1 (old ID 6)
    ("knob", "black"),    # ID 2 (old ID 5)
    ("knob", "red"),      # ID 3 (old ID 4)
    ("button", "red"),    # ID 4 (old ID 3)
    ("button", "red"),    # ID 5 (old ID 2)
    ("button", "green"),  # ID 6 (old ID 1)
]

# Second (lower) row, left to right: IDs 8..10.  Each target has a distinct
# class/colour pair, so it is identified on its own; a movable target keeps its
# ID regardless of where it sits relative to the main row.
ROW2_LAYOUT = [
    ("button", "blue"),   # ID 8
    ("button", "green"),  # ID 9
    ("knob", "white"),    # ID 10
]

# Explicit lookup keeps the main row's valid ID 0 distinct from the unchanged
# second-row IDs 8..10.
PANEL_LAYOUT_BY_ID = dict(zip(ROW1_IDS, ROW1_LAYOUT))
PANEL_LAYOUT_BY_ID.update(dict(zip(range(8, 11), ROW2_LAYOUT)))


def _classify_color(
    color_image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    green_hue_range: Tuple[int, int] = (35, 85),
    minimum_ratio: float = 0.15,
) -> str:
    """Classify the centre half of a box as green, red, black or unknown."""

    image_height, image_width = color_image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    crop_x1 = max(0, x1 + box_width // 4)
    crop_x2 = min(image_width, x2 - box_width // 4)
    crop_y1 = max(0, y1 + box_height // 4)
    crop_y2 = min(image_height, y2 - box_height // 4)
    roi = color_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    pixel_count = int(hue.size)
    if pixel_count == 0:
        return "unknown"

    green_min, green_max = green_hue_range
    green = (
        (hue >= int(green_min))
        & (hue <= int(green_max))
        & (saturation > 50)
    )
    red = (
        ((hue <= 10) | (hue >= 160))
        & (saturation > 50)
    )
    black = value < 80
    ratios = {
        "green": float(np.sum(green)) / pixel_count,
        "red": float(np.sum(red)) / pixel_count,
        "black": float(np.sum(black)) / pixel_count,
    }
    best_color = max(ratios, key=ratios.get)
    if ratios[best_color] < float(minimum_ratio):
        return "unknown"
    return best_color


def _classify_second_row_color(
    color_image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    minimum_ratio: float = 0.15,
) -> str:
    """Classify a lower-row button/knob as blue, green, white or unknown.

    Kept separate from :func:`_classify_color` so the main-row colour logic is
    untouched.  The lower row uses blue / green / white, which the main-row
    classifier would report as ``unknown``.
    """

    image_height, image_width = color_image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    box_width = x2 - x1
    box_height = y2 - y1
    crop_x1 = max(0, x1 + box_width // 4)
    crop_x2 = min(image_width, x2 - box_width // 4)
    crop_y1 = max(0, y1 + box_height // 4)
    crop_y2 = min(image_height, y2 - box_height // 4)
    roi = color_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    pixel_count = int(hue.size)
    if pixel_count == 0:
        return "unknown"

    blue = (hue >= 95) & (hue <= 135) & (saturation > 50)
    green = (hue >= 35) & (hue <= 85) & (saturation > 60)
    white = (saturation < 65) & (value > 150)
    ratios = {
        "blue": float(np.sum(blue)) / pixel_count,
        "green": float(np.sum(green)) / pixel_count,
        "white": float(np.sum(white)) / pixel_count,
    }
    best_color = max(ratios, key=ratios.get)
    if ratios[best_color] < float(minimum_ratio):
        return "unknown"
    return best_color


def _match_subsequence_scored(
    observed: List[Tuple[str, str]],
) -> Tuple[List[int], int, bool]:
    """Find the best contiguous layout window for an observed sequence."""

    observed_count = len(observed)
    layout_count = len(ROW1_LAYOUT)
    if observed_count == 0:
        return [], -10000, False
    if observed_count > layout_count:
        observed = observed[:layout_count]
        observed_count = layout_count

    best_score = -10000
    best_offset = -1
    for offset in range(layout_count - observed_count + 1):
        score = 20
        valid = True
        for observed_index, (observed_class, observed_color) in enumerate(
            observed
        ):
            layout_class, layout_color = ROW1_LAYOUT[
                offset + observed_index
            ]
            if observed_class != layout_class:
                valid = False
                break
            if observed_color == layout_color:
                score += 3
            elif observed_color != "unknown":
                score -= 2
        if valid and score > best_score:
            best_score = score
            best_offset = offset

    if best_offset < 0:
        return [], -10000, False
    return (
        [ROW1_IDS[best_offset + index] for index in range(observed_count)],
        best_score,
        True,
    )


def _match_individual(observed: List[Tuple[str, str]]) -> List[int]:
    """Fallback for non-contiguous detections, with strict class ID pools."""

    id_pools = {
        "button": [0, 1, 4, 5, 6],
        "knob": [2, 3],
    }
    expected_colors = {
        target_id: color
        for target_id, (_, color) in zip(ROW1_IDS, ROW1_LAYOUT)
    }
    used_ids = set()
    result = []

    for observed_class, observed_color in observed:
        candidates = [
            target_id
            for target_id in id_pools.get(observed_class, [])
            if target_id not in used_ids
        ]
        color_candidates = [
            target_id
            for target_id in candidates
            if expected_colors[target_id] == observed_color
        ]
        target_id = (
            color_candidates[0]
            if color_candidates
            else (candidates[0] if candidates else -1)
        )
        if target_id >= 0:
            used_ids.add(target_id)
        result.append(target_id)
    return result


def _match_individual_allowed(
    observed: List[Tuple[str, str]],
    allowed_ids: List[int],
) -> List[int]:
    """Match a restricted camera view only against explicitly allowed IDs.

    This is used by the right wrist camera, whose valid physical targets are
    ID5/6. It prevents a solitary red ID5 button from being assigned to the
    first red button in the full-panel layout (ID1).
    """

    used_ids = set()
    result = []
    for observed_class, observed_color in observed:
        candidates = [
            target_id
            for target_id in allowed_ids
            if target_id not in used_ids
            and PANEL_LAYOUT_BY_ID[target_id][0] == observed_class
        ]
        color_candidates = [
            target_id
            for target_id in candidates
            if PANEL_LAYOUT_BY_ID[target_id][1] == observed_color
        ]
        if len(color_candidates) == 1:
            target_id = color_candidates[0]
        elif len(candidates) == 1:
            target_id = candidates[0]
        else:
            target_id = -1
        if target_id >= 0:
            used_ids.add(target_id)
        result.append(target_id)
    return result


class TargetRegistry:
    """Assign stable absolute IDs to the visible panel row."""

    def __init__(
        self,
        stable_frames: int = 15,
        green_hue_range: Tuple[int, int] = (35, 85),
        match_distance_thresh: float = 80.0,
        color_minimum_ratio: float = 0.15,
        allowed_ids: Optional[List[int]] = None,
    ):
        # stable_frames is kept for source/config compatibility.  This matcher
        # identifies every frame and does not require a registration pause.
        self._stable_frames = max(1, int(stable_frames))
        self._green_hue_range = (
            int(green_hue_range[0]),
            int(green_hue_range[1]),
        )
        self._color_minimum_ratio = float(color_minimum_ratio)
        self._match_distance_thresh = float(match_distance_thresh)
        self._allowed_ids = [int(value) for value in (allowed_ids or [])]
        if (
            len(set(self._allowed_ids)) != len(self._allowed_ids)
            or any(
                target_id not in ROW1_IDS
                for target_id in self._allowed_ids
            )
        ):
            raise ValueError("allowed_ids must be unique main-row IDs in 0..6")
        self._has_complete_reference = False
        self._last_by_id: Dict[int, RegisteredTarget] = {}
        self._registered: List[RegisteredTarget] = []

    @property
    def registered(self) -> bool:
        """Compatibility with the previous ROS1 registry API."""

        return True

    @property
    def is_registered(self) -> bool:
        return True

    @property
    def targets(self) -> List[RegisteredTarget]:
        return self._registered

    @property
    def has_complete_reference(self) -> bool:
        return self._has_complete_reference

    def reset(self) -> None:
        self._has_complete_reference = False
        self._last_by_id.clear()
        self._registered = []

    def update(
        self,
        detections: List[FrameDetection],
        color_image: Optional[np.ndarray] = None,
    ):
        """Compatibility wrapper returning the existing ``{id: detection}``."""

        if color_image is None:
            return {}
        return dict(self.identify(detections, color_image))

    def identify(
        self,
        detections: List[FrameDetection],
        color_image: np.ndarray,
        axis_origin: Optional[Tuple[float, float]] = None,
        axis_vector: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[int, FrameDetection]]:
        if not detections:
            return []

        sorted_detections = self._sort_along_axis(
            detections, axis_origin, axis_vector
        )
        observed = [
            (
                det.class_name,
                _classify_color(
                    color_image,
                    det.bbox,
                    green_hue_range=self._green_hue_range,
                    minimum_ratio=self._color_minimum_ratio,
                ),
            )
            for det in sorted_detections
        ]
        if self._allowed_ids:
            ids = _match_individual_allowed(observed, self._allowed_ids)
        else:
            ids = self._match_contiguous_ids(sorted_detections, observed)
        matched = [
            (target_id, det)
            for target_id, det in zip(ids, sorted_detections)
            if target_id >= 0
        ]
        self._update_recent_reference(matched)
        return matched

    def identify_second_row(
        self,
        detections: List[FrameDetection],
        color_image: np.ndarray,
        exclude_detection_ids: Optional[set] = None,
    ) -> List[Tuple[int, FrameDetection]]:
        """Assign IDs 8..10 from the lower row using colour (class-tolerant).

        Blue (8) and white (10) have globally unique colours, so they match
        regardless of class.  Their detections must NOT be in ``exclude``
        (main-row detections), so main-row IDs 0..6 are never stolen.  Green (9)
        may be a main-row detection, but only right next to a non-main anchor.
        The lower-row knob is assigned ID 10 when it is not a main-row detection.
        """
        if not LOWER_ROW_DETECTION_ENABLED:
            return []
        if not detections or color_image is None:
            return []

        classified = [
            (
                detection,
                detection.class_name,
                _classify_second_row_color(
                    color_image,
                    detection.bbox,
                    minimum_ratio=self._color_minimum_ratio,
                ),
            )
            for detection in detections
        ]
        used = set()
        exclude = exclude_detection_ids or set()
        result: List[Tuple[int, FrameDetection]] = []

        def take(target_id, wanted_color, wanted_classes):
            for detection, class_name, color_name in classified:
                if id(detection) in used or id(detection) in exclude:
                    continue
                if class_name in wanted_classes and color_name == wanted_color:
                    used.add(id(detection))
                    result.append((target_id, detection))
                    return True
            return False

        take(8, "blue", ("button", "light"))
        take(10, "white", ("knob", "button", "light"))
        anchor = next(
            (
                detection
                for target_id, detection in result
                if target_id in (8, 10)
            ),
            None,
        )
        if anchor is not None:
            anchor_width = max(1.0, anchor.bbox[2] - anchor.bbox[0])
            anchor_height = max(1.0, anchor.bbox[3] - anchor.bbox[1])
            max_gap = 3.0 * max(anchor_width, anchor_height)
            best = None
            best_distance = None
            for detection, class_name, color_name in classified:
                if id(detection) in used:
                    continue
                if class_name not in ("button", "light") or color_name != "green":
                    continue
                distance = float(
                    (
                        (detection.center_x - anchor.center_x) ** 2
                        + (detection.center_y - anchor.center_y) ** 2
                    )
                    ** 0.5
                )
                if distance > max_gap:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best = detection
            if best is not None:
                used.add(id(best))
                result.append((9, best))
        if not any(target_id == 10 for target_id, _ in result):
            # RELAX-10: the white knob is often detected as button/light with
            # an unclassified colour.  After 8 (blue) and 9 (green) are taken,
            # 10 is the remaining lower-row detection: prefer a white/unknown
            # one, else accept a lone leftover.
            remaining = [
                (detection, class_name, color_name)
                for detection, class_name, color_name in classified
                if id(detection) not in used
                and id(detection) not in exclude
                and class_name in ("knob", "button", "light")
            ]
            best = None
            # 10 is now a black knob with a white handle (like ID 2), so it is
            # detected as class "knob".  Green is 9 and blue is 8, so only a
            # black knob or an unclassified detection may be 10.
            knob_like = [d for d, c, col in remaining if c == "knob"]
            white_like = [d for d, c, col in remaining
                          if col in ("white", "unknown")]
            if len(knob_like) == 1:
                best = knob_like[0]
            elif len(white_like) == 1:
                best = white_like[0]
            if best is not None:
                used.add(id(best))
                result.append((10, best))
        return result

    def classify_lower_row(
        self,
        detections: List[FrameDetection],
        color_image: np.ndarray,
    ) -> List[Tuple[FrameDetection, str, str]]:
        """Return (detection, class, colour) for diagnostics."""

        if color_image is None:
            return []
        return [
            (
                detection,
                detection.class_name,
                _classify_second_row_color(
                    color_image,
                    detection.bbox,
                    minimum_ratio=self._color_minimum_ratio,
                ),
            )
            for detection in detections
        ]

    @staticmethod
    def _sort_along_axis(
        detections: List[FrameDetection],
        axis_origin: Optional[Tuple[float, float]],
        axis_vector: Optional[Tuple[float, float]],
    ) -> List[FrameDetection]:
        if axis_origin is None or axis_vector is None:
            return sorted(detections, key=lambda det: det.center_x)

        axis = _normalize_axis(axis_vector)
        if axis is None:
            return sorted(detections, key=lambda det: det.center_x)
        origin = np.asarray(axis_origin, dtype=np.float64)
        return sorted(
            detections,
            key=lambda det: float(
                np.dot(
                    np.asarray(
                        [det.center_x, det.center_y], dtype=np.float64
                    )
                    - origin,
                    axis,
                )
            ),
        )

    def _match_contiguous_ids(
        self,
        sorted_detections: List[FrameDetection],
        observed: List[Tuple[str, str]],
    ) -> List[int]:
        # Cold start trusts class/colour layout.  A contiguous match prevents
        # an anchor from silently skipping an unseen middle target.
        if not self._has_complete_reference:
            ids, _, contiguous = _match_subsequence_scored(observed)
            return ids if contiguous else _match_individual(observed)

        ids, _, contiguous = self._match_reference_window_scored(
            sorted_detections, observed
        )
        if contiguous:
            return ids
        ids, _, contiguous = _match_subsequence_scored(observed)
        return ids if contiguous else _match_individual(observed)

    def _match_reference_window_scored(
        self,
        sorted_detections: List[FrameDetection],
        observed: List[Tuple[str, str]],
    ) -> Tuple[List[int], int, bool]:
        observed_count = len(observed)
        layout_count = len(ROW1_LAYOUT)
        if observed_count == 0 or observed_count > layout_count:
            return [], -10000, False

        best_score = -10000
        best_offset = -1
        for offset in range(layout_count - observed_count + 1):
            score = 20
            valid = True
            for observed_index, (
                observed_class,
                observed_color,
            ) in enumerate(observed):
                layout_index = offset + observed_index
                target_id = ROW1_IDS[layout_index]
                layout_class, layout_color = ROW1_LAYOUT[layout_index]
                if observed_class != layout_class:
                    valid = False
                    break

                if observed_color == layout_color:
                    score += 3
                elif observed_color != "unknown":
                    score -= 2

                previous = self._last_by_id.get(target_id)
                if (
                    previous is None
                    or previous.class_name != observed_class
                ):
                    continue

                det = sorted_detections[observed_index]
                current_xy = np.asarray(
                    [det.center_x, det.center_y], dtype=np.float64
                )
                previous_xy = np.asarray(
                    [previous.center_x, previous.center_y],
                    dtype=np.float64,
                )
                distance = float(np.linalg.norm(current_xy - previous_xy))
                threshold = self._dynamic_match_threshold(det)
                if distance <= threshold:
                    score += int(
                        round(30.0 * (1.0 - distance / threshold))
                    )
                else:
                    score -= int(
                        round(
                            min(
                                30.0,
                                10.0 * distance / threshold,
                            )
                        )
                    )

            if valid and score > best_score:
                best_score = score
                best_offset = offset

        if best_offset < 0:
            return [], -10000, False
        return (
            [ROW1_IDS[best_offset + index] for index in range(observed_count)],
            best_score,
            True,
        )

    def _dynamic_match_threshold(self, det: FrameDetection) -> float:
        width = max(1.0, det.bbox[2] - det.bbox[0])
        height = max(1.0, det.bbox[3] - det.bbox[1])
        return max(
            self._match_distance_thresh,
            2.5 * max(width, height),
        )

    def _update_recent_reference(
        self,
        matches: List[Tuple[int, FrameDetection]],
    ) -> None:
        if not matches:
            return

        current_ids = {target_id for target_id, _ in matches}
        if set(ROW1_IDS).issubset(current_ids):
            self._has_complete_reference = True
        if not self._has_complete_reference:
            return

        for target_id, det in matches:
            self._last_by_id[target_id] = RegisteredTarget(
                target_id=target_id,
                class_name=det.class_name,
                center_x=float(det.center_x),
                center_y=float(det.center_y),
                bbox_w=float(det.bbox[2] - det.bbox[0]),
                bbox_h=float(det.bbox[3] - det.bbox[1]),
            )
        self._registered = [
            self._last_by_id[target_id]
            for target_id in sorted(self._last_by_id)
        ]
