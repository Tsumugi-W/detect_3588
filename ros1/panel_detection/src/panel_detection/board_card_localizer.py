"""Stable tag-relative board-card state and position estimation.

T07 supplies a scale-normalised slot frame.  Colour confirms the binary state
(black card fully inserted vs empty slot).  In the inserted state the published
centre is the centre of the **outer black card silhouette**, not the centre of
the narrow black strip used by earlier versions.  The silhouette is measured
from the low-value/low-saturation card body, neighbouring card fragments are
merged across its white label, and a robust tag-relative centre/axis/width/
length lock is projected through the current T07 homography.

Depth is measured on the card front itself (the white label and illuminated
edges usually provide valid pixels), with the panel plane only as a fallback.
The empty state publishes the blue-backplane slot centre on the panel plane.
No fixed tag-relative card top, length, or slot anchor is used: the tag may be
mounted with a different offset on another panel.

The module is deliberately independent of ROS.  It accepts a plain BGR image,
a depth image, a camera matrix and a tag object with ``corners``/``tag_id``
attributes (or a mapping with the same fields).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


CANONICAL_TAG = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
)

DEFAULT_CONFIG = {
    "enable": True,
    "tag_id": 7,
    # Fixed tag-relative slot ROI, in tag-side units.  +x is image-right and
    # +y is image-down after the tag orientation is normalised.
    "roi_x": (-3.5, 3.5),
    "roi_y": (1.0, 11.0),
    "roi_size": (560, 800),
    # Search band inside the rectified ROI.  The central band excludes the
    # recess bevel and its cast shadow on both sides.
    "slot_x": (0.36, 0.64),
    "slot_y": (0.16, 0.99),
    # Black card body only.  The validation site has one black card; there is
    # no blue-card or partially-pulled state in production.
    "dark_v_max": 105,
    "dark_s_max": 165,
    # Geometry gates.  The fully inserted card is long and its top edge sits
    # below the card's white handle.  A short dark blob near the bottom of an
    # empty recess must not be accepted as a card.
    "min_visible_rows": 30,
    "min_width_px": 70,
    "max_width_px": 150,
    "min_confidence": 0.30,
    "max_row_gap": 24,
    "min_inserted_length_tag": 3.8,
    "min_black_fraction": 0.50,
    # Full outer-card silhouette.  The black card body is much darker and less
    # saturated than the blue backplane and the white recess.  Neighbouring
    # card fragments separated by the card's white label are merged inside a
    # narrow band around the main body.
    "card_dark_v_max": 70,
    "card_dark_s_max": 100,
    "card_min_area_px": 5000,
    "card_min_width_px": 50,
    "card_max_width_px": 260,
    "card_candidate_max_width_px": 520,
    "card_trim_margin_px": 12,
    "card_min_length_px": 180,
    "card_min_fill_ratio": 0.45,
    "card_merge_gap_px": 28,
    "card_merge_min_area_px": 300,
    "card_merge_min_height_px": 40,
    "card_merge_min_overlap": 0.40,
    "card_protrusion_min_m": 0.015,
    "card_component_min_protrusion_fraction": 0.45,
    "card_min_protrusion_pixels": 500,
    "card_outer_min_confidence": 0.35,
    "card_outer_min_black_fraction": 0.55,
    # Card-front depth plane.  A plane fit is accepted only when its residual
    # is small and its centre is consistent with the measured median depth.
    "card_depth_min_points": 80,
    "card_depth_max_rms_m": 0.010,
    "card_depth_gate_m": 0.08,
    "card_plane_inlier_m": 0.006,
    "card_plane_min_inliers": 150,
    # Robust tag-relative pose lock.  No absolute tag-relative position prior:
    # the tag may be mounted with a different offset on another panel.
    "pose_lock_frames": 5,
    "pose_lock_max_mad_tag": 0.30,
    # Binary state confirmation.  Insertion motion is out of scope, so only a
    # stable empty/inserted state is published.
    "inserted_confirm_frames": 3,
    "empty_confirm_frames": 5,
    # Panel plane (depth must be aligned to colour).
    "panel_plane_radius_tag": 1.2,
    "panel_plane_min_points": 60,
    "panel_plane_min_inliers": 600,
    "panel_z_range_m": (0.15, 1.20),
    "panel_depth_offset_m": 0.0,
    "plane_hold_frames": 15,
    # Stillness-gated final position lock.
    "still_frames": 5,
    "still_pixel_thresh": 2.0,
    "still_window": 15,
    "plane_depth_gate_m": 0.12,
    # Temporal smoothing of the 2D anchor.
    "smooth_alpha": 0.35,
    "max_anchor_jump_px": 35.0,
    "tag_cache_frames": 12,
    # Empty-slot centre is measured from the blue backplane; no fixed
    # tag-relative slot anchor is used.
    "slot_blue_h": (80, 138),
    "slot_blue_s_min": 70,
    "slot_blue_v_min": 18,
    "slot_min_blue_area": 300,
    "slot_min_blue_height": 80,
    "slot_max_blue_width": 180,
    # Depth / plane fitting.
    "depth_min_m": 0.05,
    "depth_max_m": 2.0,
    "depth_scale": 0.001,
    "plane_inlier_m": 0.004,
    "plane_iterations": 80,
    "plane_min_points": 40,
}


def _cfg_value(cfg: Optional[dict], key: str) -> Any:
    if cfg and key in cfg:
        return cfg[key]
    return DEFAULT_CONFIG[key]


def _tag_field(tag: Any, name: str, default=None):
    if tag is None:
        return default
    if isinstance(tag, dict):
        return tag.get(name, default)
    return getattr(tag, name, default)


def _tag_corners(tag: Any) -> Optional[np.ndarray]:
    corners = _tag_field(tag, "corners")
    if corners is None:
        return None
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        return None
    return points


def _tag_id(tag: Any) -> Optional[int]:
    value = _tag_field(tag, "tag_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _intrinsics(camera_info: Any) -> Optional[np.ndarray]:
    """Accept ROS CameraInfo, a mapping or a 3x3 matrix."""

    if camera_info is None:
        return None
    if isinstance(camera_info, np.ndarray) and camera_info.shape == (3, 3):
        matrix = camera_info.astype(np.float64)
    elif isinstance(camera_info, dict):
        matrix = np.asarray(camera_info.get("K", camera_info), dtype=np.float64)
        if matrix.size == 9:
            matrix = matrix.reshape(3, 3)
        else:
            return None
    else:
        values = getattr(camera_info, "K", None)
        if values is None:
            return None
        matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        return None
    return matrix


def _depth_metres(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth is None:
        return None
    array = np.asarray(depth)
    if array.ndim != 2:
        return None
    if array.dtype == np.float32 or array.dtype == np.float64:
        return array.astype(np.float64)
    return array.astype(np.float64) * float(depth_scale)


def _normalise_tag_frame(marker: Any, dst: np.ndarray) -> np.ndarray:
    """Make the tag frame image-right / image-down before warping."""

    corners = _tag_corners(marker)
    if corners is None:
        return dst
    homography, _ = cv2.findHomography(corners, dst, 0)
    if homography is None:
        return dst
    inverse = np.linalg.inv(homography)
    centre = cv2.perspectiveTransform(
        np.array([[[0.5, 0.5]]], dtype=np.float32), inverse
    )[0, 0]
    px = cv2.perspectiveTransform(
        np.array([[[1.5, 0.5]]], dtype=np.float32), inverse
    )[0, 0]
    py = cv2.perspectiveTransform(
        np.array([[[0.5, 1.5]]], dtype=np.float32), inverse
    )[0, 0]
    oriented = dst.copy()
    if px[0] < centre[0]:
        oriented[:, 0] = 1.0 - oriented[:, 0]
    if py[1] < centre[1]:
        oriented[:, 1] = 1.0 - oriented[:, 1]
    return oriented


def rectify_board_card_roi(
    color: np.ndarray, marker: Any, cfg: Optional[dict] = None
) -> Optional[dict]:
    """Warp the fixed tag-relative slot ROI and its validity mask."""

    if color is None or color.size == 0 or marker is None:
        return None
    corners = _tag_corners(marker)
    if corners is None:
        return None
    dst = _normalise_tag_frame(marker, CANONICAL_TAG.copy())
    homography, _ = cv2.findHomography(corners, dst, 0)
    if homography is None or not np.isfinite(homography).all():
        return None

    roi_x = _cfg_value(cfg, "roi_x")
    roi_y = _cfg_value(cfg, "roi_y")
    width, height = _cfg_value(cfg, "roi_size")
    width, height = int(width), int(height)
    x0, x1 = float(roi_x[0]) + 0.5, float(roi_x[1]) + 0.5
    y0, y1 = float(roi_y[0]) + 0.5, float(roi_y[1]) + 0.5
    sx, sy = width / max(x1 - x0, 1e-6), height / max(y1 - y0, 1e-6)
    transform = np.array(
        [[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image_to_roi = transform @ homography
    roi = cv2.warpPerspective(
        color, image_to_roi, (width, height), flags=cv2.INTER_LINEAR
    )
    valid = cv2.warpPerspective(
        np.ones(color.shape[:2], dtype=np.uint8) * 255,
        image_to_roi,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ) > 0
    return {
        "roi": roi,
        "valid": valid,
        "image_to_roi": image_to_roi,
        "roi_to_image": np.linalg.inv(image_to_roi),
        "transform": transform,
        "tag": marker,
    }


def _row_runs(mask_row: np.ndarray, min_width: int = 8) -> list:
    changes = np.diff(np.r_[0, mask_row.astype(np.int8), 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_width
    ]


def _merge_runs(runs: list, max_gap: int, max_width: int) -> list:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if (
            start - merged[-1][1] - 1 <= max_gap
            and end - merged[-1][0] + 1 <= max_width
        ):
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(run) for run in merged]


def _fit_line(ys: np.ndarray, xs: np.ndarray) -> Optional[Tuple[float, float]]:
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if len(ys) < 8:
        return None
    keep = np.ones(len(ys), dtype=bool)
    for _ in range(4):
        if keep.sum() < 8:
            break
        slope, intercept = np.polyfit(ys[keep], xs[keep], 1)
        residual = np.abs(xs - (slope * ys + intercept))
        limit = max(1.0, 3.0 * float(np.median(residual[keep])))
        updated = residual <= limit
        if updated.sum() < 8 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def _refine_x(
    gray: np.ndarray, ys: np.ndarray, xs: np.ndarray, radius: int = 6
) -> np.ndarray:
    refined = []
    height, width = gray.shape[:2]
    for y, x in zip(ys, xs):
        yy = int(round(float(y)))
        if yy < 1 or yy >= height - 1:
            refined.append(float(x))
            continue
        lo = max(1, int(round(float(x))) - radius)
        hi = min(width - 2, int(round(float(x))) + radius + 1)
        if hi - lo < 4:
            refined.append(float(x))
            continue
        gradient = np.abs(
            cv2.Sobel(gray[yy : yy + 1, lo:hi], cv2.CV_32F, 1, 0, ksize=3)
        ).ravel()
        index = int(np.argmax(gradient))
        peak = float(lo + index)
        if 0 < index < len(gradient) - 1:
            denominator = (
                gradient[index - 1] - 2.0 * gradient[index] + gradient[index + 1]
            )
            if abs(denominator) > 1e-6:
                offset = 0.5 * (gradient[index - 1] - gradient[index + 1]) / denominator
                peak += float(np.clip(offset, -0.6, 0.6))
        refined.append(peak if abs(peak - float(x)) <= radius else float(x))
    return np.asarray(refined, dtype=np.float64)


def _refine_y(
    gray: np.ndarray, y: float, left_x: float, right_x: float, radius: int = 6
) -> float:
    height, width = gray.shape[:2]
    yy = int(round(float(y)))
    x0 = max(0, int(round(float(left_x))))
    x1 = min(width - 1, int(round(float(right_x))))
    if x1 - x0 < 10:
        return float(y)
    lo = max(1, yy - radius)
    hi = min(height - 2, yy + radius + 1)
    if hi - lo < 3:
        return float(y)
    gradient = np.abs(
        cv2.Sobel(gray[lo:hi, x0:x1], cv2.CV_32F, 0, 1, ksize=3)
    ).mean(axis=1)
    index = int(np.argmax(gradient))
    peak = float(lo + index)
    if 0 < index < len(gradient) - 1:
        denominator = gradient[index - 1] - 2.0 * gradient[index] + gradient[index + 1]
        if abs(denominator) > 1e-6:
            offset = 0.5 * (gradient[index - 1] - gradient[index + 1]) / denominator
            peak += float(np.clip(offset, -0.6, 0.6))
    return peak


def _row_span(
    mask: np.ndarray,
    y: float,
    centre_x: float,
    half_width: float,
    max_gap: int,
    max_width: int,
) -> Optional[Tuple[float, float]]:
    height, width = mask.shape[:2]
    yy = int(round(np.clip(y, 0, height - 1)))
    x0 = max(0, int(round(centre_x - half_width)))
    x1 = min(width, int(round(centre_x + half_width)))
    runs = [
        run
        for run in _merge_runs(_row_runs(mask[yy], 8), max_gap, max_width)
        if run[1] >= x0 and run[0] <= x1
    ]
    if not runs:
        return None
    run = min(runs, key=lambda item: abs((item[0] + item[1]) * 0.5 - centre_x))
    return float(run[0]), float(run[1])


def locate_card_strip(
    roi: np.ndarray, valid: Optional[np.ndarray] = None, cfg: Optional[dict] = None
) -> Optional[dict]:
    """Fit the card's two long colour edges and its top/bottom endpoints.

    The returned points are in rectified-ROI pixel coordinates.  Depth is not
    accepted here on purpose: depth edges on a thin dark/blue card are dominated
    by mixed foreground/background pixels.
    """

    if roi is None or roi.size == 0:
        return None
    height, width = roi.shape[:2]
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    dark = (value < float(_cfg_value(cfg, "dark_v_max"))) & (
        saturation < float(_cfg_value(cfg, "dark_s_max"))
    )
    mask = (dark & valid).astype(np.uint8)

    slot_x = _cfg_value(cfg, "slot_x")
    slot_y = _cfg_value(cfg, "slot_y")
    x_min, x_max = int(float(slot_x[0]) * width), int(float(slot_x[1]) * width)
    y_min, y_max = int(float(slot_y[0]) * height), int(float(slot_y[1]) * height)
    region = np.zeros_like(mask)
    region[y_min:y_max, x_min:x_max] = 1
    mask *= region
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 17), np.uint8)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    max_width = int(_cfg_value(cfg, "max_width_px"))
    min_width = int(_cfg_value(cfg, "min_width_px"))
    max_gap = int(_cfg_value(cfg, "max_row_gap"))
    rows = []
    for y in range(y_min, y_max):
        runs = [
            run
            for run in _merge_runs(_row_runs(mask[y], max(6, min_width // 2)), max_gap, max_width)
            if min_width <= run[1] - run[0] + 1 <= max_width
            and 0.30 * width <= (run[0] + run[1]) * 0.5 <= 0.70 * width
        ]
        if not runs:
            continue
        runs.sort(
            key=lambda item: (
                -(item[1] - item[0]),
                abs((item[0] + item[1]) * 0.5 - 0.5 * width),
            )
        )
        start, end = runs[0]
        rows.append((float(y), float(start), float(end)))
    if len(rows) < int(_cfg_value(cfg, "min_visible_rows")):
        return None

    array = np.asarray(rows, dtype=np.float64)
    ys, left, right = array[:, 0], array[:, 1], array[:, 2]
    widths = right - left
    median_width = float(np.median(widths))
    median_centre = float(np.median((left + right) * 0.5))
    keep = (
        (widths >= max(min_width, 0.40 * median_width))
        & (widths <= 1.55 * median_width)
        & (np.abs((left + right) * 0.5 - median_centre) < 0.14 * width)
    )
    array, ys, left, right = array[keep], ys[keep], left[keep], right[keep]
    if len(ys) < int(_cfg_value(cfg, "min_visible_rows")):
        return None
    widths = right - left
    median_width = float(np.median(widths))
    median_centre = float(np.median((left + right) * 0.5))

    good = np.flatnonzero(widths >= 0.60 * median_width)
    if len(good) < 10:
        return None
    top_y = float(ys[int(good[0])])

    # Long edges are most reliable above the lower recess shadow.  A 55% window
    # still spans enough rows for a stable line fit but avoids the cast shadow
    # that can merge with the bottom of a black card.
    cut_y = top_y + 0.55 * (ys[-1] - top_y)
    fit_mask = ys <= cut_y
    if fit_mask.sum() < 20:
        fit_mask = np.ones(len(ys), dtype=bool)
    left_fit = _fit_line(ys[fit_mask], left[fit_mask])
    right_fit = _fit_line(ys[fit_mask], right[fit_mask])
    if left_fit is None or right_fit is None:
        return None

    left_refined = _refine_x(gray, ys[fit_mask], left[fit_mask])
    right_refined = _refine_x(gray, ys[fit_mask], right[fit_mask])
    left_refined = np.where(
        np.abs(left_refined - left[fit_mask]) <= 6, left_refined, left[fit_mask]
    )
    right_refined = np.where(
        np.abs(right_refined - right[fit_mask]) <= 6, right_refined, right[fit_mask]
    )
    left_fit = _fit_line(ys[fit_mask], left_refined)
    right_fit = _fit_line(ys[fit_mask], right_refined)
    if left_fit is None or right_fit is None:
        return None
    left_slope, left_intercept = left_fit
    right_slope, right_intercept = right_fit

    left_top = left_slope * top_y + left_intercept
    right_top = right_slope * top_y + right_intercept
    top_y = _refine_y(gray, top_y, left_top, right_top)
    left_top = left_slope * top_y + left_intercept
    right_top = right_slope * top_y + right_intercept
    top_centre_x = 0.5 * (left_top + right_top)

    # Bottom from a narrow centre-band occupancy profile.  Unlike a full-row
    # run, this does not merge the card with the darker recess shadow on the
    # left and therefore remains valid for a black card with a bright grip.
    half_width = max(12.0, 0.62 * median_width)
    occupancy = []
    for y in range(int(round(top_y)), y_max):
        x0 = max(0, int(round(top_centre_x - half_width)))
        x1 = min(width, int(round(top_centre_x + half_width)))
        occupancy.append((float(y), float(mask[y, x0:x1].mean())))
    if not occupancy:
        return None
    occupancy = np.asarray(occupancy, dtype=np.float64)
    smoothed = np.convolve(occupancy[:, 1], np.ones(5) / 5.0, mode="same")
    active = np.flatnonzero(smoothed >= 0.22)
    if len(active) == 0:
        return None
    gaps = np.flatnonzero(np.diff(active) > 12)
    if len(gaps):
        starts = np.r_[0, gaps + 1]
        ends = np.r_[gaps, len(active) - 1]
        segments = [(int(starts[i]), int(ends[i])) for i in range(len(starts))]
        best = max(
            segments,
            key=lambda item: float(occupancy[active[item[0] : item[1] + 1], 1].sum()),
        )
        bottom_index = int(active[best[1]])
    else:
        bottom_index = int(active[-1])
    bottom_y = float(occupancy[bottom_index, 0])

    left_bottom = left_slope * bottom_y + left_intercept
    right_bottom = right_slope * bottom_y + right_intercept
    bottom_y = _refine_y(gray, bottom_y, left_bottom, right_bottom)

    clipped = False
    valid_columns = valid[:, int(round(np.clip(top_centre_x, 0, width - 1)))]
    valid_rows = np.flatnonzero(valid_columns)
    if len(valid_rows) and bottom_y >= valid_rows[-1] - 2:
        bottom_y = float(min(bottom_y, valid_rows[-1]))
        clipped = True

    if bottom_y - top_y < 20:
        return None

    # The right long edge is the high-contrast edge against the white recess;
    # the left edge can merge with the cast shadow.  Its slope is therefore the
    # most reliable centreline direction.  Keep the observed bottom row and put
    # the bottom endpoint on that centreline, so a shadow cannot pull the bottom
    # centre sideways.
    centreline_slope = right_slope
    axis_length = math.hypot(1.0, centreline_slope)
    axis = np.array([centreline_slope / axis_length, 1.0 / axis_length])
    visible_length = float(bottom_y - top_y)
    top = (float(top_centre_x), float(top_y))
    bottom = (
        float(top_centre_x + centreline_slope * visible_length),
        float(bottom_y),
    )
    visible_center = (
        float(top[0] + 0.5 * (bottom[0] - top[0])),
        float(top[1] + 0.5 * visible_length),
    )
    half_width = 0.5 * median_width
    left_bottom = (float(bottom[0] - half_width), float(bottom_y))
    right_bottom = (float(bottom[0] + half_width), float(bottom_y))
    left_top = (float(top[0] - half_width), float(top_y))
    right_top = (float(top[0] + half_width), float(top_y))

    # Confidence uses row coverage, edge contrast and observed width.  It is a
    # quality gate, not a calibrated probability.
    coverage = len(ys) / max(1.0, float(ys[-1] - ys[0] + 1))
    left_contrasts = []
    right_contrasts = []
    sample_bottom = min(bottom_y - 8.0, ys[-1] - 2.0)
    for y in np.linspace(top_y + 8.0, max(top_y + 9.0, sample_bottom), 15):
        yy = int(round(y))
        xl = int(round(left_slope * yy + left_intercept))
        xr = int(round(right_slope * yy + right_intercept))
        if xl - 8 < 0 or xr + 8 >= width or xr - xl < 8:
            continue
        inside = float(np.median(gray[yy, xl + 2 : xr - 1]))
        left_out = float(np.median(gray[yy, xl - 6 : xl - 1]))
        right_out = float(np.median(gray[yy, xr + 2 : xr + 7]))
        left_contrasts.append(abs(left_out - inside))
        right_contrasts.append(abs(right_out - inside))
    contrasts = [max(a, b) for a, b in zip(left_contrasts, right_contrasts)]
    contrast = float(np.median(contrasts)) if contrasts else 0.0
    left_contrast = (
        float(np.median(left_contrasts)) if left_contrasts else 0.0
    )
    right_contrast = (
        float(np.median(right_contrasts)) if right_contrasts else 0.0
    )
    confidence = float(
        np.clip(
            0.40 * coverage
            + 0.35 * np.clip(contrast / 55.0, 0.0, 1.0)
            + 0.25 * np.clip(median_width / 85.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    bbox = (
        float(min(left_top[0], left_bottom[0])),
        float(top_y),
        float(max(right_top[0], right_bottom[0])),
        float(bottom_y),
    )

    # Fraction of the fitted card polygon that is actually low-saturation
    # black.  This separates the black card body from the empty slot's blue
    # backplane/connector and from the recess shadow around it.
    polygon = np.array(
        [left_top, right_top, right_bottom, left_bottom], dtype=np.float32
    )
    interior = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(interior, np.rint(polygon).astype(np.int32), 255)
    interior = (interior > 0) & valid
    interior_count = int(interior.sum())
    black_fraction = (
        float(np.count_nonzero(dark & interior)) / interior_count
        if interior_count
        else 0.0
    )

    return {
        "top_px": top,
        "bottom_px": bottom,
        "visible_center_px": visible_center,
        "left_top_px": left_top,
        "right_top_px": right_top,
        "left_bottom_px": left_bottom,
        "right_bottom_px": right_bottom,
        "axis_px": (float(axis[0]), float(axis[1])),
        "width_px": float(median_width),
        "visible_length_px": visible_length,
        "black_fraction": black_fraction,
        "clipped": bool(clipped),
        "confidence": confidence,
        "coverage": float(coverage),
        "contrast": contrast,
        "left_contrast": left_contrast,
        "right_contrast": right_contrast,
        "bbox": bbox,
        "rows": int(len(ys)),
        "source": "tag_rectified_color_edges",
    }


def _box_geometry(box: np.ndarray) -> dict:
    """Return centre/axis/width/length and ordered corners for a box."""

    box = np.asarray(box, dtype=np.float64).reshape(4, 2)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_index = int(np.argmax(lengths))
    axis = edges[long_index] / max(lengths[long_index], 1e-9)
    if axis[1] < 0:
        axis = -axis
    length = float(max(lengths))
    width = float(min(lengths))
    if width > length:
        width, length = length, width
        axis = np.array([-axis[1], axis[0]], dtype=np.float64)
        if axis[1] < 0:
            axis = -axis
    centre = box.mean(axis=0)
    # Right-handed perpendicular with +y down: for a downward card axis this
    # points to image-right, so the four corners read TL, TR, BR, BL.
    perp = np.array([axis[1], -axis[0]], dtype=np.float64)
    half_width = 0.5 * width
    top = centre - axis * (0.5 * length)
    bottom = centre + axis * (0.5 * length)
    quad = np.array(
        [
            top - perp * half_width,
            top + perp * half_width,
            bottom + perp * half_width,
            bottom - perp * half_width,
        ],
        dtype=np.float64,
    )
    return {
        "center": centre,
        "axis": axis,
        "width": width,
        "length": length,
        "quad": quad,
        "top": quad[0] + (quad[1] - quad[0]) * 0.5,
        "bottom": quad[3] + (quad[2] - quad[3]) * 0.5,
    }


def _protrusion_masks(
    depth_m: Optional[np.ndarray],
    camera_matrix: Optional[np.ndarray],
    panel: Optional[dict],
    context: dict,
    cfg: Optional[dict] = None,
):
    """Warp a card-front protrusion mask and a depth-validity mask to the ROI."""

    if depth_m is None or camera_matrix is None or panel is None:
        return None, None
    height, width = depth_m.shape[:2]
    ys, xs = np.indices((height, width))
    z = np.asarray(depth_m, dtype=np.float64)
    valid = (
        np.isfinite(z)
        & (z > float(_cfg_value(cfg, "depth_min_m")))
        & (z < float(_cfg_value(cfg, "depth_max_m")))
    )
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    normal = np.asarray(panel["normal"], dtype=np.float64)
    residual = normal[0] * x + normal[1] * y + normal[2] * z - float(panel["d"])
    threshold = float(_cfg_value(cfg, "card_protrusion_min_m"))
    protrusion = (valid & (residual < -threshold)).astype(np.uint8) * 255
    validity = valid.astype(np.uint8) * 255
    shape = (context["roi"].shape[1], context["roi"].shape[0])
    transform = context["image_to_roi"]
    protrusion_roi = cv2.warpPerspective(
        protrusion, transform, shape, flags=cv2.INTER_NEAREST
    ) > 0
    validity_roi = cv2.warpPerspective(
        validity, transform, shape, flags=cv2.INTER_NEAREST
    ) > 0
    return protrusion_roi, validity_roi


def locate_card_outer(
    roi: np.ndarray,
    valid: Optional[np.ndarray] = None,
    context: Optional[dict] = None,
    cfg: Optional[dict] = None,
    depth_m: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    panel: Optional[dict] = None,
) -> Optional[dict]:
    """Measure the full outer silhouette of the inserted black card.

    The black body is the darkest, least-saturated object in the slot.  A white
    label can split the body into several dark components, so components that
    are vertically aligned with and close to the main body are merged before
    the rotated rectangle is fitted.  The white recess frame is much brighter
    and is not merged; the blue backplane is more saturated and is rejected by
    the saturation gate.

    When depth and a panel plane are available, a component is accepted only
    when its valid depth pixels are in front of the panel.  This rejects the
    dark gap between the card and the recess wall, which has the same colour as
    the card but lies on the panel.
    """

    if roi is None or roi.size == 0:
        return None
    height, width = roi.shape[:2]
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)

    protrusion = validity = None
    if context is not None:
        protrusion, validity = _protrusion_masks(
            depth_m, camera_matrix, panel, context, cfg
        )
        if (
            protrusion is not None
            and int(np.count_nonzero(protrusion & valid))
            < int(_cfg_value(cfg, "card_min_protrusion_pixels"))
        ):
            # Depth is unusable/aligned to a completely flat scene.  Fall back
            # to the colour silhouette rather than rejecting every card.
            protrusion = validity = None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    dark = (
        (value < float(_cfg_value(cfg, "card_dark_v_max")))
        & (saturation < float(_cfg_value(cfg, "card_dark_s_max")))
        & valid
    ).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return None

    min_width = int(_cfg_value(cfg, "card_min_width_px"))
    max_width = int(_cfg_value(cfg, "card_max_width_px"))
    candidate_max_width = int(
        _cfg_value(cfg, "card_candidate_max_width_px")
    )
    min_length = int(_cfg_value(cfg, "card_min_length_px"))
    min_area = int(_cfg_value(cfg, "card_min_area_px"))
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x, y, component_width, component_height = [
            int(v) for v in stats[index, :4]
        ]
        if area < min_area:
            continue
        if not (min_width <= component_width <= candidate_max_width):
            continue
        if component_height < min_length:
            continue
        centre_x = x + 0.5 * component_width
        if not (0.15 * width <= centre_x <= 0.85 * width):
            continue
        near_fraction = None
        if protrusion is not None and validity is not None:
            component = labels == index
            valid_count = int(np.count_nonzero(validity & component))
            if valid_count <= 0:
                near_fraction = 0.0
            else:
                near_fraction = float(
                    np.count_nonzero(protrusion & component)
                ) / float(valid_count)
            if near_fraction < float(
                _cfg_value(cfg, "card_component_min_protrusion_fraction")
            ):
                continue
        candidates.append(
            (
                area,
                index,
                (x, y, component_width, component_height),
                near_fraction,
            )
        )
    if not candidates:
        return None

    _, best_index, base_bbox, _ = max(candidates)
    base_x, base_y, base_w, base_h = base_bbox
    base_component = (labels == best_index).astype(np.uint8)
    if protrusion is not None and validity is not None:
        ys, xs = np.nonzero(protrusion & (base_component > 0))
        if len(xs) >= int(_cfg_value(cfg, "card_min_protrusion_pixels")):
            # The dark component can merge the card with the left recess wall
            # during a fast pan.  Trim it to the measured card-front
            # protrusion, with a small margin for the black edge/holes.
            margin = int(_cfg_value(cfg, "card_trim_margin_px"))
            x0 = max(0, int(np.floor(np.percentile(xs, 1))) - margin)
            x1 = min(width, int(np.ceil(np.percentile(xs, 99))) + margin)
            # Trim only the horizontal extent.  The card's top/bottom edges
            # often have no valid depth (black/dark), so trimming vertically
            # would cut the real card ends and bias the centre downward.
            y0 = max(0, base_y)
            y1 = min(height, base_y + base_h)
            trim = np.zeros_like(base_component)
            trim[y0:y1, x0:x1] = 1
            trimmed = base_component & trim
            if int(trimmed.sum()) >= max(min_area // 2, 1000):
                base_component = trimmed
                rows = np.flatnonzero(base_component.any(axis=1))
                cols = np.flatnonzero(base_component.any(axis=0))
                if len(rows) and len(cols):
                    base_y, base_h = int(rows[0]), int(rows[-1] - rows[0] + 1)
                    base_x, base_w = int(cols[0]), int(cols[-1] - cols[0] + 1)
                    base_bbox = (base_x, base_y, base_w, base_h)
    base_area = float(
        sum(
            cv2.contourArea(contour)
            for contour in cv2.findContours(
                base_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
        )
    )
    if base_area / max(float(base_w * base_h), 1.0) < float(
        _cfg_value(cfg, "card_min_fill_ratio")
    ):
        return None

    merge_gap = float(_cfg_value(cfg, "card_merge_gap_px"))
    merge_min_area = int(_cfg_value(cfg, "card_merge_min_area_px"))
    merge_min_height = int(_cfg_value(cfg, "card_merge_min_height_px"))
    merge_min_overlap = float(_cfg_value(cfg, "card_merge_min_overlap"))
    union = base_component.copy()
    if protrusion is not None and validity is not None:
        for index in range(1, count):
            if index == best_index:
                continue
            area = int(stats[index, cv2.CC_STAT_AREA])
            x, y, component_width, component_height = [
                int(v) for v in stats[index, :4]
            ]
            if area < merge_min_area or component_height < merge_min_height:
                continue
            overlap = max(
                0,
                min(base_y + base_h, y + component_height) - max(base_y, y),
            )
            if overlap < merge_min_overlap * min(base_h, component_height):
                continue
            gap = max(
                0, max(x - (base_x + base_w), base_x - (x + component_width))
            )
            if gap > merge_gap:
                continue
            fragment_centre_x = x + 0.5 * component_width
            if not (
                base_x - 0.25 * base_w
                <= fragment_centre_x
                <= base_x + 1.25 * base_w
            ):
                continue
            component = labels == index
            valid_count = int(np.count_nonzero(validity & component))
            if valid_count <= 0:
                continue
            near_fraction = float(
                np.count_nonzero(protrusion & component)
            ) / float(valid_count)
            if near_fraction < 0.50:
                continue
            union[component] = 1

    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(
        union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    if not contours:
        return None
    points = np.vstack([contour.reshape(-1, 2) for contour in contours])
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    geometry = _box_geometry(box)
    if (
        geometry["width"] < min_width
        or geometry["width"] > max_width
        or geometry["length"] < min_length
        or geometry["length"] < 1.8 * geometry["width"]
    ):
        return None

    quad = geometry["quad"].astype(np.float32)
    interior = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(interior, np.rint(quad).astype(np.int32), 255)
    interior = (interior > 0) & valid
    interior_count = int(interior.sum())
    if interior_count <= 0:
        return None
    black_fraction = (
        float(np.count_nonzero(dark & interior)) / float(interior_count)
    )
    silhouette_area = float(sum(cv2.contourArea(c) for c in contours))
    fill_ratio = silhouette_area / max(
        geometry["width"] * geometry["length"], 1e-9
    )
    if fill_ratio < float(_cfg_value(cfg, "card_min_fill_ratio")):
        return None
    if black_fraction < float(_cfg_value(cfg, "card_outer_min_black_fraction")):
        return None

    confidence = float(
        np.clip(
            0.45 * np.clip(fill_ratio / 0.70, 0.0, 1.0)
            + 0.35 * np.clip(black_fraction / 0.75, 0.0, 1.0)
            + 0.20 * np.clip(geometry["length"] / 420.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    if confidence < float(_cfg_value(cfg, "card_outer_min_confidence")):
        return None

    centre_x = float(geometry["center"][0])
    valid_rows = np.flatnonzero(valid[:, int(np.clip(round(centre_x), 0, width - 1))])
    clipped = False
    if len(valid_rows):
        top_y = float(geometry["quad"][:, 1].min())
        bottom_y = float(geometry["quad"][:, 1].max())
        if top_y <= float(valid_rows[0]) + 1.5:
            clipped = True
        if bottom_y >= float(valid_rows[-1]) - 1.5:
            clipped = True

    return {
        "center_px": tuple(float(v) for v in geometry["center"]),
        "axis_px": tuple(float(v) for v in geometry["axis"]),
        "top_px": tuple(float(v) for v in geometry["top"]),
        "bottom_px": tuple(float(v) for v in geometry["bottom"]),
        "width_px": float(geometry["width"]),
        "length_px": float(geometry["length"]),
        "visible_length_px": float(geometry["length"]),
        "quad_px": [tuple(float(v) for v in point) for point in geometry["quad"]],
        "left_top_px": tuple(float(v) for v in geometry["quad"][0]),
        "right_top_px": tuple(float(v) for v in geometry["quad"][1]),
        "right_bottom_px": tuple(float(v) for v in geometry["quad"][2]),
        "left_bottom_px": tuple(float(v) for v in geometry["quad"][3]),
        "bbox": (
            float(geometry["quad"][:, 0].min()),
            float(geometry["quad"][:, 1].min()),
            float(geometry["quad"][:, 0].max()),
            float(geometry["quad"][:, 1].max()),
        ),
        "area_px": int(silhouette_area),
        "black_fraction": black_fraction,
        "fill_ratio": float(fill_ratio),
        "clipped": bool(clipped),
        "confidence": confidence,
        "rows": int(geometry["length"]),
        "source": "tag_rectified_outer_silhouette",
    }



def _measure_slot_anchor(
    roi: np.ndarray,
    valid: Optional[np.ndarray],
    context: dict,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Find the empty slot centre from its blue backplane/connector."""

    if roi is None or roi.size == 0:
        return None
    height, width = roi.shape[:2]
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    h_min, h_max = (int(v) for v in _cfg_value(cfg, "slot_blue_h"))
    mask = (
        (hue >= h_min)
        & (hue <= h_max)
        & (saturation > int(_cfg_value(cfg, "slot_blue_s_min")))
        & (value > int(_cfg_value(cfg, "slot_blue_v_min")))
        & valid
    ).astype(np.uint8)
    # Restrict to the broad central slot band.  The rectified ROI also contains
    # blue panel furniture (buttons, cables); without this gate the largest
    # blue blob can be far away from the slot when the slot is partially out of
    # view.
    slot_x = _cfg_value(cfg, "slot_x")
    slot_y = _cfg_value(cfg, "slot_y")
    x_min = int(float(slot_x[0]) * width)
    x_max = int(float(slot_x[1]) * width)
    y_min = int(float(slot_y[0]) * height)
    y_max = int(float(slot_y[1]) * height)
    region = np.zeros_like(mask)
    region[y_min:y_max, x_min:x_max] = 1
    mask *= region
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    best = max(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
    )
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < int(_cfg_value(cfg, "slot_min_blue_area")):
        return None
    component_width = int(stats[best, cv2.CC_STAT_WIDTH])
    component_height = int(stats[best, cv2.CC_STAT_HEIGHT])
    if component_height < int(_cfg_value(cfg, "slot_min_blue_height")):
        return None
    if component_width > int(_cfg_value(cfg, "slot_max_blue_width")):
        return None
    centre = centroids[best]
    if not (x_min - 8 <= float(centre[0]) <= x_max + 8):
        return None
    anchor_roi = (float(centre[0]), float(centre[1]))
    anchor_tag = _roi_point_to_tag(anchor_roi, context)
    return {
        "roi": anchor_roi,
        "tag": anchor_tag,
        "area": area,
    }


def _plane_from_points(points: np.ndarray, cfg: Optional[dict] = None) -> Optional[dict]:
    """RANSAC plane in camera coordinates; n.dot(X) = d."""

    if points is None or len(points) < 12:
        return None
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 12:
        return None
    threshold = float(_cfg_value(cfg, "plane_inlier_m"))
    iterations = int(_cfg_value(cfg, "plane_iterations"))
    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        indices = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        distance = float(normal @ p0)
        residual = np.abs(points @ normal - distance)
        inliers = residual < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 12:
        return None
    selected = points[best_inliers]
    centre = selected.mean(axis=0)
    _, _, vectors = np.linalg.svd(selected - centre, full_matrices=False)
    normal = vectors[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal /= norm
    distance = float(normal @ centre)
    residual = np.abs(points @ normal - distance)
    inliers = residual < 1.5 * threshold
    if inliers.sum() >= 12:
        selected = points[inliers]
        centre = selected.mean(axis=0)
        _, _, vectors = np.linalg.svd(selected - centre, full_matrices=False)
        normal = vectors[-1]
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        distance = float(normal @ centre)
        residual = np.abs(points @ normal - distance)
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2))) if inliers.any() else None
    return {
        "normal": normal,
        "d": distance,
        "inliers": int(inliers.sum()),
        "rms": rms,
    }


def _depth_plane(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    polygon: np.ndarray,
    cfg: Optional[dict] = None,
    expected_z: Optional[float] = None,
) -> Optional[dict]:
    if depth_m is None or camera_matrix is None:
        return None
    height, width = depth_m.shape[:2]
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    centre = points.mean(axis=0)
    shrunk = centre + (points - centre) * 0.60
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(shrunk).astype(np.int32), 255)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    stride = max(1, int(math.sqrt(len(xs) / 900.0)))
    xs, ys = xs[::stride], ys[::stride]
    z = depth_m[ys, xs]
    valid = (
        np.isfinite(z)
        & (z > float(_cfg_value(cfg, "depth_min_m")))
        & (z < float(_cfg_value(cfg, "depth_max_m")))
    )
    if expected_z is not None:
        valid &= np.abs(z - float(expected_z)) <= float(
            _cfg_value(cfg, "plane_depth_gate_m")
        )
    if valid.sum() < int(_cfg_value(cfg, "plane_min_points")):
        return None
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    z = z[valid]
    x = (xs[valid].astype(np.float64) - cx) * z / fx
    y = (ys[valid].astype(np.float64) - cy) * z / fy
    points_3d = np.column_stack([x, y, z])
    return _plane_from_points(points_3d, cfg)


def _card_surface_depth(
    depth_m: Optional[np.ndarray],
    camera_matrix: Optional[np.ndarray],
    quad: np.ndarray,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Measure the card-front depth from pixels inside the outer quad."""

    if depth_m is None or camera_matrix is None:
        return None
    height, width = depth_m.shape[:2]
    polygon = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < int(_cfg_value(cfg, "card_depth_min_points")):
        return None
    z = np.asarray(depth_m[ys, xs], dtype=np.float64)
    valid = (
        np.isfinite(z)
        & (z > float(_cfg_value(cfg, "depth_min_m")))
        & (z < float(_cfg_value(cfg, "depth_max_m")))
    )
    if int(valid.sum()) < int(_cfg_value(cfg, "card_depth_min_points")):
        return None
    z_valid = z[valid]
    median_z = float(np.median(z_valid))
    plane = _depth_plane(
        depth_m,
        camera_matrix,
        polygon,
        {
            **dict(cfg or {}),
            "plane_min_points": int(_cfg_value(cfg, "card_depth_min_points")),
            "plane_inlier_m": float(_cfg_value(cfg, "card_plane_inlier_m")),
            "plane_depth_gate_m": float(_cfg_value(cfg, "card_depth_gate_m")),
        },
        expected_z=median_z,
    )
    if plane is not None:
        rms = plane.get("rms")
        centre = polygon.mean(axis=0)
        centre_xyz = _ray_plane(centre, camera_matrix, plane)
        if (
            (rms is None or float(rms) <= float(_cfg_value(cfg, "card_depth_max_rms_m")))
            and int(plane["inliers"]) >= int(_cfg_value(cfg, "card_plane_min_inliers"))
            and centre_xyz is not None
            and abs(float(centre_xyz[2]) - median_z)
            <= float(_cfg_value(cfg, "card_depth_gate_m"))
        ):
            return {
                "plane": plane,
                "median_z": median_z,
                "source": "card_plane",
            }
    return {
        "plane": None,
        "median_z": median_z,
        "source": "card_depth_z",
    }



def _panel_plane(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    marker: Any,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Fit the panel plane around T07, not the black card surface."""

    corners = _tag_corners(marker)
    if corners is None:
        return None
    centre = np.mean(corners, axis=0)
    side = float(
        np.mean([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    )
    if side < 4.0:
        return None
    radius = float(_cfg_value(cfg, "panel_plane_radius_tag"))
    polygon = centre + (corners - centre) * (2.0 * radius)
    marker_centre = _tag_field(marker, "center_xy")
    expected_z = None
    if marker_centre is not None:
        point = _depth_point(marker_centre, depth_m, camera_matrix, radius=15)
        if point is not None:
            expected_z = float(point[2])
    z_min, z_max = (float(v) for v in _cfg_value(cfg, "panel_z_range_m"))
    if expected_z is None or not (z_min <= expected_z <= z_max):
        return None
    plane = _depth_plane(
        depth_m,
        camera_matrix,
        polygon,
        {
            **dict(cfg or {}),
            "plane_min_points": int(_cfg_value(cfg, "panel_plane_min_points")),
        },
        expected_z=expected_z,
    )
    if plane is None:
        return None
    if int(plane["inliers"]) < int(_cfg_value(cfg, "panel_plane_min_inliers")):
        return None
    if float(plane["d"]) < 0.0:
        plane["normal"] = -np.asarray(plane["normal"], dtype=np.float64)
        plane["d"] = -float(plane["d"])
    # A tilted/background plane can still pass RANSAC near a hand.  Reject it
    # when the tag-centre ray does not agree with the measured tag depth.
    if marker_centre is not None and expected_z is not None:
        check = _ray_plane(marker_centre, camera_matrix, plane)
        if (
            check is None
            or abs(float(check[2]) - expected_z)
            > float(_cfg_value(cfg, "plane_depth_gate_m"))
        ):
            return None
    plane["expected_z"] = expected_z
    return plane


def _ray_plane(
    point: Sequence[float],
    camera_matrix: np.ndarray,
    plane: Optional[dict],
) -> Optional[np.ndarray]:
    if plane is None or camera_matrix is None:
        return None
    u, v = float(point[0]), float(point[1])
    ray = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0], dtype=np.float64)
    normal = np.asarray(plane["normal"], dtype=np.float64)
    distance = float(plane["d"])
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-9:
        return None
    scale = distance / denominator
    if scale <= 0:
        return None
    xyz = ray * scale
    if not np.isfinite(xyz).all() or not (
        float(_cfg_value(None, "depth_min_m"))
        <= xyz[2]
        <= float(_cfg_value(None, "depth_max_m"))
    ):
        return None
    return xyz


def _depth_point(
    point: Sequence[float],
    depth_m: Optional[np.ndarray],
    camera_matrix: Optional[np.ndarray],
    radius: int = 7,
) -> Optional[np.ndarray]:
    if depth_m is None or camera_matrix is None:
        return None
    u, v = int(round(float(point[0]))), int(round(float(point[1])))
    height, width = depth_m.shape[:2]
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 2.0)]
    if values.size < 6:
        return None
    z = float(np.median(values))
    if z <= 0:
        return None
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array(
        [(float(point[0]) - cx) * z / fx, (float(point[1]) - cy) * z / fy, z],
        dtype=np.float64,
    )


def _point_at_z(
    point: Sequence[float], camera_matrix: np.ndarray, z: float
) -> Optional[np.ndarray]:
    if camera_matrix is None or z is None or z <= 0:
        return None
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array(
        [
            (float(point[0]) - cx) * float(z) / fx,
            (float(point[1]) - cy) * float(z) / fy,
            float(z),
        ],
        dtype=np.float64,
    )


def _to_image(point: Sequence[float], roi_to_image: np.ndarray) -> Tuple[float, float]:
    transformed = cv2.perspectiveTransform(
        np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32),
        roi_to_image,
    )[0, 0]
    return float(transformed[0]), float(transformed[1])


def _tag_roi_point(
    tag_point: Sequence[float], context: dict, cfg: Optional[dict] = None
) -> Tuple[float, float]:
    """Map tag-side coordinates to rectified-ROI pixels."""

    roi_x = _cfg_value(cfg, "roi_x")
    roi_y = _cfg_value(cfg, "roi_y")
    width, height = _cfg_value(cfg, "roi_size")
    x0 = float(roi_x[0]) + 0.5
    y0 = float(roi_y[0]) + 0.5
    sx = float(width) / max(float(roi_x[1]) - float(roi_x[0]), 1e-6)
    sy = float(height) / max(float(roi_y[1]) - float(roi_y[0]), 1e-6)
    return (
        (float(tag_point[0]) + 0.5 - x0) * sx,
        (float(tag_point[1]) + 0.5 - y0) * sy,
    )


def _roi_point_to_tag(
    point: Sequence[float], context: dict
) -> Optional[Tuple[float, float]]:
    """Map rectified-ROI pixels back to tag-side coordinates."""

    transform = context.get("transform")
    if transform is None:
        return None
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64))
    mapped = inverse @ np.array(
        [float(point[0]), float(point[1]), 1.0], dtype=np.float64
    )
    if abs(float(mapped[2])) < 1e-9:
        return None
    return (
        float(mapped[0] / mapped[2] - 0.5),
        float(mapped[1] / mapped[2] - 0.5),
    )


def _tag_polygon_image(
    top_tag: Sequence[float],
    axis_tag: Sequence[float],
    width_tag: float,
    length_tag: float,
    context: dict,
    cfg: Optional[dict] = None,
) -> Optional[list]:
    """Project a tag-relative card quad into the source image."""

    axis = np.asarray(axis_tag, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    axis /= norm
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    half = 0.5 * float(width_tag)
    top = np.asarray(top_tag, dtype=np.float64)
    bottom = top + axis * float(length_tag)
    corners = [
        top - perp * half,
        top + perp * half,
        bottom + perp * half,
        bottom - perp * half,
    ]
    points = []
    for corner in corners:
        roi = _tag_roi_point(corner, context, cfg)
        points.append(_to_image(roi, context["roi_to_image"]))
    return points


def _observation_to_image(observation: dict, context: dict, cfg: Optional[dict] = None) -> dict:
    roi_to_image = context["roi_to_image"]
    transform = context.get("transform")
    if transform is not None:
        roi_to_tag = np.linalg.inv(transform)
        for name in (
            "top_px",
            "bottom_px",
            "visible_center_px",
            "center_px",
            "anchor_px",
            "slot_px",
        ):
            value = observation.get(name)
            if value is None:
                continue
            canonical = roi_to_tag @ np.array(
                [float(value[0]), float(value[1]), 1.0], dtype=np.float64
            )
            if abs(canonical[2]) < 1e-9:
                continue
            observation[name[:-3] + "_tag"] = (
                float(canonical[0] / canonical[2] - 0.5),
                float(canonical[1] / canonical[2] - 0.5),
            )
    for name in (
        "top_px",
        "bottom_px",
        "visible_center_px",
        "center_px",
        "left_top_px",
        "right_top_px",
        "left_bottom_px",
        "right_bottom_px",
        "anchor_px",
        "slot_px",
    ):
        if observation.get(name) is not None:
            observation[name] = _to_image(observation[name], roi_to_image)
    if observation.get("bbox") is not None:
        points = np.array(
            [
                [observation["bbox"][0], observation["bbox"][1]],
                [observation["bbox"][2], observation["bbox"][1]],
                [observation["bbox"][2], observation["bbox"][3]],
                [observation["bbox"][0], observation["bbox"][3]],
            ],
            dtype=np.float32,
        )
        transformed = cv2.perspectiveTransform(
            points.reshape(1, -1, 2), roi_to_image
        )[0]
        observation["bbox"] = (
            float(transformed[:, 0].min()),
            float(transformed[:, 1].min()),
            float(transformed[:, 0].max()),
            float(transformed[:, 1].max()),
        )
    if observation.get("quad_px") is not None:
        observation["quad_px"] = [
            _to_image(point, roi_to_image) for point in observation["quad_px"]
        ]
    return observation


class BoardCardLocalizer:
    """Stateful per-tag localizer with nominal-length and anchor smoothing."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(DEFAULT_CONFIG)
        if cfg:
            self.cfg.update(cfg)
        self._frame_index = 0
        self._tag_cache: Dict[int, tuple] = {}
        self._state: Dict[int, dict] = {}

    def _cached_marker(self, tag_id: int, frame_index: int):
        cached = self._tag_cache.get(int(tag_id))
        if cached is None:
            return None
        marker, last_frame = cached
        if frame_index - last_frame > int(self.cfg["tag_cache_frames"]):
            return None
        return marker

    def _remember_marker(self, tag_id: int, marker: Any) -> None:
        self._tag_cache[int(tag_id)] = (marker, self._frame_index)

    def _effective_panel_plane(
        self,
        state: dict,
        depth_m: Optional[np.ndarray],
        camera_matrix: Optional[np.ndarray],
        marker: Any,
    ) -> Optional[dict]:
        if depth_m is None or camera_matrix is None:
            return None
        panel = _panel_plane(depth_m, camera_matrix, marker, self.cfg)
        if panel is not None:
            state["panel_plane"] = panel
            state["panel_plane_frame"] = self._frame_index
            if panel.get("expected_z") is not None:
                state["last_panel_z"] = float(panel["expected_z"])
            return panel
        if (
            state.get("panel_plane") is not None
            and self._frame_index - int(state.get("panel_plane_frame", -10**9))
            <= int(self.cfg["plane_hold_frames"])
        ):
            return state["panel_plane"]
        return None

    def _state_for(self, tag_id: int) -> dict:
        return self._state.setdefault(
            int(tag_id),
            {
                "candidate_state": None,
                "candidate_count": 0,
                "stable_state": None,
                "anchor_px": None,
                "axis_px": None,
                "last_panel_z": None,
                "missed": 0,
                "pose_samples": [],
                "pose_center_tag": None,
                "pose_axis_tag": None,
                "pose_width_tag": None,
                "pose_length_tag": None,
                "pose_spread_tag": None,
                "pose_locked": False,
                "panel_plane": None,
                "panel_plane_frame": -10**9,
                "last_tag_centre": None,
                "still_count": 0,
                "still_buffer": [],
                "center_lock": None,
                "anchor_lock": None,
            },
        )

    def observe(
        self,
        color: np.ndarray,
        depth: Optional[np.ndarray] = None,
        camera_info: Any = None,
        marker: Any = None,
        tag_id: Optional[int] = None,
    ) -> Optional[dict]:
        self._frame_index += 1
        if tag_id is None:
            tag_id = _tag_id(marker)
        if tag_id is None and marker is not None:
            tag_id = _tag_id(marker)
        if marker is None and tag_id is not None:
            marker = self._cached_marker(int(tag_id), self._frame_index)
        if marker is None or _tag_corners(marker) is None:
            if tag_id is not None:
                state = self._state_for(int(tag_id))
                state["missed"] += 1
            return None
        resolved_tag_id = tag_id if tag_id is not None else _tag_id(marker)
        if resolved_tag_id is None:
            return None
        tag_id = int(resolved_tag_id)
        self._remember_marker(tag_id, marker)

        context = rectify_board_card_roi(color, marker, self.cfg)
        if context is None:
            return None
        strip = locate_card_strip(context["roi"], context["valid"], self.cfg)
        state = self._state_for(tag_id)
        camera_matrix = _intrinsics(camera_info)
        depth_m = _depth_metres(depth, float(self.cfg["depth_scale"]))
        panel = self._effective_panel_plane(
            state, depth_m, camera_matrix, marker
        )
        outer = locate_card_outer(
            context["roi"],
            context["valid"],
            context,
            self.cfg,
            depth_m,
            camera_matrix,
            panel,
        )

        roi_width = context["roi"].shape[1]
        roi_height = context["roi"].shape[0]
        roi_x0, roi_x1 = (float(v) for v in self.cfg["roi_x"])
        roi_y0, roi_y1 = (float(v) for v in self.cfg["roi_y"])
        sx = roi_width / max(roi_x1 - roi_x0, 1e-6)
        sy = roi_height / max(roi_y1 - roi_y0, 1e-6)
        # No fixed card length: report the length actually measured this frame.
        nominal = float(
            outer["length_px"]
            if outer is not None
            else (strip["visible_length_px"] if strip is not None else 0.0)
        )
        outer_length_tag = (
            float(outer["length_px"]) * (roi_y1 - roi_y0) / roi_height
            if outer is not None
            else 0.0
        )
        strip_length_tag = (
            float(strip["visible_length_px"]) * (roi_y1 - roi_y0) / roi_height
            if strip is not None
            else 0.0
        )

        outer_inserted = bool(
            outer is not None
            and outer["confidence"]
            >= float(self.cfg["card_outer_min_confidence"])
            and outer_length_tag >= float(self.cfg["min_inserted_length_tag"])
            and outer["black_fraction"]
            >= float(self.cfg["card_outer_min_black_fraction"])
        )
        strip_inserted = bool(
            strip is not None
            and strip["confidence"] >= float(self.cfg["min_confidence"])
            and strip_length_tag >= float(self.cfg["min_inserted_length_tag"])
            and float(strip.get("black_fraction", 0.0))
            >= float(self.cfg["min_black_fraction"])
        )
        depth_available = bool(
            depth_m is not None and camera_matrix is not None and panel is not None
        )
        if depth_available:
            # With depth, the outer silhouette is authoritative for entering
            # the inserted state.  The colour strip may only *hold* an already
            # confirmed inserted state while the silhouette is temporarily
            # missing (motion blur, tag near the image edge).
            inserted_measurement = bool(
                outer_inserted
                or (
                    state.get("stable_state") == "inserted"
                    and strip_inserted
                )
            )
        else:
            inserted_measurement = bool(outer_inserted or strip_inserted)
        candidate = "inserted" if inserted_measurement else "empty"

        if state["candidate_state"] == candidate:
            state["candidate_count"] = int(state["candidate_count"]) + 1
        else:
            state["candidate_state"] = candidate
            state["candidate_count"] = 1
        confirm_frames = (
            int(self.cfg["inserted_confirm_frames"])
            if candidate == "inserted"
            else int(self.cfg["empty_confirm_frames"])
        )
        if int(state["candidate_count"]) >= max(1, confirm_frames):
            state["stable_state"] = candidate
        stable_state = state.get("stable_state")
        if stable_state == "inserted" and not inserted_measurement:
            # A short/absent measurement is not yet a confirmed empty slot.
            # Do not reuse the previous inserted geometry with this frame's
            # unrelated dark blob; publish unknown until empty confirms.
            stable_state = None
        state_confidence = min(
            1.0, float(state["candidate_count"]) / max(1, confirm_frames)
        )
        state["missed"] = 0

        if stable_state is None:
            observation = {
                "tag_id": tag_id,
                "card_present": False,
                "insertion_state": "unknown",
                "state": "unknown",
                "stable": False,
                "state_confidence": 0.0,
                "state_frames": int(state["candidate_count"]),
                "anchor_source": "none",
                "anchor_px": None,
                "center_px": None,
                "top_px": None,
                "bottom_px": None,
                "visible_center_px": None,
                "left_top_px": None,
                "right_top_px": None,
                "left_bottom_px": None,
                "right_bottom_px": None,
                "bbox": None,
                "quad_px": None,
                "quad_tag": None,
                "geometry_source": "none",
                "axis_px": (0.0, 1.0),
                "width_px": None,
                "visible_length_px": 0.0,
                "nominal_length_px": nominal,
                "length_estimated": False,
                "clipped": False,
                "confidence": 0.0,
                "source": "state_unknown",
                "slot_px": None,
                "position": None,
                "slot_position": None,
                "anchor_position": None,
                "top_position": None,
                "bottom_position": None,
                "plane": None,
                "plane_rms_m": None,
                "depth_source": None,
            }
            return _observation_to_image(observation, context, self.cfg)

        if stable_state == "empty":
            slot_measure = _measure_slot_anchor(
                context["roi"], context["valid"], context, self.cfg
            )
            anchor_roi = (
                slot_measure["roi"] if slot_measure is not None else None
            )
            slot_tag = (
                slot_measure["tag"] if slot_measure is not None else None
            )
            observation = {
                "tag_id": tag_id,
                "card_present": False,
                "insertion_state": "empty",
                "state": "empty",
                "stable": True,
                "state_confidence": state_confidence,
                "state_frames": int(state["candidate_count"]),
                "anchor_source": "slot_blue" if anchor_roi is not None else "none",
                "anchor_px": anchor_roi,
                "center_px": None,
                "top_px": None,
                "bottom_px": None,
                "visible_center_px": None,
                "left_top_px": None,
                "right_top_px": None,
                "left_bottom_px": None,
                "right_bottom_px": None,
                "bbox": None,
                "quad_px": None,
                "quad_tag": None,
                "geometry_source": "none",
                "axis_px": (0.0, 1.0),
                "width_px": None,
                "visible_length_px": 0.0,
                "nominal_length_px": nominal,
                "length_estimated": False,
                "clipped": False,
                "confidence": 0.0,
                "source": "tag_slot_blue" if anchor_roi is not None else "tag_slot_unmeasured",
                "slot_px": anchor_roi,
                "slot_tag": slot_tag,
                "position": None,
                "slot_position": None,
                "anchor_position": None,
                "top_position": None,
                "bottom_position": None,
                "plane": None,
                "plane_rms_m": None,
                "depth_source": None,
            }
            if anchor_roi is not None:
                observation["center_px"] = anchor_roi
            if camera_matrix is not None and anchor_roi is not None:
                anchor_img = _to_image(anchor_roi, context["roi_to_image"])
                slot_xyz = None
                if panel is not None:
                    card_plane = {
                        "normal": np.asarray(panel["normal"], dtype=np.float64),
                        "d": float(panel["d"])
                        - float(self.cfg["panel_depth_offset_m"]),
                    }
                    slot_xyz = _ray_plane(anchor_img, camera_matrix, card_plane)
                    observation["depth_source"] = "panel_plane"
                if slot_xyz is None:
                    expected_z = panel.get("expected_z") if panel else None
                    if expected_z is None:
                        expected_z = state.get("last_panel_z")
                    if expected_z is not None:
                        slot_xyz = _point_at_z(
                            anchor_img, camera_matrix, float(expected_z)
                        )
                        observation["depth_source"] = "tag_z_fallback"
                if slot_xyz is not None:
                    observation["slot_position"] = tuple(
                        float(v) for v in slot_xyz
                    )
                    # Empty state: the published position is the measured slot
                    # centre on the panel plane, not a card position.
                    observation["position"] = observation["slot_position"]
                    observation["anchor_position"] = observation["slot_position"]
                    observation["center_px"] = anchor_roi
            observation["plane"] = (
                {
                    "normal": [float(v) for v in panel["normal"]],
                    "d": float(panel["d"]),
                    "inliers": int(panel["inliers"]),
                }
                if panel is not None
                else None
            )
            observation["plane_rms_m"] = (
                float(panel["rms"])
                if panel is not None and panel.get("rms") is not None
                else None
            )
            state["still_buffer"] = []
            state["still_count"] = 0
            state["center_lock"] = None
            state["anchor_lock"] = None
            state["pose_samples"] = []
            state["pose_center_tag"] = None
            state["pose_axis_tag"] = None
            state["pose_width_tag"] = None
            state["pose_length_tag"] = None
            state["pose_spread_tag"] = None
            state["pose_locked"] = False
            return _observation_to_image(observation, context, self.cfg)

        # Inserted card: measure the full outer silhouette in tag coordinates,
        # lock its robust median pose, then project that pose through the
        # current T07 homography.  The strip fit remains only a state check and
        # a fallback when the outer silhouette is unavailable.
        if outer is not None:
            geometry_measure = outer
            geometry_source = "outer_silhouette"
        else:
            geometry_measure = strip
            geometry_source = "strip_fallback"
        bbox_measure = geometry_measure["bbox"]
        center_roi_meas = np.array(
            [
                float((bbox_measure[0] + bbox_measure[2]) * 0.5),
                float((bbox_measure[1] + bbox_measure[3]) * 0.5),
            ],
            dtype=np.float64,
        )
        axis_roi_meas = np.asarray(geometry_measure["axis_px"], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis_roi_meas))
        if axis_norm > 1e-9:
            axis_roi_meas /= axis_norm
        else:
            axis_roi_meas = np.array([0.0, 1.0], dtype=np.float64)
        width_meas_px = float(geometry_measure["width_px"])
        length_meas_px = float(
            geometry_measure.get(
                "length_px", geometry_measure.get("visible_length_px", 0.0)
            )
        )
        perp_roi = np.array([-axis_roi_meas[1], axis_roi_meas[0]])
        centre_tag_meas = _roi_point_to_tag(center_roi_meas, context)
        axis_probe_tag = _roi_point_to_tag(
            center_roi_meas + axis_roi_meas * 40.0, context
        )
        width_a_tag = _roi_point_to_tag(
            center_roi_meas + perp_roi * 0.5 * width_meas_px, context
        )
        width_b_tag = _roi_point_to_tag(
            center_roi_meas - perp_roi * 0.5 * width_meas_px, context
        )
        top_tag_meas = _roi_point_to_tag(
            center_roi_meas - axis_roi_meas * 0.5 * length_meas_px, context
        )
        bottom_tag_meas = _roi_point_to_tag(
            center_roi_meas + axis_roi_meas * 0.5 * length_meas_px, context
        )
        axis_tag_meas = None
        width_tag_meas = 0.0
        length_tag_meas = 0.0
        if centre_tag_meas is not None and axis_probe_tag is not None:
            axis_tag_meas = np.asarray(axis_probe_tag) - np.asarray(
                centre_tag_meas
            )
            axis_norm = float(np.linalg.norm(axis_tag_meas))
            if axis_norm > 1e-9:
                axis_tag_meas /= axis_norm
            else:
                axis_tag_meas = None
        if axis_tag_meas is not None and axis_tag_meas[1] < 0:
            axis_tag_meas = -axis_tag_meas
        if width_a_tag is not None and width_b_tag is not None:
            width_tag_meas = float(
                np.linalg.norm(
                    np.asarray(width_a_tag) - np.asarray(width_b_tag)
                )
            )
        if top_tag_meas is not None and bottom_tag_meas is not None:
            length_tag_meas = float(
                np.linalg.norm(
                    np.asarray(bottom_tag_meas) - np.asarray(top_tag_meas)
                )
            )

        pose_ok = bool(
            geometry_source == "outer_silhouette"
            and centre_tag_meas is not None
            and axis_tag_meas is not None
            and width_tag_meas > 0.1
            and length_tag_meas > 0.1
            and float(geometry_measure.get("confidence", 0.0)) >= 0.45
            and float(axis_tag_meas[1]) > 0.7
        )
        if pose_ok:
            state["pose_samples"].append(
                (
                    tuple(float(v) for v in centre_tag_meas),
                    tuple(float(v) for v in axis_tag_meas),
                    float(width_tag_meas),
                    float(length_tag_meas),
                )
            )
            state["pose_samples"] = state["pose_samples"][-30:]
            sample_centres = np.asarray(
                [sample[0] for sample in state["pose_samples"]],
                dtype=np.float64,
            )
            sample_median = np.median(sample_centres, axis=0)
            state["pose_spread_tag"] = float(
                np.median(
                    np.linalg.norm(sample_centres - sample_median, axis=1)
                )
            )
        if not state["pose_locked"] and len(state["pose_samples"]) >= int(
            self.cfg["pose_lock_frames"]
        ):
            centres = np.asarray(
                [sample[0] for sample in state["pose_samples"]], dtype=np.float64
            )
            axes = np.asarray(
                [sample[1] for sample in state["pose_samples"]], dtype=np.float64
            )
            widths = np.asarray(
                [sample[2] for sample in state["pose_samples"]], dtype=np.float64
            )
            lengths = np.asarray(
                [sample[3] for sample in state["pose_samples"]], dtype=np.float64
            )
            centre_med = np.median(centres, axis=0)
            axis_med = np.median(axes, axis=0)
            axis_norm = float(np.linalg.norm(axis_med))
            if axis_norm > 1e-9:
                axis_med /= axis_norm
            spread = float(
                np.median(np.linalg.norm(centres - centre_med, axis=1))
            )
            width_med = float(np.median(widths))
            length_med = float(np.median(lengths))
            width_spread = float(
                np.median(np.abs(widths - width_med))
            ) / max(width_med, 1e-6)
            length_spread = float(
                np.median(np.abs(lengths - length_med))
            ) / max(length_med, 1e-6)
            if (
                spread <= float(self.cfg["pose_lock_max_mad_tag"])
                and width_spread <= 0.25
                and length_spread <= 0.25
                and float(axis_med[1]) > 0.7
            ):
                state["pose_center_tag"] = tuple(
                    float(v) for v in centre_med
                )
                state["pose_axis_tag"] = tuple(float(v) for v in axis_med)
                state["pose_width_tag"] = width_med
                state["pose_length_tag"] = length_med
                state["pose_spread_tag"] = spread
                state["pose_locked"] = True

        if state["pose_locked"]:
            centre_tag_use = np.asarray(
                state["pose_center_tag"], dtype=np.float64
            )
            axis_tag_use = np.asarray(state["pose_axis_tag"], dtype=np.float64)
            width_tag_use = float(state["pose_width_tag"])
            length_tag_use = float(state["pose_length_tag"])
            pose_locked = True
            output_geometry_source = (
                "outer_pose_lock" if outer is None else "outer_silhouette"
            )
        elif centre_tag_meas is None or axis_tag_meas is None:
            # Cannot express the measured geometry in tag coordinates; do not
            # invent a position from a fixed prior.
            return None
        else:
            centre_tag_use = np.asarray(centre_tag_meas, dtype=np.float64)
            axis_tag_use = np.asarray(axis_tag_meas, dtype=np.float64)
            width_tag_use = float(width_tag_meas)
            length_tag_use = float(length_tag_meas)
            pose_locked = False
            output_geometry_source = geometry_source

        # Project the measured (or locked) tag-relative outer rectangle through
        # the current homography.  No fixed top/length prior is involved.
        axis_tag_use = axis_tag_use / max(
            float(np.linalg.norm(axis_tag_use)), 1e-9
        )
        perp_tag_use = np.array(
            [axis_tag_use[1], -axis_tag_use[0]], dtype=np.float64
        )
        top_tag_use = centre_tag_use - axis_tag_use * (0.5 * length_tag_use)
        bottom_tag_use = centre_tag_use + axis_tag_use * (0.5 * length_tag_use)
        quad_tag = [
            top_tag_use - perp_tag_use * (0.5 * width_tag_use),
            top_tag_use + perp_tag_use * (0.5 * width_tag_use),
            bottom_tag_use + perp_tag_use * (0.5 * width_tag_use),
            bottom_tag_use - perp_tag_use * (0.5 * width_tag_use),
        ]
        quad_roi = [
            _tag_roi_point(point, context, self.cfg) for point in quad_tag
        ]
        quad_roi = [np.asarray(point, dtype=np.float64) for point in quad_roi]
        center_roi = np.asarray(
            _tag_roi_point(centre_tag_use, context, self.cfg),
            dtype=np.float64,
        )
        axis_roi_use = quad_roi[3] - quad_roi[0]
        axis_roi_use = axis_roi_use / max(
            float(np.linalg.norm(axis_roi_use)), 1e-9
        )
        width_px_use = float(np.linalg.norm(quad_roi[1] - quad_roi[0]))
        length_px_use = float(np.linalg.norm(quad_roi[3] - quad_roi[0]))
        top_roi = np.asarray(
            _tag_roi_point(top_tag_use, context, self.cfg), dtype=np.float64
        )
        bottom_roi = np.asarray(
            _tag_roi_point(bottom_tag_use, context, self.cfg),
            dtype=np.float64,
        )
        bbox = (
            float(min(point[0] for point in quad_roi)),
            float(min(point[1] for point in quad_roi)),
            float(max(point[0] for point in quad_roi)),
            float(max(point[1] for point in quad_roi)),
        )
        visible_center = center_roi

        observation = {
            "tag_id": tag_id,
            "card_present": True,
            "insertion_state": "inserted",
            "state": "inserted",
            "stable": True,
            "state_confidence": state_confidence,
            "state_frames": int(state["candidate_count"]),
            "pose_locked": pose_locked,
            "pose_samples": len(state["pose_samples"]),
            "pose_spread_tag": state.get("pose_spread_tag"),
            "anchor_source": "card_top",
            "anchor_px": tuple(float(v) for v in top_roi),
            "top_px": tuple(float(v) for v in top_roi),
            "bottom_px": tuple(float(v) for v in bottom_roi),
            "center_px": tuple(float(v) for v in center_roi),
            "visible_center_px": tuple(float(v) for v in visible_center),
            "axis_px": tuple(float(v) for v in axis_roi_use),
            "width_px": width_px_use,
            "visible_length_px": length_px_use,
            "nominal_length_px": length_px_use,
            "length_estimated": False,
            "clipped": bool(
                outer["clipped"] if outer is not None else strip["clipped"]
            ),
            "confidence": float(geometry_measure.get("confidence", 0.0)),
            "source": geometry_measure.get(
                "source", "tag_rectified_outer_silhouette"
            ),
            "geometry_source": output_geometry_source,
            "black_fraction": float(
                geometry_measure.get("black_fraction", 0.0)
            ),
            "fill_ratio": float(geometry_measure.get("fill_ratio", 0.0)),
            "bbox": bbox,
            "quad_px": [tuple(float(v) for v in point) for point in quad_roi],
            "quad_tag": [tuple(float(v) for v in point) for point in quad_tag],
            "left_top_px": tuple(float(v) for v in quad_roi[0]),
            "right_top_px": tuple(float(v) for v in quad_roi[1]),
            "right_bottom_px": tuple(float(v) for v in quad_roi[2]),
            "left_bottom_px": tuple(float(v) for v in quad_roi[3]),
            "width_tag": width_tag_use,
            "length_tag": length_tag_use,
            "slot_px": None,
            "position": None,
            "slot_position": None,
            "anchor_position": None,
            "top_position": None,
            "bottom_position": None,
            "plane": None,
            "plane_rms_m": None,
            "panel_plane_rms_m": None,
            "depth_source": None,
            "position_source": None,
            "card_depth_m": None,
            "card_plane_rms_m": None,
        }

        # 3D: measure the card-front depth from the outer quad.  The panel
        # plane is only a fallback; it is not the card surface.
        center_xyz = None
        anchor_xyz = None
        expected_z = panel.get("expected_z") if panel is not None else None
        if expected_z is None:
            expected_z = state.get("last_panel_z")
        center_img = _to_image(center_roi, context["roi_to_image"])
        top_img = _to_image(top_roi, context["roi_to_image"])
        quad_img = np.asarray(
            [
                _to_image(point, context["roi_to_image"])
                for point in quad_roi
            ],
            dtype=np.float64,
        )
        card_depth = _card_surface_depth(
            depth_m, camera_matrix, quad_img, self.cfg
        )
        if card_depth is not None:
            observation["card_depth_m"] = float(card_depth["median_z"])
            if card_depth.get("plane") is not None:
                center_xyz = _ray_plane(
                    center_img, camera_matrix, card_depth["plane"]
                )
                anchor_xyz = _ray_plane(
                    top_img, camera_matrix, card_depth["plane"]
                )
                observation["card_plane_rms_m"] = (
                    float(card_depth["plane"]["rms"])
                    if card_depth["plane"].get("rms") is not None
                    else None
                )
            if center_xyz is None:
                center_xyz = _point_at_z(
                    center_img, camera_matrix, float(card_depth["median_z"])
                )
                anchor_xyz = _point_at_z(
                    top_img, camera_matrix, float(card_depth["median_z"])
                )
            observation["depth_source"] = card_depth["source"]
        if center_xyz is None and panel is not None:
            plane_card = {
                "normal": np.asarray(panel["normal"], dtype=np.float64),
                "d": float(panel["d"])
                - float(self.cfg["panel_depth_offset_m"]),
            }
            center_xyz = _ray_plane(center_img, camera_matrix, plane_card)
            anchor_xyz = _ray_plane(top_img, camera_matrix, plane_card)
            observation["depth_source"] = "panel_plane"
        if center_xyz is None and expected_z is not None:
            center_xyz = _point_at_z(
                center_img, camera_matrix, float(expected_z)
            )
            anchor_xyz = _point_at_z(
                top_img, camera_matrix, float(expected_z)
            )
            observation["depth_source"] = "tag_z_fallback"
        observation["plane"] = (
            {
                "normal": [float(v) for v in panel["normal"]],
                "d": float(panel["d"]),
                "inliers": int(panel["inliers"]),
            }
            if panel is not None
            else None
        )
        observation["plane_rms_m"] = (
            float(panel["rms"])
            if panel is not None and panel.get("rms") is not None
            else None
        )
        observation["panel_plane_rms_m"] = observation["plane_rms_m"]

        # Final stillness lock: if T07 is stationary, the card position in the
        # camera frame is also stationary.  Lock the median 3D point and stop
        # publishing residual depth/tag jitter until the camera moves again.
        corners_now = _tag_corners(marker)
        tag_centre = (
            np.mean(corners_now, axis=0)
            if corners_now is not None
            else None
        )
        previous_centre = state.get("last_tag_centre")
        motion = 0.0
        if tag_centre is not None and previous_centre is not None:
            motion = float(
                np.linalg.norm(np.asarray(tag_centre) - np.asarray(previous_centre))
            )
        if tag_centre is not None:
            state["last_tag_centre"] = tuple(float(v) for v in tag_centre)
        if motion <= float(self.cfg["still_pixel_thresh"]):
            state["still_count"] = int(state.get("still_count", 0)) + 1
        else:
            state["still_count"] = 1
            state["still_buffer"] = []
            state["center_lock"] = None
            state["anchor_lock"] = None

        if center_xyz is not None and anchor_xyz is not None:
            state["still_buffer"].append(
                (
                    tuple(float(v) for v in center_xyz),
                    tuple(float(v) for v in anchor_xyz),
                )
            )
            state["still_buffer"] = state["still_buffer"][
                -int(self.cfg["still_window"]) :
            ]
            if int(state["still_count"]) >= int(self.cfg["still_frames"]):
                centers = np.asarray(
                    [item[0] for item in state["still_buffer"]], dtype=np.float64
                )
                anchors = np.asarray(
                    [item[1] for item in state["still_buffer"]], dtype=np.float64
                )
                state["center_lock"] = tuple(
                    float(v) for v in np.median(centers, axis=0)
                )
                state["anchor_lock"] = tuple(
                    float(v) for v in np.median(anchors, axis=0)
                )

        center_out = center_xyz
        anchor_out = anchor_xyz
        if state.get("center_lock") is not None:
            center_out = np.asarray(state["center_lock"], dtype=np.float64)
        if state.get("anchor_lock") is not None:
            anchor_out = np.asarray(state["anchor_lock"], dtype=np.float64)

        observation["position"] = (
            tuple(float(v) for v in center_out) if center_out is not None else None
        )
        observation["anchor_position"] = (
            tuple(float(v) for v in anchor_out) if anchor_out is not None else None
        )
        observation["top_position"] = observation["anchor_position"]
        observation["bottom_position"] = None
        observation["position_stable"] = bool(
            state.get("center_lock") is not None
        )
        observation["position_source"] = observation["depth_source"]
        return _observation_to_image(observation, context, self.cfg)


def draw_board_card_observation(
    canvas: np.ndarray, observation: Optional[dict]
) -> None:
    """Draw the refined card geometry on a BGR canvas."""

    if canvas is None or not observation:
        return
    colour = (0, 165, 255) if observation.get("card_present") else (128, 128, 128)
    quad = observation.get("quad_px")
    if quad is not None and len(quad) == 4:
        points = np.rint(np.asarray(quad, dtype=np.float64)).astype(np.int32)
        cv2.polylines(canvas, [points], True, colour, 2, cv2.LINE_AA)
    else:
        for name in (
            "left_top_px",
            "right_top_px",
            "right_bottom_px",
            "left_bottom_px",
        ):
            point = observation.get(name)
            if point is None:
                continue
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                3,
                colour,
                -1,
            )
        for start, end in (
            ("left_top_px", "left_bottom_px"),
            ("right_top_px", "right_bottom_px"),
            ("left_top_px", "right_top_px"),
            ("left_bottom_px", "right_bottom_px"),
        ):
            a, b = observation.get(start), observation.get(end)
            if a is None or b is None:
                continue
            cv2.line(
                canvas,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                colour,
                2,
                cv2.LINE_AA,
            )
    anchor = observation.get("anchor_px")
    if anchor is not None:
        point = (int(round(anchor[0])), int(round(anchor[1])))
        cv2.drawMarker(canvas, point, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        label = "{} {:.2f}".format(
            observation.get("insertion_state", "unknown"),
            float(observation.get("confidence", 0.0)),
        )
        cv2.putText(
            canvas,
            label,
            (point[0] + 8, max(18, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    center = observation.get("center_px")
    if center is not None:
        point = (int(round(center[0])), int(round(center[1])))
        cv2.drawMarker(canvas, point, (255, 255, 0), cv2.MARKER_CROSS, 12, 2)
