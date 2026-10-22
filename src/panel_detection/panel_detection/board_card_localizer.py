"""Stable tag-relative board-card state and position estimation.

T07 supplies a scale-normalised slot frame.  Colour is used to confirm the
binary state (black card fully inserted vs empty slot) and to feed a robust
tag-relative pose lock.  Once inserted, the fixed card pose is projected
through the current T07 homography; per-frame colour edges are no longer used
as the published centre.  Depth is used only for the panel plane around T07,
not for the black card surface (which is often a depth hole).  A final
stillness-gated lock removes residual camera-frame jitter while the robot is
stationary.

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
    "top_tag_y_range": (2.2, 4.4),
    "min_black_fraction": 0.50,
    # Full card length in tag-side units, measured once for the site.  It is
    # used for the canonical centre even when the bottom edge is clipped.
    "card_length_tag": 7.13,
    # Site prior and robust tag-relative pose lock.
    "card_top_tag_prior": (0.23, 3.70),
    "pose_lock_frames": 5,
    "pose_lock_max_mad_tag": 0.20,
    "pose_lock_prior_gate_tag": 0.80,
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
    # Empty-slot anchor in tag-side units (directly below the tag).
    "slot_anchor_tag": (0.0, 6.0),
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
                "pose_top_tag": None,
                "pose_axis_tag": None,
                "pose_width_tag": None,
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

        roi_height = context["roi"].shape[0]
        roi_y0, roi_y1 = (float(v) for v in self.cfg["roi_y"])
        tag_y_span = max(roi_y1 - roi_y0, 1e-6)
        nominal = float(self.cfg["card_length_tag"]) * roi_height / tag_y_span

        top_tag_y = None
        visible_length_tag = 0.0
        if strip is not None:
            top_tag_y = roi_y0 + (float(strip["top_px"][1]) / roi_height) * tag_y_span
            visible_length_tag = float(strip["visible_length_px"]) * tag_y_span / roi_height

        y_min, y_max = (float(v) for v in self.cfg["top_tag_y_range"])
        inserted_measurement = bool(
            strip is not None
            and strip["confidence"] >= float(self.cfg["min_confidence"])
            and top_tag_y is not None
            and y_min <= top_tag_y <= y_max
            and visible_length_tag >= float(self.cfg["min_inserted_length_tag"])
            and float(strip.get("black_fraction", 0.0))
            >= float(self.cfg["min_black_fraction"])
        )
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
        panel = self._effective_panel_plane(
            state, depth_m, camera_matrix, marker
        )

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
            anchor_roi = _tag_roi_point(
                self.cfg["slot_anchor_tag"], context, self.cfg
            )
            observation = {
                "tag_id": tag_id,
                "card_present": False,
                "insertion_state": "empty",
                "state": "empty",
                "stable": True,
                "state_confidence": state_confidence,
                "state_frames": int(state["candidate_count"]),
                "anchor_source": "slot",
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
                "axis_px": (0.0, 1.0),
                "width_px": None,
                "visible_length_px": 0.0,
                "nominal_length_px": nominal,
                "length_estimated": False,
                "clipped": False,
                "confidence": 0.0,
                "source": "tag_slot_prior",
                "slot_px": anchor_roi,
                "position": None,
                "slot_position": None,
                "anchor_position": None,
                "top_position": None,
                "bottom_position": None,
                "plane": None,
                "plane_rms_m": None,
                "depth_source": None,
            }
            if camera_matrix is not None:
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
            return _observation_to_image(observation, context, self.cfg)

        # Inserted card: measure once in tag coordinates, lock the robust
        # median pose, then project that fixed pose through the current T07
        # homography.  Per-frame colour edges are only used to confirm the
        # state and to feed the lock.
        top_roi = np.asarray(strip["top_px"], dtype=np.float64)
        axis_roi_meas = np.asarray(strip["axis_px"], dtype=np.float64)
        top_tag = _roi_point_to_tag(top_roi, context)
        axis_probe_roi = top_roi + axis_roi_meas * 40.0
        axis_probe_tag = _roi_point_to_tag(axis_probe_roi, context)
        roi_width = context["roi"].shape[1]
        roi_x0, roi_x1 = (float(v) for v in self.cfg["roi_x"])
        sx = roi_width / max(roi_x1 - roi_x0, 1e-6)
        width_tag = float(strip["width_px"]) / max(sx, 1e-6)
        axis_tag = None
        if top_tag is not None and axis_probe_tag is not None:
            axis_tag = np.asarray(axis_probe_tag, dtype=np.float64) - np.asarray(
                top_tag, dtype=np.float64
            )
            norm = float(np.linalg.norm(axis_tag))
            if norm > 1e-9:
                axis_tag /= norm
            else:
                axis_tag = None
        if axis_tag is not None and axis_tag[1] < 0:
            axis_tag = -axis_tag

        prior_top = np.asarray(self.cfg["card_top_tag_prior"], dtype=np.float64)
        pose_ok = bool(
            top_tag is not None
            and axis_tag is not None
            and float(strip["confidence"]) >= 0.45
            and np.linalg.norm(np.asarray(top_tag) - prior_top)
            <= float(self.cfg["pose_lock_prior_gate_tag"])
            and float(axis_tag[1]) > 0.7
        )
        if pose_ok:
            state["pose_samples"].append(
                (tuple(top_tag), tuple(float(v) for v in axis_tag), float(width_tag))
            )
            state["pose_samples"] = state["pose_samples"][-30:]
        if not state["pose_locked"] and len(state["pose_samples"]) >= int(
            self.cfg["pose_lock_frames"]
        ):
            tops = np.asarray([s[0] for s in state["pose_samples"]], dtype=np.float64)
            axes = np.asarray([s[1] for s in state["pose_samples"]], dtype=np.float64)
            widths = np.asarray([s[2] for s in state["pose_samples"]], dtype=np.float64)
            top_med = np.median(tops, axis=0)
            axis_med = np.median(axes, axis=0)
            axis_norm = float(np.linalg.norm(axis_med))
            if axis_norm > 1e-9:
                axis_med /= axis_norm
            spread = float(
                np.median(np.linalg.norm(tops - top_med, axis=1))
            )
            if (
                spread <= float(self.cfg["pose_lock_max_mad_tag"])
                and np.linalg.norm(top_med - prior_top)
                <= float(self.cfg["pose_lock_prior_gate_tag"])
            ):
                state["pose_top_tag"] = tuple(float(v) for v in top_med)
                state["pose_axis_tag"] = tuple(float(v) for v in axis_med)
                state["pose_width_tag"] = float(np.median(widths))
                state["pose_spread_tag"] = spread
                state["pose_locked"] = True

        if state["pose_locked"]:
            top_use = np.asarray(state["pose_top_tag"], dtype=np.float64)
            axis = np.asarray(state["pose_axis_tag"], dtype=np.float64)
            width_use = float(state["pose_width_tag"])
            center_tag_use = top_use + axis * (float(self.cfg["card_length_tag"]) * 0.5)
            anchor_roi = _tag_roi_point(top_use, context, self.cfg)
            center_roi = _tag_roi_point(center_tag_use, context, self.cfg)
            axis_end_roi = _tag_roi_point(top_use + axis, context, self.cfg)
            axis_vec = np.asarray(axis_end_roi) - np.asarray(anchor_roi)
            norm = float(np.linalg.norm(axis_vec))
            axis_px_use = axis_vec / norm if norm > 1e-9 else (0.0, 1.0)
            half = 0.5 * width_use
            perp = np.array([-axis[1], axis[0]], dtype=np.float64)
            corners = [
                top_use - perp * half,
                top_use + perp * half,
                center_tag_use + axis * (self.cfg["card_length_tag"] * 0.5) + perp * half,
                center_tag_use + axis * (self.cfg["card_length_tag"] * 0.5) - perp * half,
            ]
            corners_roi = np.asarray(
                [_tag_roi_point(p, context, self.cfg) for p in corners],
                dtype=np.float64,
            )
            bbox = (
                float(corners_roi[:, 0].min()),
                float(corners_roi[:, 1].min()),
                float(corners_roi[:, 0].max()),
                float(corners_roi[:, 1].max()),
            )
            top_px_use = anchor_roi
            width_px_use = width_use * sx
            pose_locked = True
        else:
            previous_anchor = state.get("anchor_px")
            if previous_anchor is not None and float(
                np.linalg.norm(top_roi - np.asarray(previous_anchor, dtype=np.float64))
            ) <= float(self.cfg["max_anchor_jump_px"]):
                alpha = float(self.cfg["smooth_alpha"])
                smooth_top = (
                    np.asarray(previous_anchor, dtype=np.float64) * (1.0 - alpha)
                    + top_roi * alpha
                )
            else:
                smooth_top = top_roi.copy()
            previous_axis = state.get("axis_px")
            if previous_axis is not None:
                previous_axis = np.asarray(previous_axis, dtype=np.float64)
                if float(previous_axis @ axis_roi_meas) > 0.8:
                    alpha = float(self.cfg["smooth_alpha"])
                    axis_roi_meas = (
                        previous_axis * (1.0 - alpha) + axis_roi_meas * alpha
                    )
                    axis_roi_meas /= max(
                        float(np.linalg.norm(axis_roi_meas)), 1e-9
                    )
            state["anchor_px"] = tuple(float(v) for v in smooth_top)
            state["axis_px"] = tuple(float(v) for v in axis_roi_meas)
            anchor_roi = tuple(float(v) for v in smooth_top)
            center_roi = (
                float(smooth_top[0] + axis_roi_meas[0] * nominal * 0.5),
                float(smooth_top[1] + axis_roi_meas[1] * nominal * 0.5),
            )
            top_px_use = tuple(float(v) for v in top_roi)
            axis_px_use = tuple(float(v) for v in axis_roi_meas)
            width_px_use = float(strip["width_px"])
            bbox = strip["bbox"]
            pose_locked = False

        visible_center = np.asarray(strip["visible_center_px"], dtype=np.float64)
        observation = dict(strip)
        observation.update(
            {
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
                "anchor_px": tuple(float(v) for v in anchor_roi),
                "top_px": tuple(float(v) for v in top_px_use),
                "center_px": (float(center_roi[0]), float(center_roi[1])),
                "visible_center_px": tuple(float(v) for v in visible_center),
                "axis_px": tuple(float(v) for v in axis_px_use),
                "width_px": width_px_use,
                "nominal_length_px": float(nominal),
                "length_estimated": False,
                "clipped": bool(strip["clipped"]),
                "bbox": bbox,
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
            }
        )

        # 3D: use the panel plane around T07, not the black card's depth (which
        # is frequently full of holes).  Card thickness is a fixed site offset.
        plane_card = None
        center_xyz = None
        anchor_xyz = None
        expected_z = panel.get("expected_z") if panel is not None else None
        if expected_z is None:
            expected_z = state.get("last_panel_z")
        if panel is not None:
            plane_card = {
                "normal": np.asarray(panel["normal"], dtype=np.float64),
                "d": float(panel["d"])
                - float(self.cfg["panel_depth_offset_m"]),
            }
        if camera_matrix is not None:
            anchor_img = _to_image(anchor_roi, context["roi_to_image"])
            center_img = _to_image(center_roi, context["roi_to_image"])
            if plane_card is not None:
                anchor_xyz = _ray_plane(anchor_img, camera_matrix, plane_card)
                center_xyz = _ray_plane(center_img, camera_matrix, plane_card)
                observation["depth_source"] = "panel_plane"
            if center_xyz is None and expected_z is not None:
                center_xyz = _point_at_z(center_img, camera_matrix, float(expected_z))
                anchor_xyz = _point_at_z(
                    anchor_img, camera_matrix, float(expected_z)
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
    for name in ("left_top_px", "right_top_px", "right_bottom_px", "left_bottom_px"):
        point = observation.get(name)
        if point is None:
            continue
        cv2.circle(canvas, (int(round(point[0])), int(round(point[1]))), 3, colour, -1)
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
