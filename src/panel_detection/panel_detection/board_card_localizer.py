"""Tag-relative board-card localization from colour edges and depth planes.

The board-card tag gives a stable, scale-normalised coordinate frame.  Inside a
fixed tag-relative slot ROI the card is a long, narrow object whose two long
edges are photometric edges.  Those edges and the card's top/bottom endpoints
are therefore fitted in the **colour** image; depth is used only to recover the
card plane and to turn the 2D points into 3D points.

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
    # Fixed tag-relative slot ROI, in tag-side units.  +x is image-right and
    # +y is image-down after the tag orientation is normalised.
    "roi_x": (-3.5, 3.5),
    "roi_y": (1.0, 11.0),
    "roi_size": (560, 800),
    # Search band inside the rectified ROI.  The central band excludes the
    # recess bevel and its cast shadow on both sides.
    "slot_x": (0.36, 0.64),
    "slot_y": (0.16, 0.99),
    "max_top_ratio": 0.45,
    # Colour model: black card body or blue PCB/connector.
    "dark_v_max": 105,
    "dark_s_max": 165,
    "blue_h": (80, 138),
    "blue_s_min": 70,
    "blue_v_min": 18,
    # Geometry gates.
    "min_visible_rows": 30,
    "min_width_px": 18,
    "max_width_px": 150,
    "min_confidence": 0.30,
    "max_row_gap": 24,
    "blue_width_px": 58.0,
    "black_width_px": 105.0,
    "blue_ratio_threshold": 0.45,
    "edge_width_ratio": 0.75,
    "plane_depth_gate_m": 0.12,
    # Top-centre is the stable insertion/grasp anchor.  The canonical centre
    # uses the longest reliably observed length so a clipped bottom does not
    # drag the centre towards the camera.
    "nominal_length_memory": True,
    "design_width_change_ratio": 0.30,
    "state_inserted_ratio": 0.85,
    "state_pulled_ratio": 0.72,
    "state_pulled_bottom_roi_ratio": 0.78,
    "state_wide_width_px": 80.0,
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
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    dark = (value < float(_cfg_value(cfg, "dark_v_max"))) & (
        saturation < float(_cfg_value(cfg, "dark_s_max"))
    )
    blue_h = _cfg_value(cfg, "blue_h")
    blue = (
        (hue >= int(blue_h[0]))
        & (hue <= int(blue_h[1]))
        & (saturation > int(_cfg_value(cfg, "blue_s_min")))
        & (value > int(_cfg_value(cfg, "blue_v_min")))
    )
    mask = ((dark | blue) & valid).astype(np.uint8)

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
    mask_count = int(mask.sum())
    blue_ratio = (
        float(np.count_nonzero(blue & (mask > 0))) / mask_count
        if mask_count else 0.0
    )

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
    if top_y > float(_cfg_value(cfg, "max_top_ratio")) * height:
        return None

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
        "blue_ratio": blue_ratio,
        "visible_length_px": visible_length,
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


def _plane_from_points(points: np.ndarray) -> Optional[dict]:
    """RANSAC plane in camera coordinates; n.dot(X) = d."""

    if points is None or len(points) < 12:
        return None
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 12:
        return None
    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    for _ in range(80):
        indices = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        distance = float(normal @ p0)
        residual = np.abs(points @ normal - distance)
        inliers = residual < 0.004
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
    inliers = residual < 0.006
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
    return _plane_from_points(points_3d)


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

    def _state_for(self, tag_id: int) -> dict:
        return self._state.setdefault(
            int(tag_id),
            {
                "nominal_length_px": None,
                "design_width_px": None,
                "anchor_px": None,
                "axis_px": None,
                "last_panel_z": None,
                "missed": 0,
            },
        )

    @staticmethod
    def _edge_support(gray: np.ndarray, x: float, y0: float, y1: float) -> float:
        height, width = gray.shape[:2]
        x0 = max(1, int(round(x)) - 3)
        x1 = min(width - 1, int(round(x)) + 4)
        yy0 = max(1, int(round(y0)))
        yy1 = min(height - 1, int(round(y1)))
        if x1 - x0 < 3 or yy1 - yy0 < 8:
            return 0.0
        gradient = np.abs(
            cv2.Sobel(gray[yy0:yy1, x0:x1], cv2.CV_32F, 1, 0, ksize=3)
        )
        return float(np.median(gradient.max(axis=1)))

    def _correct_strip_width(
        self, strip: dict, context: dict, design_width: float
    ) -> None:
        """Re-anchor a strip whose mask width disagrees with the card prior.

        A shadowed blue card can expose only its dark contact strip, while a
        loose threshold can also merge one side with the recess shadow.  Test
        both width-prior hypotheses against the colour gradient and shift the
        measured geometry onto the better-supported pair.
        """

        width = float(strip["width_px"])
        if design_width <= 0 or abs(width - design_width) <= 0.25 * design_width:
            return
        left = float(strip["left_top_px"][0])
        right = float(strip["right_top_px"][0])
        top_y = float(strip["top_px"][1])
        bottom_y = float(strip["bottom_px"][1])
        y0 = top_y + 0.08 * max(1.0, bottom_y - top_y)
        y1 = top_y + 0.85 * max(1.0, bottom_y - top_y)
        gray = cv2.cvtColor(context["roi"], cv2.COLOR_BGR2GRAY)
        image_width = gray.shape[1]
        left_center = left + 0.5 * design_width
        right_center = right - 0.5 * design_width
        if not (5 <= left_center <= image_width - 5):
            left_center = None
        if not (5 <= right_center <= image_width - 5):
            right_center = None
        score_left = (
            self._edge_support(gray, left + design_width, y0, y1)
            if left_center is not None else -1.0
        )
        score_right = (
            self._edge_support(gray, right - design_width, y0, y1)
            if right_center is not None else -1.0
        )
        if max(score_left, score_right) < 6.0:
            return
        if score_left >= score_right * 1.10:
            target_center = left_center
            source = "left_edge_plus_width"
        elif score_right >= score_left * 1.10:
            target_center = right_center
            source = "right_edge_minus_width"
        else:
            return
        delta = float(target_center) - float(strip["top_px"][0])

        def shift(name):
            point = strip.get(name)
            if point is not None:
                strip[name] = (float(point[0]) + delta, float(point[1]))

        for name in (
            "top_px", "bottom_px", "visible_center_px",
            "left_top_px", "right_top_px", "left_bottom_px", "right_bottom_px",
        ):
            shift(name)
        bbox = strip.get("bbox")
        if bbox is not None:
            strip["bbox"] = (
                float(bbox[0]) + delta, float(bbox[1]),
                float(bbox[2]) + delta, float(bbox[3]),
            )
        strip["width_px"] = float(design_width)
        strip["width_source"] = source
        strip["edge_prior_delta"] = delta
        strip["edge_prior_scores"] = (score_left, score_right)

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

        if strip is None or strip["confidence"] < float(self.cfg["min_confidence"]):
            # No card: publish the fixed slot anchor, explicitly marked empty.
            state["missed"] += 1
            anchor_roi = _tag_roi_point(self.cfg["slot_anchor_tag"], context, self.cfg)
            observation = {
                "tag_id": tag_id,
                "card_present": False,
                "insertion_state": "empty",
                "anchor_source": "slot",
                "anchor_px": anchor_roi,
                "top_px": None,
                "bottom_px": None,
                "center_px": anchor_roi,
                "visible_center_px": None,
                "left_top_px": None,
                "right_top_px": None,
                "left_bottom_px": None,
                "right_bottom_px": None,
                "bbox": None,
                "axis_px": (0.0, 1.0),
                "width_px": None,
                "visible_length_px": 0.0,
                "nominal_length_px": state["nominal_length_px"],
                "length_estimated": state["nominal_length_px"] is not None,
                "clipped": False,
                "confidence": 0.0,
                "source": "tag_slot_prior",
                "position": None,
                "anchor_position": None,
                "top_position": None,
                "bottom_position": None,
                "plane": None,
                "plane_rms_m": None,
                "depth_source": None,
            }
            if camera_matrix is not None and depth_m is not None:
                point = _depth_point(anchor_roi, depth_m, camera_matrix, radius=10)
                if point is not None:
                    observation["position"] = tuple(float(v) for v in point)
                    observation["anchor_position"] = observation["position"]
                    observation["depth_source"] = "slot_depth"
            return _observation_to_image(observation, context, self.cfg)

        # Nominal full length is stable in the tag-normalised ROI because the
        # tag homography removes camera distance.  A width change means a
        # different card design, so the length memory is reset.
        nominal = state["nominal_length_px"]
        design_width = state["design_width_px"]
        base_width = (
            float(self.cfg["blue_width_px"])
            if float(strip.get("blue_ratio", 0.0))
            >= float(self.cfg["blue_ratio_threshold"])
            else float(self.cfg["black_width_px"])
        )
        if design_width is None or abs(design_width - base_width) > 0.25 * base_width:
            nominal = None
            design_width = base_width
        self._correct_strip_width(strip, context, design_width)
        width = float(strip["width_px"])
        if design_width is None:
            design_width = width
        if not strip["clipped"]:
            if nominal is None or strip["visible_length_px"] > nominal:
                nominal = float(strip["visible_length_px"])
        elif nominal is None:
            nominal = float(strip["visible_length_px"])
        state["nominal_length_px"] = nominal
        state["design_width_px"] = design_width
        estimated = bool(strip["clipped"])

        # Top centre is the physical stable anchor.  Canonical centre uses the
        # remembered full length so a clipped bottom cannot bias it upward.
        axis = np.asarray(strip["axis_px"], dtype=np.float64)
        top = np.asarray(strip["top_px"], dtype=np.float64)
        center_roi = top + axis * (float(nominal) * 0.5)
        top_roi = tuple(float(v) for v in top)
        center_roi = (float(center_roi[0]), float(center_roi[1]))

        # Smooth the anchor only if it moved plausibly; a tag/ID switch or a
        # genuine fast motion is accepted immediately.
        previous_anchor = state.get("anchor_px")
        if previous_anchor is not None and float(
            np.linalg.norm(top - np.asarray(previous_anchor, dtype=np.float64))
        ) <= float(self.cfg["max_anchor_jump_px"]):
            alpha = float(self.cfg["smooth_alpha"])
            smooth_top = np.asarray(previous_anchor, dtype=np.float64) * (1.0 - alpha) + top * alpha
        else:
            smooth_top = top.copy()
        previous_axis = state.get("axis_px")
        if previous_axis is not None:
            previous_axis = np.asarray(previous_axis, dtype=np.float64)
            if float(previous_axis @ axis) > 0.8:
                alpha = float(self.cfg["smooth_alpha"])
                axis = previous_axis * (1.0 - alpha) + axis * alpha
                axis /= max(float(np.linalg.norm(axis)), 1e-9)
        state["anchor_px"] = tuple(float(v) for v in smooth_top)
        state["axis_px"] = tuple(float(v) for v in axis)
        state["missed"] = 0

        center_roi = smooth_top + axis * (float(nominal) * 0.5)
        visible_center = np.asarray(strip["visible_center_px"], dtype=np.float64)
        ratio = float(strip["visible_length_px"]) / max(float(nominal), 1e-6)
        pulled_by_bottom = False
        # The narrow blue card body clears the slot bottom when it is pulled
        # towards the camera.  The wide black handle already extends below the
        # frame when seated, so this cue is only used for the narrow design.
        if (
            float(strip["width_px"]) < float(self.cfg["state_wide_width_px"])
            and float(strip.get("blue_ratio", 0.0))
            >= float(self.cfg["blue_ratio_threshold"])
        ):
            roi_height = float(context["roi"].shape[0])
            bottom_ratio = float(strip["bottom_px"][1]) / max(roi_height, 1.0)
            if bottom_ratio > float(self.cfg["state_pulled_bottom_roi_ratio"]):
                pulled_by_bottom = True
        if (
            strip["clipped"]
            or ratio < float(self.cfg["state_pulled_ratio"])
            or pulled_by_bottom
        ):
            insertion_state = "pulled"
        elif ratio >= float(self.cfg["state_inserted_ratio"]):
            insertion_state = "inserted"
        else:
            insertion_state = "transition"

        observation = dict(strip)
        observation.update(
            {
                "tag_id": tag_id,
                "card_present": True,
                "insertion_state": insertion_state,
                "anchor_source": "card_top",
                "anchor_px": tuple(float(v) for v in smooth_top),
                "top_px": top_roi,
                "center_px": (float(center_roi[0]), float(center_roi[1])),
                "visible_center_px": tuple(float(v) for v in visible_center),
                "axis_px": tuple(float(v) for v in axis),
                "nominal_length_px": float(nominal),
                "length_estimated": estimated,
                "position": None,
                "anchor_position": None,
                "top_position": None,
                "bottom_position": None,
                "plane": None,
                "plane_rms_m": None,
                "depth_source": None,
            }
        )

        # 3D: fit the card plane from interior depth points, then intersect the
        # colour-derived rays with that plane.  This keeps depth out of the 2D
        # edge decision while giving a metric 3D point.  A depth sample just
        # inside the top edge supplies an expected Z so a plane fit cannot jump
        # to the white frame or the wall behind a pulled card.
        plane = None
        anchor_depth_xyz = None
        panel_depth_xyz = None
        expected_z = None
        if camera_matrix is not None and depth_m is not None:
            sample = np.asarray(observation["anchor_px"]) + np.asarray(
                observation["axis_px"]
            ) * max(8.0, min(0.15 * float(strip["visible_length_px"]), 40.0))
            anchor_depth_xyz = _depth_point(
                sample, depth_m, camera_matrix, radius=9
            )
            marker_centre = _tag_field(marker, "center_xy")
            if marker_centre is not None:
                panel_depth_xyz = _depth_point(
                    marker_centre, depth_m, camera_matrix, radius=15
                )
            if anchor_depth_xyz is not None:
                expected_z = float(anchor_depth_xyz[2])
            if panel_depth_xyz is not None:
                panel_z = float(panel_depth_xyz[2])
                state["last_panel_z"] = panel_z
                if (
                    expected_z is None
                    or abs(expected_z - panel_z)
                    > float(self.cfg["plane_depth_gate_m"])
                ):
                    # Blue/black card surfaces are frequently depth holes; the
                    # tag is high-contrast and sits on the same panel plane, so
                    # it is a safer Z prior than a background pixel behind the
                    # card.
                    expected_z = panel_z
            elif state.get("last_panel_z") is not None:
                expected_z = float(state["last_panel_z"])
            polygon = np.array(
                [
                    observation["left_top_px"],
                    observation["right_top_px"],
                    observation["right_bottom_px"],
                    observation["left_bottom_px"],
                ],
                dtype=np.float32,
            )
            plane = _depth_plane(
                depth_m, camera_matrix, polygon, self.cfg,
                expected_z=expected_z,
            )
            if plane is not None and expected_z is not None:
                plane_anchor = _ray_plane(
                    observation["anchor_px"], camera_matrix, plane
                )
                if (
                    plane_anchor is not None
                    and abs(float(plane_anchor[2]) - expected_z)
                    > float(self.cfg["plane_depth_gate_m"])
                ):
                    plane = None
        observation["plane"] = (
            {
                "normal": [float(v) for v in plane["normal"]],
                "d": float(plane["d"]),
                "inliers": int(plane["inliers"]),
            }
            if plane is not None
            else None
        )
        observation["plane_rms_m"] = float(plane["rms"]) if plane and plane["rms"] is not None else None

        anchor_xyz = _ray_plane(observation["anchor_px"], camera_matrix, plane) if camera_matrix is not None else None
        center_xyz = _ray_plane(observation["center_px"], camera_matrix, plane) if camera_matrix is not None else None
        top_xyz = _ray_plane(observation["top_px"], camera_matrix, plane) if camera_matrix is not None else None
        bottom_xyz = _ray_plane(observation["bottom_px"], camera_matrix, plane) if camera_matrix is not None else None
        if center_xyz is not None:
            observation["depth_source"] = "card_plane"
        elif camera_matrix is not None and expected_z is not None:
            center_xyz = _point_at_z(
                observation["center_px"], camera_matrix, expected_z
            )
            anchor_xyz = _point_at_z(
                observation["anchor_px"], camera_matrix, expected_z
            )
            top_xyz = _point_at_z(
                observation["top_px"], camera_matrix, expected_z
            )
            if not strip["clipped"]:
                bottom_xyz = _point_at_z(
                    observation["bottom_px"], camera_matrix, expected_z
                )
            observation["depth_source"] = "anchor_depth"
        if top_xyz is None:
            top_xyz = anchor_xyz
        if anchor_xyz is None:
            anchor_xyz = center_xyz
        if bottom_xyz is None and not strip["clipped"] and plane is not None:
            bottom_xyz = _ray_plane(
                observation["bottom_px"], camera_matrix, plane
            )

        observation["position"] = (
            tuple(float(v) for v in center_xyz) if center_xyz is not None else None
        )
        observation["anchor_position"] = (
            tuple(float(v) for v in anchor_xyz) if anchor_xyz is not None else None
        )
        observation["top_position"] = (
            tuple(float(v) for v in top_xyz) if top_xyz is not None else None
        )
        observation["bottom_position"] = (
            tuple(float(v) for v in bottom_xyz) if bottom_xyz is not None else None
        )
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
