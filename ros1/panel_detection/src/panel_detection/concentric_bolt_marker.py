"""Detection and RGB-D localization for the printed concentric bolt target."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ConcentricBoltMarker:
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    axes: Tuple[float, float]
    ellipse_angle_deg: float
    radius: float
    gap_angle_deg: Optional[float]
    gap_point: Optional[Tuple[float, float]]
    confidence: float
    contour: np.ndarray
    depth_mask: np.ndarray


@dataclass
class ConcentricBoltPose:
    point_3d: Optional[np.ndarray]
    direction_3d: Optional[np.ndarray]
    plane_offset: Optional[float]
    depth_source: str


def _direct_children(hierarchy, index):
    child = int(hierarchy[index][2])
    while child >= 0:
        yield child
        child = int(hierarchy[child][0])


def _ellipse_point(center, axes, angle_deg, rho, theta):
    phi = np.deg2rad(angle_deg)
    a = 0.5 * float(axes[0]) * rho
    b = 0.5 * float(axes[1]) * rho
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return (
        center[0] + a * ct * cp - b * st * sp,
        center[1] + a * ct * sp + b * st * cp,
    )


def _estimate_gap(gray, center, axes, angle_deg, min_axis_for_angle):
    if min(axes) < min_axis_for_angle:
        return None, None, 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    rho = np.linspace(0.36, 0.68, 18)
    theta_grid, rho_grid = np.meshgrid(theta, rho, indexing='ij')
    phi = np.deg2rad(angle_deg)
    a = 0.5 * float(axes[0])
    b = 0.5 * float(axes[1])
    xmap = (center[0]
            + a * rho_grid * np.cos(theta_grid) * np.cos(phi)
            - b * rho_grid * np.sin(theta_grid) * np.sin(phi))
    ymap = (center[1]
            + a * rho_grid * np.cos(theta_grid) * np.sin(phi)
            + b * rho_grid * np.sin(theta_grid) * np.cos(phi))
    polar = cv2.remap(
        gray, xmap.astype(np.float32), ymap.astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    angular = polar.mean(axis=1)
    half_window = 10
    padded = np.r_[angular[-half_window:], angular, angular[:half_window]]
    smooth = np.convolve(
        padded, np.ones(2 * half_window + 1) / (2 * half_window + 1),
        mode='valid')[:360]
    peak_index = int(np.argmax(smooth))
    prominence = float(smooth[peak_index] - np.percentile(smooth, 35))
    if prominence < 8.0:
        return None, None, prominence

    theta_peak = theta[peak_index]
    gap_point = _ellipse_point(center, axes, angle_deg, 0.55, theta_peak)
    image_angle = float(
        np.degrees(np.arctan2(
            gap_point[1] - center[1], gap_point[0] - center[0])) % 360.0)
    return image_angle, gap_point, prominence


def detect_concentric_bolt_markers(color_image, cfg=None):
    """Detect printed targets without relying on a learned object detector."""
    cfg = cfg or {}
    if color_image is None or color_image.size == 0:
        return []

    gray = (color_image if color_image.ndim == 2
            else cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = float(cfg.get('min_area_px', 45.0))
    max_area = float(cfg.get('max_area_ratio', 0.04)) * image_area
    min_circularity = float(cfg.get('min_circularity', 0.72))
    min_axis_ratio = float(cfg.get('min_axis_ratio', 0.60))
    min_axis_for_angle = float(cfg.get('min_axis_for_angle_px', 22.0))
    candidates = []

    for index, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if not (min_area <= area <= max_area) or perimeter <= 0.0:
            continue
        if len(contour) < 5:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        center, axes, ellipse_angle = cv2.fitEllipse(contour)
        min_axis, max_axis = min(axes), max(axes)
        if min_axis < 7.0 or min_axis / max(max_axis, 1e-6) < min_axis_ratio:
            continue
        radius = 0.25 * (float(axes[0]) + float(axes[1]))

        centered_dots = []
        centered_rings = []
        for child_index in _direct_children(hierarchy, index):
            child = contours[child_index]
            child_area = float(cv2.contourArea(child))
            child_perimeter = float(cv2.arcLength(child, True))
            if child_area <= 2.0 or child_perimeter <= 0.0:
                continue
            moments = cv2.moments(child)
            if moments['m00'] == 0.0:
                continue
            child_center = (
                moments['m10'] / moments['m00'],
                moments['m01'] / moments['m00'],
            )
            child_circularity = (
                4.0 * np.pi * child_area / (child_perimeter * child_perimeter))
            center_offset = float(np.hypot(
                child_center[0] - center[0], child_center[1] - center[1]))
            area_ratio = child_area / area
            if (center_offset <= 0.22 * radius
                    and 0.008 <= area_ratio <= 0.16
                    and child_circularity >= 0.55):
                centered_dots.append((child_circularity, area_ratio))
            if (center_offset <= 0.30 * radius
                    and 0.16 <= area_ratio <= 0.65
                    and child_circularity <= 0.62):
                centered_rings.append((child_circularity, area_ratio))
        # The small round child is the center dot; the larger low-circularity
        # child is the C-shaped orientation ring. Requiring both prevents a
        # plain button or an ordinary bullseye from becoming a bolt marker.
        if not centered_dots or not centered_rings:
            continue

        gap_angle, gap_point, gap_prominence = _estimate_gap(
            gray, center, axes, ellipse_angle, min_axis_for_angle)
        size_score = float(np.clip((min_axis - 7.0) / 18.0, 0.0, 1.0))
        gap_score = float(np.clip(gap_prominence / 35.0, 0.0, 1.0))
        child_score = max(item[0] for item in centered_dots)
        confidence = float(np.clip(
            0.35 * circularity + 0.25 * child_score
            + 0.20 * size_score + 0.20 * gap_score, 0.0, 1.0))

        half_w = 0.58 * float(axes[0])
        half_h = 0.58 * float(axes[1])
        bbox = (
            max(0.0, center[0] - half_w),
            max(0.0, center[1] - half_h),
            min(float(gray.shape[1] - 1), center[0] + half_w),
            min(float(gray.shape[0] - 1), center[1] + half_h),
        )
        depth_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.ellipse(
            depth_mask,
            (int(round(center[0])), int(round(center[1]))),
            (max(2, int(round(0.18 * axes[0]))),
             max(2, int(round(0.18 * axes[1])))),
            float(ellipse_angle), 0.0, 360.0, 255, -1)
        candidates.append(ConcentricBoltMarker(
            center=(float(center[0]), float(center[1])),
            bbox=bbox,
            axes=(float(axes[0]), float(axes[1])),
            ellipse_angle_deg=float(ellipse_angle),
            radius=radius,
            gap_angle_deg=gap_angle,
            gap_point=gap_point,
            confidence=confidence,
            contour=contour,
            depth_mask=depth_mask,
        ))

    kept = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            np.hypot(candidate.center[0] - item.center[0],
                     candidate.center[1] - item.center[1])
            <= 0.5 * max(candidate.radius, item.radius)
            for item in kept)
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.center[1], item.center[0]))


def _pixel_ray(intrin, pixel):
    return np.array([
        (float(pixel[0]) - intrin.cx) / intrin.fx,
        (float(pixel[1]) - intrin.cy) / intrin.fy,
        1.0,
    ], dtype=np.float64)


def estimate_concentric_bolt_poses(markers, depth_image, intrin, normal,
                                   depth_scale=0.001, cfg=None):
    """Intersect marker rays with one bolt-head plane constrained by Tag normal."""
    cfg = cfg or {}
    if not markers:
        return []
    if depth_image is None or intrin is None or normal is None:
        return [ConcentricBoltPose(None, None, None, 'unavailable') for _ in markers]

    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not np.isfinite(norm) or norm < 1e-9:
        return [ConcentricBoltPose(None, None, None, 'unavailable') for _ in markers]
    normal = normal / norm
    if normal[2] > 0.0:
        normal = -normal

    min_valid = int(cfg.get('min_valid_depth_px', 8))
    per_marker_offsets = []
    for marker_index, marker in enumerate(markers):
        valid = (marker.depth_mask > 0) & (depth_image > 0)
        ys, xs = np.where(valid)
        if xs.size < min_valid:
            continue
        raw_depths = depth_image[ys, xs].astype(np.float64) * depth_scale
        lo, hi = np.percentile(raw_depths, [10.0, 90.0])
        keep = (raw_depths >= lo) & (raw_depths <= hi)
        xs = xs[keep]
        ys = ys[keep]
        depths = raw_depths[keep]
        if depths.size < min_valid:
            continue
        x = (xs.astype(np.float64) - intrin.cx) / intrin.fx * depths
        y = (ys.astype(np.float64) - intrin.cy) / intrin.fy * depths
        points = np.column_stack([x, y, depths])
        offsets = points @ normal
        per_marker_offsets.append((marker_index, float(np.median(offsets))))

    if not per_marker_offsets:
        return [ConcentricBoltPose(None, None, None, 'no_valid_depth') for _ in markers]

    values = np.array([item[1] for item in per_marker_offsets], dtype=np.float64)
    initial_offset = float(np.median(values))
    max_residual = float(cfg.get('max_plane_offset_residual_m', 0.008))
    inliers = values[np.abs(values - initial_offset) <= max_residual]
    plane_offset = float(np.median(inliers if inliers.size else values))
    source = ('shared_plane' if len(per_marker_offsets) >= 2
              else 'single_depth_plane')

    poses = []
    for marker in markers:
        center_ray = _pixel_ray(intrin, marker.center)
        denominator = float(np.dot(normal, center_ray))
        if abs(denominator) < 1e-8:
            poses.append(ConcentricBoltPose(None, None, plane_offset, 'parallel_ray'))
            continue
        center_point = center_ray * (plane_offset / denominator)

        direction = None
        if marker.gap_point is not None:
            gap_ray = _pixel_ray(intrin, marker.gap_point)
            gap_denominator = float(np.dot(normal, gap_ray))
            if abs(gap_denominator) >= 1e-8:
                gap_point = gap_ray * (plane_offset / gap_denominator)
                direction = gap_point - center_point
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1e-9:
                    direction = direction / direction_norm
                else:
                    direction = None
        poses.append(ConcentricBoltPose(
            point_3d=center_point,
            direction_3d=direction,
            plane_offset=plane_offset,
            depth_source=source,
        ))
    return poses
