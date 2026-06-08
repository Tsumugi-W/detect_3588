"""
CV post-processing for hex nut localization inside a YOLO detection.

The YOLO bbox center can land on the central screw/shaft.  This module looks
for the outer hexagonal nut contour and returns the operation point at that
outer contour center.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class NutLocalization:
    center: Tuple[float, float]
    angle: Optional[float]
    confidence: float
    crop_bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    depth_mask: np.ndarray


def _expanded_bbox(bbox, image_shape, expand_ratio=0.15):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * expand_ratio
    ex1 = max(0, int(round(x1 - pad)))
    ey1 = max(0, int(round(y1 - pad)))
    ex2 = min(w, int(round(x2 + pad)))
    ey2 = min(h, int(round(y2 + pad)))
    return ex1, ey1, ex2, ey2


def _normalize_hex_angle(angle):
    if angle is None:
        return None
    angle = float(angle) % 60.0
    if angle < 0:
        angle += 60.0
    return angle


def _regular_hex_contour(center, radius, angle_deg=0.0):
    # angle_deg is the dominant edge angle; vertices are offset by 30 degrees.
    angles = np.deg2rad(np.arange(6) * 60.0 + angle_deg + 30.0)
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _edge_angle(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 4:
        return None

    angles = []
    for i in range(len(pts)):
        p1 = pts[i].astype(np.float64)
        p2 = pts[(i + 1) % len(pts)].astype(np.float64)
        dx, dy = p2 - p1
        length = np.hypot(dx, dy)
        if length < 4.0:
            continue
        angle = np.degrees(np.arctan2(dy, dx)) % 180.0
        angles.append(angle % 60.0)
    if not angles:
        return None

    scale = 360.0 / 60.0
    rads = np.radians(np.array(angles, dtype=np.float64) * scale)
    mean = np.degrees(np.arctan2(np.sin(rads).sum(), np.cos(rads).sum())) / scale
    return _normalize_hex_angle(mean)


def _candidate_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Bright, low-saturation metal, plus edge support for darker bevels.
    metal = (((s_ch < 95) & (v_ch > 70)) | ((s_ch < 55) & (v_ch > 45))).astype(np.uint8) * 255
    metal = cv2.medianBlur(metal, 5)

    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(metal, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _score_contour(contour, roi_shape, bbox_center_roi):
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    h, w = roi_shape[:2]
    roi_area = float(h * w)
    if area < max(80.0, roi_area * 0.04) or area > roi_area * 0.92:
        return None

    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    if solidity < 0.72:
        return None

    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    vertices = len(approx)
    vertex_score = max(0.0, 1.0 - abs(vertices - 6) / 5.0)

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ = rect
    if rw <= 1 or rh <= 1:
        return None
    aspect = min(rw, rh) / max(rw, rh)
    if aspect < 0.45:
        return None

    dist = np.hypot(cx - bbox_center_roi[0], cy - bbox_center_roi[1])
    max_dist = max(8.0, min(w, h) * 0.35)
    center_score = max(0.0, 1.0 - dist / max_dist)

    circularity = 4.0 * np.pi * area / (peri * peri)
    non_circle_score = float(np.clip((0.95 - circularity) / 0.35, 0.0, 1.0))
    area_score = float(np.clip(area / (roi_area * 0.35), 0.0, 1.0))

    score = (
        0.30 * vertex_score +
        0.25 * aspect +
        0.20 * solidity +
        0.15 * center_score +
        0.10 * non_circle_score
    )
    score *= 0.65 + 0.35 * area_score
    return score, (cx, cy), vertices


def localize_nut(color_image: np.ndarray, bbox, expand_ratio=0.15) -> Optional[NutLocalization]:
    """
    Refine a nut detection to the outer hex contour center.

    Args:
        color_image: Full BGR frame.
        bbox: YOLO bbox (x1, y1, x2, y2) in full-frame pixel coordinates.
        expand_ratio: ROI expansion around the bbox to recover clipped hex edges.

    Returns:
        NutLocalization or None when no reliable outer contour is found.
    """
    if color_image is None or color_image.size == 0:
        return None

    x1, y1, x2, y2 = _expanded_bbox(bbox, color_image.shape, expand_ratio)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    roi = color_image[y1:y2, x1:x2]
    mask = _candidate_mask(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    bbox_center_roi = ((bx1 + bx2) * 0.5 - x1, (by1 + by2) * 0.5 - y1)

    best = None
    for contour in contours:
        scored = _score_contour(contour, roi.shape, bbox_center_roi)
        if scored is None:
            continue
        score, center_roi, _ = scored
        if best is None or score > best[0]:
            best = (score, contour, center_roi)

    if best is None:
        return None

    confidence, contour_roi, center_roi = best

    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    bbox_center = np.array([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5], dtype=np.float64)
    bbox_w = max(1.0, bx2 - bx1)
    bbox_h = max(1.0, by2 - by1)
    bbox_min = min(bbox_w, bbox_h)

    raw_center = np.array([center_roi[0] + x1, center_roi[1] + y1], dtype=np.float64)
    offset = raw_center - bbox_center
    offset_norm = float(np.linalg.norm(offset))
    max_offset = max(3.0, bbox_min * 0.22)
    if offset_norm > max_offset:
        raw_center = bbox_center + offset * (max_offset / max(offset_norm, 1e-6))
        confidence *= 0.85

    angle = _edge_angle(contour_roi)
    rect = cv2.minAreaRect(contour_roi)
    (_, _), (rw, rh), _ = rect
    raw_radius = 0.5 * min(max(rw, 1.0), max(rh, 1.0))
    radius = float(np.clip(raw_radius, bbox_min * 0.30, bbox_min * 0.55))

    contour_full = _regular_hex_contour(raw_center, radius, angle or 0.0)
    contour_full[:, 0, 0] = np.clip(contour_full[:, 0, 0], 0, color_image.shape[1] - 1)
    contour_full[:, 0, 1] = np.clip(contour_full[:, 0, 1], 0, color_image.shape[0] - 1)

    full_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(full_mask, [contour_full], -1, 255, -1)

    depth_mask = full_mask.copy()
    cx_full = float(raw_center[0])
    cy_full = float(raw_center[1])
    inner_radius = max(3, int(round(radius * 0.38)))
    cv2.circle(depth_mask, (int(round(cx_full)), int(round(cy_full))), inner_radius, 0, -1)
    depth_mask = cv2.erode(depth_mask, np.ones((3, 3), np.uint8), iterations=1)

    return NutLocalization(
        center=(cx_full, cy_full),
        angle=angle,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        crop_bbox=(x1, y1, x2, y2),
        contour=contour_full,
        mask=full_mask,
        depth_mask=depth_mask,
    )
