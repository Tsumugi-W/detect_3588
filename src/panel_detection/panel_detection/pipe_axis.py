from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class PipeAxisResult:
    leak_point: Tuple[int, int]
    roi: Tuple[int, int, int, int]
    angle_deg: float
    vector_xy: Tuple[float, float]
    segments: list


def fold_axis_angle(angle_rad: float) -> float:
    """Fold a directed line angle into the undirected axis range."""
    while angle_rad < -math.pi / 2:
        angle_rad += math.pi
    while angle_rad > math.pi / 2:
        angle_rad -= math.pi
    return angle_rad


def axial_angle_delta_deg(a_deg: float, b_deg: float) -> float:
    """Smallest angular difference for axes with 180 degree ambiguity."""
    delta = abs((a_deg - b_deg + 90.0) % 180.0 - 90.0)
    return float(delta)


def weighted_axial_mean(angles_rad: Sequence[float],
                        weights: Sequence[float]) -> float:
    if len(angles_rad) == 0:
        raise ValueError('angles_rad must not be empty')

    angles = np.asarray(angles_rad, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if angles.shape != weights_arr.shape:
        raise ValueError('angles_rad and weights must have the same shape')

    sin_sum = float(np.sum(weights_arr * np.sin(2.0 * angles)))
    cos_sum = float(np.sum(weights_arr * np.cos(2.0 * angles)))
    return fold_axis_angle(0.5 * math.atan2(sin_sum, cos_sum))


def estimate_pipe_axis_from_image(
    image: np.ndarray,
    leak_point: Tuple[int, int],
    roi_half_size: Tuple[int, int] = (170, 80),
    canny_thresholds: Tuple[int, int] = (40, 130),
    hough_threshold: int = 35,
    min_line_length: int = 55,
    max_line_gap: int = 12,
    max_line_distance: float = 50.0,
    angle_prior_deg: Optional[float] = None,
    angle_tolerance_deg: float = 45.0,
    consensus_tolerance_deg: float = 18.0,
) -> Optional[PipeAxisResult]:
    """
    Estimate a pipe axis from image edges around a simulated or measured leak.

    The result is a 2D image-plane axis. It has 180 degree ambiguity by design:
    both vector_xy and -vector_xy describe the same physical pipe axis.
    """
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    leak_x = int(round(leak_point[0]))
    leak_y = int(round(leak_point[1]))
    if not (0 <= leak_x < width and 0 <= leak_y < height):
        return None

    half_w = max(1, int(roi_half_size[0]))
    half_h = max(1, int(roi_half_size[1]))
    x1 = max(0, leak_x - half_w)
    x2 = min(width, leak_x + half_w)
    y1 = max(0, leak_y - half_h)
    y2 = min(height, leak_y + half_h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    point_x = leak_x - x1
    point_y = leak_y - y1
    segments = []
    angles = []
    weights = []
    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [float(v) for v in line]
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue

        distance = abs(
            dy * point_x - dx * point_y + lx2 * ly1 - ly2 * lx1
        ) / (length + 1e-6)
        if distance > max_line_distance:
            continue

        angle = fold_axis_angle(math.atan2(dy, dx))
        angle_deg = math.degrees(angle)
        if angle_prior_deg is not None:
            if axial_angle_delta_deg(angle_deg, angle_prior_deg) > angle_tolerance_deg:
                continue

        segment = {
            'segment': (
                int(round(lx1 + x1)),
                int(round(ly1 + y1)),
                int(round(lx2 + x1)),
                int(round(ly2 + y1)),
            ),
            'length': float(length),
            'angle_deg': float(angle_deg),
            'distance_px': float(distance),
        }
        segments.append(segment)
        angles.append(angle)
        weights.append(length)

    if not segments:
        return None

    if len(segments) >= 3 and consensus_tolerance_deg > 0.0:
        best_indices = list(range(len(segments)))
        best_score = -1.0
        for seed_angle in angles:
            seed_deg = math.degrees(seed_angle)
            indices = [
                idx for idx, angle in enumerate(angles)
                if axial_angle_delta_deg(math.degrees(angle), seed_deg)
                <= consensus_tolerance_deg
            ]
            score = sum(weights[idx] for idx in indices)
            if score > best_score:
                best_score = score
                best_indices = indices

        if len(best_indices) >= 2:
            segments = [segments[idx] for idx in best_indices]
            angles = [angles[idx] for idx in best_indices]
            weights = [weights[idx] for idx in best_indices]

    mean = weighted_axial_mean(angles, weights)
    vector = (math.cos(mean), math.sin(mean))
    segments.sort(key=lambda item: item['length'], reverse=True)
    return PipeAxisResult(
        leak_point=(leak_x, leak_y),
        roi=(x1, y1, x2, y2),
        angle_deg=float(math.degrees(mean)),
        vector_xy=(float(vector[0]), float(vector[1])),
        segments=segments,
    )


def draw_pipe_axis_result(image: np.ndarray,
                          result: PipeAxisResult,
                          output_size: int = 150) -> np.ndarray:
    vis = image.copy()
    x1, y1, x2, y2 = result.roi
    leak_x, leak_y = result.leak_point
    vec_x, vec_y = result.vector_xy

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
    for item in result.segments[:12]:
        sx1, sy1, sx2, sy2 = item['segment']
        cv2.line(vis, (sx1, sy1), (sx2, sy2), (255, 255, 0), 2)

    cv2.circle(vis, (leak_x, leak_y), 7, (0, 0, 255), -1)
    start = (
        int(round(leak_x - vec_x * output_size)),
        int(round(leak_y - vec_y * output_size)),
    )
    end = (
        int(round(leak_x + vec_x * output_size)),
        int(round(leak_y + vec_y * output_size)),
    )
    cv2.line(vis, start, (leak_x, leak_y), (255, 0, 255), 3)
    cv2.arrowedLine(
        vis,
        (leak_x, leak_y),
        end,
        (255, 0, 255),
        4,
        tipLength=0.18,
    )

    text = (
        f'pipe axis: {result.angle_deg:.1f} deg  '
        f'v=({vec_x:.3f},{vec_y:.3f})'
    )
    cv2.rectangle(vis, (20, 18), (min(vis.shape[1] - 1, 680), 82), (0, 0, 0), -1)
    cv2.putText(
        vis,
        text,
        (32, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        'simulated leak',
        (leak_x + 12, leak_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def result_to_dict(result: PipeAxisResult) -> dict:
    return {
        'leak_point_px': list(result.leak_point),
        'roi': list(result.roi),
        'pipe_axis_angle_deg_image_x': result.angle_deg,
        'pipe_axis_vector_image_xy': list(result.vector_xy),
        'num_segments': len(result.segments),
        'segments': result.segments,
    }
