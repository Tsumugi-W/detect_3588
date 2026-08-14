import math

import cv2
import numpy as np

from .depth_utils import deproject_pixels_to_points, fit_plane_ransac_quality


def angle_between_normals_deg(normal_a, normal_b):
    a = np.asarray(normal_a, dtype=np.float64)
    b = np.asarray(normal_b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    a = a / na
    b = b / nb
    if np.dot(a, b) < 0:
        b = -b
    return float(math.degrees(math.acos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _get_apriltag_dictionary(name):
    if not hasattr(cv2, 'aruco'):
        return None
    dictionary_id = getattr(cv2.aruco, str(name), None)
    if dictionary_id is None:
        dictionary_id = getattr(cv2.aruco, 'DICT_APRILTAG_36h11', None)
    if dictionary_id is None:
        return None
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def _detect_aruco_markers(gray, dictionary):
    """Detect markers through either the modern or legacy OpenCV API."""
    aruco = cv2.aruco
    if hasattr(aruco, 'ArucoDetector'):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        result = detector.detectMarkers(gray)
        return _normalize_aruco_result(result)

    if hasattr(aruco, 'DetectorParameters_create'):
        params = aruco.DetectorParameters_create()
    elif hasattr(aruco, 'DetectorParameters'):
        params = aruco.DetectorParameters()
    else:
        return (), None, ()

    if not hasattr(aruco, 'detectMarkers'):
        return (), None, ()
    result = aruco.detectMarkers(gray, dictionary, parameters=params)
    return _normalize_aruco_result(result)


def _normalize_aruco_result(result):
    """Keep marker IDs compatible across OpenCV versions."""
    corners, ids, rejected = result
    if ids is not None:
        ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    return corners, ids, rejected


def _detect_tag_like_fallback_corners(gray, cfg=None):
    """Fallback for printed tag boards that are visible but not decodable."""
    cfg = cfg or {}
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = gray.shape[:2]
    white_thresh = int(cfg.get('fallback_white_thresh', 150))
    min_white_ratio = float(cfg.get('fallback_min_white_border_ratio', 0.45))
    candidates = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < 1500 or bw < 28 or bh < 28:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        aspect = float(bw) / float(max(1, bh))
        density = float(area) / float(max(1, bw * bh))
        if not (0.55 <= aspect <= 1.8 and density >= 0.18):
            continue

        pad = max(8, int(round(max(bw, bh) * 0.25)))
        ox1 = max(0, x - pad)
        oy1 = max(0, y - pad)
        ox2 = min(w, x + bw + pad)
        oy2 = min(h, y + bh + pad)
        outer = np.zeros((h, w), dtype=np.uint8)
        outer[oy1:oy2, ox1:ox2] = 1
        inner = np.zeros((h, w), dtype=np.uint8)
        inner[y:y + bh, x:x + bw] = 1
        ring = (outer > 0) & (inner == 0)
        ring_count = int(np.count_nonzero(ring))
        if ring_count < 100:
            continue
        white_ratio = float(np.count_nonzero(gray[ring] >= white_thresh)) / float(ring_count)
        if white_ratio < min_white_ratio:
            continue

        ys, xs = np.where(labels == label)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        corners = cv2.boxPoints(rect).astype(np.float32)
        square_score = 1.0 - min(1.0, abs(math.log(max(aspect, 1e-6))))
        score = float(area) * max(0.1, white_ratio) * max(0.1, square_score)
        candidates.append((score, corners))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidates[0][1].reshape(1, 4, 2)]


def detect_apriltag_reference_axis(color_image, depth_image, intrin,
                                   depth_scale=0.001, cfg=None):
    cfg = cfg or {}
    if not cfg.get('enable', True):
        return None
    if color_image is None or depth_image is None or intrin is None:
        return None
    if not hasattr(cv2, 'aruco'):
        return None

    dictionary = _get_apriltag_dictionary(
        cfg.get('dictionary', 'DICT_APRILTAG_36h11'))
    if dictionary is None:
        return None
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detect_aruco_markers(gray, dictionary)
    if ids is None or len(corners) == 0:
        if not cfg.get('fallback_enable', True):
            return None
        corners = _detect_tag_like_fallback_corners(gray, cfg)
        ids = np.array([[-1]], dtype=np.int32) if corners else None
        source = 'tag_like_depth_plane'
    else:
        source = 'apriltag_depth_plane'
    if ids is None or len(corners) == 0:
        return None

    h, w = depth_image.shape[:2]
    marker_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    candidates = []
    for idx, corner in enumerate(corners):
        if idx >= marker_ids.size:
            continue
        pts = corner.reshape(4, 2).astype(np.float32)
        area = abs(float(cv2.contourArea(pts)))
        if area < 100.0:
            continue

        center = np.mean(pts, axis=0)
        margin = float(cfg.get('border_margin_ratio', 0.12))
        inner_pts = center + (pts - center) * max(0.1, 1.0 - margin)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(inner_pts).astype(np.int32), 255)
        valid = (mask > 0) & (depth_image > 0)
        ys, xs = np.where(valid)
        min_points = int(cfg.get('min_points', 80))
        if len(xs) < min_points:
            continue

        step = max(1, int(cfg.get('sample_stride', 3)))
        xs_sample = xs[::step]
        ys_sample = ys[::step]
        if len(xs_sample) < min_points:
            xs_sample = xs
            ys_sample = ys
        if len(xs_sample) < min_points:
            continue

        depths = depth_image[ys_sample, xs_sample].astype(np.float64) * depth_scale
        pixels = np.column_stack([xs_sample, ys_sample])
        points_3d = deproject_pixels_to_points(intrin, pixels, depths)
        fit = fit_plane_ransac_quality(
            points_3d,
            min_points=min(min_points, len(points_3d)),
            ransac_iter=120,
            ransac_thresh=float(cfg.get('ransac_thresh', 0.008)),
            random_seed=idx,
        )
        if fit is None:
            continue
        normal, centroid, inlier_count, inlier_ratio, rms_error = fit
        if float(inlier_ratio) < float(cfg.get('min_inlier_ratio', 0.45)):
            continue
        if float(rms_error) > float(cfg.get('max_rms_m', 0.012)):
            continue
        if normal[2] > 0:
            normal = -normal
        if abs(float(normal[2])) < float(cfg.get('min_abs_z', 0.50)):
            continue
        if source == 'tag_like_depth_plane':
            if int(inlier_count) < int(cfg.get('fallback_min_point_count', 2000)):
                continue
            if float(inlier_ratio) < float(cfg.get('fallback_min_inlier_ratio', 0.75)):
                continue
        candidates.append({
            'tag_id': int(marker_ids[idx]),
            'source': source,
            'corners': pts,
            'normal': normal,
            'centroid': centroid,
            'point_count': int(inlier_count),
            'inlier_ratio': float(inlier_ratio),
            'rms_error': float(rms_error),
            'area': area,
        })

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item['point_count'], item['area']), reverse=True)
    return candidates[0]
