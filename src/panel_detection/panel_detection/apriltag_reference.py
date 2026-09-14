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


def estimate_apriltag_pnp_pose(corners, intrin):
    """Estimate a metric-size-independent Tag orientation from four corners.

    The Tag origin is intentionally not taken from ``translation_vector``:
    unit-square object coordinates make that translation scale-ambiguous.
    Callers combine these axes with a metric origin estimated from depth.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    object_points = np.array([
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ], dtype=np.float32)
    camera_matrix = np.array([
        [intrin.fx, 0.0, intrin.cx],
        [0.0, intrin.fy, intrin.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    distortion = np.asarray(getattr(intrin, 'coeffs', []), dtype=np.float64)
    flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, pts, camera_matrix, distortion, flags=flag)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    x_axis = rotation[:, 0].astype(np.float64)
    normal = rotation[:, 2].astype(np.float64)
    if normal[2] > 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    x_axis -= np.dot(x_axis, normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-9:
        return None
    x_axis /= x_norm
    # Keep a right-handed frame after forcing Z to face the camera.
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    projected, _ = cv2.projectPoints(
        object_points, rotation_vector, translation_vector,
        camera_matrix, distortion)
    reprojection_error = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(4, 2) - pts) ** 2, axis=1))))
    return {
        'normal': normal,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'reprojection_error_px': reprojection_error,
    }


def estimate_apriltag_pnp_normal(corners, intrin):
    """Backward-compatible normal-only view of the full PnP pose."""
    pose = estimate_apriltag_pnp_pose(corners, intrin)
    if pose is None:
        return None
    return pose['normal'], pose['reprojection_error_px']


def _intersect_pixel_with_plane(intrin, pixel, normal, point_on_plane):
    ray = np.array([
        (float(pixel[0]) - intrin.cx) / intrin.fx,
        (float(pixel[1]) - intrin.cy) / intrin.fy,
        1.0,
    ], dtype=np.float64)
    plane_normal = np.asarray(normal, dtype=np.float64)
    plane_point = np.asarray(point_on_plane, dtype=np.float64)
    denominator = float(np.dot(plane_normal, ray))
    if abs(denominator) < 1e-9:
        return None
    distance = float(np.dot(plane_normal, plane_point)) / denominator
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    return ray * distance


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
    decoded = ids is not None and len(corners) > 0
    if ids is None or len(corners) == 0:
        if not cfg.get('fallback_enable', True):
            return None
        corners = _detect_tag_like_fallback_corners(gray, cfg)
        ids = np.array([[-1]], dtype=np.int32) if corners else None
        source = 'tag_like_depth_plane'
    else:
        source = 'apriltag_pnp'
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

        pnp_pose = estimate_apriltag_pnp_pose(pts, intrin) if decoded else None
        if decoded and pnp_pose is None:
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
        depth_normal, centroid, inlier_count, inlier_ratio, rms_error = fit
        if float(inlier_ratio) < float(cfg.get('min_inlier_ratio', 0.45)):
            continue
        if float(rms_error) > float(cfg.get('max_rms_m', 0.012)):
            continue
        normal = pnp_pose['normal'] if pnp_pose is not None else depth_normal
        reprojection_error = (
            pnp_pose['reprojection_error_px']
            if pnp_pose is not None else None)
        if normal[2] > 0:
            normal = -normal
        if abs(float(normal[2])) < float(cfg.get('min_abs_z', 0.50)):
            continue
        if source == 'tag_like_depth_plane':
            if int(inlier_count) < int(cfg.get('fallback_min_point_count', 2000)):
                continue
            if float(inlier_ratio) < float(cfg.get('fallback_min_inlier_ratio', 0.75)):
                continue
        origin = _intersect_pixel_with_plane(
            intrin, center, depth_normal, centroid)
        if origin is None:
            origin = centroid
        candidates.append({
            'tag_id': int(marker_ids[idx]),
            'source': source,
            'corners': pts,
            'normal': normal,
            'centroid': centroid,
            'origin': origin,
            'x_axis': None if pnp_pose is None else pnp_pose['x_axis'],
            'y_axis': None if pnp_pose is None else pnp_pose['y_axis'],
            'point_count': int(inlier_count),
            'inlier_ratio': float(inlier_ratio),
            'rms_error': float(rms_error),
            'reprojection_error_px': reprojection_error,
            'area': area,
        })

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item['point_count'], item['area']), reverse=True)
    return candidates[0]
