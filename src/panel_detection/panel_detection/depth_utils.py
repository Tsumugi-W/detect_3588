"""
公共深度处理工具函数

从 rstest.py / rstest3.py 提取的与相机 SDK 无关的纯数学/图像处理函数。
适用于 RealSense D435i / Orbbec Gemini 336 等任意深度相机。
"""
import numpy as np
import cv2

from .camera.base import CameraIntrinsics


def deproject_pixel_to_point(intrin: CameraIntrinsics,
                             pixel: tuple, depth: float) -> list:
    """
    将像素坐标 + 深度 反投影为 3D 相机坐标（pinhole 模型）

    等价于 rs2_deproject_pixel_to_point()，但不依赖任何 SDK。

    Args:
        intrin: 相机内参（通常用深度相机内参）
        pixel: (u, v) 像素坐标
        depth: 深度值（米）

    Returns:
        [x, y, z] 相机坐标系下的 3D 点（米）
    """
    x = (pixel[0] - intrin.cx) / intrin.fx
    y = (pixel[1] - intrin.cy) / intrin.fy
    return [depth * x, depth * y, depth]


def deproject_pixels_to_points(intrin: CameraIntrinsics,
                               pixels: np.ndarray,
                               depths: np.ndarray) -> np.ndarray:
    """
    向量化批量反投影：将 N 个像素坐标 + 深度转为 3D 点云

    Args:
        intrin: 相机内参
        pixels: (N, 2) 数组，每行 [u, v]
        depths: (N,) 数组，深度值（米）

    Returns:
        (N, 3) 数组，每行 [x, y, z]
    """
    us = pixels[:, 0].astype(np.float64)
    vs = pixels[:, 1].astype(np.float64)
    x = (us - intrin.cx) / intrin.fx * depths
    y = (vs - intrin.cy) / intrin.fy * depths
    return np.column_stack([x, y, depths])


def undistort_pixel(u, v, intrin: CameraIntrinsics, iterations=5):
    """
    将畸变的像素坐标转换为理想坐标（去畸变）

    使用 Brown-Conrady 畸变模型的迭代反演（牛顿法）

    Args:
        u, v: 实际像素坐标
        intrin: 统一内参对象
        iterations: 迭代次数（通常 3-5 次收敛）

    Returns:
        (u_ideal, v_ideal): 去畸变后的理想像素坐标
    """
    fx, fy = intrin.fx, intrin.fy
    cx, cy = intrin.cx, intrin.cy

    coeffs = intrin.coeffs
    k1 = coeffs[0] if len(coeffs) > 0 else 0
    k2 = coeffs[1] if len(coeffs) > 1 else 0
    p1 = coeffs[2] if len(coeffs) > 2 else 0
    p2 = coeffs[3] if len(coeffs) > 3 else 0
    k3 = coeffs[4] if len(coeffs) > 4 else 0

    # 转换为归一化坐标
    x = (u - cx) / fx
    y = (v - cy) / fy

    # 迭代求解理想坐标
    for _ in range(iterations):
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        dy = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        x_new = (x - dx) / radial
        y_new = (y - dy) / radial

        if abs(x_new - x) < 1e-6 and abs(y_new - y) < 1e-6:
            break
        x, y = x_new, y_new

    return x * fx + cx, y * fy + cy


def filter_depth(depth_image: np.ndarray, method='bilateral', kernel_size=5):
    """
    对深度图进行滤波降噪

    Args:
        depth_image: 深度图 numpy 数组 (H, W) uint16
        method: 滤波方法 'bilateral' | 'median' | 'gaussian'
        kernel_size: 核大小（必须为奇数）

    Returns:
        滤波后的深度图 numpy 数组
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(
            depth_image.astype(np.float32), kernel_size, 75, 75
        ).astype(depth_image.dtype)
    elif method == 'median':
        return cv2.medianBlur(depth_image, kernel_size)
    elif method == 'gaussian':
        return cv2.GaussianBlur(depth_image, (kernel_size, kernel_size), 0)
    return depth_image


def get_robust_depth(depth_image, ux, uy, sample_radius=3, depth_scale=0.001):
    """
    从中心点周围采样多个深度值，取中位数

    Args:
        depth_image: 深度图 numpy 数组
        ux, uy: 中心像素坐标
        sample_radius: 采样半径（像素）
        depth_scale: 深度缩放因子

    Returns:
        鲁棒的深度值（米）
    """
    h, w = depth_image.shape
    x1 = max(0, ux - sample_radius)
    x2 = min(w, ux + sample_radius + 1)
    y1 = max(0, uy - sample_radius)
    y2 = min(h, uy + sample_radius + 1)
    patch = depth_image[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * depth_scale


def get_bbox_robust_depth(depth_image, bbox, depth_scale=0.001,
                          center_ratio=0.45, min_valid=6,
                          quantile=0.5):
    """
    从检测框中心区域估计目标表面深度。

    比单点 3x3 更稳：按钮/旋钮表面常有深度空洞、反光和边缘混入；
    只取 bbox 中央区域并用分位数统计，可减少跳变和背景污染。

    Args:
        depth_image: 深度图 numpy 数组
        bbox: (x1, y1, x2, y2)
        depth_scale: 深度缩放因子
        center_ratio: 采样区域占 bbox 宽高比例
        min_valid: 最少有效深度点数量
        quantile: 分位数，0.5 为中位数；更小更偏向近处表面

    Returns:
        深度值（米），无有效深度返回 0.0
    """
    h, w = depth_image.shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    half_w = max(2.0, bw * center_ratio * 0.5)
    half_h = max(2.0, bh * center_ratio * 0.5)

    sx1 = max(0, int(round(cx - half_w)))
    sx2 = min(w, int(round(cx + half_w + 1)))
    sy1 = max(0, int(round(cy - half_h)))
    sy2 = min(h, int(round(cy + half_h + 1)))
    if sx1 >= sx2 or sy1 >= sy2:
        return 0.0

    patch = depth_image[sy1:sy2, sx1:sx2]
    valid = patch[patch > 0].astype(np.float64)
    if valid.size < min_valid:
        return get_robust_depth(depth_image, int(round(cx)), int(round(cy)),
                                sample_radius=3, depth_scale=depth_scale)

    lo, hi = np.percentile(valid, [10, 90])
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    if trimmed.size >= min_valid:
        valid = trimmed

    q = float(np.clip(quantile, 0.05, 0.95))
    return float(np.percentile(valid, q * 100.0)) * depth_scale


def get_masked_robust_depth(depth_image, mask, depth_scale=0.001,
                            min_valid=12, trim_percentiles=(10, 90),
                            quantile=0.5):
    """
    Estimate depth from pixels selected by a binary mask.

    This is used for nuts so depth comes from the outer hex ring instead of
    the central screw/shaft surface.
    """
    if depth_image is None or mask is None:
        return 0.0
    if depth_image.shape[:2] != mask.shape[:2]:
        return 0.0

    valid_mask = (mask > 0) & (depth_image > 0)
    valid = depth_image[valid_mask].astype(np.float64)
    if valid.size < min_valid:
        return 0.0

    lo_p, hi_p = trim_percentiles
    lo, hi = np.percentile(valid, [lo_p, hi_p])
    trimmed = valid[(valid >= lo) & (valid <= hi)]
    if trimmed.size >= min_valid:
        valid = trimmed

    q = float(np.clip(quantile, 0.05, 0.95))
    return float(np.percentile(valid, q * 100.0)) * depth_scale


def fit_plane_ransac_quality(points_3d, min_points=50, ransac_iter=100,
                             ransac_thresh=0.005, random_seed=0):
    """
    RANSAC + SVD 平面拟合

    Args:
        points_3d: (N, 3) 点云数组
        min_points: 最少内点数
        ransac_iter: RANSAC 迭代次数
        ransac_thresh: 内点距离阈值（米）
        random_seed: 固定随机种子，避免同一帧点云多次拟合结果抖动

    Returns:
        (normal, centroid, inlier_count, inlier_ratio, rms_error) 或 None
    """
    if len(points_3d) < min_points:
        return None

    best_normal = None
    best_inliers = 0
    best_inlier_mask = None

    rng = np.random.default_rng(random_seed)
    for _ in range(ransac_iter):
        idx = rng.choice(len(points_3d), 3, replace=False)
        p0, p1, p2 = points_3d[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len
        dists = np.abs((points_3d - p0) @ normal)
        inlier_mask = dists < ransac_thresh
        inlier_count = np.sum(inlier_mask)
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = normal
            best_inlier_mask = inlier_mask

    if best_normal is None or best_inliers < min_points:
        return None

    # SVD 精化
    inlier_pts = points_3d[best_inlier_mask]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = Vt[2]
    if normal[2] > 0:
        normal = -normal
    residuals = np.abs((inlier_pts - centroid) @ normal)
    rms_error = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else 0.0
    inlier_ratio = float(best_inliers) / float(len(points_3d))
    return normal, centroid, int(best_inliers), inlier_ratio, rms_error


def fit_plane_ransac(points_3d, min_points=50, ransac_iter=100,
                     ransac_thresh=0.005, random_seed=0):
    """
    RANSAC + SVD 平面拟合

    Args:
        points_3d: (N, 3) 点云数组
        min_points: 最少内点数
        ransac_iter: RANSAC 迭代次数
        ransac_thresh: 内点距离阈值（米）
        random_seed: 固定随机种子，避免同一帧点云多次拟合结果抖动

    Returns:
        (normal, centroid) 或 None
    """
    result = fit_plane_ransac_quality(
        points_3d,
        min_points=min_points,
        ransac_iter=ransac_iter,
        ransac_thresh=ransac_thresh,
        random_seed=random_seed,
    )
    if result is None:
        return None
    normal, centroid, _, _, _ = result
    return normal, centroid


def fit_plane_svd(points_3d, min_points=3):
    """
    Fit a plane with SVD for a small deterministic set of points.

    This is useful for a local fastener group where only 3-4 visible
    nut/bolt centers may be available.
    """
    points = np.asarray(points_3d, dtype=np.float64)
    if len(points) < min_points:
        return None
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    try:
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(s_vals) < 3 or s_vals[1] < 1e-6:
        return None
    normal = vt[2]
    if normal[2] > 0:
        normal = -normal
    return normal, centroid


def estimate_fastener_group_axis_direction(target_xy, candidates,
                                           max_distance_px=360.0,
                                           min_points=3,
                                           max_points=6,
                                           merge_distance_px=12.0,
                                           min_abs_z=0.45):
    """
    Estimate axis direction from nearby fastener/valve 3D points.

    Args:
        target_xy: (x, y) center of the target in image pixels.
        candidates: iterable of dicts with keys:
            center_xy: (x, y), point_3d: (x, y, z), class_name: str
        max_distance_px: candidate neighborhood radius in the image.
        min_points: minimum visible neighboring objects required.
        max_points: use nearest N points to avoid crossing to another plane.

    Returns:
        (normal, centroid, point_count) or None.
    """
    tx, ty = [float(v) for v in target_xy]
    usable = []
    for cand in candidates:
        point = cand.get('point_3d')
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0:
            continue
        cx, cy = cand.get('center_xy', (None, None))
        if cx is None or cy is None:
            continue
        cx = float(cx)
        cy = float(cy)
        dist = float(np.hypot(cx - tx, cy - ty))
        if dist <= max_distance_px:
            frame = int(cand.get('frame', 0))
            usable.append((dist, -frame, (cx, cy), point))

    if len(usable) < min_points:
        return None

    usable.sort(key=lambda item: (item[0], item[1]))
    merged = []
    merged_centers = []
    for _, _, center_xy, point in usable:
        if any(np.hypot(center_xy[0] - existing[0],
                        center_xy[1] - existing[1]) < merge_distance_px
               for existing in merged_centers):
            continue
        merged_centers.append(center_xy)
        merged.append(point)
        if len(merged) >= max_points:
            break

    if len(merged) < min_points:
        return None

    points = np.array(merged, dtype=np.float64)
    result = fit_plane_svd(points, min_points=min_points)
    if result is None:
        return None
    normal, centroid = result
    if abs(float(normal[2])) < float(min_abs_z):
        return None
    return normal, centroid, int(len(points))


def estimate_fastener_line_constrained_axis(target_xy, candidates, base_normal,
                                            max_distance_px=360.0,
                                            merge_distance_px=12.0,
                                            min_abs_z=0.45):
    """
    Use two nearby fastener points to constrain a candidate axis normal.

    With only two visible coplanar objects, a plane is underdetermined.  The
    line through those two object points still lies in the mounting plane, so
    the mounting-plane normal must be perpendicular to that line.  This helper
    projects an existing normal estimate onto the subspace perpendicular to the
    fastener line.
    """
    if base_normal is None:
        return None

    tx, ty = [float(v) for v in target_xy]
    usable = []
    for cand in candidates:
        point = cand.get('point_3d')
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0:
            continue
        cx, cy = cand.get('center_xy', (None, None))
        if cx is None or cy is None:
            continue
        cx = float(cx)
        cy = float(cy)
        dist = float(np.hypot(cx - tx, cy - ty))
        if dist <= max_distance_px:
            frame = int(cand.get('frame', 0))
            usable.append((dist, -frame, (cx, cy), point))

    if len(usable) < 2:
        return None

    usable.sort(key=lambda item: (item[0], item[1]))
    merged_points = []
    merged_centers = []
    for _, _, center_xy, point in usable:
        if any(np.hypot(center_xy[0] - existing[0],
                        center_xy[1] - existing[1]) < merge_distance_px
               for existing in merged_centers):
            continue
        merged_centers.append(center_xy)
        merged_points.append(point)
        if len(merged_points) >= 2:
            break

    if len(merged_points) < 2:
        return None

    line = merged_points[1] - merged_points[0]
    line_norm = np.linalg.norm(line)
    if line_norm < 1e-6:
        return None
    line = line / line_norm

    normal = np.asarray(base_normal, dtype=np.float64)
    normal = normal - np.dot(normal, line) * line
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-6:
        return None
    normal = normal / normal_norm
    if normal[2] > 0:
        normal = -normal
    if abs(float(normal[2])) < float(min_abs_z):
        return None

    centroid = np.mean(np.asarray(merged_points, dtype=np.float64), axis=0)
    return normal, centroid, int(len(merged_points))


def _bbox_mask(shape, bbox, shrink_ratio=0.08):
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    sx1 = max(0, int(round(x1 + bw * shrink_ratio)))
    sy1 = max(0, int(round(y1 + bh * shrink_ratio)))
    sx2 = min(w, int(round(x2 - bw * shrink_ratio)))
    sy2 = min(h, int(round(y2 - bh * shrink_ratio)))
    mask = np.zeros((h, w), dtype=np.uint8)
    if sx1 < sx2 and sy1 < sy2:
        mask[sy1:sy2, sx1:sx2] = 255
    return mask


def _valve_annulus_mask(shape, bbox):
    """
    Build a depth mask for hollow valve-like parts.

    The center of a valve bbox often sees through to the panel/background, so
    the mask keeps an elliptical outer ring and removes the central hole.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    outer_axes = (max(2, int(round(bw * 0.46))),
                  max(2, int(round(bh * 0.46))))
    inner_axes = (max(1, int(round(bw * 0.22))),
                  max(1, int(round(bh * 0.22))))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), outer_axes, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx, cy), inner_axes, 0, 0, 360, 0, -1)
    return mask


def estimate_object_axis_direction(depth_image, depth_intrin, bbox,
                                   depth_scale=0.001, mask=None,
                                   object_class='',
                                   sample_stride=2,
                                   min_points=24):
    """
    Estimate a target axis direction from local depth points.

    For bolt/nut/valve, the mechanical axis is approximated by the normal of
    the visible front face.  Hollow valves need special treatment: the center
    of the bbox may contain background depth, so only an annular mask is used
    and deeper outliers are trimmed away.

    Returns:
        (normal, centroid, inlier_count) or None.
        normal points toward the camera (negative z in the camera optical frame).
    """
    if depth_image is None or depth_intrin is None:
        return None

    h, w = depth_image.shape[:2]
    if mask is None:
        if object_class == 'valve':
            mask = _valve_annulus_mask((h, w), bbox)
        else:
            mask = _bbox_mask((h, w), bbox)
    elif mask.shape[:2] != depth_image.shape[:2]:
        return None

    valid = (mask > 0) & (depth_image > 0)
    raw_depths = depth_image[valid].astype(np.float64)
    if raw_depths.size < min_points:
        return None

    if object_class == 'valve':
        # Keep the nearer depth cluster.  The center opening and far panel
        # pixels otherwise pull the fitted plane away from the valve face.
        lo, hi = np.percentile(raw_depths, [3, 70])
    else:
        lo, hi = np.percentile(raw_depths, [5, 95])
    depth_band = valid & (depth_image >= lo) & (depth_image <= hi)

    ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        ys, xs = np.where(valid)
    if len(xs) < min_points:
        return None

    step = max(1, int(sample_stride))
    ys = ys[::step]
    xs = xs[::step]
    if len(xs) < min_points:
        ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        return None

    depths = depth_image[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([xs, ys])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    result = fit_plane_ransac(
        points_3d,
        min_points=min(min_points, len(points_3d)),
        ransac_iter=80,
        ransac_thresh=0.006,
        random_seed=0,
    )
    if result is None:
        return None
    normal, centroid = result
    return normal, centroid, int(len(points_3d))


def estimate_valve_wheel_axis_direction(depth_image, depth_intrin, bbox,
                                        depth_scale=0.001,
                                        sample_stride=1,
                                        min_points=36,
                                        min_abs_z=0.55):
    """
    Estimate a hollow valve wheel axis from the wheel itself.

    Valve wheels are hollow, so surrounding-plane fitting is unreliable.  This
    samples only the wheel's annular region inside the detection bbox, removes
    the central opening, keeps the near depth cluster, and fits the wheel plane.
    """
    if depth_image is None or depth_intrin is None:
        return None

    h, w = depth_image.shape[:2]
    mask = _valve_annulus_mask((h, w), bbox)
    valid = (mask > 0) & (depth_image > 0)
    raw_depths = depth_image[valid].astype(np.float64)
    if raw_depths.size < min_points:
        return None

    lo, hi = np.percentile(raw_depths, [2, 55])
    depth_band = valid & (depth_image >= lo) & (depth_image <= hi)
    ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        lo, hi = np.percentile(raw_depths, [2, 70])
        depth_band = valid & (depth_image >= lo) & (depth_image <= hi)
        ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        return None

    step = max(1, int(sample_stride))
    ys = ys[::step]
    xs = xs[::step]
    if len(xs) < min_points:
        ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        return None

    depths = depth_image[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([xs, ys])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    result = fit_plane_ransac(
        points_3d,
        min_points=min(min_points, len(points_3d)),
        ransac_iter=120,
        ransac_thresh=0.006,
        random_seed=0,
    )
    if result is None:
        return None
    normal, centroid = result
    if abs(float(normal[2])) < float(min_abs_z):
        return None
    return normal, centroid, int(len(points_3d))


def estimate_mounting_plane_axis_direction(depth_image, depth_intrin, bbox,
                                           all_bboxes=None,
                                           depth_scale=0.001,
                                           expand_ratio=1.4,
                                           object_margin_ratio=0.12,
                                           sample_stride=3,
                                           min_points=45):
    """
    Estimate an object's axis from its local mounting plane.

    Bolts, nuts, and valves are assumed to be mounted perpendicular to a local
    support plane.  A scene may contain multiple support planes, so this samples
    only an expanded region around one target, removes all detected object boxes
    from that region, and fits the remaining local surface.

    Returns:
        (normal, centroid, point_count) or None.
    """
    if depth_image is None or depth_intrin is None:
        return None

    h, w = depth_image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * float(expand_ratio)

    rx1 = max(0, int(round(x1 - pad)))
    ry1 = max(0, int(round(y1 - pad)))
    rx2 = min(w, int(round(x2 + pad)))
    ry2 = min(h, int(round(y2 + pad)))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[ry1:ry2, rx1:rx2] = True

    boxes = all_bboxes or [bbox]
    for b in boxes:
        bx1, by1, bx2, by2 = [float(v) for v in b]
        bbw = max(1.0, bx2 - bx1)
        bbh = max(1.0, by2 - by1)
        margin = max(bbw, bbh) * float(object_margin_ratio)
        ex1 = max(0, int(round(bx1 - margin)))
        ey1 = max(0, int(round(by1 - margin)))
        ex2 = min(w, int(round(bx2 + margin)))
        ey2 = min(h, int(round(by2 + margin)))
        mask[ey1:ey2, ex1:ex2] = False

    valid = mask & (depth_image > 0)
    raw_depths = depth_image[valid].astype(np.float64)
    if raw_depths.size < min_points:
        return None

    lo, hi = np.percentile(raw_depths, [4, 96])
    depth_band = valid & (depth_image >= lo) & (depth_image <= hi)
    ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        ys, xs = np.where(valid)
    if len(xs) < min_points:
        return None

    step = max(1, int(sample_stride))
    ys = ys[::step]
    xs = xs[::step]
    if len(xs) < min_points:
        ys, xs = np.where(depth_band)
    if len(xs) < min_points:
        return None

    depths = depth_image[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([xs, ys])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    result = fit_plane_ransac(
        points_3d,
        min_points=min(min_points, len(points_3d)),
        ransac_iter=120,
        ransac_thresh=0.008,
        random_seed=0,
    )
    if result is None:
        return None
    normal, centroid = result
    return normal, centroid, int(len(points_3d))


def estimate_fastener_patch_axis_direction(depth_image, depth_intrin, bbox,
                                           all_bboxes=None,
                                           depth_scale=0.001,
                                           expand_ratio=1.8,
                                           object_margin_ratio=0.10,
                                           seed_ring_ratio=0.22,
                                           depth_window_m=0.08,
                                           sample_stride=2,
                                           min_points=45,
                                           min_inlier_ratio=0.55,
                                           max_rms_m=0.010,
                                           min_abs_z=0.45):
    """
    Estimate bolt/nut axis from a depth-continuous local mounting patch.

    This is stricter than estimate_mounting_plane_axis_direction: it seeds from
    a thin ring just outside the target bbox, keeps only a depth-continuous
    connected component, and rejects low-quality plane fits.
    """
    if depth_image is None or depth_intrin is None:
        return None

    h, w = depth_image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * float(expand_ratio)

    rx1 = max(0, int(round(x1 - pad)))
    ry1 = max(0, int(round(y1 - pad)))
    rx2 = min(w, int(round(x2 + pad)))
    ry2 = min(h, int(round(y2 + pad)))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None

    candidate = np.zeros((h, w), dtype=np.uint8)
    candidate[ry1:ry2, rx1:rx2] = 1

    boxes = all_bboxes or [bbox]
    for b in boxes:
        bx1, by1, bx2, by2 = [float(v) for v in b]
        bbw = max(1.0, bx2 - bx1)
        bbh = max(1.0, by2 - by1)
        margin = max(bbw, bbh) * float(object_margin_ratio)
        ex1 = max(0, int(round(bx1 - margin)))
        ey1 = max(0, int(round(by1 - margin)))
        ex2 = min(w, int(round(bx2 + margin)))
        ey2 = min(h, int(round(by2 + margin)))
        candidate[ey1:ey2, ex1:ex2] = 0

    ring_margin = max(2.0, max(bw, bh) * float(seed_ring_ratio))
    ox1 = max(0, int(round(x1 - ring_margin)))
    oy1 = max(0, int(round(y1 - ring_margin)))
    ox2 = min(w, int(round(x2 + ring_margin)))
    oy2 = min(h, int(round(y2 + ring_margin)))
    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(w, int(round(x2)))
    iy2 = min(h, int(round(y2)))
    seed_mask = np.zeros((h, w), dtype=bool)
    seed_mask[oy1:oy2, ox1:ox2] = True
    seed_mask[iy1:iy2, ix1:ix2] = False
    seed_mask &= candidate.astype(bool)

    seed_depths = depth_image[seed_mask & (depth_image > 0)].astype(np.float64)
    if seed_depths.size < max(8, min_points // 5):
        return None
    lo, hi = np.percentile(seed_depths, [15, 85])
    trimmed = seed_depths[(seed_depths >= lo) & (seed_depths <= hi)]
    if trimmed.size >= max(8, min_points // 5):
        seed_depths = trimmed
    seed_depth = float(np.median(seed_depths))
    depth_window_raw = max(1.0, float(depth_window_m) / float(depth_scale))

    valid = (
        candidate.astype(bool)
        & (depth_image > 0)
        & (np.abs(depth_image.astype(np.float64) - seed_depth) <= depth_window_raw)
    )
    valid_roi = valid[ry1:ry2, rx1:rx2].astype(np.uint8)
    if int(np.count_nonzero(valid_roi)) < min_points:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid_roi, connectivity=8)
    if num_labels <= 1:
        return None

    seed_roi = seed_mask[ry1:ry2, rx1:rx2]
    best_label = 0
    best_score = -1.0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_points:
            continue
        component = labels == label
        seed_overlap = int(np.count_nonzero(component & seed_roi))
        if seed_overlap == 0:
            continue
        score = float(seed_overlap) * 4.0 + float(area)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        return None

    component_roi = labels == best_label
    ys_roi, xs_roi = np.where(component_roi)
    if len(xs_roi) < min_points:
        return None
    xs = xs_roi + rx1
    ys = ys_roi + ry1

    step = max(1, int(sample_stride))
    xs_sample = xs[::step]
    ys_sample = ys[::step]
    if len(xs_sample) < min_points:
        xs_sample = xs
        ys_sample = ys
    if len(xs_sample) < min_points:
        return None

    depths = depth_image[ys_sample, xs_sample].astype(np.float64) * depth_scale
    pixels = np.column_stack([xs_sample, ys_sample])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    result = fit_plane_ransac_quality(
        points_3d,
        min_points=min(min_points, len(points_3d)),
        ransac_iter=140,
        ransac_thresh=0.007,
        random_seed=0,
    )
    if result is None:
        return None
    normal, centroid, inlier_count, inlier_ratio, rms_error = result
    if abs(float(normal[2])) < float(min_abs_z):
        return None
    if float(inlier_ratio) < float(min_inlier_ratio):
        return None
    if float(rms_error) > float(max_rms_m):
        return None
    return normal, centroid, int(inlier_count)


def compute_panel_normal(color_image, depth_image, depth_intrin,
                         all_bboxes, panel_color_thresh=40,
                         depth_scale=0.001, sample_stride=4):
    """
    通过操作面板区域的点云拟合法向量

    策略：
    1. 以所有检测框的联合外扩区域作为面板 ROI
    2. 排除按钮 bbox 内的像素，只保留面板本身
    3. 用 ROI 边缘像素采样面板颜色，做颜色筛选
    4. 向量化反投影为 3D 点云，RANSAC + SVD 拟合平面

    Args:
        color_image: BGR 彩色图
        depth_image: 深度图 (H, W) uint16（已滤波）
        depth_intrin: 深度相机内参 CameraIntrinsics
        all_bboxes: 检测框列表 [(x1,y1,x2,y2), ...]
        panel_color_thresh: 面板颜色筛选阈值
        depth_scale: 深度缩放因子
        sample_stride: 采样步长

    Returns:
        (normal, centroid) 或 None
    """
    if not all_bboxes:
        return None

    h, w = depth_image.shape
    margin = 60

    xs1 = [int(b[0]) for b in all_bboxes]
    ys1 = [int(b[1]) for b in all_bboxes]
    xs2 = [int(b[2]) for b in all_bboxes]
    ys2 = [int(b[3]) for b in all_bboxes]
    roi_x1 = max(0, min(xs1) - margin)
    roi_y1 = max(0, min(ys1) - margin)
    roi_x2 = min(w, max(xs2) + margin)
    roi_y2 = min(h, max(ys2) + margin)

    # 构建按钮遮罩
    button_mask = np.zeros((h, w), dtype=bool)
    for b in all_bboxes:
        bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        button_mask[by1:by2, bx1:bx2] = True

    # 采样面板颜色
    edge_pixels = []
    for x in range(roi_x1, roi_x2, 4):
        for y in [roi_y1, roi_y2 - 1]:
            if not button_mask[y, x]:
                edge_pixels.append(color_image[y, x].astype(np.float32))
    for y in range(roi_y1, roi_y2, 4):
        for x in [roi_x1, roi_x2 - 1]:
            if not button_mask[y, x]:
                edge_pixels.append(color_image[y, x].astype(np.float32))

    if len(edge_pixels) < 10:
        return None
    panel_color = np.median(edge_pixels, axis=0)

    # 在 ROI 内筛选面板像素
    roi_color = color_image[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_button_mask = button_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    color_diff = np.linalg.norm(roi_color - panel_color, axis=2)
    pixel_mask = (color_diff < panel_color_thresh) & (roi_depth > 0) & (~roi_button_mask)

    ys, xs = np.where(pixel_mask)
    if len(xs) < 50:
        return None
    step_idx = np.arange(0, len(xs), sample_stride)
    ys, xs = ys[step_idx], xs[step_idx]

    # 向量化反投影
    us = xs + roi_x1
    vs = ys + roi_y1
    depths = roi_depth[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([us, vs])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    return fit_plane_ransac(points_3d, min_points=50,
                            ransac_iter=100, ransac_thresh=0.008,
                            random_seed=0)
