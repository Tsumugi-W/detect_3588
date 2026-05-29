"""
旋钮角度估计模块

通过传统 CV 方法检测旋钮上的白色指针条方向，估计旋转角度。
仅依赖 OpenCV + numpy，无需额外模型。

用法:
    from knob_angle import estimate_knob_angle
    angle = estimate_knob_angle(knob_roi)  # 输入 BGR 裁剪图，返回角度 (°) 或 None
"""
import math
import cv2
import numpy as np


def estimate_knob_angle(
    color_roi: np.ndarray,
    binary_thresh: int = 180,
    min_pointer_area_ratio: float = 0.01,
    max_pointer_area_ratio: float = 0.25,
    circle_mask_ratio: float = 0.85,
    angle_range: tuple = None,
    debug: bool = False,
) -> float | None:
    """
    估计旋钮指针角度

    自动判断旋钮类型：
      - 白色指针旋钮（黑色旋钮上有白色标记）：灰度二值化提取
      - 彩色把手旋钮（红色/棕色旋转把手）：HSV 颜色隔离提取

    Args:
        color_roi: BGR 格式的旋钮裁剪图 (来自 YOLO bbox)
        binary_thresh: 二值化阈值，用于提取白色指针（高亮区域）
        min_pointer_area_ratio: 指针最小面积占圆形区域的比例
        max_pointer_area_ratio: 指针最大面积占圆形区域的比例
        circle_mask_ratio: 圆形 mask 半径相对于 ROI 短边半径的比例
                           用于去除旋钮边框（银色金属圈）干扰
        angle_range: (min_angle, max_angle) 物理角度范围，用于消歧和 clamp
                     例如 (-45, 135) 表示黑色旋钮，(135, 315) 表示红色旋钮
                     超出范围的值会被 clamp 到边界
        debug: 是否返回调试中间结果

    Returns:
        角度值 (°)，以 12 点钟方向为 0°，顺时针增加
        如果设定了 angle_range，返回值在该范围内（可能为负数）
        如果无法检测到有效指针则返回 None
        当 debug=True 时返回 (angle, debug_dict)
    """
    if color_roi is None or color_roi.size == 0:
        return (None, {}) if debug else None

    h, w = color_roi.shape[:2]
    if h < 10 or w < 10:
        return (None, {}) if debug else None

    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * circle_mask_ratio)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
    circle_area = math.pi * radius * radius

    # 先尝试白色指针方法
    angle, pointer_contour, binary = _try_white_pointer(
        color_roi, circle_mask, radius, circle_area,
        binary_thresh, min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is not None:
        angle = _constrain_angle(angle, angle_range)
        if debug:
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            dbg = _build_debug(gray, circle_mask, binary, pointer_contour)
            dbg['angle'] = angle
            dbg['method'] = 'white_pointer'
            return angle, dbg
        return angle

    # 白色指针失败，尝试径向亮度扫描法
    angle, _, _ = _try_color_handle(
        color_roi, circle_mask, circle_area,
        min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is not None:
        angle = _constrain_angle(angle, angle_range)
        if debug:
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            dbg = _build_debug(gray, circle_mask, np.zeros_like(circle_mask), None)
            dbg['angle'] = angle
            dbg['method'] = 'radial_scan'
            return angle, dbg
        return angle

    if debug:
        gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
        return None, _build_debug(gray, circle_mask,
                                  binary if binary is not None else np.zeros_like(circle_mask),
                                  None)
    return None


def _try_white_pointer(color_roi, circle_mask, radius, circle_area,
                       binary_thresh, min_area_ratio, max_area_ratio, cx, cy):
    """尝试用灰度二值化提取白色指针"""
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # OTSU 自适应 + 固定阈值双策略
    _, binary_otsu = cv2.threshold(masked_gray, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_otsu = cv2.bitwise_and(binary_otsu, circle_mask)

    _, binary_fixed = cv2.threshold(masked_gray, binary_thresh, 255,
                                    cv2.THRESH_BINARY)
    binary_fixed = cv2.bitwise_and(binary_fixed, circle_mask)

    otsu_white = cv2.countNonZero(binary_otsu)
    if otsu_white > circle_area * 0.4:
        binary = binary_fixed
    else:
        binary = binary_otsu

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 轮廓筛选
    contour = _find_pointer_contour(
        binary, circle_area, min_area_ratio, max_area_ratio)
    if contour is None:
        return None, None, binary

    angle = _compute_angle_from_contour(contour, cx, cy)
    return angle, contour, binary


def _try_color_handle(color_roi, circle_mask, circle_area,
                      min_area_ratio, max_area_ratio, cx, cy):
    """
    径向亮度扫描法检测把手方向

    把手比底座更亮。将圆形区域分为 36 个扇区（每 10°），
    计算外圈（半径 40%-90%）每个扇区的平均亮度，
    最亮的方向就是把手指向。
    """
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    radius = int(min(cx, cy) * 0.85)

    # 构建角度和距离映射
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # 只看外圈（40%~90% 半径），排除中心和边缘
    inner_r = radius * 0.4
    outer_r = radius * 0.9
    ring_mask = (dist >= inner_r) & (dist <= outer_r)

    # 计算每个像素的角度（0°=12点钟方向，顺时针）
    angles = np.degrees(np.arctan2(dx, -dy)) % 360

    # 分成 36 个扇区，每个 10°
    n_sectors = 36
    sector_brightness = np.zeros(n_sectors)
    sector_counts = np.zeros(n_sectors)

    for s in range(n_sectors):
        a_lo = s * 10.0
        a_hi = (s + 1) * 10.0
        mask = ring_mask & (angles >= a_lo) & (angles < a_hi)
        pixels = gray[mask]
        if len(pixels) > 0:
            sector_brightness[s] = np.mean(pixels)
            sector_counts[s] = len(pixels)

    # 用高斯平滑扇区亮度（循环卷积，消除噪声）
    from scipy.ndimage import uniform_filter1d
    # 简单用滑窗平均代替（避免依赖 scipy）
    smooth = np.zeros(n_sectors)
    kernel_half = 2  # ±2 扇区 = ±20° 平滑
    for s in range(n_sectors):
        total = 0.0
        count = 0
        for k in range(-kernel_half, kernel_half + 1):
            idx = (s + k) % n_sectors
            if sector_counts[idx] > 0:
                total += sector_brightness[idx]
                count += 1
        smooth[s] = total / count if count > 0 else 0

    # 找最亮扇区
    best_sector = int(np.argmax(smooth))
    angle = (best_sector + 0.5) * 10.0  # 扇区中心角度

    # 返回兼容接口（无 contour）
    return angle, None, None


def _find_pointer_contour(binary, circle_area, min_area_ratio, max_area_ratio):
    """从二值图中找到符合指针特征的轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = circle_area * min_area_ratio
    max_area = circle_area * max_area_ratio

    best_contour = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        box_w, box_h = rect[1]
        if min(box_w, box_h) < 1:
            continue
        aspect = max(box_w, box_h) / min(box_w, box_h)
        if aspect < 1.5:
            continue

        score = aspect * math.sqrt(area)
        if score > best_score:
            best_score = score
            best_contour = cnt

    return best_contour


def _compute_angle_from_contour(contour, cx, cy) -> float:
    """
    从轮廓计算指针角度

    策略：取轮廓上距离中心最远的 top 10% 点，计算它们的平均方向。
    比单个最远点更稳定，不容易被噪声干扰。
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    # 取距离最远的 top 10% 点（至少 3 个）
    n_top = max(3, len(pts) // 10)
    top_indices = np.argsort(dists)[-n_top:]
    top_pts = pts[top_indices]

    # 计算这些点相对于中心的平均方向
    dx = np.mean(top_pts[:, 0]) - cx
    dy = np.mean(top_pts[:, 1]) - cy

    # 计算角度：以 12 点钟方向 (向上) 为 0°，顺时针增加
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0

    return angle_deg


def _constrain_angle(angle: float, angle_range: tuple = None) -> float:
    """
    将角度约束到物理范围内

    Args:
        angle: 原始角度 [0, 360)
        angle_range: (min_angle, max_angle)，允许负数和 >360 的值
                     例如 (-45, 135) 或 (135, 315)

    Returns:
        约束后的角度值，超出范围 clamp 到边界
        如果检测到 180° 翻转（距离范围中心 > 90°），先折回再 clamp
    """
    if angle_range is None:
        return angle

    lo, hi = angle_range
    mid = (lo + hi) / 2.0

    # 将 angle 转换到以 mid 为中心的 [-180, 180) 区间
    # 这样可以正确处理跨 0°/360° 的情况
    diff = angle - mid
    # 归一化 diff 到 [-180, 180)
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360

    # 如果偏离中心超过 90°，说明方向反了，折回 180°
    half_span = (hi - lo) / 2.0
    if abs(diff) > half_span + 90:
        diff -= 180 if diff > 0 else -180
        # 再次归一化
        while diff > 180:
            diff -= 360
        while diff <= -180:
            diff += 360

    # 转回绝对角度
    result = mid + diff

    # Clamp 到范围边界
    result = max(lo, min(hi, result))

    return result


def _build_debug(gray, circle_mask, binary, pointer_contour):
    """构造调试信息字典"""
    return {
        'gray': gray,
        'circle_mask': circle_mask,
        'binary': binary,
        'pointer_contour': pointer_contour,
    }


def draw_knob_angle(image, bbox, angle, color=(0, 255, 255), thickness=2):
    """
    在图像上绘制旋钮角度标注

    Args:
        image: 原始图像 (会被原地修改)
        bbox: (x1, y1, x2, y2) 旋钮边界框
        angle: 角度 (°)，范围 [0, 360)
        color: 标注颜色
        thickness: 线条粗细
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = min(x2 - x1, y2 - y1) // 3

    # 画指针方向线
    angle_rad = math.radians(angle)
    ex = int(cx + r * math.sin(angle_rad))
    ey = int(cy - r * math.cos(angle_rad))
    cv2.line(image, (cx, cy), (ex, ey), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 3, color, -1, cv2.LINE_AA)

    # 标注角度数值
    label = f'{angle:.0f} deg'
    cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
