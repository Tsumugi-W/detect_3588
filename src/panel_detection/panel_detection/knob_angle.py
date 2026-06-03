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
    pointer_color: str = 'auto',
    debug: bool = False,
) -> float | None:
    """
    估计旋钮指针角度

    Args:
        color_roi: BGR 格式的旋钮裁剪图 (来自 YOLO bbox)
        binary_thresh: 二值化阈值
        min_pointer_area_ratio: 指针最小面积占圆形区域的比例
        max_pointer_area_ratio: 指针最大面积占圆形区域的比例
        circle_mask_ratio: 圆形 mask 半径比例
        angle_range: (min_angle, max_angle) 物理角度范围
        pointer_color: 'red_handle'=红色长柄, 'dark'=暗色指针,
                       'bright'=亮色把手, 'auto'=自动
        debug: 是否返回调试中间结果

    Returns:
        角度值 (°)，以 12 点钟方向为 0°，顺时针增加
        如果无法检测到有效指针则返回 None
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

    if pointer_color == 'red_handle':
        angle, pointer_contour, binary = _try_red_handle_pointer(
            color_roi, radius, circle_area,
            min_pointer_area_ratio, max_pointer_area_ratio, cx, cy,
            angle_range=angle_range)
    else:
        # 当前黑色旋钮的稳定特征是“深色圆形底座 + 亮色径向指示条”。
        # 优先检测亮色细长结构，避免旧的暗色外轮廓法把底座边缘当成指针。
        angle, pointer_contour, binary = _try_bright_radial_pointer(
            color_roi, circle_mask, radius, circle_area,
            binary_thresh, min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is None and pointer_color in ('dark', 'auto'):
        angle, pointer_contour, binary = _try_dark_pointer(
            color_roi, circle_mask, radius, circle_area,
            min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is None:
        angle, pointer_contour, binary = _try_white_pointer(
            color_roi, circle_mask, radius, circle_area,
            binary_thresh, min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is not None:
        angle = _constrain_angle(angle, angle_range)
        if debug:
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            dbg = _build_debug(gray, circle_mask, binary, pointer_contour)
            dbg['angle'] = angle
            dbg['method'] = 'red_handle_pointer' if pointer_color == 'red_handle' else 'bright_radial_pointer'
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


def _try_red_handle_pointer(color_roi, radius, circle_area,
                            min_area_ratio, max_area_ratio, cx, cy,
                            angle_range=None):
    """
    红色大旋钮专用：根据红色本体中相对中心最突出的区域估计长柄方向。

    红色旋钮的手柄本身是暗红/红色，不是亮色指针；bbox 又包含黄色底座。
    因此先用 HSV 只保留红色像素，排除黄色底座和叠加标注，再按角度
    扇区寻找红色像素的最大径向长度峰值。
    """
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_mask = (((h_ch <= 12) | (h_ch >= 165)) & (s_ch >= 45) & (v_ch >= 25))
    binary = red_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, binary

    min_area = max(20.0, circle_area * min_area_ratio * 0.5)
    max_area = circle_area * max_area_ratio * 3.0
    red_contours = [
        cnt for cnt in contours
        if min_area <= cv2.contourArea(cnt) <= max_area
    ]
    if not red_contours:
        red_contours = [max(contours, key=cv2.contourArea)]

    pts = np.vstack(red_contours).reshape(-1, 2).astype(np.float64)
    if len(pts) < 20:
        return None, None, binary

    angle_deg = _red_handle_angle_from_radial_profile(
        pts, cx, cy, radius, angle_range)
    if angle_deg is None:
        return None, None, binary

    contour = max(red_contours, key=cv2.contourArea)
    return angle_deg, contour, binary


def _red_handle_angle_from_radial_profile(points, cx, cy, radius, angle_range=None):
    """
    在红色像素的极坐标径向长度图上找峰值。

    圆环会在许多方向产生相近半径；真实长柄会在少数相邻方向产生
    更大的 95 分位径向长度。若给定物理范围，则只在该范围附近搜索。
    """
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dists = np.sqrt(dx ** 2 + dy ** 2)
    if np.max(dists) < radius * 0.35:
        return None

    angles = np.degrees(np.arctan2(dx, -dy)) % 360.0
    min_count = max(8, int(len(points) * 0.004))
    sector_width = 6.0
    candidates = []

    for angle in np.arange(0.0, 360.0, 2.0):
        if angle_range is not None and not _angle_in_range(
                angle, angle_range, margin=25.0):
            continue
        diff = np.abs(((angles - angle + 180.0) % 360.0) - 180.0)
        sector_dists = dists[diff <= sector_width]
        if len(sector_dists) < min_count:
            continue
        p95 = float(np.percentile(sector_dists, 95))
        if p95 < radius * 0.45:
            continue
        count_bonus = min(len(sector_dists), 300) * 0.01
        protrusion_bonus = max(0.0, p95 - radius * 0.75) * 1.5
        candidates.append((p95 + protrusion_bonus + count_bonus, angle, p95))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_angle, best_radius = candidates[0]
    if best_radius < radius * 0.55:
        return None
    return float(best_angle % 360.0)


def _angle_in_range(angle, angle_range, margin=0.0):
    lo, hi = angle_range
    lo -= margin
    hi += margin
    angle = angle % 360.0
    lo = lo % 360.0
    hi = hi % 360.0
    if lo <= hi:
        return lo <= angle <= hi
    return angle >= lo or angle <= hi


def _try_bright_radial_pointer(color_roi, circle_mask, radius, circle_area,
                               binary_thresh, min_area_ratio, max_area_ratio,
                               cx, cy):
    """
    检测亮色径向指示条/拨杆。

    旋钮底座通常是红色或黑色，指示条明显更亮。这里先用亮度分割提取
    细长亮色连通域，再用长轴端点到旋钮中心的距离消除 180° 歧义；
    如果连通域被反光切碎，则退回到按亮度加权的径向向量。
    """
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    pointer_region = (dist >= radius * 0.08) & (dist <= radius * 0.95)
    valid_mask = ((circle_mask > 0) & pointer_region).astype(np.uint8) * 255

    valid_pixels = gray[valid_mask > 0]
    if valid_pixels.size < 20:
        return None, None, np.zeros_like(gray)

    # 用分位数阈值适应红/黑两种底座；固定阈值作为下限，防止暗噪声入选。
    percentile_thresh = float(np.percentile(valid_pixels, 88))
    thresh = max(float(binary_thresh), percentile_thresh)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(8.0, circle_area * min_area_ratio * 0.25)
    max_area = circle_area * max_area_ratio

    best_contour = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        short_side = min(rw, rh)
        long_side = max(rw, rh)
        if short_side < 1 or long_side < radius * 0.18:
            continue

        aspect = long_side / short_side
        moments = cv2.moments(cnt)
        if moments['m00'] == 0:
            continue
        mx = moments['m10'] / moments['m00']
        my = moments['m01'] / moments['m00']
        radial = math.hypot(mx - cx, my - cy)

        # 指示条可以穿过中心，不强制质心离中心很远；长、细、亮面积更可信。
        score = aspect * 2.0 + long_side / max(radius, 1) + radial / max(radius, 1)
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is not None:
        return _compute_angle_from_contour(best_contour, cx, cy), best_contour, binary

    angle = _weighted_radial_angle(gray, valid_mask, cx, cy, radius)
    return angle, None, binary


def _weighted_radial_angle(gray, valid_mask, cx, cy, radius):
    """用高亮像素的径向质心兜底估计方向。"""
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)

    pixels = gray[valid_mask > 0].astype(np.float64)
    if pixels.size < 20:
        return None

    base = np.percentile(pixels, 75)
    weights = np.maximum(gray.astype(np.float64) - base, 0.0)
    weights[(valid_mask == 0) | (dist < radius * 0.12)] = 0.0

    total = float(np.sum(weights))
    if total < 1e-6:
        return None

    vx = float(np.sum(weights * dx) / total)
    vy = float(np.sum(weights * dy) / total)
    if math.hypot(vx, vy) < radius * 0.08:
        return None

    angle_rad = math.atan2(vx, -vy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg


def _try_dark_pointer(color_roi, circle_mask, radius, circle_area,
                      min_area_ratio, max_area_ratio, cx, cy):
    """
    最远点法：手柄比圆形底座直径长，
    轮廓上离中心最远的一簇点即为手柄方向。

    步骤：
    1. 边缘检测拿到旋钮轮廓
    2. 计算轮廓上每个点到 bbox 中心的距离
    3. 取距离 > 中位数 * 1.1 的远点簇
    4. 远点簇质心相对中心的方向 = 手柄角度
    """
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 高斯模糊 + Canny 边缘
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # 只保留圆形区域内的边缘
    edges = cv2.bitwise_and(edges, circle_mask)

    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, edges

    # 合并所有轮廓点
    all_pts = np.vstack(contours).squeeze()
    if all_pts.ndim != 2 or len(all_pts) < 20:
        return None, None, edges

    # 计算每个轮廓点到 bbox 中心的距离
    dists = np.sqrt((all_pts[:, 0] - cx) ** 2 + (all_pts[:, 1] - cy) ** 2)

    # 用中位距离作为 "底座半径" 的估计
    median_dist = np.median(dists)
    if median_dist < 5:
        return None, None, edges

    # 取距离超过中位数 * 1.15 的点作为远点（手柄突出部分）
    threshold = median_dist * 1.15
    far_mask = dists > threshold

    far_pts = all_pts[far_mask]
    if len(far_pts) < 5:
        return None, None, edges

    # 远点质心
    mass_x = np.mean(far_pts[:, 0])
    mass_y = np.mean(far_pts[:, 1])

    # 从中心到远点质心的角度
    dx = mass_x - cx
    dy = mass_y - cy
    angle_rad = math.atan2(dx, -dy)  # 0° = 12点钟，顺时针
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0

    return angle_deg, None, edges


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

    策略：
    1. 用 minAreaRect 获取长轴方向（对细长指针非常准确）
    2. minAreaRect 有 180° 歧义，通过比较长轴两端哪端离旋钮中心更远来消歧
       — 指针尖端离中心更远
    """
    rect = cv2.minAreaRect(contour)
    (rect_cx, rect_cy), (w, h), angle_rect = rect

    # minAreaRect 的 angle 是短边与 x 轴的夹角
    # 转换为长轴方向
    if w < h:
        # 长轴方向 = angle_rect + 90
        axis_angle = angle_rect + 90
    else:
        axis_angle = angle_rect

    # axis_angle 是长轴与 x 轴正方向的夹角（逆时针）
    # 转换为方向向量
    rad = math.radians(axis_angle)
    vx = math.cos(rad)
    vy = math.sin(rad)

    # 沿长轴两个方向各取一个端点，看哪端离旋钮中心更远
    half_len = max(w, h) / 2.0
    end1_x = rect_cx + vx * half_len
    end1_y = rect_cy + vy * half_len
    end2_x = rect_cx - vx * half_len
    end2_y = rect_cy - vy * half_len

    dist1 = math.sqrt((end1_x - cx) ** 2 + (end1_y - cy) ** 2)
    dist2 = math.sqrt((end2_x - cx) ** 2 + (end2_y - cy) ** 2)

    # 指针指向离中心更远的那端
    if dist2 > dist1:
        vx, vy = -vx, -vy

    # 计算角度：以 12 点钟方向 (向上) 为 0°，顺时针增加
    # 图像坐标系 y 向下，所以 12 点钟是 (0, -1)
    angle_rad = math.atan2(vx, -vy)
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


def estimate_hex_angle(color_roi: np.ndarray, circle_mask_ratio: float = 0.9,
                       n_sides: int = 6) -> float | None:
    """
    估计正多边形螺母/螺栓/阀门的旋转角度

    正六边形有 60° 对称性，正八边形有 45° 对称性。
    输出 [0, 360/n_sides) 之间的角度。
    0° 表示有一条边是水平的（平边朝上）。

    Args:
        color_roi: BGR 格式的裁剪图
        circle_mask_ratio: 圆形 mask 比例，排除背景
        n_sides: 边数（6=六边形 nut/bolt，8=八边形 valve）

    Returns:
        角度 [0, 360/n_sides)°，None 表示无法检测
    """
    if color_roi is None or color_roi.size == 0:
        return None

    h, w = color_roi.shape[:2]
    if h < 15 or w < 15:
        return None

    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * circle_mask_ratio)

    # 圆形 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # 边缘检测
    edges = cv2.Canny(masked, 50, 150)
    edges = cv2.bitwise_and(edges, mask)

    # Hough 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=15,
                            minLineLength=radius * 0.3,
                            maxLineGap=radius * 0.2)
    if lines is None or len(lines) < 2:
        return None

    # 对称周期角度
    sym_angle = 360.0 / n_sides  # 60° for hex, 45° for octagon

    # 收集所有线段角度，归一化到 [0, sym_angle)
    angles_mod = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 线段角度（相对水平，范围 [0, 180)）
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        # 利用对称性归一化到 [0, sym_angle)
        angle_m = angle % sym_angle
        angles_mod.append(angle_m)

    if not angles_mod:
        return None

    # 用圆形平均处理跨界情况（如 1° 和 59° 应该平均为 0°）
    # 将角度映射到单位圆取平均方向
    scale = 360.0 / sym_angle  # 放大到 [0, 360)
    rads = [math.radians(a * scale) for a in angles_mod]
    sin_sum = sum(math.sin(r) for r in rads)
    cos_sum = sum(math.cos(r) for r in rads)

    mean_rad = math.atan2(sin_sum, cos_sum)
    mean_deg = math.degrees(mean_rad) / scale  # 缩回 [0, sym_angle)
    if mean_deg < 0:
        mean_deg += sym_angle

    return mean_deg


def draw_hex_angle(image, bbox, angle, color=(0, 0, 0), thickness=1):
    """
    在图像上绘制六边形角度标注

    Args:
        image: 原始图像
        bbox: (x1, y1, x2, y2)
        angle: 角度 [0, 60)°
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    label = f'{angle:.1f}deg'
    cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA)


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

    # 画直线（从中心向两侧延伸）
    angle_rad = math.radians(angle)
    dx = r * math.sin(angle_rad)
    dy = -r * math.cos(angle_rad)
    ex1 = int(cx + dx)
    ey1 = int(cy + dy)
    ex2 = int(cx - dx)
    ey2 = int(cy - dy)
    cv2.line(image, (ex1, ey1), (ex2, ey2), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 3, color, -1, cv2.LINE_AA)

    # 标注角度数值
    label = f'{angle:.0f} deg'
    cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
