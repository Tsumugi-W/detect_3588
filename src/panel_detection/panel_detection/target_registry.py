"""
操作目标识别模块

通过类别 + 颜色特征，每帧独立识别绝对编号，无需注册阶段。
支持视野中只出现部分目标的情况。

面板布局（从左到右）：
  ID=1: button 绿色
  ID=2: button 红色
  ID=3: button 红色
  ID=4: knob   红色
  ID=5: knob   黑色
  ID=6: button 红色
  ID=7: button 绿色

识别策略：
  1. 按 x 坐标排序当前帧检测结果
  2. 对每个检测提取颜色特征（绿/红/黑）
  3. 构建当前帧的 (类别, 颜色) 序列
  4. 在完整布局中做子序列匹配，利用 knob 颜色差异（红/黑）作为锚点消歧
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RegisteredTarget:
    """已注册的目标（保留兼容）"""
    target_id: int
    class_name: str
    center_x: float
    center_y: float
    bbox_w: float
    bbox_h: float


@dataclass
class FrameDetection:
    """单帧中一个检测结果"""
    class_name: str
    center_x: float
    center_y: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


# 完整面板布局：(类别, 颜色)
PANEL_LAYOUT = [
    ('button', 'green'),   # ID=1
    ('button', 'red'),     # ID=2
    ('button', 'red'),     # ID=3
    ('knob', 'red'),       # ID=4
    ('knob', 'black'),     # ID=5
    ('button', 'red'),     # ID=6
    ('button', 'green'),   # ID=7
]


def _classify_color(color_image: np.ndarray, bbox: Tuple[float, float, float, float]) -> str:
    """
    判断 bbox 区域的主色调：green / red / black

    使用 bbox 中心区域（50%）进行颜色分析，减少面板背景干扰
    """
    h, w = color_image.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # 取中心 50% 区域
    bw, bh = x2 - x1, y2 - y1
    cx1 = x1 + bw // 4
    cx2 = x2 - bw // 4
    cy1 = y1 + bh // 4
    cy2 = y2 - bh // 4
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w, cx2), min(h, cy2)

    roi = color_image[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    total_pixels = h_ch.size
    if total_pixels == 0:
        return 'unknown'

    # 绿色：H 在 35-85，S > 50
    green_mask = (h_ch >= 35) & (h_ch <= 85) & (s_ch > 50)
    green_ratio = np.sum(green_mask) / total_pixels

    # 红色：H 在 0-10 或 160-180，S > 50
    red_mask = ((h_ch <= 10) | (h_ch >= 160)) & (s_ch > 50)
    red_ratio = np.sum(red_mask) / total_pixels

    # 黑色：V < 80（暗区域）
    black_mask = v_ch < 80
    black_ratio = np.sum(black_mask) / total_pixels

    # 取最大占比
    ratios = {'green': green_ratio, 'red': red_ratio, 'black': black_ratio}
    best = max(ratios, key=ratios.get)

    # 至少要有 15% 的占比才认定
    if ratios[best] < 0.15:
        return 'unknown'

    return best


def _match_subsequence(observed: List[Tuple[str, str]]) -> List[int]:
    """
    将观测到的 (类别, 颜色) 序列匹配到完整布局，返回对应的绝对 ID 列表

    使用滑动窗口 + 评分机制，找到最佳匹配位置。
    强制约束：button 只能匹配 #1#2#3#6#7，knob 只能匹配 #4#5
    """
    ids, _, contiguous = _match_subsequence_scored(observed)
    if not contiguous:
        # 无法找到类别完全匹配的连续子序列，退回逐个匹配
        return _match_individual(observed)
    return ids


def _match_subsequence_scored(observed: List[Tuple[str, str]]) -> Tuple[List[int], int, bool]:
    """
    匹配连续布局子序列并返回分数。

    Returns:
        (ids, score, contiguous_match)
    """
    n_obs = len(observed)
    n_layout = len(PANEL_LAYOUT)

    if n_obs == 0:
        return [], -10_000, False
    if n_obs > n_layout:
        observed = observed[:n_layout]
        n_obs = n_layout

    best_score = -10_000
    best_offset = -1

    for offset in range(n_layout - n_obs + 1):
        score = 0
        valid = True

        for i in range(n_obs):
            layout_cls, layout_color = PANEL_LAYOUT[offset + i]
            obs_cls, obs_color = observed[i]

            if obs_cls != layout_cls:
                valid = False
                break

            if obs_color == layout_color:
                score += 3
            elif obs_color == 'unknown':
                score += 0
            else:
                score -= 2

        if not valid:
            continue

        # 连续子序列比逐个颜色匹配更可信，避免两个绿色按钮互换 #1/#7。
        score += 20
        if score > best_score:
            best_score = score
            best_offset = offset

    if best_offset < 0:
        return [], -10_000, False

    return [best_offset + i + 1 for i in range(n_obs)], best_score, True


def _match_best_direction(
        sorted_dets: List[FrameDetection],
        color_image: np.ndarray) -> List[Tuple[int, FrameDetection]]:
    """同时评估轴线正反两个方向，选择更符合完整布局的一边。"""
    candidates = []
    for dets in (sorted_dets, list(reversed(sorted_dets))):
        observed = []
        for det in dets:
            color = _classify_color(color_image, det.bbox)
            observed.append((det.class_name, color))

        ids, score, contiguous = _match_subsequence_scored(observed)
        if not contiguous:
            ids = _match_individual(observed)
            score = _score_individual_match(observed, ids)
        candidates.append((score, ids, dets))

    score, ids, dets = max(candidates, key=lambda item: item[0])
    results = []
    for i, det in enumerate(dets):
        if i < len(ids):
            results.append((ids[i], det))
    return results


def _score_individual_match(observed: List[Tuple[str, str]], ids: List[int]) -> int:
    """给逐个匹配一个较低分，仅在两边都无法连续匹配时使用。"""
    if not observed or not ids:
        return -10_000
    score = -100
    for (obs_cls, obs_color), target_id in zip(observed, ids):
        if target_id <= 0 or target_id > len(PANEL_LAYOUT):
            score -= 5
            continue
        layout_cls, layout_color = PANEL_LAYOUT[target_id - 1]
        if obs_cls == layout_cls:
            score += 1
        if obs_color == layout_color:
            score += 2
        elif obs_color != 'unknown':
            score -= 1
    return score


def _match_individual(observed: List[Tuple[str, str]]) -> List[int]:
    """
    当无法做连续子序列匹配时，逐个独立匹配每个检测结果

    强制约束：button→{1,2,3,6,7}, knob→{4,5}
    """
    # 可用 ID 池
    button_ids = [1, 2, 3, 6, 7]  # 对应颜色: green, red, red, red, green
    knob_ids = [4, 5]              # 对应颜色: red, black
    button_colors = {1: 'green', 2: 'red', 3: 'red', 6: 'red', 7: 'green'}
    knob_colors = {4: 'red', 5: 'black'}

    results = []
    used_ids = set()

    for obs_cls, obs_color in observed:
        if obs_cls == 'knob':
            # 优先按颜色匹配
            best_id = None
            for kid in knob_ids:
                if kid in used_ids:
                    continue
                if obs_color == knob_colors[kid]:
                    best_id = kid
                    break
            # 颜色没匹配上就取剩余的
            if best_id is None:
                for kid in knob_ids:
                    if kid not in used_ids:
                        best_id = kid
                        break
            if best_id is not None:
                used_ids.add(best_id)
                results.append(best_id)
            else:
                results.append(0)  # 无可用 ID
        elif obs_cls == 'button':
            # 优先按颜色匹配
            best_id = None
            for bid in button_ids:
                if bid in used_ids:
                    continue
                if obs_color == button_colors[bid]:
                    best_id = bid
                    break
            if best_id is None:
                for bid in button_ids:
                    if bid not in used_ids:
                        best_id = bid
                        break
            if best_id is not None:
                used_ids.add(best_id)
                results.append(best_id)
            else:
                results.append(0)
        else:
            results.append(0)

    return results


class TargetRegistry:
    """
    目标识别器（每帧独立识别，无注册阶段）
    """

    def __init__(self, stable_frames=15, green_hue_range=(35, 85),
                 match_distance_thresh=80):
        # 保留参数兼容，但不再使用注册逻辑
        self._is_registered = True  # 始终认为已注册
        self._registered: List[RegisteredTarget] = []
        self._stable_frames = stable_frames
        self._stable_count = 0

    @property
    def is_registered(self) -> bool:
        return True

    @property
    def targets(self) -> List[RegisteredTarget]:
        return self._registered

    def reset(self):
        pass

    def update(self, detections: List[FrameDetection],
               color_image: np.ndarray) -> bool:
        """兼容接口，始终返回 True"""
        return True

    def identify(self, detections: List[FrameDetection],
                 color_image: np.ndarray,
                 axis_origin: Tuple[float, float] | None = None,
                 axis_vector: Tuple[float, float] | None = None) -> List[Tuple[int, FrameDetection]]:
        """
        每帧独立识别：根据检测结果的类别和颜色确定绝对编号

        Args:
            detections: 当前帧的检测结果列表
            color_image: BGR 彩色图像

        Returns:
            [(absolute_id, detection), ...] 带绝对编号的结果
        """
        if not detections:
            return []

        # 按面板轴线排序。没有轴线时退回按 x 坐标排序，兼容旧调用。
        if axis_origin is not None and axis_vector is not None:
            origin = np.array(axis_origin, dtype=np.float64)
            axis = np.array(axis_vector, dtype=np.float64)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                sorted_dets = sorted(
                    detections,
                    key=lambda d: float(np.dot(
                        np.array([d.center_x, d.center_y], dtype=np.float64) - origin,
                        axis)))
            else:
                sorted_dets = sorted(detections, key=lambda d: d.center_x)
        else:
            sorted_dets = sorted(detections, key=lambda d: d.center_x)

        return _match_best_direction(sorted_dets, color_image)

    def match(self, detections: List[FrameDetection]) -> List[Tuple[int, FrameDetection]]:
        """兼容旧接口 — 不应该再被调用"""
        return []
