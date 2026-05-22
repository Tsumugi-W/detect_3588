"""
操作目标注册模块

两阶段设计：
  阶段一（注册）：积累多帧检测结果，聚类去抖动，按 x 坐标排序 + 类别/颜色验证后分配编号 1-7
  阶段二（跟踪）：每帧检测结果与注册表做最近邻匹配，输出带编号的结果

场景假设：一列7个操作对象，从左到右为 button, button, button, knob, knob, button, button
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RegisteredTarget:
    """已注册的目标"""
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


EXPECTED_LAYOUT = ['button', 'button', 'button', 'knob', 'knob', 'button', 'button']


class TargetRegistry:
    """
    操作目标注册器

    Args:
        stable_frames: 需要连续多少帧检测稳定才完成注册
        green_hue_range: 绿色按钮 HSV 色调范围 (low, high)
        match_distance_thresh: 跟踪匹配最大像素距离
    """

    def __init__(self, stable_frames=15, green_hue_range=(35, 85),
                 match_distance_thresh=80):
        self._stable_frames = stable_frames
        self._green_hue_range = green_hue_range
        self._match_distance_thresh = match_distance_thresh

        self._registered: List[RegisteredTarget] = []
        self._history: List[List[FrameDetection]] = []
        self._stable_count = 0
        self._is_registered = False

    @property
    def is_registered(self) -> bool:
        return self._is_registered

    @property
    def targets(self) -> List[RegisteredTarget]:
        return self._registered

    def reset(self):
        """重置注册状态"""
        self._registered = []
        self._history = []
        self._stable_count = 0
        self._is_registered = False

    def update(self, detections: List[FrameDetection],
               color_image: np.ndarray) -> bool:
        """
        注册阶段：输入一帧检测结果，尝试完成注册

        Returns:
            True 表示注册完成
        """
        if self._is_registered:
            return True

        buttons = [d for d in detections if d.class_name == 'button']
        knobs = [d for d in detections if d.class_name == 'knob']

        if len(buttons) == 5 and len(knobs) == 2:
            self._history.append(detections)
            self._stable_count += 1
        else:
            self._stable_count = 0
            self._history = []

        if self._stable_count >= self._stable_frames:
            return self._try_register(color_image)

        return False

    def _try_register(self, color_image: np.ndarray) -> bool:
        """尝试从积累的历史帧中完成注册"""
        # 取最近 stable_frames 帧做聚类
        recent = self._history[-self._stable_frames:]

        # 对每帧按 x 排序，取各位置中位数
        all_sorted = []
        for frame_dets in recent:
            sorted_dets = sorted(frame_dets, key=lambda d: d.center_x)
            all_sorted.append(sorted_dets)

        # 验证类别布局一致性
        for frame_dets in all_sorted:
            layout = [d.class_name for d in frame_dets]
            if layout != EXPECTED_LAYOUT:
                self._stable_count = 0
                self._history = []
                return False

        # 计算每个位置的中位坐标
        n_targets = 7
        median_centers = []
        median_bboxes = []
        for i in range(n_targets):
            xs = [all_sorted[f][i].center_x for f in range(len(all_sorted))]
            ys = [all_sorted[f][i].center_y for f in range(len(all_sorted))]
            ws = [all_sorted[f][i].bbox[2] - all_sorted[f][i].bbox[0]
                  for f in range(len(all_sorted))]
            hs = [all_sorted[f][i].bbox[3] - all_sorted[f][i].bbox[1]
                  for f in range(len(all_sorted))]
            median_centers.append((np.median(xs), np.median(ys)))
            median_bboxes.append((np.median(ws), np.median(hs)))

        # 验证第1个按钮是绿色
        first_button_cx = int(median_centers[0][0])
        first_button_cy = int(median_centers[0][1])
        if not self._is_green(color_image, first_button_cx, first_button_cy,
                              int(median_bboxes[0][0]), int(median_bboxes[0][1])):
            self._stable_count = 0
            self._history = []
            return False

        # 注册成功
        self._registered = []
        for i in range(n_targets):
            self._registered.append(RegisteredTarget(
                target_id=i + 1,
                class_name=EXPECTED_LAYOUT[i],
                center_x=median_centers[i][0],
                center_y=median_centers[i][1],
                bbox_w=median_bboxes[i][0],
                bbox_h=median_bboxes[i][1],
            ))
        self._is_registered = True
        return True

    def _is_green(self, color_image: np.ndarray, cx: int, cy: int,
                  w: int, h: int) -> bool:
        """检查指定区域是否为绿色"""
        half_w, half_h = w // 4, h // 4
        y1 = max(0, cy - half_h)
        y2 = min(color_image.shape[0], cy + half_h)
        x1 = max(0, cx - half_w)
        x2 = min(color_image.shape[1], cx + half_w)

        roi = color_image[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]

        low, high = self._green_hue_range
        green_mask = (h_channel >= low) & (h_channel <= high) & (s_channel > 50)
        green_ratio = np.sum(green_mask) / green_mask.size

        return green_ratio > 0.2

    def match(self, detections: List[FrameDetection]) -> List[Tuple[int, FrameDetection]]:
        """
        跟踪阶段：将当前帧检测结果匹配到注册表

        Returns:
            [(target_id, detection), ...] 匹配成功的结果列表
            未匹配的检测会被丢弃，未匹配的注册目标不输出
        """
        if not self._is_registered:
            return []

        results = []
        used_reg = set()
        used_det = set()

        # 贪心最近邻匹配（先按距离排序）
        pairs = []
        for di, det in enumerate(detections):
            for ri, reg in enumerate(self._registered):
                if det.class_name != reg.class_name:
                    continue
                dist = np.sqrt((det.center_x - reg.center_x) ** 2 +
                               (det.center_y - reg.center_y) ** 2)
                pairs.append((dist, di, ri))

        pairs.sort(key=lambda x: x[0])

        for dist, di, ri in pairs:
            if di in used_det or ri in used_reg:
                continue
            if dist > self._match_distance_thresh:
                continue
            used_det.add(di)
            used_reg.add(ri)
            results.append((self._registered[ri].target_id, detections[di]))

        return results
