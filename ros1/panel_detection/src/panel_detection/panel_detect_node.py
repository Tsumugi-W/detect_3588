"""
ROS1 Noetic 面板检测节点

两阶段工作流：
  1. 注册阶段：积累多帧检测结果，按 x 坐标排序 + 类别/颜色验证后为 7 个目标分配编号 1-7
  2. 跟踪阶段：每帧检测结果匹配注册表，发布带编号的检测结果

话题:
  /panel/targets   (String, JSON) — 带编号的检测结果（位置 + 类别 + ID）
  /panel/knob_angles (String, JSON) — 旋钮角度信息（带编号）
  /objects/geometry (String, JSON) — 非面板目标角度与轴线方向（阀门/螺栓/螺母）
  /panel/distance  (String, JSON) — 相机到操作面板平面的垂直距离
  /panel/status    (String)       — 注册状态（registering / registered）
"""
import os
import math
import json
import time
import threading
import copy
from collections import defaultdict, deque

import yaml
import numpy as np
import cv2
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo

from .ros1_compat import Node

from .apriltag_reference import (
    angle_between_normals_deg,
    detect_apriltag_reference_axis,
)
from .camera import create_backend
from .camera.base import CameraIntrinsics
from .depth_utils import (
    deproject_pixel_to_point,
    filter_depth, get_robust_depth, get_bbox_robust_depth, get_masked_robust_depth,
    compute_panel_normal, estimate_fastener_group_axis_direction,
    estimate_fastener_line_constrained_axis,
    estimate_fastener_patch_axis_direction,
    estimate_object_axis_direction,
    estimate_valve_wheel_axis_direction,
)
from .knob_angle import (
    estimate_knob_angle, draw_knob_angle, estimate_hex_angle,
    estimate_valve_angle, estimate_valve_angle_candidates, draw_hex_angle,
)
from .fastener_registry import FastenerGroupRegistry, FastenerObservation
from .fastener_axis_stabilizer import (
    FastenerAxisMeasurement,
    FastenerAxisStabilizer,
)
# 铭牌 OCR 暂停使用。保留 nameplate_ocr.py，后续需要时再恢复接入。
# from .nameplate_ocr import NameplateRecognizer
from .nut_localizer import localize_nut
from .panel_apriltag import (
    PanelAprilTagTracker,
    detect_panel_tags,
    draw_panel_tag_assignments,
    reclassify_buttons_without_tags,
)
from .target_registry import PersistentPanelAxis, TargetRegistry, FrameDetection


# ─── 配置 ─────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'camera_backend': 'orbbec',
    'camera': {'color_width': 1280, 'color_height': 720,
               'depth_width': 1280, 'depth_height': 720, 'fps': 30},
    'inference_backend': 'onnx',
    'onnx_model': '0813.onnx',
    'onnx_threads': 8,
    'weight': '0813.pt',
    'input_size': 640,
    'class_num': 9,
    'class_name': ['light', 'knob', 'bolt', 'nut', 'valve', 'pump',
                   'button', 'door_button', 'air_switch'],
    'threshold': {'iou': 0.01, 'confidence': 0.3},
    'knob_angle': {'enable': True, 'binary_thresh': 180,
                   'circle_mask_ratio': 0.85, 'knob_class': 'knob',
                   'use_constraint': 1},
    'position_stabilizer': {'enable': True, 'still_time': 3.0,
                            'pixel_thresh': 5.0, 'window_size': 45,
                            'ema_alpha': 0.25, 'depth_std_thresh': 0.01},
    'panel_line': {'initial_dist_ratio': 1.1, 'dist_ratio': 0.85,
                   'min_dist': 45.0, 'max_dist': 110.0,
                   'proj_margin_ratio': 3.0,
                   'min_proj_margin': 450.0},
    'panel_apriltag': {
        'enable': True,
        'dictionary': 'DICT_APRILTAG_36h11',
        'min_id': 0,
        'max_id': 39,
        'min_area_px': 64.0,
        'max_horizontal_ratio': 1.35,
        'max_vertical_ratio': 3.0,
        'overlap_tolerance_ratio': 0.35,
        'min_horizontal_gate_px': 24.0,
        'min_vertical_gate_px': 30.0,
        'track_distance_px': 90.0,
        'track_size_ratio': 1.3,
        'stale_frames': 8,
    },
    'panel_normal_interval': 10,
    'topic_sync': {'max_dt': 0.05, 'registered_depth': True},
    'valve_axis': {
        'edge_margin_px': 4,
    },
    'fastener_registry': {
        'min_init_observations': 2,
        'max_group_distance_m': 0.35,
        'max_slot_distance_m': 0.12,
        'slot_match_ratio': 0.45,
        'normal_angle_thresh_deg': 25.0,
        'ema_alpha': 0.35,
        'stale_frames': 120,
    },
    'fastener_axis_stabilizer': {
        'enable': True,
        'ema_alpha': 0.18,
        'max_jump_deg': 12.0,
        'confirm_frames': 5,
        'max_output_step_deg': 3.0,
        'match_distance_px': 60.0,
        'match_size_ratio': 1.2,
        'stale_frames': 60,
    },
    'fastener_position_stabilizer': {
        'enable': True,
        'still_time': 1.0,
        'pixel_thresh': 5.0,
        'window_size': 30,
        'ema_alpha': 0.20,
        'depth_std_thresh': 0.008,
    },
    'valve_angle_stabilizer': {
        'enable': False,
        'max_jump_deg': 10.0,
        'confirm_frames': 5,
    },
    # 铭牌 OCR 配置暂停使用。
    # 'nameplate_ocr': {
    #     'enable': False,
    # },
    'detection_mode': 'all',
    'publish_legacy_topics': False,
    'apriltag_reference': {
        'enable': False,
        'dictionary': 'DICT_APRILTAG_36h11',
        'sample_stride': 3,
        'border_margin_ratio': 0.12,
        'min_points': 80,
        'ransac_thresh': 0.008,
        'min_inlier_ratio': 0.45,
        'max_rms_m': 0.012,
        'min_abs_z': 0.50,
        'fallback_enable': True,
        'fallback_min_point_count': 2000,
        'fallback_min_inlier_ratio': 0.75,
        'fallback_min_white_border_ratio': 0.45,
        'fallback_white_thresh': 150,
        'log_enable': True,
        'log_path': 'axis_reference_log.jsonl',
    },
}

CLASS_TOPIC_MAP = {
    'light': '/panel/lights',
    'knob': '/panel/knobs',
    'bolt': '/panel/bolts',
    'nut': '/panel/nuts',
    'valve': '/panel/valves',
    'pump': '/panel/pumps',
    'button': '/panel/buttons',
    'door_button': '/panel/door_buttons',
    'air_switch': '/panel/air_switches',
}

MODE_CLASS_FILTERS = {
    'panel_controls': {'button', 'door_button', 'knob', 'light'},
    'valve': {'valve'},
    'fastener': {'bolt', 'nut'},
    'all': None,
}

MODE_TOPIC_MAP = {
    'panel_controls': {
        'targets': '/panel/targets',
        'angles': '/panel/knob_angles',
        'status': '/panel/status',
        'distance': '/panel/distance',
    },
    'valve': {
        'targets': '/valve/targets',
        'geometry': '/valve/geometry',
        'status': '/valve/status',
    },
    'fastener': {
        'targets': '/fasteners/targets',
        'geometry': '/fasteners/geometry',
        'status': '/fasteners/status',
    },
    'all': {
        'targets': '/panel/targets',
        'angles': '/panel/knob_angles',
        'geometry': '/objects/geometry',
        'status': '/panel/status',
        'distance': '/panel/distance',
    },
}


def _merge_config(base, overrides):
    """Recursively merge a partial YAML config into the built-in defaults."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


class AsyncCamera:
    def __init__(self, cam):
        self._cam = cam
        self._frame = (None, None, None, None)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                result = self._cam.get_aligned_frames()
                if result[0] is not None:
                    with self._lock:
                        self._frame = result
            except Exception:
                time.sleep(0.1)

    def get_aligned_frames(self):
        with self._lock:
            return self._frame

    def get_depth_scale(self):
        return self._cam.get_depth_scale()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cam.stop()


def _normal_to_quaternion(normal):
    """将法向量转换为四元数 (x, y, z, w)"""
    n = np.array(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, -1.0])
    cross = np.cross(ref, n)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        return [0.0, 0.0, 0.0, 1.0] if np.dot(ref, n) > 0 else [1.0, 0.0, 0.0, 0.0]
    axis = cross / cross_norm
    angle = math.acos(np.clip(np.dot(ref, n), -1.0, 1.0))
    s = math.sin(angle / 2)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]


def _panel_plane_distance(normal, centroid):
    """计算相机原点到面板平面的垂直距离，单位米。"""
    n = np.array(normal, dtype=np.float64)
    p = np.array(centroid, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        return None
    n = n / n_norm
    return float(abs(np.dot(n, p)))


def _stamp_to_sec(stamp):
    sec, nanosec = _stamp_parts(stamp)
    return float(sec) + float(nanosec) * 1e-9


def _stamp_parts(stamp):
    """Return (seconds, nanoseconds) for ROS1 or ROS2 time objects."""
    sec = getattr(stamp, 'sec', getattr(stamp, 'secs', 0))
    nanosec = getattr(stamp, 'nanosec', getattr(stamp, 'nsecs', 0))
    return int(sec), int(nanosec)


def _pop_synced_frame_pair(color_queue, depth_queue, max_dt):
    """Pop the oldest timestamp-compatible color/depth pair."""
    while color_queue and depth_queue:
        color_item = color_queue[0]
        depth_item = depth_queue[0]
        delta = color_item[0] - depth_item[0]
        if abs(delta) <= max_dt:
            return color_queue.popleft(), depth_queue.popleft()
        if delta < 0.0:
            color_queue.popleft()
        else:
            depth_queue.popleft()
    return None


def _estimate_detection_point(det, depth_image, intrin, depth_scale):
    """
    估计检测目标中心 3D 点。

    话题模式下深度图已通过 depth_registration 对齐到彩色图，采样深度
    应使用检测框原始像素坐标；不在采样前做 undistort，避免取错深度像素。
    """
    ux = int(round(det.center_x))
    uy = int(round(det.center_y))
    if det.class_name in ('button', 'door_button', 'knob'):
        depth = get_bbox_robust_depth(
            depth_image, det.bbox, depth_scale=depth_scale,
            center_ratio=0.45, min_valid=8, quantile=0.5)
    else:
        depth = get_robust_depth(
            depth_image, ux, uy, sample_radius=3, depth_scale=depth_scale)
    xyz = deproject_pixel_to_point(intrin, (ux, uy), depth)
    return ux, uy, xyz


def _estimate_nut_detection_point(localization, depth_image, intrin, depth_scale):
    ux = int(round(localization.center[0]))
    uy = int(round(localization.center[1]))
    depth = get_masked_robust_depth(
        depth_image, localization.depth_mask,
        depth_scale=depth_scale, min_valid=12, quantile=0.5)
    if depth <= 0.0:
        return ux, uy, None
    xyz = deproject_pixel_to_point(intrin, (ux, uy), depth)
    return ux, uy, xyz


def _estimate_bbox_grouping_point(det, depth_image, intrin, depth_scale):
    """Estimate a robust center point used only for fastener grouping."""
    ux = int(round(det.center_x))
    uy = int(round(det.center_y))
    depth = get_bbox_robust_depth(
        depth_image, det.bbox, depth_scale=depth_scale,
        center_ratio=0.70, min_valid=12, quantile=0.5)
    if depth <= 0.0:
        return ux, uy, None
    return ux, uy, deproject_pixel_to_point(intrin, (ux, uy), depth)


def _valid_point_3d(xyz):
    if xyz is None:
        return False
    point = np.asarray(xyz, dtype=np.float64)
    return point.shape == (3,) and np.all(np.isfinite(point)) and point[2] > 0.0


def _is_button_like_detection(det):
    """允许被当作按钮的几何形状：近似方形/圆形，排除细长指示灯。"""
    w = max(1.0, det.bbox[2] - det.bbox[0])
    h = max(1.0, det.bbox[3] - det.bbox[1])
    aspect = w / h
    return 0.55 <= aspect <= 1.8


def _bbox_inside_image(bbox, image_shape, margin_px=4):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin = float(margin_px)
    return x1 > margin and y1 > margin and x2 < (w - margin) and y2 < (h - margin)


def _publish_pose(pub, stamp, xyz, quat):
    if pub is None or xyz is None:
        return
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = 'camera_color_optical_frame'
    msg.pose.position.x = float(xyz[0])
    msg.pose.position.y = float(xyz[1])
    msg.pose.position.z = float(xyz[2])
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    pub.publish(msg)


def _draw_axis_direction(canvas, bbox, normal, label='axis',
                         color=(255, 255, 0)):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    nx, ny, nz = [float(v) for v in normal]

    scale = max(24.0, min(x2 - x1, y2 - y1) * 0.45)
    xy_norm = math.hypot(nx, ny)
    if xy_norm >= 0.03:
        ex = int(round(cx + nx / xy_norm * scale))
        ey = int(round(cy + ny / xy_norm * scale))
        cv2.arrowedLine(canvas, (cx, cy), (ex, ey), color, 2,
                        cv2.LINE_AA, tipLength=0.25)
    else:
        cv2.drawMarker(canvas, (cx, cy), color,
                       markerType=cv2.MARKER_CROSS, markerSize=16,
                       thickness=2)

    text = f'{label} n=({nx:.2f},{ny:.2f},{nz:.2f})'
    text_y = max(15, y1 - 24)
    cv2.putText(canvas, text, (x1, text_y), 0, 0.42,
                color, 1, cv2.LINE_AA)


def _draw_apriltag_reference(canvas, reference, valve_errors=None):
    if reference is None:
        return
    pts = np.round(reference['corners']).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (0, 180, 255), 2, cv2.LINE_AA)
    center = np.mean(pts, axis=0).astype(int)
    normal = reference['normal']
    _draw_axis_direction(
        canvas,
        (center[0] - 24, center[1] - 24, center[0] + 24, center[1] + 24),
        normal,
        label=f"tag_ref[{reference['tag_id']}]",
        color=(0, 180, 255),
    )
    lines = [
        f"{reference['source']} n=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})",
        f"inlier={reference['inlier_ratio']:.2f} rms={reference['rms_error']*1000:.1f}mm",
    ]
    for err in valve_errors or []:
        lines.append(f"valve diff={err['angle_deg']:.1f}deg")
    x = int(np.min(pts[:, 0]))
    y = int(np.max(pts[:, 1])) + 18
    for i, text in enumerate(lines[:5]):
        cv2.putText(canvas, text, (x, y + i * 16), 0, 0.42,
                    (0, 180, 255), 1, cv2.LINE_AA)


def _regular_polygon_points(bbox, n_sides, angle_deg=0.0, angle_mode='edge'):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    radius = max(2.0, min(x2 - x1, y2 - y1) * 0.47)
    if angle_mode == 'vertex':
        start_deg = angle_deg
    else:
        # Offset by half a sector so the fitted angle follows the detected edge
        # orientation rather than a vertex orientation.
        start_deg = angle_deg + 180.0 / max(3, n_sides)
    start = math.radians(start_deg)
    angles = start + np.arange(n_sides, dtype=np.float64) * (2.0 * math.pi / n_sides)
    pts = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])
    return np.round(pts).astype(np.int32)


def _draw_regular_polygon(canvas, bbox, n_sides, angle, color=(0, 255, 255),
                          angle_mode='edge'):
    pts = _regular_polygon_points(bbox, n_sides, angle, angle_mode=angle_mode)
    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
    x1, y1 = int(round(bbox[0])), int(round(bbox[1]))
    cv2.putText(canvas, f'{n_sides}-gon {angle:.1f}deg',
                (x1, max(15, y1 - 8)), 0, 0.42,
                color, 1, cv2.LINE_AA)


def _draw_axis_3d_view(canvas, axis_directions):
    if not axis_directions:
        return

    panel_w = 330
    panel_h = min(235, max(155, 72 + 24 * min(len(axis_directions), 5)))
    margin = 10
    x1 = max(0, canvas.shape[1] - panel_w - margin)
    y1 = margin
    x2 = min(canvas.shape[1] - margin, x1 + panel_w)
    y2 = min(canvas.shape[0] - margin, y1 + panel_h)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.putText(canvas, 'Camera-facing axis view', (x1 + 10, y1 + 20),
                0, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    origin = (x1 + 76, y1 + 92)
    radius = 54
    cv2.circle(canvas, origin, radius, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.circle(canvas, origin, int(radius * 0.5), (65, 65, 65), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin[0] - radius, origin[1]),
             (origin[0] + radius, origin[1]), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin[0], origin[1] - radius),
             (origin[0], origin[1] + radius), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, '+X', (origin[0] + radius + 4, origin[1] + 4),
                0, 0.36, (80, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, '+Y', (origin[0] + 4, origin[1] + radius + 14),
                0, 0.36, (80, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, '-Z to camera', (origin[0] - 43, origin[1] - radius - 8),
                0, 0.36, (255, 180, 80), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, origin, (255, 180, 80),
                   markerType=cv2.MARKER_CROSS, markerSize=12,
                   thickness=1)

    shown = axis_directions[:5]
    for idx, item in enumerate(shown):
        normal = item.get('axis_direction')
        if normal is None:
            continue
        cls_name = item.get('class', 'axis')
        source = item.get('source', '')
        nx, ny, nz = [float(v) for v in normal]
        align = float(np.clip(-nz, 0.0, 1.0))
        if align >= 0.9:
            color = (80, 255, 80)
        elif align >= 0.65:
            color = (0, 220, 255)
        else:
            color = (0, 80, 255)

        px = int(round(origin[0] + np.clip(nx, -1.0, 1.0) * radius))
        py = int(round(origin[1] + np.clip(ny, -1.0, 1.0) * radius))
        cv2.arrowedLine(canvas, origin, (px, py), color, 2,
                        cv2.LINE_AA, tipLength=0.22)
        dot_radius = int(round(4 + 7 * align))
        cv2.circle(canvas, (px, py), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), dot_radius, (20, 20, 20), 1, cv2.LINE_AA)

        tx = x1 + 145
        ty = y1 + 48 + idx * 24
        label = f'{cls_name}[{source}] ({nx:.2f},{ny:.2f},{nz:.2f})'
        cv2.putText(canvas, label, (tx, ty),
                    0, 0.38, color, 1, cv2.LINE_AA)

    if len(axis_directions) > len(shown):
        cv2.putText(canvas, f'+{len(axis_directions) - len(shown)} more',
                    (x1 + 145, y2 - 12), 0, 0.38,
                    (210, 210, 210), 1, cv2.LINE_AA)


def _fit_axis_from_points(points, fallback_axis):
    """用 PCA 拟合目标行方向，并固定为从左到右。"""
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 2:
        axis = np.array(fallback_axis, dtype=np.float64)
    else:
        centered = pts - np.mean(pts, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]

    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        axis = np.array(fallback_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
    axis = axis / max(norm, 1e-6)
    if axis[0] < 0:
        axis = -axis
    return axis


class PositionStabilizer:
    """画面静止后对 3D 坐标做时序滤波，降低 topic 输出跳变。"""

    def __init__(self, enabled=True, still_time=3.0,
                 pixel_thresh=5.0, window_size=45, ema_alpha=0.25,
                 depth_std_thresh=0.01):
        self.enabled = enabled
        self.still_time = float(still_time)
        self.pixel_thresh = float(pixel_thresh)
        self.window_size = int(window_size)
        self.ema_alpha = float(ema_alpha)
        self.depth_std_thresh = float(depth_std_thresh)
        self._last_pixels = {}
        self._still_since = {}
        self._histories = defaultdict(lambda: deque(maxlen=self.window_size))
        self._stable_xyz = {}

    def update(self, measurements, now):
        """
        Args:
            measurements: [(target_id, pixel_xy, xyz), ...]
            now: time.time() 秒

        Returns:
            {target_id: filtered_xyz}
        """
        raw = {tid: np.array(xyz, dtype=np.float64)
               for tid, _, xyz in measurements}
        if not self.enabled or not measurements:
            return {tid: xyz.tolist() for tid, xyz in raw.items()}

        pixels = {tid: np.array(pixel_xy, dtype=np.float64)
                  for tid, pixel_xy, _ in measurements}
        current_ids = set(raw)

        for tid in list(self._last_pixels):
            if tid not in current_ids:
                self._last_pixels.pop(tid, None)
                self._still_since.pop(tid, None)

        filtered = {}
        for tid, xyz in raw.items():
            last_pixel = self._last_pixels.get(tid)
            is_still = (
                last_pixel is not None and
                np.linalg.norm(pixels[tid] - last_pixel) <= self.pixel_thresh
            )
            if is_still:
                self._still_since.setdefault(tid, now)
                self._histories[tid].append(xyz)
            else:
                self._still_since[tid] = now
                self._histories[tid].clear()
                self._histories[tid].append(xyz)
                self._stable_xyz.pop(tid, None)

            self._last_pixels[tid] = pixels[tid]
            history = self._histories[tid]
            static_ready = (
                now - self._still_since.get(tid, now) >= self.still_time and
                len(history) >= 3
            )

            if static_ready:
                stacked = np.vstack(history)
                depth_stable = float(np.std(stacked[:, 2])) <= self.depth_std_thresh
                if depth_stable:
                    median_xyz = np.median(stacked, axis=0)
                    prev = self._stable_xyz.get(tid)
                    stable = median_xyz if prev is None else (
                        (1.0 - self.ema_alpha) * prev + self.ema_alpha * median_xyz)
                    self._stable_xyz[tid] = stable
                    filtered[tid] = stable.tolist()
                    continue

            filtered[tid] = self._stable_xyz.get(tid, xyz).tolist()

        return filtered


class PlaneStabilizer:
    """对面板平面结果做确定性时序平滑，减少 /panel/distance 抖动。"""

    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self._normal = None
        self._centroid = None

    def update(self, normal, centroid):
        normal = np.array(normal, dtype=np.float64)
        centroid = np.array(centroid, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            return None
        normal = normal / norm

        if self._normal is None:
            self._normal = normal
            self._centroid = centroid
        else:
            if np.dot(normal, self._normal) < 0:
                normal = -normal
            self._normal = (1.0 - self.alpha) * self._normal + self.alpha * normal
            self._normal = self._normal / max(np.linalg.norm(self._normal), 1e-10)
            self._centroid = (1.0 - self.alpha) * self._centroid + self.alpha * centroid
        return self._normal.copy(), self._centroid.copy()


def _parse_angle_constraint_mode(value):
    """
    use_constraint 三档模式:
      1: 默认，白色手柄线方向离散输出 0/90，并做时序稳定
      2: 兼容旧 true，启用原物理范围约束
      3: 兼容旧 false，不启用原物理范围约束
    """
    if isinstance(value, bool):
        return 2 if value else 3
    if isinstance(value, (int, float)):
        mode = int(value)
    else:
        text = str(value).strip().lower()
        if text in ('true', 'yes', 'on'):
            return 2
        if text in ('false', 'no', 'off'):
            return 3
        try:
            mode = int(text)
        except ValueError:
            mode = 1
    return mode if mode in (1, 2, 3) else 1


class KnobDiscreteAngleStabilizer:
    """将白色手柄线方向稳定为 0/90，避免临界帧抖动。"""

    def __init__(self, switch_margin=8.0, confirm_frames=3):
        self.switch_margin = float(switch_margin)
        self.confirm_frames = int(confirm_frames)
        self._stable = {}
        self._pending = {}
        self._pending_count = defaultdict(int)

    def update(self, target_id, raw_angle):
        if raw_angle is None:
            return self._stable.get(target_id)

        line_angle = float(raw_angle) % 180.0
        vertical_diff = min(line_angle, 180.0 - line_angle)
        candidate = 0.0 if vertical_diff <= 45.0 else 90.0

        stable = self._stable.get(target_id)
        if stable is None:
            self._stable[target_id] = candidate
            self._clear_pending(target_id)
            return candidate

        if candidate == stable:
            self._clear_pending(target_id)
            return stable

        if not self._beyond_hysteresis(vertical_diff, stable, candidate):
            self._clear_pending(target_id)
            return stable

        if self._pending.get(target_id) != candidate:
            self._pending[target_id] = candidate
            self._pending_count[target_id] = 1
        else:
            self._pending_count[target_id] += 1

        if self._pending_count[target_id] >= self.confirm_frames:
            self._stable[target_id] = candidate
            self._clear_pending(target_id)

        return self._stable[target_id]

    def prune(self, active_ids):
        active_ids = set(active_ids)
        for target_id in list(self._stable):
            if target_id not in active_ids:
                self._stable.pop(target_id, None)
                self._clear_pending(target_id)

    def _clear_pending(self, target_id):
        self._pending.pop(target_id, None)
        self._pending_count.pop(target_id, None)

    def _beyond_hysteresis(self, vertical_diff, stable, candidate):
        if stable == 0.0 and candidate == 90.0:
            return vertical_diff >= 45.0 + self.switch_margin
        if stable == 90.0 and candidate == 0.0:
            return vertical_diff <= 45.0 - self.switch_margin
        return True


def _symmetric_angle_diff_deg(a, b, period=45.0):
    return abs(((float(a) - float(b) + period * 0.5) % period) - period * 0.5)


class ValveAngleStabilizer:
    """Select valve angle candidates using temporal continuity."""

    def __init__(self, enabled=True, max_jump_deg=10.0, confirm_frames=5):
        self.enabled = bool(enabled)
        self.max_jump_deg = float(max_jump_deg)
        self.confirm_frames = int(confirm_frames)
        self._stable = {}
        self._pending = {}
        self._pending_count = defaultdict(int)

    def update(self, target_key, raw_angle, candidates=None):
        if raw_angle is None:
            return self._stable.get(target_key)
        if not self.enabled:
            return float(raw_angle) % 45.0

        options = []
        for angle in list(candidates or []) + [raw_angle]:
            if angle is None:
                continue
            angle = float(angle) % 45.0
            if not any(_symmetric_angle_diff_deg(angle, existing) < 1e-3
                       for existing in options):
                options.append(angle)
        if not options:
            return self._stable.get(target_key)

        stable = self._stable.get(target_key)
        if stable is None:
            selected = options[0]
            self._stable[target_key] = selected
            self._clear_pending(target_key)
            return selected

        selected = min(options, key=lambda angle: _symmetric_angle_diff_deg(angle, stable))
        if _symmetric_angle_diff_deg(selected, stable) <= self.max_jump_deg:
            self._stable[target_key] = selected
            self._clear_pending(target_key)
            return selected

        pending = options[0]
        if self._pending.get(target_key) is None or (
                _symmetric_angle_diff_deg(pending, self._pending[target_key]) > self.max_jump_deg):
            self._pending[target_key] = pending
            self._pending_count[target_key] = 1
        else:
            self._pending[target_key] = pending
            self._pending_count[target_key] += 1

        if self._pending_count[target_key] >= self.confirm_frames:
            self._stable[target_key] = pending
            self._clear_pending(target_key)

        return self._stable[target_key]

    def _clear_pending(self, target_key):
        self._pending.pop(target_key, None)
        self._pending_count.pop(target_key, None)


class PanelDetectionNode(Node):
    def __init__(self):
        super().__init__('panel_detection_node')

        # 加载配置
        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        if config_path and os.path.isfile(config_path):
            self.get_logger().info(f'加载配置: {config_path}')
            with open(config_path, 'r', encoding='utf-8') as f:
                overrides = yaml.load(f.read(), Loader=yaml.SafeLoader) or {}
            self.cfg = _merge_config(copy.deepcopy(DEFAULT_CONFIG), overrides)
        else:
            self.get_logger().info('使用默认配置')
            self.cfg = copy.deepcopy(DEFAULT_CONFIG)

        self.declare_parameter(
            'onnx_model', self.cfg.get('onnx_model', '0813.onnx'))
        self.declare_parameter(
            'onnx_threads', self.cfg.get('onnx_threads', 8))
        self.cfg['onnx_model'] = self.get_parameter('onnx_model').value
        self.cfg['onnx_threads'] = int(self.get_parameter('onnx_threads').value)

        self.declare_parameter('detection_mode', self.cfg.get('detection_mode', 'all'))
        mode_param = self.get_parameter('detection_mode').get_parameter_value().string_value
        self._detection_mode = mode_param if mode_param in MODE_CLASS_FILTERS else 'all'
        if mode_param != self._detection_mode:
            self.get_logger().warn(f'未知 detection_mode={mode_param}, 回退 all')
        self._active_classes = MODE_CLASS_FILTERS[self._detection_mode]
        self._process_panel_controls = self._detection_mode in ('panel_controls', 'all')
        self._process_valves = self._detection_mode in ('valve', 'all')
        self._process_fasteners = self._detection_mode in ('fastener', 'all')
        self.declare_parameter(
            'publish_legacy_topics', self.cfg.get('publish_legacy_topics', False))
        self._publish_legacy_topics = (
            self.get_parameter('publish_legacy_topics').get_parameter_value().bool_value)
        self.get_logger().info(
            f'检测模式: {self._detection_mode}, legacy topics={self._publish_legacy_topics}')

        topics = MODE_TOPIC_MAP[self._detection_mode]

        # 创建话题 publisher (每类一个 PoseStamped，默认关闭，仅兼容旧系统)
        self._pose_pubs = {}
        if self._publish_legacy_topics:
            for cls_name, topic in CLASS_TOPIC_MAP.items():
                if self._active_classes is None or cls_name in self._active_classes:
                    self._pose_pubs[cls_name] = self.create_publisher(PoseStamped, topic, 10)

        self._targets_pub = (
            self.create_publisher(String, topics['targets'], 10)
            if 'targets' in topics else None)
        self._status_pub = (
            self.create_publisher(String, topics['status'], 10)
            if 'status' in topics else None)
        self._angle_pub = (
            self.create_publisher(String, topics['angles'], 10)
            if 'angles' in topics else None)
        self._object_geometry_pub = (
            self.create_publisher(String, topics['geometry'], 10)
            if 'geometry' in topics else None)
        self._distance_pub = (
            self.create_publisher(String, topics['distance'], 10)
            if 'distance' in topics else None)
        # 铭牌 OCR 暂停使用，不创建 /panel/labels publisher。
        self._labels_pub = None

        # 目标注册器
        reg_cfg = self.cfg.get('registry', {})
        self._registry = TargetRegistry(
            stable_frames=reg_cfg.get('stable_frames', 15),
            green_hue_range=tuple(reg_cfg.get('green_hue_range', [35, 85])),
            match_distance_thresh=reg_cfg.get('match_distance_thresh', 80),
        )
        panel_tag_cfg = self.cfg.get('panel_apriltag', {})
        self.declare_parameter(
            'use_panel_tags', panel_tag_cfg.get('enable', True))
        self._panel_tags_enabled = (
            self._process_panel_controls and
            self.get_parameter('use_panel_tags').get_parameter_value().bool_value)
        self._panel_tag_cfg = panel_tag_cfg
        self._panel_tag_tracker = PanelAprilTagTracker(panel_tag_cfg)
        if self._panel_tags_enabled:
            if hasattr(cv2, 'aruco'):
                self.get_logger().info(
                    '面板 AprilTag 编号已启用 (tag36h11, ID 00-39)')
            else:
                self.get_logger().warn(
                    '当前 OpenCV 不包含 aruco，面板 AprilTag 编号不可用')
        fastener_reg_cfg = self.cfg.get('fastener_registry', {})
        self._fastener_registry = FastenerGroupRegistry(
            min_init_observations=fastener_reg_cfg.get('min_init_observations', 2),
            max_group_distance_m=fastener_reg_cfg.get('max_group_distance_m', 0.35),
            max_slot_distance_m=fastener_reg_cfg.get('max_slot_distance_m', 0.12),
            slot_match_ratio=fastener_reg_cfg.get('slot_match_ratio', 0.45),
            normal_angle_thresh_deg=fastener_reg_cfg.get(
                'normal_angle_thresh_deg', 25.0),
            ema_alpha=fastener_reg_cfg.get('ema_alpha', 0.35),
            stale_frames=fastener_reg_cfg.get('stale_frames', 120),
        )
        fastener_axis_cfg = self.cfg.get('fastener_axis_stabilizer', {})
        self._fastener_axis_stabilizer = FastenerAxisStabilizer(
            enabled=fastener_axis_cfg.get('enable', True),
            ema_alpha=fastener_axis_cfg.get('ema_alpha', 0.18),
            max_jump_deg=fastener_axis_cfg.get('max_jump_deg', 12.0),
            confirm_frames=fastener_axis_cfg.get('confirm_frames', 5),
            max_output_step_deg=fastener_axis_cfg.get(
                'max_output_step_deg', 3.0),
            match_distance_px=fastener_axis_cfg.get('match_distance_px', 60.0),
            match_size_ratio=fastener_axis_cfg.get('match_size_ratio', 1.2),
            stale_frames=fastener_axis_cfg.get('stale_frames', 60),
        )

        # 相机来源：直接连接 or 话题订阅
        self.declare_parameter('use_topic', True)
        self._use_topic = self.get_parameter('use_topic').get_parameter_value().bool_value

        self._camera = None
        # Orbbec 官方驱动深度图单位为 mm，× 0.001 转米
        self.declare_parameter('depth_scale', self.cfg.get('depth_scale', 0.001))
        self._depth_scale = float(self.get_parameter('depth_scale').value)
        self._camera_ready = False

        # 话题订阅模式的缓存
        self._topic_color = None
        self._topic_depth = None
        self._topic_color_stamp = None
        self._topic_depth_stamp = None
        self._topic_color_stamp_msg = None
        self._topic_depth_stamp_msg = None
        self._topic_color_queue = deque(maxlen=10)
        self._topic_depth_queue = deque(maxlen=10)
        self._topic_pair_queue = deque(maxlen=10)
        self._last_topic_enqueue_stamp = None
        self._topic_color_intrin = None
        self._topic_depth_intrin = None
        self._topic_frame_lock = threading.Lock()
        sync_cfg = self.cfg.get('topic_sync', {})
        self.declare_parameter('registered_depth', sync_cfg.get('registered_depth', True))
        self._registered_depth = (
            self.get_parameter('registered_depth').get_parameter_value().bool_value)
        self._topic_sync_max_dt = float(sync_cfg.get('max_dt', 0.05))
        self._last_sync_warn = 0.0
        self._intrinsics_checked = False

        # 直连模式下转发图像话题，供其他节点使用
        self._color_pub = None
        self._depth_pub = None
        self._color_info_pub = None
        self._depth_info_pub = None

        if self._use_topic:
            self._init_subscribers()
            self._camera_ready = True
            self.get_logger().info('使用话题订阅模式')
        else:
            self._init_camera()
            # 直连模式下发布图像话题
            self._color_pub = self.create_publisher(Image, '/camera/color/image_raw', 5)
            self._depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 5)
            self._color_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 5)
            self._depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 5)
            self.get_logger().info('直连模式：同时发布相机话题供其他节点使用')

        # 初始化检测器
        self._detector = self._create_detector()
        self._class_names = getattr(self._detector, 'class_names', None) or \
            self.cfg.get('class_name', [])

        # 旋钮角度配置
        angle_cfg = self.cfg.get('knob_angle', {})
        self._angle_enable = angle_cfg.get('enable', False)
        self._angle_binary_thresh = angle_cfg.get('binary_thresh', 180)
        self._angle_circle_mask = angle_cfg.get('circle_mask_ratio', 0.85)
        self._angle_knob_class = angle_cfg.get('knob_class', 'knob')
        self.declare_parameter('use_constraint', str(angle_cfg.get('use_constraint', 1)))
        self.declare_parameter('use_constrain', '')
        constraint_alias = self.get_parameter('use_constrain').value
        constraint_param = constraint_alias or self.get_parameter('use_constraint').value
        self._angle_constraint_mode = _parse_angle_constraint_mode(constraint_param)
        self._knob_angle_stabilizer = KnobDiscreteAngleStabilizer(
            switch_margin=angle_cfg.get('discrete_switch_margin', 8.0),
            confirm_frames=angle_cfg.get('discrete_confirm_frames', 3),
        )
        valve_angle_cfg = self.cfg.get('valve_angle_stabilizer', {})
        self._valve_angle_stabilizer = ValveAngleStabilizer(
            enabled=valve_angle_cfg.get('enable', True),
            max_jump_deg=valve_angle_cfg.get('max_jump_deg', 10.0),
            confirm_frames=valve_angle_cfg.get('confirm_frames', 5),
        )
        self.get_logger().info(
            f'旋钮角度模式: {self._angle_constraint_mode} '
            '(1=0/90稳定输出, 2=旧约束, 3=旧无约束)')

        # 铭牌 OCR 暂停使用，不初始化 RapidOCR/PaddleOCR，也不绘制 OCR ROI。
        self._nameplate_enable = False
        self._nameplate_recognizer = None

        pos_cfg = self.cfg.get('position_stabilizer', {})
        self._position_stabilizer = PositionStabilizer(
            enabled=pos_cfg.get('enable', True),
            still_time=pos_cfg.get('still_time', 3.0),
            pixel_thresh=pos_cfg.get('pixel_thresh', 5.0),
            window_size=pos_cfg.get('window_size', 45),
            ema_alpha=pos_cfg.get('ema_alpha', 0.25),
            depth_std_thresh=pos_cfg.get('depth_std_thresh', 0.01),
        )
        fastener_pos_cfg = self.cfg.get('fastener_position_stabilizer', {})
        self._fastener_position_stabilizer = PositionStabilizer(
            enabled=fastener_pos_cfg.get('enable', True),
            still_time=fastener_pos_cfg.get('still_time', 1.0),
            pixel_thresh=fastener_pos_cfg.get('pixel_thresh', 5.0),
            window_size=fastener_pos_cfg.get('window_size', 30),
            ema_alpha=fastener_pos_cfg.get('ema_alpha', 0.20),
            depth_std_thresh=fastener_pos_cfg.get('depth_std_thresh', 0.008),
        )
        self._panel_line_cfg = self.cfg.get('panel_line', {})
        self._persistent_panel_axis = PersistentPanelAxis(
            dist_ratio=self._panel_line_cfg.get('dist_ratio', 0.85),
            min_dist=self._panel_line_cfg.get('min_dist', 45.0),
            max_dist=self._panel_line_cfg.get('max_dist', 110.0),
            min_proj_margin=self._panel_line_cfg.get('min_proj_margin', 450.0),
        )

        # 面板法向量
        self._panel_normal_cache = None
        self._panel_plane_stabilizer = PlaneStabilizer(
            alpha=self.cfg.get('panel_normal_alpha', 0.25))
        self._frame_count = 0
        self._normal_interval = self.cfg.get('panel_normal_interval', 10)
        self._last_status_publish_time = 0.0
        ref_cfg = self.cfg.get('apriltag_reference', {})
        self._apriltag_reference_enabled = bool(ref_cfg.get('enable', False))
        self._axis_ref_log_fp = None
        self._axis_ref_log_path = ''
        if self._apriltag_reference_enabled and ref_cfg.get('log_enable', True):
            self._axis_ref_log_path = os.path.abspath(
                ref_cfg.get('log_path', 'axis_reference_log.jsonl'))
            try:
                self._axis_ref_log_fp = open(self._axis_ref_log_path, 'w',
                                             encoding='utf-8', buffering=1)
                self.get_logger().info(
                    f'AprilTag 参考轴线日志: {self._axis_ref_log_path}')
            except OSError as exc:
                self.get_logger().warn(f'无法写入参考轴线日志: {exc}')

        # 重连
        self._reconnect_interval = 5.0
        self._last_reconnect_time = 0.0

        # 可视化
        self.display_frame = None
        self.declare_parameter('capture_dir', '')
        self.declare_parameter('capture_hz', 1.0)
        self.declare_parameter('show_gui', False)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '~debug_image')
        publish_debug_image = self.get_parameter(
            'publish_debug_image').get_parameter_value().bool_value
        debug_image_topic = self.get_parameter(
            'debug_image_topic').get_parameter_value().string_value
        self._debug_image_pub = (
            self.create_publisher(Image, debug_image_topic, 1)
            if publish_debug_image else None)
        self._capture_dir = self.get_parameter(
            'capture_dir').get_parameter_value().string_value
        capture_hz = self.get_parameter(
            'capture_hz').get_parameter_value().double_value
        self._capture_interval = 1.0 / capture_hz if capture_hz > 0.0 else None
        self._last_capture_stamp = None
        if self._capture_dir:
            self._capture_dir = os.path.abspath(os.path.expanduser(self._capture_dir))
            os.makedirs(self._capture_dir, exist_ok=True)
            self.get_logger().info(
                f'检测画面保存: {self._capture_dir} ({capture_hz:g} Hz)')

        self._gui_enabled = False
        if self.get_parameter('show_gui').get_parameter_value().bool_value:
            try:
                cv2.namedWindow(
                    'panel_detection',
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self._gui_enabled = True
            except cv2.error:
                self.get_logger().warn('无显示环境，禁用 GUI 可视化')

        status = '运行中' if self._camera_ready else '等待相机'
        self.get_logger().info(f'面板检测节点已启动 ({status})')

    def _write_axis_reference_log(self, stamp, axis_reference, valve_errors):
        if self._axis_ref_log_fp is None or axis_reference is None or not valve_errors:
            return
        stamp_secs, stamp_nanosecs = _stamp_parts(stamp)
        stamp_sec = _stamp_to_sec(stamp)
        for err in valve_errors:
            record = {
                'stamp': stamp_sec,
                'stamp_sec': stamp_secs,
                'stamp_nanosec': stamp_nanosecs,
                'frame': int(self._frame_count),
                'reference': {
                    'source': axis_reference.get('source'),
                    'tag_id': axis_reference.get('tag_id'),
                    'normal': axis_reference.get('normal'),
                    'centroid': axis_reference.get('centroid'),
                    'point_count': axis_reference.get('point_count'),
                    'inlier_ratio': axis_reference.get('inlier_ratio'),
                    'rms_error_m': axis_reference.get('rms_error_m'),
                },
                'valve': err,
            }
            self._axis_ref_log_fp.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _update_display_frame(self, canvas, stamp=None):
        self.display_frame = canvas

        if self._debug_image_pub is not None:
            debug_msg = Image()
            debug_msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
            debug_msg.header.frame_id = 'camera_color_optical_frame'
            debug_msg.height, debug_msg.width = canvas.shape[:2]
            debug_msg.encoding = 'bgr8'
            debug_msg.is_bigendian = 0
            debug_msg.step = debug_msg.width * 3
            debug_msg.data = np.ascontiguousarray(canvas).tobytes()
            self._debug_image_pub.publish(debug_msg)

        if not self._capture_dir or self._capture_interval is None:
            return

        if stamp is None:
            stamp_seconds = time.time()
            stamp_sec = int(stamp_seconds)
            stamp_nanosec = int(round(
                (stamp_seconds - stamp_sec) * 1e9))
        else:
            stamp_sec, stamp_nanosec = _stamp_parts(stamp)
            stamp_seconds = stamp_sec + stamp_nanosec * 1e-9

        if (self._last_capture_stamp is not None and
                stamp_seconds - self._last_capture_stamp <
                self._capture_interval - 1e-6):
            return

        filename = f'{stamp_sec}_{stamp_nanosec:09d}.jpg'
        output_path = os.path.join(self._capture_dir, filename)
        if cv2.imwrite(
                output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            self._last_capture_stamp = stamp_seconds
        else:
            self.get_logger().warn(f'检测画面保存失败: {output_path}')

    def destroy_node(self):
        if self._axis_ref_log_fp is not None:
            try:
                self._axis_ref_log_fp.close()
            finally:
                self._axis_ref_log_fp = None
        super().destroy_node()

    def _init_subscribers(self):
        """订阅相机话题，检测循环按 header stamp 取近似同步帧。"""
        self.declare_parameter(
            'color_topic', '/camera_d435_0/color/image_raw')
        self.declare_parameter(
            'depth_topic', '/camera_d435_0/aligned_depth_to_color/image_raw')
        self.declare_parameter(
            'color_info_topic', '/camera_d435_0/color/camera_info')
        self.declare_parameter(
            'depth_info_topic', '/camera_d435_0/color/camera_info')
        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        color_info_topic = self.get_parameter('color_info_topic').value
        depth_info_topic = self.get_parameter('depth_info_topic').value

        # 保留短队列，供推理循环按时间戳消费同步帧。
        self._sub_depth_info = None
        self._sub_color = self.create_subscription(
            Image, color_topic, self._color_callback, 10)
        self._sub_depth = self.create_subscription(
            Image, depth_topic, self._depth_callback, 10)
        self._sub_color_info = self.create_subscription(
            CameraInfo, color_info_topic, self._color_info_callback, 5)
        if depth_info_topic == color_info_topic and self._registered_depth:
            self._sub_depth_info = None
        else:
            self._sub_depth_info = self.create_subscription(
                CameraInfo, depth_info_topic, self._depth_info_callback, 5)
        self.get_logger().info(
            f'相机话题: color={color_topic}, depth={depth_topic}, '
            f'camera_info={color_info_topic}')

    def _color_callback(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        # OrbbecSDK_ROS2 发布 rgb8，转换为 BGR 供 OpenCV/YOLO 使用
        if msg.encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        with self._topic_frame_lock:
            self._topic_color = image
            self._topic_color_stamp = _stamp_to_sec(msg.header.stamp)
            self._topic_color_stamp_msg = msg.header.stamp
            self._topic_color_queue.append((
                self._topic_color_stamp, msg.header.stamp, image))
            self._queue_synced_pairs_locked()

    def _depth_callback(self, msg):
        if msg.encoding in ('16UC1', 'mono16'):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding == '32FC1':
            depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            depth = np.nan_to_num(depth_m / max(self._depth_scale, 1e-9)).astype(np.uint16)
        else:
            self.get_logger().warn(f'不支持的深度图编码: {msg.encoding}')
            return
        with self._topic_frame_lock:
            self._topic_depth = depth
            self._topic_depth_stamp = _stamp_to_sec(msg.header.stamp)
            self._topic_depth_stamp_msg = msg.header.stamp
            self._topic_depth_queue.append((
                self._topic_depth_stamp, msg.header.stamp, depth))
            self._queue_synced_pairs_locked()

    def _color_info_callback(self, msg):
        matrix = msg.K if hasattr(msg, 'K') else msg.k
        distortion = msg.D if hasattr(msg, 'D') else msg.d
        intrin = CameraIntrinsics(
            fx=matrix[0], fy=matrix[4],
            cx=matrix[2], cy=matrix[5],
            width=msg.width, height=msg.height,
            coeffs=list(distortion) if distortion else [0.0]*5,
        )
        with self._topic_frame_lock:
            self._topic_color_intrin = intrin
            if self._registered_depth and self._sub_depth_info is None:
                self._topic_depth_intrin = intrin

    def _depth_info_callback(self, msg):
        matrix = msg.K if hasattr(msg, 'K') else msg.k
        distortion = msg.D if hasattr(msg, 'D') else msg.d
        intrin = CameraIntrinsics(
            fx=matrix[0], fy=matrix[4],
            cx=matrix[2], cy=matrix[5],
            width=msg.width, height=msg.height,
            coeffs=list(distortion) if distortion else [0.0]*5,
        )
        with self._topic_frame_lock:
            self._topic_depth_intrin = intrin

    def _queue_synced_pairs_locked(self):
        while True:
            pair = _pop_synced_frame_pair(
                self._topic_color_queue,
                self._topic_depth_queue,
                self._topic_sync_max_dt,
            )
            if pair is None:
                return
            color_item, depth_item = pair
            if self._capture_dir and self._capture_interval is not None:
                if (self._last_topic_enqueue_stamp is not None and
                        color_item[0] - self._last_topic_enqueue_stamp <
                        self._capture_interval - 1e-6):
                    continue
                self._last_topic_enqueue_stamp = color_item[0]
            else:
                # Normal live mode prioritizes the freshest complete pair.
                self._topic_pair_queue.clear()
            self._topic_pair_queue.append((color_item, depth_item))

    def _get_synced_topic_frames(self):
        with self._topic_frame_lock:
            if (self._topic_color_intrin is None or
                    self._topic_depth_intrin is None or
                    not self._topic_pair_queue):
                return None
            color_item, depth_item = self._topic_pair_queue.popleft()
            return (
                self._topic_color_intrin,
                self._topic_depth_intrin,
                color_item[2],
                depth_item[2],
                color_item[1],
                depth_item[1],
            )

    def _select_deprojection_intrin(self, color_intrin, depth_intrin,
                                    color_shape, depth_shape):
        if not self._use_topic:
            return depth_intrin
        same_size = color_shape[:2] == depth_shape[:2]
        intrin_diff = (
            abs(color_intrin.fx - depth_intrin.fx) > 1.0 or
            abs(color_intrin.fy - depth_intrin.fy) > 1.0 or
            abs(color_intrin.cx - depth_intrin.cx) > 1.0 or
            abs(color_intrin.cy - depth_intrin.cy) > 1.0
        )
        if not self._intrinsics_checked:
            self._intrinsics_checked = True
            if same_size and intrin_diff and self._registered_depth:
                self.get_logger().warn(
                    '深度图和彩色图尺寸相同但内参不同；registered_depth=true，'
                    '将使用彩色内参反投影注册后的深度图。')
            elif same_size and intrin_diff:
                self.get_logger().warn(
                    '深度图和彩色图尺寸相同但内参不同；registered_depth=false，'
                    '仍使用深度内参反投影，请确认话题配置。')
        if same_size and self._registered_depth:
            return color_intrin
        return depth_intrin

    def _init_camera(self):
        backend_type = self.cfg.get('camera_backend', 'orbbec')
        cam_cfg = self.cfg.get('camera', {})
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                raw_cam = create_backend(backend_type)
                raw_cam.initialize(
                    cam_cfg.get('color_width', 640),
                    cam_cfg.get('color_height', 480),
                    cam_cfg.get('depth_width', 640),
                    cam_cfg.get('depth_height', 480),
                    cam_cfg.get('fps', 30),
                )
                self._camera = AsyncCamera(raw_cam)
                self._depth_scale = self._camera.get_depth_scale()
                self._camera_ready = True
                self.get_logger().info(
                    f'相机已启动: {backend_type}, depth_scale={self._depth_scale}')
                return
            except Exception as e:
                self.get_logger().warn(
                    f'相机初始化失败 ({attempt}/{max_retries}): {e}')
                if attempt < max_retries:
                    time.sleep(3.0)
        self.get_logger().error('相机初始化失败，节点将等待相机连接')

    def _try_reconnect_camera(self):
        now = time.time()
        if now - self._last_reconnect_time < self._reconnect_interval:
            return
        self._last_reconnect_time = now
        backend_type = self.cfg.get('camera_backend', 'orbbec')
        cam_cfg = self.cfg.get('camera', {})
        try:
            raw_cam = create_backend(backend_type)
            raw_cam.initialize(
                cam_cfg.get('color_width', 640),
                cam_cfg.get('color_height', 480),
                cam_cfg.get('depth_width', 640),
                cam_cfg.get('depth_height', 480),
                cam_cfg.get('fps', 30),
            )
            self._camera = AsyncCamera(raw_cam)
            self._depth_scale = self._camera.get_depth_scale()
            self._camera_ready = True
            self.get_logger().info(f'相机已连接: {backend_type}')
        except Exception:
            pass

    def _create_detector(self):
        backend = self.cfg.get('inference_backend', 'onnx')
        pkg_dir = os.path.dirname(os.path.abspath(__file__))

        if backend == 'onnx':
            from .detector_onnx import YoloV5ORT
            onnx_path = self.cfg.get('onnx_model', '0813.onnx')
            if not os.path.isabs(onnx_path):
                module_path = os.path.join(pkg_dir, onnx_path)
                if os.path.isfile(module_path):
                    onnx_path = module_path
                else:
                    import rospkg
                    package_path = rospkg.RosPack().get_path('panel_detection')
                    onnx_path = os.path.join(package_path, 'models', onnx_path)
            threads = self.cfg.get('onnx_threads', 4)
            self.get_logger().info(f'推理后端: ONNX Runtime ({onnx_path})')

            import tempfile
            tmp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.cfg, tmp_cfg)
            tmp_cfg.close()

            det = YoloV5ORT(onnx_path=onnx_path, config_path=tmp_cfg.name, threads=threads)
            os.unlink(tmp_cfg.name)

            # 从配置文件设置类别名
            class_names = self.cfg.get('class_name', [])
            if class_names:
                import random
                det.class_names = class_names
                det.class_num = len(class_names)
                det.colors = [[random.randint(0, 255) for _ in range(3)]
                              for _ in range(det.class_num)]
            self.get_logger().info(f'模型类别: {det.class_names}')
            return det

        if backend == 'rknn':
            from .detector_rknn import YoloV5RKNN
            rknn_path = self.cfg.get('rknn_model', '0520.rknn')
            if not os.path.isabs(rknn_path):
                rknn_path = os.path.join(pkg_dir, rknn_path)
            self.get_logger().info(f'推理后端: RKNN ({rknn_path})')

            import tempfile
            tmp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(self.cfg, tmp_cfg)
            tmp_cfg.close()
            det = YoloV5RKNN(rknn_model_path=rknn_path, config_path=tmp_cfg.name)
            os.unlink(tmp_cfg.name)
            return det

        return None

    def _publish_camera_topics(self, color_image, depth_image, color_intrin, depth_intrin):
        """直连模式下转发图像和内参话题"""
        stamp = self.get_clock().now().to_msg()

        # 发布彩色图像
        color_msg = Image()
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = 'camera_color_optical_frame'
        color_msg.height, color_msg.width = color_image.shape[:2]
        color_msg.encoding = 'bgr8'
        color_msg.step = color_msg.width * 3
        color_msg.data = color_image.tobytes()
        self._color_pub.publish(color_msg)

        # 发布深度图像
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'camera_depth_optical_frame'
        depth_msg.height, depth_msg.width = depth_image.shape[:2]
        depth_msg.encoding = '16UC1'
        depth_msg.step = depth_msg.width * 2
        depth_msg.data = depth_image.tobytes()
        self._depth_pub.publish(depth_msg)

        # 发布彩色相机内参
        color_info_msg = CameraInfo()
        color_info_msg.header.stamp = stamp
        color_info_msg.header.frame_id = 'camera_color_optical_frame'
        color_info_msg.width = color_intrin.width
        color_info_msg.height = color_intrin.height
        color_info_msg.K = [color_intrin.fx, 0.0, color_intrin.cx,
                            0.0, color_intrin.fy, color_intrin.cy,
                            0.0, 0.0, 1.0]
        color_info_msg.D = color_intrin.coeffs
        self._color_info_pub.publish(color_info_msg)

        # 发布深度相机内参
        depth_info_msg = CameraInfo()
        depth_info_msg.header.stamp = stamp
        depth_info_msg.header.frame_id = 'camera_depth_optical_frame'
        depth_info_msg.width = depth_intrin.width
        depth_info_msg.height = depth_intrin.height
        depth_info_msg.K = [depth_intrin.fx, 0.0, depth_intrin.cx,
                            0.0, depth_intrin.fy, depth_intrin.cy,
                            0.0, 0.0, 1.0]
        depth_info_msg.D = depth_intrin.coeffs
        self._depth_info_pub.publish(depth_info_msg)

    def _publish_status(self, status, force=False):
        if self._status_pub is None:
            return
        now = time.time()
        if not force and now - self._last_status_publish_time < 1.0:
            return
        self._last_status_publish_time = now
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _detection_callback(self):
        if not self._camera_ready:
            self._publish_status('waiting_camera')
            self._try_reconnect_camera()
            return

        if self._use_topic:
            topic_frames = self._get_synced_topic_frames()
            if topic_frames is None:
                self._publish_status('waiting_topic_frames')
                return
            color_intrin, depth_intrin, color_image, depth_image, color_stamp, _ = topic_frames
            deproj_intrin = self._select_deprojection_intrin(
                color_intrin, depth_intrin, color_image.shape, depth_image.shape)
        else:
            color_intrin, depth_intrin, color_image, depth_image = \
                self._camera.get_aligned_frames()
            if color_intrin is None:
                return
            deproj_intrin = depth_intrin
            color_stamp = None

        if color_image.size == 0 or depth_image.size == 0:
            return

        # 直连模式下转发图像话题（仅在有订阅者时发布）
        if self._color_pub is not None and (
                self._color_pub.get_subscription_count() > 0 or
                self._depth_pub.get_subscription_count() > 0):
            self._publish_camera_topics(color_image, depth_image, color_intrin, depth_intrin)

        t0 = time.time()
        canvas, class_id_list, xyxy_list, conf_list = self._detector.detect(color_image)
        t1 = time.time()

        # Detect panel tags independently of YOLO. This keeps decoded tag
        # outlines visible while commissioning a tag sheet or when the
        # associated component is temporarily missed by the detector.
        panel_tag_markers = []
        if self._panel_tags_enabled:
            panel_tag_markers = detect_panel_tags(color_image, self._panel_tag_cfg)

        if not xyxy_list:
            if self._panel_tags_enabled:
                self._panel_tag_tracker.update([], panel_tag_markers)
                draw_panel_tag_assignments(
                    canvas, panel_tag_markers, {}, [])
            self._update_display_frame(canvas, color_stamp)
            self._publish_status('no_detection')
            return

        filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)
        t2 = time.time()
        if not self._use_topic:
            self._depth_scale = self._camera.get_depth_scale()

        # 面板法向量
        self._frame_count += 1
        if (self._panel_normal_cache is None or
                self._frame_count % self._normal_interval == 0):
            result = compute_panel_normal(
                color_image, filtered_depth, deproj_intrin, xyxy_list,
                depth_scale=self._depth_scale)
            if result is not None:
                stable_plane = self._panel_plane_stabilizer.update(*result)
                if stable_plane is not None:
                    self._panel_normal_cache = stable_plane
        t3 = time.time()

        # 每 30 帧打印一次耗时
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'耗时: 推理={int((t1-t0)*1000)}ms 滤波={int((t2-t1)*1000)}ms '
                f'法向量={int((t3-t2)*1000)}ms 总={int((t3-t0)*1000)}ms')

        # 获取四元数（面板法向量）
        quat = [0.0, 0.0, 0.0, 1.0]
        if self._panel_normal_cache is not None:
            normal, _ = self._panel_normal_cache
            quat = _normal_to_quaternion(normal)

        stamp = color_stamp if color_stamp is not None else self.get_clock().now().to_msg()

        # 构建当前帧的 FrameDetection 列表
        frame_detections = []
        for i, xyxy in enumerate(xyxy_list):
            cls_id = class_id_list[i]
            cls_name = self._class_names[cls_id] if cls_id < len(self._class_names) else 'unknown'
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            frame_detections.append(FrameDetection(
                class_name=cls_name,
                center_x=cx,
                center_y=cy,
                bbox=tuple(xyxy[:4]),
                confidence=float(conf_list[i]),
            ))

        if self._active_classes is not None:
            frame_detections = [
                det for det in frame_detections
                if det.class_name in self._active_classes
            ]
        active_xyxy_list = [det.bbox for det in frame_detections]
        if not frame_detections:
            if self._panel_tags_enabled:
                self._panel_tag_tracker.update([], panel_tag_markers)
                draw_panel_tag_assignments(
                    canvas, panel_tag_markers, {}, [])
            self._update_display_frame(canvas, color_stamp)
            self._publish_status('no_detection')
            return

        panel_tag_assignments = {}
        tagged_detection_ids = set()
        no_tag_light_ids = set()
        if self._panel_tags_enabled:
            if panel_tag_markers:
                panel_tag_assignments = self._panel_tag_tracker.update(
                    frame_detections, panel_tag_markers)
                for det_index, assignment in panel_tag_assignments.items():
                    frame_detections[det_index].class_name = assignment.forced_class
                    tagged_detection_ids.add(id(frame_detections[det_index]))
            else:
                # Current-frame Tag visibility is authoritative. Do not let a
                # stale track keep a YOLO button classified as a button.
                self._panel_tag_tracker.update([], [])
                changed = reclassify_buttons_without_tags(
                    frame_detections, panel_tag_markers)
                no_tag_light_ids = {id(detection) for detection in changed}
                for detection in changed:
                    x1, y1, x2, y2 = [
                        int(round(value)) for value in detection.bbox]
                    cv2.rectangle(canvas, (x1, max(0, y1 - 22)),
                                  (min(canvas.shape[1] - 1, x1 + 64), y1),
                                  (0, 0, 0), -1)
                    cv2.putText(canvas, 'light', (x1 + 2, max(14, y1 - 5)),
                                0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2),
                                  (255, 255, 0), 2, cv2.LINE_AA)
            draw_panel_tag_assignments(
                canvas, panel_tag_markers, panel_tag_assignments, frame_detections)

        nut_localizations = {}
        for i, det in enumerate(frame_detections):
            if self._process_fasteners and det.class_name == 'nut':
                nut_localizations[i] = localize_nut(color_image, det.bbox)

        # 检测到两个 knob 时：
        # 1. 用两枚 knob 连线粗定位目标行
        # 2. 用目标行附近的 button/knob/button-like light 重新拟合整排轴线
        # 3. 只有最终轴线附近的 button/knob 参与编号；细长 light 不参与，避免编号贴到灯上
        knobs = [d for d in frame_detections if d.class_name == 'knob']
        panel_detections = []  # 只有在面板行上的目标才参与编号
        panel_axis_origin = None
        panel_axis_vector = None
        matched = []

        if not self._process_panel_controls:
            matched = []
        elif len(knobs) >= 2:
            k1, k2 = sorted(knobs, key=lambda d: d.center_x)[:2]
            line_vec = np.array([k2.center_x - k1.center_x, k2.center_y - k1.center_y])
            line_len = np.linalg.norm(line_vec)
            if line_len > 1:
                line_unit = line_vec / line_len
                line_normal = np.array([-line_unit[1], line_unit[0]])

                size_candidates = [d for d in frame_detections
                                   if d.class_name in ('button', 'knob')]
                median_h = np.median([max(1.0, d.bbox[3] - d.bbox[1])
                                      for d in size_candidates]) if size_candidates else 80.0
                initial_thresh = float(np.clip(
                    median_h * self._panel_line_cfg.get('initial_dist_ratio', 1.1),
                    self._panel_line_cfg.get('min_dist', 45.0),
                    self._panel_line_cfg.get('max_dist', 110.0)))
                proj_margin = max(
                    self._panel_line_cfg.get('min_proj_margin', 450.0),
                    line_len * self._panel_line_cfg.get('proj_margin_ratio', 3.0))

                rough_row = []
                for d in frame_detections:
                    can_join_row = (
                        d.class_name in ('button', 'knob') or
                        (d.class_name == 'light' and _is_button_like_detection(d))
                    )
                    if not can_join_row:
                        continue
                    pt = np.array([d.center_x - k1.center_x, d.center_y - k1.center_y])
                    dist = abs(np.dot(pt, line_normal))
                    proj = float(np.dot(pt, line_unit))
                    if dist <= initial_thresh and -proj_margin <= proj <= line_len + proj_margin:
                        rough_row.append(d)

                if len(rough_row) >= 3:
                    fitted_axis = _fit_axis_from_points(
                        [(d.center_x, d.center_y) for d in rough_row],
                        line_unit)
                    axis_origin_arr = np.mean(
                        np.array([(d.center_x, d.center_y) for d in rough_row],
                                 dtype=np.float64),
                        axis=0)
                else:
                    fitted_axis = line_unit
                    axis_origin_arr = np.array([k1.center_x, k1.center_y], dtype=np.float64)

                fitted_normal = np.array([-fitted_axis[1], fitted_axis[0]])
                projections = [
                    float(np.dot(
                        np.array([d.center_x, d.center_y], dtype=np.float64) - axis_origin_arr,
                        fitted_axis))
                    for d in rough_row
                ]
                if projections:
                    proj_min = min(projections) - self._panel_line_cfg.get('min_proj_margin', 450.0)
                    proj_max = max(projections) + self._panel_line_cfg.get('min_proj_margin', 450.0)
                else:
                    proj_min = -proj_margin
                    proj_max = line_len + proj_margin

                line_dist_thresh = float(np.clip(
                    median_h * self._panel_line_cfg.get('dist_ratio', 0.85),
                    self._panel_line_cfg.get('min_dist', 45.0),
                    self._panel_line_cfg.get('max_dist', 110.0)))
                panel_axis_origin = (float(axis_origin_arr[0]), float(axis_origin_arr[1]))
                panel_axis_vector = (float(fitted_axis[0]), float(fitted_axis[1]))
                self._persistent_panel_axis.update(panel_axis_origin, panel_axis_vector)

                for d in frame_detections:
                    can_join_row = (
                        d.class_name in ('button', 'knob') or
                        (d.class_name == 'light' and _is_button_like_detection(d))
                    )
                    if not can_join_row:
                        continue
                    pt = np.array([d.center_x, d.center_y], dtype=np.float64) - axis_origin_arr
                    dist = abs(np.dot(pt, fitted_normal))
                    proj = float(np.dot(pt, fitted_axis))
                    on_line = (
                        dist <= line_dist_thresh and
                        proj_min <= proj <= proj_max
                    )
                    if (d.class_name == 'light' and on_line and
                            id(d) not in tagged_detection_ids and
                            id(d) not in no_tag_light_ids):
                        d.class_name = 'button'
                        x1, y1 = int(d.bbox[0]), int(d.bbox[1])
                        cv2.rectangle(canvas, (x1, y1 - 20), (x1 + 80, y1), (0, 0, 0), -1)
                        cv2.putText(canvas, 'button', (x1, y1 - 5),
                                    0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # 只有在连线上的 button/knob 参与编号
                    if on_line and d.class_name in ('button', 'knob'):
                        panel_detections.append(d)
            else:
                # knob 重叠退化，取所有 button/knob
                axis_candidates = [
                    d for d in frame_detections
                    if id(d) not in no_tag_light_ids and (
                        id(d) not in tagged_detection_ids or
                        d.class_name in ('button', 'knob'))
                ]
                cached_axis_result = self._persistent_panel_axis.select(axis_candidates)
                if cached_axis_result is not None:
                    panel_detections, panel_axis_origin, panel_axis_vector = cached_axis_result
                else:
                    panel_detections = [d for d in frame_detections
                                        if d.class_name in ('button', 'knob')]
        else:
            # 没有两个 knob 可见，取所有 button/knob
            axis_candidates = [
                d for d in frame_detections
                if id(d) not in no_tag_light_ids and (
                    id(d) not in tagged_detection_ids or
                    d.class_name in ('button', 'knob'))
            ]
            cached_axis_result = self._persistent_panel_axis.select(axis_candidates)
            if cached_axis_result is not None:
                panel_detections, panel_axis_origin, panel_axis_vector = cached_axis_result
            else:
                panel_detections = [d for d in frame_detections
                                    if d.class_name in ('button', 'knob')]

        if self._process_panel_controls:
            direct_matches = [
                (assignment.target_id, frame_detections[det_index])
                for det_index, assignment in panel_tag_assignments.items()
            ]
            fallback_detections = [
                detection for detection in panel_detections
                if id(detection) not in tagged_detection_ids
            ]
            fallback_matches = self._registry.identify(
                fallback_detections, color_image,
                axis_origin=panel_axis_origin,
                axis_vector=panel_axis_vector)
            direct_target_ids = {target_id for target_id, _ in direct_matches}
            fallback_matches = [
                item for item in fallback_matches
                if item[0] not in direct_target_ids
            ]
            matched = sorted(
                direct_matches + fallback_matches, key=lambda item: item[0])

        tag_assignment_by_detection = {
            id(frame_detections[det_index]): assignment
            for det_index, assignment in panel_tag_assignments.items()
        }

        targets_output = []
        knob_angles = []
        matched_xyz = {}  # 缓存 3D 坐标避免重复计算
        position_measurements = []

        for target_id, det in matched:
            ux, uy, xyz = _estimate_detection_point(
                det, filtered_depth, deproj_intrin, self._depth_scale)
            matched_xyz[target_id] = xyz
            position_measurements.append((target_id, (ux, uy), xyz))

        stable_xyz = self._position_stabilizer.update(position_measurements, time.time())
        active_knob_ids = [
            target_id for target_id, det in matched
            if det.class_name == self._angle_knob_class
        ]
        self._knob_angle_stabilizer.prune(active_knob_ids)

        for target_id, det in matched:
            xyz = stable_xyz.get(target_id, matched_xyz[target_id])
            matched_xyz[target_id] = xyz
            tag_assignment = tag_assignment_by_detection.get(id(det))

            target_info = {
                'id': target_id,
                'class': det.class_name,
                'tag_id': (tag_assignment.tag_id
                           if tag_assignment is not None else None),
                'classification_source': (
                    tag_assignment.source
                    if tag_assignment is not None else 'yolo_layout'),
                'position': {'x': round(xyz[0], 4),
                             'y': round(xyz[1], 4),
                             'z': round(xyz[2], 4)},
                'orientation': {'x': quat[0], 'y': quat[1],
                                'z': quat[2], 'w': quat[3]},
                'confidence': round(det.confidence, 3),
            }
            targets_output.append(target_info)

            # 兼容旧话题
            _publish_pose(self._pose_pubs.get(det.class_name), stamp, xyz, quat)

            # 旋钮角度
            if det.class_name == self._angle_knob_class and self._angle_enable:
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                x2, y2 = int(det.bbox[2]), int(det.bbox[3])
                roi = color_image[y1:y2, x1:x2]
                # 模式 1 只根据白色手柄线相对竖直线的夹角输出 0/90。
                # 模式 2/3 保留旧逻辑：红色旋钮可回退到红色长柄，黑色旋钮用白色胶带。
                mode_0_90 = self._angle_constraint_mode == 1
                ptr_color = 'white_line' if mode_0_90 else (
                    'red_handle' if target_id == 4 else 'white_tape')
                knob_range = None
                if self._angle_constraint_mode == 2:
                    knob_range = (180.0, 270.0) if target_id == 4 else (0.0, 90.0)
                angle = estimate_knob_angle(
                    roi,
                    binary_thresh=self._angle_binary_thresh,
                    circle_mask_ratio=self._angle_circle_mask,
                    angle_range=knob_range,
                    pointer_color=ptr_color,
                )
                if mode_0_90:
                    angle = self._knob_angle_stabilizer.update(target_id, angle)
                if angle is not None:
                    output_angle = int(angle) if mode_0_90 else round(angle, 1)
                    knob_angles.append({
                        'id': target_id,
                        'position': {'x': round(xyz[0], 4),
                                     'y': round(xyz[1], 4),
                                     'z': round(xyz[2], 4)},
                        'angle': output_angle,
                        'confidence': round(det.confidence, 3),
                    })


        # 对所有非编号目标计算 3D 坐标并标注。nut 只发布 refined 可靠点。
        object_targets_output = []
        fastener_observation_base = {}
        fastener_target_items = {}
        matched_detection_ids = {id(det) for _, det in matched}
        for det_idx, det in enumerate(frame_detections):
            if (id(det) in matched_detection_ids or
                    det.class_name in ('button', 'knob')):
                continue  # 已在 matched 中处理

            if det.class_name == 'nut':
                loc = nut_localizations.get(det_idx)
                bbox_cx = int(round(det.center_x))
                bbox_cy = int(round(det.center_y))
                cv2.circle(canvas, (bbox_cx, bbox_cy), 3, (160, 160, 160), -1)

                if loc is None or loc.confidence < 0.45:
                    x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                    conf_text = 0.0 if loc is None else loc.confidence
                    cv2.putText(canvas, f'nut refine fail {conf_text:.2f}',
                                (x1, max(15, y1 - 8)), 0, 0.4,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    ux, uy, fallback_xyz = _estimate_bbox_grouping_point(
                        det, filtered_depth, deproj_intrin, self._depth_scale)
                    if _valid_point_3d(fallback_xyz):
                        fastener_observation_base[det_idx] = {
                            'det_idx': det_idx,
                            'class_name': det.class_name,
                            'center_xy': (float(det.center_x), float(det.center_y)),
                            'bbox': tuple(float(v) for v in det.bbox),
                            'confidence': float(det.confidence) * 0.75,
                            'point_3d': [float(value) for value in fallback_xyz],
                        }
                    continue

                ux, uy, xyz = _estimate_nut_detection_point(
                    loc, filtered_depth, deproj_intrin, self._depth_scale)
                cv2.drawContours(canvas, [loc.contour], -1, (0, 255, 255), 2)
                cv2.drawMarker(canvas, (ux, uy), (0, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=14,
                               thickness=2)
                angle_text = 'na' if loc.angle is None else f'{loc.angle:.1f}'
                cv2.putText(canvas, f'nut_refined_conf={loc.confidence:.2f} angle={angle_text}',
                            (ux + 10, uy - 8), 0, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)

                if not _valid_point_3d(xyz):
                    cv2.putText(canvas, 'nut depth fail',
                                (ux + 10, uy + 18), 0, 0.4,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    continue

                # Fastener legacy pose is published after ID-based position filtering.
            else:
                ux, uy, xyz = _estimate_detection_point(
                    det, filtered_depth, deproj_intrin, self._depth_scale)
                if not _valid_point_3d(xyz):
                    x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                    cv2.putText(canvas, f'{det.class_name} depth fail',
                                (x1, max(15, y1 - 8)), 0, 0.4,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    continue
                if det.class_name != 'bolt':
                    _publish_pose(self._pose_pubs.get(det.class_name), stamp, xyz, quat)

            if det.class_name in ('valve', 'bolt', 'nut'):
                target_item = {
                    'class': det.class_name,
                    'bbox': det.bbox,
                    'position': {'x': round(xyz[0], 4),
                                 'y': round(xyz[1], 4),
                                 'z': round(xyz[2], 4)},
                    'orientation': {'x': quat[0], 'y': quat[1],
                                    'z': quat[2], 'w': quat[3]},
                    'confidence': round(det.confidence, 3),
                }
                object_targets_output.append(target_item)
                if det.class_name in ('bolt', 'nut'):
                    fastener_target_items[det_idx] = target_item
                    fastener_observation_base[det_idx] = {
                        'det_idx': det_idx,
                        'class_name': det.class_name,
                        'center_xy': (float(det.center_x), float(det.center_y)),
                        'bbox': tuple(float(v) for v in det.bbox),
                        'confidence': float(det.confidence),
                        'point_3d': [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    }

            if det.class_name not in ('bolt', 'nut'):
                cv2.putText(canvas, f'({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})',
                            (ux + 10, uy + 5), 0, 0.4,
                            (225, 255, 255), 1, cv2.LINE_AA)

        apriltag_reference = None
        if self._process_valves and self._apriltag_reference_enabled:
            apriltag_reference = detect_apriltag_reference_axis(
                color_image,
                filtered_depth,
                deproj_intrin,
                depth_scale=self._depth_scale,
                cfg=self.cfg.get('apriltag_reference', {}),
            )
        valve_reference_errors = []

        # 多边形角度检测（nut/bolt=六边形, valve=八边形）— 对所有检测结果
        hex_angles = []
        axis_directions = []
        fastener_axis_by_det_idx = {}
        fastener_geometry_items = {}
        axis_candidate_points = []
        axis_target_classes = set()
        if self._process_valves:
            axis_target_classes.add('valve')
        if self._process_fasteners:
            axis_target_classes.update(('nut', 'bolt'))
        for det_idx, det in enumerate(frame_detections):
            if det.class_name not in axis_target_classes:
                continue
            loc = nut_localizations.get(det_idx) if det.class_name == 'nut' else None
            reliable_nut = loc is not None and loc.confidence >= 0.45
            if reliable_nut:
                ux, uy, xyz = _estimate_nut_detection_point(
                    loc, filtered_depth, deproj_intrin, self._depth_scale)
            else:
                ux, uy, xyz = _estimate_detection_point(
                    det, filtered_depth, deproj_intrin, self._depth_scale)
            if xyz is None or xyz[2] <= 0:
                continue
            axis_candidate_points.append({
                'det_idx': det_idx,
                'class_name': det.class_name,
                'center_xy': (ux, uy),
                'point_3d': xyz,
                'frame': self._frame_count,
            })

        for det_idx, det in enumerate(frame_detections):
            if det.class_name in ('nut', 'bolt', 'valve'):
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                x2, y2 = int(det.bbox[2]), int(det.bbox[3])
                roi = color_image[y1:y2, x1:x2]
                n_sides = 8 if det.class_name == 'valve' else 6
                loc = nut_localizations.get(det_idx) if det.class_name == 'nut' else None
                reliable_nut = loc is not None and loc.confidence >= 0.45

                if (det.class_name == 'valve' and not _bbox_inside_image(
                        det.bbox, color_image.shape,
                        margin_px=self.cfg.get('valve_axis', {}).get('edge_margin_px', 4))):
                    axis_source = 'valve_incomplete'
                    axis_result = None
                    cv2.putText(canvas, 'valve incomplete',
                                (x1, min(canvas.shape[0] - 8, y2 + 16)),
                                0, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
                elif det.class_name == 'valve':
                    axis_source = 'valve_wheel'
                    axis_result = estimate_valve_wheel_axis_direction(
                        filtered_depth,
                        deproj_intrin,
                        det.bbox,
                        depth_scale=self._depth_scale,
                    )
                else:
                    axis_source = 'fastener_current'
                    axis_result = estimate_fastener_group_axis_direction(
                        (det.center_x, det.center_y),
                        axis_candidate_points,
                        max_distance_px=max(140.0, 3.5 * max(det.bbox[2] - det.bbox[0],
                                                             det.bbox[3] - det.bbox[1])),
                        min_points=3,
                        max_points=6,
                    )
                    if axis_result is None:
                        axis_source = 'local_patch_plane'
                        axis_result = estimate_fastener_patch_axis_direction(
                            filtered_depth,
                            deproj_intrin,
                            det.bbox,
                            all_bboxes=active_xyxy_list,
                            depth_scale=self._depth_scale,
                        )
                        if axis_result is not None:
                            constrained = estimate_fastener_line_constrained_axis(
                                (det.center_x, det.center_y),
                                axis_candidate_points,
                                axis_result[0],
                                max_distance_px=max(
                                    140.0,
                                    3.5 * max(det.bbox[2] - det.bbox[0],
                                              det.bbox[3] - det.bbox[1])),
                            )
                            if constrained is not None:
                                axis_result = constrained
                                axis_source = 'fastener_line'
                    if axis_result is None and self._panel_normal_cache is not None:
                        axis_normal, axis_centroid = self._panel_normal_cache
                        axis_points = 0
                        axis_result = (axis_normal, axis_centroid, axis_points)
                        axis_source = 'panel_plane'
                    if axis_result is None:
                        axis_result = estimate_object_axis_direction(
                            filtered_depth,
                            deproj_intrin,
                            det.bbox,
                            depth_scale=self._depth_scale,
                            mask=loc.depth_mask if reliable_nut else None,
                            object_class=det.class_name,
                        )
                        axis_source = 'local_depth'
                if axis_result is not None:
                    axis_normal, axis_centroid, axis_points = axis_result
                    axis_item = {
                        'class': det.class_name,
                        'bbox': det.bbox,
                        'source': axis_source,
                        'axis_direction': [
                            round(float(axis_normal[0]), 6),
                            round(float(axis_normal[1]), 6),
                            round(float(axis_normal[2]), 6),
                        ],
                        'centroid': [
                            round(float(axis_centroid[0]), 4),
                            round(float(axis_centroid[1]), 4),
                            round(float(axis_centroid[2]), 4),
                        ],
                        'point_count': axis_points,
                    }
                    if det.class_name == 'valve' and apriltag_reference is not None:
                        angle_deg = angle_between_normals_deg(
                            axis_normal, apriltag_reference['normal'])
                        if angle_deg is not None:
                            axis_item['reference'] = apriltag_reference['source']
                            axis_item['reference_angle_deg'] = round(angle_deg, 2)
                            valve_reference_errors.append({
                                'bbox': det.bbox,
                                'source': axis_source,
                                'normal': axis_item['axis_direction'],
                                'centroid': axis_item['centroid'],
                                'point_count': axis_points,
                                'angle_deg': round(angle_deg, 2),
                            })
                    axis_directions.append(axis_item)
                    if det.class_name in ('bolt', 'nut'):
                        fastener_axis_by_det_idx[det_idx] = [
                            float(axis_normal[0]),
                            float(axis_normal[1]),
                            float(axis_normal[2]),
                        ]
                        fastener_geometry_items.setdefault(det_idx, []).append(axis_item)
                    if det.class_name == 'valve':
                        _draw_axis_direction(
                            canvas, det.bbox, axis_normal,
                            label=f'{det.class_name}_axis[{axis_source}]',
                            color=(255, 255, 0))

                hex_angle = loc.angle if reliable_nut and loc.angle is not None else None
                if hex_angle is None:
                    if det.class_name == 'valve':
                        if not _bbox_inside_image(
                                det.bbox, color_image.shape,
                                margin_px=self.cfg.get('valve_axis', {}).get(
                                    'edge_margin_px', 4)):
                            hex_angle = None
                        else:
                            valve_candidates = estimate_valve_angle_candidates(roi)
                            raw_valve_angle = (
                                valve_candidates[0] if valve_candidates
                                else estimate_valve_angle(roi)
                            )
                            hex_angle = self._valve_angle_stabilizer.update(
                                'valve', raw_valve_angle, valve_candidates)
                    else:
                        hex_angle = estimate_hex_angle(roi, n_sides=n_sides)
                if hex_angle is not None:
                    angle_item = {
                        'class': det.class_name,
                        'bbox': det.bbox,
                        'hex_angle': round(hex_angle, 1),
                        **({'nut_refined_conf': round(loc.confidence, 3)}
                           if reliable_nut else {}),
                    }
                    if det.class_name == 'valve':
                        angle_item['valve_angle'] = round(hex_angle, 1)
                    hex_angles.append(angle_item)
                    if det.class_name in ('bolt', 'nut'):
                        fastener_geometry_items.setdefault(det_idx, []).append(angle_item)
                    if det.class_name == 'valve':
                        _draw_regular_polygon(
                            canvas, det.bbox, 8, hex_angle,
                            color=(0, 255, 255),
                            angle_mode='vertex')
                    elif det.class_name != 'nut' or not reliable_nut:
                        draw_hex_angle(canvas, det.bbox, hex_angle)

        fastener_axis_measurements = []
        for det_idx, raw_axis in fastener_axis_by_det_idx.items():
            det = frame_detections[det_idx]
            fastener_axis_measurements.append(FastenerAxisMeasurement(
                det_idx=det_idx,
                center_xy=(float(det.center_x), float(det.center_y)),
                bbox=tuple(float(value) for value in det.bbox),
                axis_direction=raw_axis,
            ))
        stabilized_fastener_axes = self._fastener_axis_stabilizer.update(
            fastener_axis_measurements, self._frame_count)
        for det_idx, stable_axis in stabilized_fastener_axes.items():
            stable_values = [float(value) for value in stable_axis]
            fastener_axis_by_det_idx[det_idx] = stable_values
            axis_item = None
            for geometry_item in fastener_geometry_items.get(det_idx, []):
                if 'axis_direction' not in geometry_item:
                    continue
                geometry_item['axis_direction'] = [
                    round(value, 6) for value in stable_values
                ]
                axis_item = geometry_item
            if axis_item is None:
                continue
            det = frame_detections[det_idx]
            _draw_axis_direction(
                canvas, det.bbox, stable_axis,
                label=f"{det.class_name}_axis[{axis_item['source']}]",
                color=(255, 0, 255))

        axis_reference = None
        if apriltag_reference is not None:
            axis_reference = {
                'source': apriltag_reference['source'],
                'tag_id': apriltag_reference['tag_id'],
                'normal': [
                    round(float(apriltag_reference['normal'][0]), 6),
                    round(float(apriltag_reference['normal'][1]), 6),
                    round(float(apriltag_reference['normal'][2]), 6),
                ],
                'centroid': [
                    round(float(apriltag_reference['centroid'][0]), 4),
                    round(float(apriltag_reference['centroid'][1]), 4),
                    round(float(apriltag_reference['centroid'][2]), 4),
                ],
                'point_count': apriltag_reference['point_count'],
                'inlier_ratio': round(float(apriltag_reference['inlier_ratio']), 4),
                'rms_error_m': round(float(apriltag_reference['rms_error']), 5),
                'valve_errors': valve_reference_errors,
            }
            _draw_apriltag_reference(canvas, apriltag_reference, valve_reference_errors)
            self._write_axis_reference_log(stamp, axis_reference, valve_reference_errors)

        _draw_axis_3d_view(canvas, axis_directions)

        assignments = {}
        if self._process_fasteners and fastener_observation_base:
            observations = []
            for det_idx, base in fastener_observation_base.items():
                observations.append(FastenerObservation(
                    axis_direction=fastener_axis_by_det_idx.get(det_idx),
                    **base,
                ))
            assignments = self._fastener_registry.update(
                observations, self._frame_count)
            for det_idx, assignment in assignments.items():
                target_item = fastener_target_items.get(det_idx)
                if target_item is not None:
                    target_item.update({
                        'id': assignment.target_id,
                        'group_id': assignment.group_id,
                        'slot': assignment.slot,
                        'registered': assignment.registered,
                        'slot_distance_m': round(assignment.distance_m, 4),
                    })
                for geometry_item in fastener_geometry_items.get(det_idx, []):
                    geometry_item.update({
                        'id': assignment.target_id,
                        'group_id': assignment.group_id,
                        'slot': assignment.slot,
                        'registered': assignment.registered,
                    })

                det = frame_detections[det_idx]
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                label = f'G{assignment.group_id} ID{assignment.target_id} {assignment.slot}'
                text_y = max(18, y1 - 24)
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(
                    canvas,
                    (x1, max(0, text_y - text_h - 5)),
                    (min(canvas.shape[1] - 1, x1 + text_w + 6), text_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(canvas, label, (x1 + 3, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (0, 255, 255), 2, cv2.LINE_AA)

        fastener_position_keys = {}
        fastener_position_measurements = []
        for det_idx, assignment in assignments.items():
            base = fastener_observation_base.get(det_idx)
            if base is None:
                continue
            filter_key = (assignment.group_id, assignment.target_id)
            fastener_position_keys[det_idx] = filter_key
            fastener_position_measurements.append((
                filter_key,
                base['center_xy'],
                base['point_3d'],
            ))
        stable_fastener_xyz = self._fastener_position_stabilizer.update(
            fastener_position_measurements, time.time())

        for det_idx, target_item in fastener_target_items.items():
            base = fastener_observation_base.get(det_idx)
            if base is None:
                continue
            filter_key = fastener_position_keys.get(det_idx)
            xyz = stable_fastener_xyz.get(filter_key, base['point_3d'])
            target_item['position'] = {
                'x': round(float(xyz[0]), 4),
                'y': round(float(xyz[1]), 4),
                'z': round(float(xyz[2]), 4),
            }
            det = frame_detections[det_idx]
            ux, uy = [int(round(value)) for value in base['center_xy']]
            cv2.putText(canvas, f'({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})',
                        (ux + 10, uy + 5), 0, 0.4,
                        (225, 255, 255), 1, cv2.LINE_AA)
            _publish_pose(self._pose_pubs.get(det.class_name), stamp, xyz, quat)

        # 发布目标结果：面板模式保持原编号格式；对象模式只包含对应类别。
        if self._process_panel_controls and targets_output and self._targets_pub is not None:
            msg = String()
            msg.data = json.dumps({
                'stamp': _stamp_to_sec(stamp),
                'targets': targets_output,
            }, ensure_ascii=False)
            self._targets_pub.publish(msg)
        elif (not self._process_panel_controls and object_targets_output and
              self._targets_pub is not None):
            msg = String()
            msg.data = json.dumps({
                'stamp': _stamp_to_sec(stamp),
                'targets': object_targets_output,
            }, ensure_ascii=False)
            self._targets_pub.publish(msg)

        stamp_value = _stamp_to_sec(stamp)

        # 保持旋钮角度话题格式不变，不混入指示灯或面板外目标信息。
        if knob_angles and self._angle_pub is not None:
            msg = String()
            msg.data = json.dumps({
                'stamp': stamp_value,
                'knob_angles': knob_angles,
            }, ensure_ascii=False)
            self._angle_pub.publish(msg)

        # 阀门/螺栓/螺母属于面板外目标，发布到独立几何话题。
        if ((hex_angles or axis_directions or axis_reference is not None) and
                self._object_geometry_pub is not None):
            msg = String()
            msg.data = json.dumps({
                'stamp': stamp_value,
                'object_angles': hex_angles,
                'axis_directions': axis_directions,
                **({'axis_reference': axis_reference} if axis_reference is not None else {}),
            }, ensure_ascii=False)
            self._object_geometry_pub.publish(msg)

        # 发布相机到操作面板平面的垂直距离
        panel_distance = None
        if self._panel_normal_cache is not None:
            normal, centroid = self._panel_normal_cache
            panel_distance = _panel_plane_distance(normal, centroid)
            if panel_distance is not None and self._distance_pub is not None:
                msg = String()
                msg.data = json.dumps({
                    'stamp': _stamp_to_sec(stamp),
                    'distance_m': round(panel_distance, 4),
                    'normal': [round(float(v), 6) for v in normal],
                    'centroid': [round(float(v), 4) for v in centroid],
                }, ensure_ascii=False)
                self._distance_pub.publish(msg)

        # ─── 可视化 ───
        has_mode_targets = bool(targets_output if self._process_panel_controls
                                else object_targets_output)
        self._publish_status('registered' if has_mode_targets else 'no_targets')

        for target_id, det in matched:
            ux = int(det.center_x)
            uy = int(det.center_y)
            cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), -1)

            cv2.putText(canvas, f'#{target_id}',
                        (ux - 10, uy - 15), 0, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)

            xyz = matched_xyz[target_id]
            cv2.putText(canvas, f'({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})',
                        (ux + 10, uy + 5), 0, 0.4,
                        (225, 255, 255), 1, cv2.LINE_AA)

        if self._panel_normal_cache is not None:
            n = np.round(self._panel_normal_cache[0], 3).tolist()
            cv2.putText(canvas, f'normal: {n}', (10, 25), 0, 0.6,
                        (0, 200, 255), 2, cv2.LINE_AA)
            if panel_distance is not None:
                cv2.putText(canvas, f'panel distance: {panel_distance:.3f}m',
                            (10, 50), 0, 0.6,
                            (0, 200, 255), 2, cv2.LINE_AA)

        for ka in knob_angles:
            for target_id, det in matched:
                if target_id == ka['id']:
                    if 'hex_angle' in ka:
                        draw_hex_angle(canvas, det.bbox, ka['hex_angle'])
                    elif 'angle' in ka:
                        draw_knob_angle(canvas, det.bbox, ka['angle'])
                    break

        # 铭牌 OCR 可视化暂停使用，Canvas 不再绘制黄色/橙色 OCR ROI 框。

        cv2.putText(canvas, 'REGISTERED', (10, canvas.shape[0] - 15),
                    0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        self._update_display_frame(canvas, stamp)


def main(args=None):
    del args
    rospy.init_node('panel_detection')
    node = None
    try:
        node = PanelDetectionNode()
        while not rospy.is_shutdown():
            node._detection_callback()
            if node._gui_enabled:
                if node.display_frame is not None:
                    cv2.imshow('panel_detection', node.display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                rospy.sleep(0.001)
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            if node._camera is not None:
                node._camera.stop()
            node.destroy_node()


if __name__ == '__main__':
    main()
