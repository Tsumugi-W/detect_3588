"""
ROS2 面板检测节点

检测操作面板上的 7 类目标，每类一个独立话题，发布 PoseStamped。
旋钮角度通过额外话题 /panel/knob_angles 发布。

话题:
  /panel/lights    (PoseStamped)  — 指示灯 3D 位姿
  /panel/knobs     (PoseStamped)  — 旋钮 3D 位姿
  /panel/buttons   (PoseStamped)  — 按钮 3D 位姿
  /panel/bolts     (PoseStamped)  — 螺栓 3D 位姿
  /panel/nuts      (PoseStamped)  — 螺母 3D 位姿
  /panel/valves    (PoseStamped)  — 阀门 3D 位姿
  /panel/pumps     (PoseStamped)  — 泵 3D 位姿
  /panel/knob_angles (String, JSON) — 旋钮角度信息
"""
import os
import math
import json
import time
import threading

import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from .camera import create_backend
from .camera.base import CameraIntrinsics
from .depth_utils import (
    deproject_pixel_to_point, undistort_pixel,
    filter_depth, get_robust_depth, compute_panel_normal,
)
from .knob_angle import estimate_knob_angle


# ─── 配置 ─────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'camera_backend': 'orbbec',
    'camera': {'color_width': 1280, 'color_height': 720,
               'depth_width': 1280, 'depth_height': 720, 'fps': 30},
    'inference_backend': 'onnx',
    'onnx_model': '0520.onnx',
    'onnx_threads': 8,
    'weight': '0520.pt',
    'input_size': 640,
    'class_num': 7,
    'class_name': ['light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button'],
    'threshold': {'iou': 0.01, 'confidence': 0.3},
    'knob_angle': {'enable': True, 'binary_thresh': 180,
                   'circle_mask_ratio': 0.85, 'knob_class': 'knob'},
    'panel_normal_interval': 10,
}

CLASS_TOPIC_MAP = {
    'light': '/panel/lights',
    'knob': '/panel/knobs',
    'bolt': '/panel/bolts',
    'nut': '/panel/nuts',
    'valve': '/panel/valves',
    'pump': '/panel/pumps',
    'button': '/panel/buttons',
}


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


class PanelDetectionNode(Node):
    def __init__(self):
        super().__init__('panel_detection_node')

        # 加载配置
        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        if config_path and os.path.isfile(config_path):
            self.get_logger().info(f'加载配置: {config_path}')
            with open(config_path, 'r', encoding='utf-8') as f:
                self.cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else:
            self.get_logger().info('使用默认配置')
            self.cfg = DEFAULT_CONFIG

        # 创建话题 publisher (每类一个 PoseStamped)
        self._pose_pubs = {}
        for cls_name, topic in CLASS_TOPIC_MAP.items():
            self._pose_pubs[cls_name] = self.create_publisher(PoseStamped, topic, 10)

        # 旋钮角度话题
        self._angle_pub = self.create_publisher(String, '/panel/knob_angles', 10)

        # 初始化相机
        self._camera = None
        self._depth_scale = 0.001
        self._camera_ready = False
        self._init_camera()

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

        # 面板法向量
        self._panel_normal_cache = None
        self._frame_count = 0
        self._normal_interval = self.cfg.get('panel_normal_interval', 10)

        # 重连
        self._reconnect_interval = 5.0
        self._last_reconnect_time = 0.0

        # 可视化
        self.display_frame = None

        status = '运行中' if self._camera_ready else '等待相机'
        self.get_logger().info(f'面板检测节点已启动 ({status})')
        cv2.namedWindow('panel_detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

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
            onnx_path = self.cfg.get('onnx_model', '0520.onnx')
            if not os.path.isabs(onnx_path):
                onnx_path = os.path.join(pkg_dir, onnx_path)
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

    def _detection_callback(self):
        if not self._camera_ready:
            self._try_reconnect_camera()
            return

        color_intrin, depth_intrin, color_image, depth_image = \
            self._camera.get_aligned_frames()
        if color_intrin is None:
            return
        if not color_image.any() or not depth_image.any():
            return

        t0 = time.time()
        canvas, class_id_list, xyxy_list, conf_list = self._detector.detect(color_image)
        t1 = time.time()
        if not xyxy_list:
            self.display_frame = canvas
            return

        filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)
        t2 = time.time()
        self._depth_scale = self._camera.get_depth_scale()

        # 面板法向量
        self._frame_count += 1
        if (self._panel_normal_cache is None or
                self._frame_count % self._normal_interval == 0):
            result = compute_panel_normal(
                color_image, filtered_depth, depth_intrin, xyxy_list,
                depth_scale=self._depth_scale)
            if result is not None:
                self._panel_normal_cache = result
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

        stamp = self.get_clock().now().to_msg()
        knob_angles = []

        for i, xyxy in enumerate(xyxy_list):
            cls_id = class_id_list[i]
            cls_name = self._class_names[cls_id] if cls_id < len(self._class_names) else 'unknown'

            # 计算 3D 坐标
            ux = int((xyxy[0] + xyxy[2]) / 2)
            uy = int((xyxy[1] + xyxy[3]) / 2)
            ux_u, uy_u = undistort_pixel(ux, uy, color_intrin)
            ux_u, uy_u = int(ux_u), int(uy_u)
            dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                   sample_radius=3, depth_scale=self._depth_scale)
            xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)

            # 发布 PoseStamped
            pub = self._pose_pubs.get(cls_name)
            if pub is not None:
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

            # 旋钮角度
            if cls_name == self._angle_knob_class and self._angle_enable:
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                roi = color_image[y1:y2, x1:x2]
                angle = estimate_knob_angle(
                    roi,
                    binary_thresh=self._angle_binary_thresh,
                    circle_mask_ratio=self._angle_circle_mask,
                )
                if angle is not None:
                    knob_angles.append({
                        'position': {'x': round(xyz[0], 4),
                                     'y': round(xyz[1], 4),
                                     'z': round(xyz[2], 4)},
                        'angle': round(angle, 1),
                        'confidence': round(float(conf_list[i]), 3),
                    })

        # 发布旋钮角度
        if knob_angles:
            msg = String()
            msg.data = json.dumps({
                'stamp': stamp.sec + stamp.nanosec * 1e-9,
                'knob_angles': knob_angles,
            }, ensure_ascii=False)
            self._angle_pub.publish(msg)

        # 可视化（复用已计算的结果，不重复计算）
        from .knob_angle import draw_knob_angle
        for ka in knob_angles:
            # 从 knob_angles 里找到对应的 xyxy 来画角度
            pass  # draw_knob_angle 在下面统一画

        for i, xyxy in enumerate(xyxy_list):
            ux = int((xyxy[0] + xyxy[2]) / 2)
            uy = int((xyxy[1] + xyxy[3]) / 2)
            cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), -1)
            # xyz 已在上面计算过，这里重新算一次（轻量操作）
            ux_u, uy_u = undistort_pixel(ux, uy, color_intrin)
            dis = get_robust_depth(filtered_depth, int(ux_u), int(uy_u),
                                   sample_radius=3, depth_scale=self._depth_scale)
            xyz = deproject_pixel_to_point(depth_intrin, (int(ux_u), int(uy_u)), dis)
            cv2.putText(canvas, f'({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})',
                        (ux + 10, uy + 5), 0, 0.4,
                        (225, 255, 255), 1, cv2.LINE_AA)

        if self._panel_normal_cache is not None:
            n = np.round(self._panel_normal_cache[0], 3).tolist()
            cv2.putText(canvas, f'normal: {n}', (10, 25), 0, 0.6,
                        (0, 200, 255), 2, cv2.LINE_AA)

        # 旋钮角度绘制（复用已计算的角度，不重复调用 estimate_knob_angle）
        knob_idx = 0
        for i, xyxy in enumerate(xyxy_list):
            cls_id = class_id_list[i]
            cls_name = self._class_names[cls_id] if cls_id < len(self._class_names) else ''
            if cls_name == self._angle_knob_class and self._angle_enable:
                if knob_idx < len(knob_angles):
                    draw_knob_angle(canvas, xyxy, knob_angles[knob_idx]['angle'])
                    knob_idx += 1

        self.display_frame = canvas


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PanelDetectionNode()
        while rclpy.ok():
            node._detection_callback()
            # 处理 ROS2 回调（参数更新等）
            rclpy.spin_once(node, timeout_sec=0.001)
            if node.display_frame is not None:
                cv2.imshow('panel_detection', node.display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            if node._camera is not None:
                node._camera.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
