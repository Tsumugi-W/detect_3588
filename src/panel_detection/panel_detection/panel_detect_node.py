"""
ROS2 面板检测节点

两阶段工作流：
  1. 注册阶段：积累多帧检测结果，按 x 坐标排序 + 类别/颜色验证后为 7 个目标分配编号 1-7
  2. 跟踪阶段：每帧检测结果匹配注册表，发布带编号的检测结果

话题:
  /panel/targets   (String, JSON) — 带编号的检测结果（位置 + 类别 + ID）
  /panel/knob_angles (String, JSON) — 旋钮角度信息（带编号）
  /panel/status    (String)       — 注册状态（registering / registered）
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
from sensor_msgs.msg import Image, CameraInfo

from .camera import create_backend
from .camera.base import CameraIntrinsics
from .depth_utils import (
    deproject_pixel_to_point, undistort_pixel,
    filter_depth, get_robust_depth, compute_panel_normal,
)
from .knob_angle import estimate_knob_angle, draw_knob_angle
from .target_registry import TargetRegistry, FrameDetection


# ─── 配置 ─────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'camera_backend': 'orbbec',
    'camera': {'color_width': 1280, 'color_height': 720,
               'depth_width': 1280, 'depth_height': 720, 'fps': 30},
    'inference_backend': 'onnx',
    'onnx_model': '0525.onnx',
    'onnx_threads': 8,
    'weight': '0525.pt',
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

        # 创建话题 publisher (每类一个 PoseStamped，保留兼容)
        self._pose_pubs = {}
        for cls_name, topic in CLASS_TOPIC_MAP.items():
            self._pose_pubs[cls_name] = self.create_publisher(PoseStamped, topic, 10)

        # 带编号的统一话题
        self._targets_pub = self.create_publisher(String, '/panel/targets', 10)
        self._status_pub = self.create_publisher(String, '/panel/status', 10)
        self._angle_pub = self.create_publisher(String, '/panel/knob_angles', 10)

        # 目标注册器
        reg_cfg = self.cfg.get('registry', {})
        self._registry = TargetRegistry(
            stable_frames=reg_cfg.get('stable_frames', 15),
            green_hue_range=tuple(reg_cfg.get('green_hue_range', [35, 85])),
            match_distance_thresh=reg_cfg.get('match_distance_thresh', 80),
        )

        # 相机来源：直接连接 or 话题订阅
        self.declare_parameter('use_topic', False)
        self._use_topic = self.get_parameter('use_topic').get_parameter_value().bool_value

        self._camera = None
        # Orbbec 官方驱动深度图单位为 mm，× 0.001 转米
        self._depth_scale = self.cfg.get('depth_scale', 0.001)
        self._camera_ready = False

        # 话题订阅模式的缓存
        self._topic_color = None
        self._topic_depth = None
        self._topic_color_intrin = None
        self._topic_depth_intrin = None

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

        # 面板法向量
        self._panel_normal_cache = None
        self._frame_count = 0
        self._normal_interval = self.cfg.get('panel_normal_interval', 10)

        # 重连
        self._reconnect_interval = 5.0
        self._last_reconnect_time = 0.0

        # 可视化
        self.display_frame = None
        self._gui_enabled = False
        try:
            cv2.namedWindow('panel_detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            self._gui_enabled = True
        except cv2.error:
            self.get_logger().warn('无显示环境，禁用 GUI 可视化')

        status = '运行中' if self._camera_ready else '等待相机'
        self.get_logger().info(f'面板检测节点已启动 ({status})')

    def _init_subscribers(self):
        """订阅相机话题（独立订阅，不依赖时间同步）"""
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        # 只保留最新1帧，避免推理跟不上时帧堆积导致 OOM
        img_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST)
        self._sub_color = self.create_subscription(
            Image, '/camera/color/image_raw', self._color_callback, img_qos)
        self._sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self._depth_callback, img_qos)
        self._sub_color_info = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self._color_info_callback, 5)
        self._sub_depth_info = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self._depth_info_callback, 5)

    def _color_callback(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        # OrbbecSDK_ROS2 发布 rgb8，转换为 BGR 供 OpenCV/YOLO 使用
        if msg.encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._topic_color = image

    def _depth_callback(self, msg):
        self._topic_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)

    def _color_info_callback(self, msg):
        self._topic_color_intrin = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4],
            cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            coeffs=list(msg.d) if msg.d else [0.0]*5,
        )

    def _depth_info_callback(self, msg):
        self._topic_depth_intrin = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4],
            cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            coeffs=list(msg.d) if msg.d else [0.0]*5,
        )

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
        color_info_msg.k = [color_intrin.fx, 0.0, color_intrin.cx,
                            0.0, color_intrin.fy, color_intrin.cy,
                            0.0, 0.0, 1.0]
        color_info_msg.d = color_intrin.coeffs
        self._color_info_pub.publish(color_info_msg)

        # 发布深度相机内参
        depth_info_msg = CameraInfo()
        depth_info_msg.header.stamp = stamp
        depth_info_msg.header.frame_id = 'camera_depth_optical_frame'
        depth_info_msg.width = depth_intrin.width
        depth_info_msg.height = depth_intrin.height
        depth_info_msg.k = [depth_intrin.fx, 0.0, depth_intrin.cx,
                            0.0, depth_intrin.fy, depth_intrin.cy,
                            0.0, 0.0, 1.0]
        depth_info_msg.d = depth_intrin.coeffs
        self._depth_info_pub.publish(depth_info_msg)

    def _detection_callback(self):
        if not self._camera_ready:
            self._try_reconnect_camera()
            return

        if self._use_topic:
            color_intrin = self._topic_color_intrin
            depth_intrin = self._topic_depth_intrin
            color_image = self._topic_color
            depth_image = self._topic_depth
            if color_intrin is None or color_image is None or depth_image is None or depth_intrin is None:
                return
        else:
            color_intrin, depth_intrin, color_image, depth_image = \
                self._camera.get_aligned_frames()
            if color_intrin is None:
                return

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
        if not xyxy_list:
            self.display_frame = canvas
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

        # ─── 每帧独立识别绝对编号 ───
        matched = self._registry.identify(frame_detections, color_image)

        targets_output = []
        knob_angles = []
        matched_xyz = {}  # 缓存 3D 坐标避免重复计算

        for target_id, det in matched:
            ux = int(det.center_x)
            uy = int(det.center_y)
            ux_u, uy_u = undistort_pixel(ux, uy, color_intrin)
            ux_u, uy_u = int(ux_u), int(uy_u)
            dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                   sample_radius=3, depth_scale=self._depth_scale)
            xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)
            matched_xyz[target_id] = xyz

            target_info = {
                'id': target_id,
                'class': det.class_name,
                'position': {'x': round(xyz[0], 4),
                             'y': round(xyz[1], 4),
                             'z': round(xyz[2], 4)},
                'orientation': {'x': quat[0], 'y': quat[1],
                                'z': quat[2], 'w': quat[3]},
                'confidence': round(det.confidence, 3),
            }
            targets_output.append(target_info)

            # 兼容旧话题
            pub = self._pose_pubs.get(det.class_name)
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
            if det.class_name == self._angle_knob_class and self._angle_enable:
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                x2, y2 = int(det.bbox[2]), int(det.bbox[3])
                roi = color_image[y1:y2, x1:x2]
                angle = estimate_knob_angle(
                    roi,
                    binary_thresh=self._angle_binary_thresh,
                    circle_mask_ratio=self._angle_circle_mask,
                )
                if angle is not None:
                    knob_angles.append({
                        'id': target_id,
                        'position': {'x': round(xyz[0], 4),
                                     'y': round(xyz[1], 4),
                                     'z': round(xyz[2], 4)},
                        'angle': round(angle, 1),
                        'confidence': round(det.confidence, 3),
                    })

        # 发布带编号的检测结果
        if targets_output:
            msg = String()
            msg.data = json.dumps({
                'stamp': stamp.sec + stamp.nanosec * 1e-9,
                'targets': targets_output,
            }, ensure_ascii=False)
            self._targets_pub.publish(msg)

        # 发布旋钮角度
        if knob_angles:
            msg = String()
            msg.data = json.dumps({
                'stamp': stamp.sec + stamp.nanosec * 1e-9,
                'knob_angles': knob_angles,
            }, ensure_ascii=False)
            self._angle_pub.publish(msg)

        # ─── 可视化 ───
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

        for ka in knob_angles:
            for target_id, det in matched:
                if target_id == ka['id']:
                    draw_knob_angle(canvas, det.bbox, ka['angle'])
                    break

        cv2.putText(canvas, 'REGISTERED', (10, canvas.shape[0] - 15),
                    0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        self.display_frame = canvas


def main(args=None):
    rclpy.init(args=args)
    node = None
    spin_thread = None
    try:
        node = PanelDetectionNode()
        # 用独立线程处理 ROS2 回调（话题订阅等）
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        while rclpy.ok():
            node._detection_callback()
            if node._gui_enabled:
                if node.display_frame is not None:
                    cv2.imshow('panel_detection', node.display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                time.sleep(0.001)
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
        if spin_thread is not None:
            spin_thread.join(timeout=2)


if __name__ == '__main__':
    main()
