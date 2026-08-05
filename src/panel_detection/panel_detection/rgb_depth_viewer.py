#!/usr/bin/env python3
"""Display and capture timestamp-synchronized color and depth images."""

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_color_image(msg):
    """Decode common sensor_msgs/Image color encodings without cv_bridge."""
    channels_by_encoding = {
        'bgr8': 3,
        'rgb8': 3,
        'bgra8': 4,
        'rgba8': 4,
        'mono8': 1,
    }
    channels = channels_by_encoding.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported color encoding: {msg.encoding}')

    row_elements = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height) * row_elements
    if raw.size < expected:
        raise ValueError(
            f'color buffer is short: {raw.size} bytes, expected {expected}')
    rows = raw[:expected].reshape(int(msg.height), row_elements)
    packed_width = int(msg.width) * channels
    image = rows[:, :packed_width].reshape(
        int(msg.height), int(msg.width), channels)

    if msg.encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'rgba8':
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if msg.encoding == 'bgra8':
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image.copy()


def decode_depth_image(msg):
    """Decode 16-bit integer or 32-bit float ROS depth images."""
    if msg.encoding in ('16UC1', 'mono16'):
        dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
    elif msg.encoding == '32FC1':
        dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
    else:
        raise ValueError(f'unsupported depth encoding: {msg.encoding}')

    row_elements = int(msg.step) // dtype.itemsize
    raw = np.frombuffer(msg.data, dtype=dtype)
    expected = int(msg.height) * row_elements
    if raw.size < expected:
        raise ValueError(
            f'depth buffer is short: {raw.size} values, expected {expected}')
    rows = raw[:expected].reshape(int(msg.height), row_elements)
    return rows[:, :int(msg.width)].astype(dtype.newbyteorder('='), copy=True)


def colorize_depth(depth_image, encoding, depth_scale, min_depth_m,
                   max_depth_m):
    """Convert a ROS depth image to a BGR pseudo-color image."""
    depth = np.asarray(depth_image)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if encoding == '32FC1' or np.issubdtype(depth.dtype, np.floating):
        depth_m = depth.astype(np.float32)
    else:
        depth_m = depth.astype(np.float32) * float(depth_scale)

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.where(
        valid, np.clip(depth_m, min_depth_m, max_depth_m), max_depth_m)
    normalized = (clipped - min_depth_m) / (max_depth_m - min_depth_m)

    # Invert the color scale so nearby pixels are warm and far pixels are cool.
    depth_u8 = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
    visual = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    visual[~valid] = (24, 24, 24)
    return visual, depth_m, valid


def build_comparison(color, depth_visual, color_stamp, depth_stamp,
                     min_depth_m, max_depth_m):
    """Build the side-by-side canvas saved by the viewer."""
    height, width = color.shape[:2]
    if depth_visual.shape[:2] != (height, width):
        depth_visual = cv2.resize(
            depth_visual, (width, height), interpolation=cv2.INTER_NEAREST)

    header_height = 46
    canvas = np.full(
        (height + header_height, width * 2, 3), 20, dtype=np.uint8)
    canvas[header_height:, :width] = color
    canvas[header_height:, width:] = depth_visual

    dt_ms = abs(color_stamp - depth_stamp) * 1000.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas, 'COLOR', (12, 29), font, 0.68, (255, 255, 255), 2,
        cv2.LINE_AA)
    cv2.putText(
        canvas, f'DEPTH  {min_depth_m:.2f}-{max_depth_m:.2f} m',
        (width + 12, 29), font, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    dt_text = f'sync dt: {dt_ms:.1f} ms'
    (text_width, _), _ = cv2.getTextSize(dt_text, font, 0.55, 1)
    cv2.putText(
        canvas, dt_text, (width - text_width - 12, 28), font, 0.55,
        (120, 230, 120), 1, cv2.LINE_AA)
    return canvas


class RgbDepthViewer(Node):
    def __init__(self):
        super().__init__('rgb_depth_viewer')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('sync_tolerance_sec', 0.02)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 3.00)
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('output_dir', 'rgb_depth_captures')
        self.declare_parameter('save_once', False)
        self.declare_parameter('capture_hz', 0.0)
        self.declare_parameter('show_window', True)

        self.color_topic = self.get_parameter(
            'color_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter(
            'depth_topic').get_parameter_value().string_value
        self.sync_tolerance = self.get_parameter(
            'sync_tolerance_sec').get_parameter_value().double_value
        self.depth_scale = self.get_parameter(
            'depth_scale').get_parameter_value().double_value
        self.min_depth_m = self.get_parameter(
            'min_depth_m').get_parameter_value().double_value
        self.max_depth_m = self.get_parameter(
            'max_depth_m').get_parameter_value().double_value
        self.display_width = self.get_parameter(
            'display_width').get_parameter_value().integer_value
        self.output_dir = Path(
            self.get_parameter('output_dir').get_parameter_value().string_value
        ).expanduser().resolve()
        self.save_once = self.get_parameter(
            'save_once').get_parameter_value().bool_value
        capture_hz = self.get_parameter(
            'capture_hz').get_parameter_value().double_value
        self.capture_interval = 1.0 / capture_hz if capture_hz > 0.0 else None
        self.last_capture_stamp = None
        self.show_window = self.get_parameter(
            'show_window').get_parameter_value().bool_value

        if self.sync_tolerance <= 0.0:
            raise ValueError('sync_tolerance_sec must be positive')
        if not 0.0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError('depth range must satisfy 0 <= min < max')

        self.color_queue = deque(maxlen=30)
        self.depth_queue = deque(maxlen=30)
        self.latest_canvas = None
        self.latest_stamps = None
        self.saved_once = False
        self.window_name = 'Synchronized color + depth'

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.color_sub = self.create_subscription(
            Image, self.color_topic, self._on_color, qos)
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self._on_depth, qos)

        self.get_logger().info(
            f'Waiting for {self.color_topic} and {self.depth_topic}; '
            'press S to save, Q or Esc to quit')

    def _on_color(self, msg):
        self.color_queue.append((stamp_to_seconds(msg.header.stamp), msg))
        self._match_frames()

    def _on_depth(self, msg):
        self.depth_queue.append((stamp_to_seconds(msg.header.stamp), msg))
        self._match_frames()

    def _match_frames(self):
        while self.color_queue and self.depth_queue:
            color_stamp, color_msg = self.color_queue[0]
            depth_stamp, depth_msg = self.depth_queue[0]
            delta = color_stamp - depth_stamp

            if abs(delta) <= self.sync_tolerance:
                self.color_queue.popleft()
                self.depth_queue.popleft()
                self._show_pair(
                    color_msg, depth_msg, color_stamp, depth_stamp)
            elif delta < 0.0:
                self.color_queue.popleft()
            else:
                self.depth_queue.popleft()

    def _show_pair(self, color_msg, depth_msg, color_stamp, depth_stamp):
        capture_due = (
            self.capture_interval is not None and
            (self.last_capture_stamp is None or
             color_stamp - self.last_capture_stamp >=
             self.capture_interval - 1e-6)
        )
        if not self.show_window and not self.save_once and not capture_due:
            return

        try:
            color = decode_color_image(color_msg)
            depth = decode_depth_image(depth_msg)
        except ValueError as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        depth_visual, _, _ = colorize_depth(
            depth, depth_msg.encoding, self.depth_scale,
            self.min_depth_m, self.max_depth_m)
        canvas = build_comparison(
            color, depth_visual, color_stamp, depth_stamp,
            self.min_depth_m, self.max_depth_m)
        self.latest_canvas = canvas
        self.latest_stamps = (color_stamp, depth_stamp)

        if self.save_once and not self.saved_once:
            self._save_canvas()
            self.saved_once = True
        if capture_due and self._save_canvas():
            self.last_capture_stamp = color_stamp

        shown = canvas
        if self.display_width > 0 and canvas.shape[1] > self.display_width:
            scale = self.display_width / float(canvas.shape[1])
            shown = cv2.resize(
                canvas, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA)
        if self.show_window:
            cv2.imshow(self.window_name, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('s'), ord('S')):
                self._save_canvas()
            elif key in (ord('q'), ord('Q'), 27):
                rclpy.shutdown()

    def _save_canvas(self):
        if self.latest_canvas is None or self.latest_stamps is None:
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        color_stamp, depth_stamp = self.latest_stamps
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        ros_ns = int(round(color_stamp * 1e9))
        path = self.output_dir / f'rgb_depth_{timestamp}_{ros_ns}.jpg'
        if cv2.imwrite(str(path), self.latest_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            self.get_logger().info(
                f'Saved {path} (sync dt '
                f'{abs(color_stamp - depth_stamp) * 1000.0:.1f} ms)')
            return True
        else:
            self.get_logger().error(f'Failed to save {path}')
            return False

    def destroy_node(self):
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RgbDepthViewer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
