#!/usr/bin/env python3
"""
螺栓识别精度测试脚本

直接订阅当前活跃话题，采集 N 帧螺栓检测数据，
在 AprilTag 坐标系下统计 3D 位置重复精度。

用法:
  python3 scripts/precision_test.py [--frames 100] [--timeout 60]

输出:
  - 终端实时打印采集进度
  - 结束后输出每颗螺栓的精度统计（std/RMS/range/P95）
  - 保存详细 CSV 和 JSON 摘要到 /tmp/bolt_precision_<timestamp>/
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'src', 'panel_detection'))
from panel_detection.fastener_precision_recorder import (
    build_tag_frame, camera_point_to_tag,
)


def parse_args():
    parser = argparse.ArgumentParser(description='螺栓识别精度测试')
    parser.add_argument('--frames', type=int, default=100,
                        help='采集帧数 (default: 100)')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='超时秒数 (default: 120)')
    parser.add_argument('--geometry-topic', default='/objects/geometry',
                        help='geometry 话题 (default: /objects/geometry)')
    parser.add_argument('--targets-topic', default='',
                        help='targets 话题; 为空时从 geometry 中提取位置')
    parser.add_argument('--output-dir', default='',
                        help='输出目录; 为空时自动生成')
    parser.add_argument('--class-filter', default='bolt',
                        help='只记录指定类别 (default: bolt)')
    return parser.parse_args()


class PrecisionCollector:
    def __init__(self, class_filter='bolt'):
        self.class_filter = class_filter
        self.frames = []
        self.positions_camera = defaultdict(list)
        self.positions_tag = defaultdict(list)
        self.axis_directions = defaultdict(list)
        self.metadata = {}

    def add_geometry_frame(self, payload):
        reference = payload.get('axis_reference')
        tag_frame = build_tag_frame(reference) if reference else None

        stamp = float(payload.get('stamp', 0.0))
        frame_record = {
            'stamp': stamp,
            'has_tag_frame': tag_frame is not None,
            'tag_source': reference.get('source') if reference else None,
            'tag_reprojection_px': (
                reference.get('reprojection_error_px') if reference else None),
            'bolts': [],
        }

        for axis_item in payload.get('axis_directions', []):
            cls = axis_item.get('class', '')
            if self.class_filter and cls != self.class_filter:
                continue

            slot = axis_item.get('slot', 'unknown')
            bolt_id = axis_item.get('id', '?')
            group_id = axis_item.get('group_id', '?')
            key = f"G{group_id}:ID{bolt_id}:{slot}"
            centroid = axis_item.get('centroid')
            axis_dir = axis_item.get('axis_direction')
            source = axis_item.get('source', 'unknown')

            if centroid is None:
                continue
            camera_point = np.array(centroid, dtype=np.float64)
            if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 0:
                continue

            self.positions_camera[key].append(camera_point)
            self.metadata[key] = {
                'group_id': group_id, 'id': bolt_id, 'slot': slot,
                'class': cls, 'source': source,
            }

            tag_point = None
            if tag_frame is not None and reference is not None:
                tag_point = camera_point_to_tag(camera_point, reference)
            if tag_point is not None:
                self.positions_tag[key].append(tag_point)

            if axis_dir is not None:
                self.axis_directions[key].append(
                    np.array(axis_dir, dtype=np.float64))

            bolt_record = {
                'key': key,
                'camera_m': camera_point.tolist(),
                'tag_m': tag_point.tolist() if tag_point is not None else None,
                'axis': axis_dir,
                'source': source,
            }
            frame_record['bolts'].append(bolt_record)

        # 从 object_angles 中也可以提取角度信息
        for angle_item in payload.get('object_angles', []):
            cls = angle_item.get('class', '')
            if self.class_filter and cls != self.class_filter:
                continue
            # 角度数据附加到 frame_record 但不单独统计

        self.frames.append(frame_record)
        return frame_record

    def add_targets_frame(self, targets_payload, geometry_payload):
        """从 targets + geometry 配对数据中提取位置。"""
        reference = geometry_payload.get('axis_reference')
        tag_frame = build_tag_frame(reference) if reference else None
        stamp = float(targets_payload.get('stamp', 0.0))

        frame_record = {
            'stamp': stamp,
            'has_tag_frame': tag_frame is not None,
            'tag_source': reference.get('source') if reference else None,
            'bolts': [],
        }

        for target in targets_payload.get('targets', []):
            cls = target.get('class', '')
            if self.class_filter and cls != self.class_filter:
                continue

            position = target.get('position', {})
            try:
                camera_point = np.array([
                    float(position['x']), float(position['y']),
                    float(position['z']),
                ], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 0:
                continue

            slot = target.get('slot', 'unknown')
            bolt_id = target.get('id', '?')
            group_id = target.get('group_id', '?')
            key = f"G{group_id}:ID{bolt_id}:{slot}"

            self.positions_camera[key].append(camera_point)
            self.metadata[key] = {
                'group_id': group_id, 'id': bolt_id, 'slot': slot,
                'class': cls,
            }

            tag_point = None
            if tag_frame is not None and reference is not None:
                tag_point = camera_point_to_tag(camera_point, reference)
            if tag_point is not None:
                self.positions_tag[key].append(tag_point)

            frame_record['bolts'].append({
                'key': key,
                'camera_m': camera_point.tolist(),
                'tag_m': tag_point.tolist() if tag_point is not None else None,
            })

        self.frames.append(frame_record)
        return frame_record

    def summary(self):
        results = {}
        for key in sorted(set(self.positions_camera) | set(self.positions_tag)):
            entry = {'metadata': self.metadata.get(key, {})}

            camera_pts = self.positions_camera.get(key, [])
            if camera_pts:
                entry['camera'] = _position_stats(camera_pts, 'camera')

            tag_pts = self.positions_tag.get(key, [])
            if tag_pts:
                entry['tag'] = _position_stats(tag_pts, 'tag')

            axes = self.axis_directions.get(key, [])
            if axes:
                entry['axis'] = _axis_stats(axes)

            results[key] = entry

        return {
            'total_frames': len(self.frames),
            'frames_with_tag': sum(1 for f in self.frames if f['has_tag_frame']),
            'targets': results,
        }


def _position_stats(points, label=''):
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    median = np.median(pts, axis=0)
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0) * 1000.0
    ptp = np.ptp(pts, axis=0) * 1000.0
    residuals_mm = np.linalg.norm(pts - median, axis=1) * 1000.0
    rms = float(math.sqrt(np.mean(residuals_mm ** 2)))
    p95 = float(np.percentile(residuals_mm, 95)) if n >= 5 else float(np.max(residuals_mm))
    return {
        'samples': n,
        'median_m': median.tolist(),
        'mean_m': mean.tolist(),
        'std_xyz_mm': std.tolist(),
        'range_xyz_mm': ptp.tolist(),
        'rms_3d_mm': round(rms, 4),
        'p95_3d_mm': round(p95, 4),
        'max_3d_mm': round(float(np.max(residuals_mm)), 4),
    }


def _axis_stats(axes):
    arr = np.array(axes, dtype=np.float64)
    n = len(arr)
    mean_axis = np.mean(arr, axis=0)
    mean_norm = np.linalg.norm(mean_axis)
    if mean_norm > 1e-9:
        mean_axis = mean_axis / mean_norm
    angles_from_mean = []
    for a in arr:
        dot = float(np.clip(np.dot(a, mean_axis), -1.0, 1.0))
        angles_from_mean.append(math.degrees(math.acos(abs(dot))))
    angles = np.array(angles_from_mean)
    return {
        'samples': n,
        'mean_axis': mean_axis.tolist(),
        'std_deg': round(float(np.std(angles)), 4),
        'max_deg': round(float(np.max(angles)), 4),
        'p95_deg': round(float(np.percentile(angles, 95)), 4) if n >= 5 else None,
    }


def print_summary(summary):
    print('\n' + '=' * 70)
    print(f"  螺栓识别精度测试结果")
    print(f"  总帧数: {summary['total_frames']}, "
          f"含 AprilTag: {summary['frames_with_tag']}")
    print('=' * 70)

    for key, data in summary['targets'].items():
        meta = data.get('metadata', {})
        print(f"\n--- {key} (class={meta.get('class')}, "
              f"source={meta.get('source', 'N/A')}) ---")

        for coord_sys in ('tag', 'camera'):
            stats = data.get(coord_sys)
            if stats is None:
                continue
            label = 'AprilTag坐标系' if coord_sys == 'tag' else '相机坐标系'
            print(f"\n  [{label}] ({stats['samples']} 帧)")
            print(f"    中位位置(m):  x={stats['median_m'][0]:.5f}  "
                  f"y={stats['median_m'][1]:.5f}  z={stats['median_m'][2]:.5f}")
            print(f"    标准差(mm):   x={stats['std_xyz_mm'][0]:.3f}  "
                  f"y={stats['std_xyz_mm'][1]:.3f}  z={stats['std_xyz_mm'][2]:.3f}")
            print(f"    极差(mm):     x={stats['range_xyz_mm'][0]:.3f}  "
                  f"y={stats['range_xyz_mm'][1]:.3f}  z={stats['range_xyz_mm'][2]:.3f}")
            print(f"    3D RMS(mm):   {stats['rms_3d_mm']:.3f}")
            print(f"    3D P95(mm):   {stats['p95_3d_mm']:.3f}")
            print(f"    3D Max(mm):   {stats['max_3d_mm']:.3f}")

        axis = data.get('axis')
        if axis:
            mean = axis['mean_axis']
            print(f"\n  [轴线方向] ({axis['samples']} 帧)")
            print(f"    均值法向:  ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
            print(f"    抖动 std:  {axis['std_deg']:.3f} deg")
            print(f"    抖动 max:  {axis['max_deg']:.3f} deg")
            if axis['p95_deg'] is not None:
                print(f"    抖动 P95:  {axis['p95_deg']:.3f} deg")

    print('\n' + '=' * 70)


def main():
    args = parse_args()

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('precision_test')

    collector = PrecisionCollector(class_filter=args.class_filter)
    bolt_frames = [0]
    running = [True]

    def on_shutdown(sig, frame):
        running[0] = False
    signal.signal(signal.SIGINT, on_shutdown)

    use_targets = bool(args.targets_topic)
    pending_targets = {}
    pending_geometry = {}

    def stamp_key(payload):
        return int(round(float(payload.get('stamp', 0.0)) * 1_000_000))

    def on_geometry(msg):
        payload = json.loads(msg.data)
        if use_targets:
            key = stamp_key(payload)
            pending_geometry[key] = payload
            if key in pending_targets:
                record = collector.add_targets_frame(
                    pending_targets.pop(key), pending_geometry.pop(key))
                if record and record['bolts']:
                    bolt_frames[0] += 1
        else:
            record = collector.add_geometry_frame(payload)
            if record and record['bolts']:
                bolt_frames[0] += 1

    def on_targets(msg):
        payload = json.loads(msg.data)
        key = stamp_key(payload)
        pending_targets[key] = payload
        if key in pending_geometry:
            record = collector.add_targets_frame(
                pending_targets.pop(key), pending_geometry.pop(key))
            if record and record['bolts']:
                bolt_frames[0] += 1

    node.create_subscription(String, args.geometry_topic, on_geometry, 20)
    if use_targets:
        node.create_subscription(String, args.targets_topic, on_targets, 20)

    print(f"开始采集... (目标 {args.frames} 帧, 超时 {args.timeout}s)")
    print(f"  geometry: {args.geometry_topic}")
    if use_targets:
        print(f"  targets:  {args.targets_topic}")
    print(f"  class:    {args.class_filter}")
    print(f"  按 Ctrl+C 提前结束\n")

    t0 = time.time()
    last_print = 0
    while running[0] and bolt_frames[0] < args.frames:
        if time.time() - t0 > args.timeout:
            print(f'\n超时 ({args.timeout}s)')
            break
        rclpy.spin_once(node, timeout_sec=0.05)
        if bolt_frames[0] > last_print:
            elapsed = time.time() - t0
            hz = bolt_frames[0] / max(elapsed, 0.01)
            print(f'\r  采集: {bolt_frames[0]}/{args.frames} 帧  '
                  f'({hz:.1f} Hz, {elapsed:.1f}s)', end='', flush=True)
            last_print = bolt_frames[0]

    node.destroy_node()
    rclpy.shutdown()

    elapsed = time.time() - t0
    print(f'\n\n采集完成: {bolt_frames[0]} 帧, 耗时 {elapsed:.1f}s')

    summary = collector.summary()
    print_summary(summary)

    # 保存结果
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'/tmp/bolt_precision_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV per-frame data
    csv_path = out_dir / 'frames.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'stamp', 'key', 'has_tag',
            'cam_x_m', 'cam_y_m', 'cam_z_m',
            'tag_x_mm', 'tag_y_mm', 'tag_z_mm',
            'axis_x', 'axis_y', 'axis_z', 'source',
        ])
        writer.writeheader()
        for i, frame in enumerate(collector.frames):
            for bolt in frame['bolts']:
                cam = bolt['camera_m']
                tag = bolt.get('tag_m')
                axis = bolt.get('axis')
                writer.writerow({
                    'frame': i,
                    'stamp': f"{frame['stamp']:.9f}",
                    'key': bolt['key'],
                    'has_tag': frame['has_tag_frame'],
                    'cam_x_m': f"{cam[0]:.6f}",
                    'cam_y_m': f"{cam[1]:.6f}",
                    'cam_z_m': f"{cam[2]:.6f}",
                    'tag_x_mm': f"{tag[0]*1000:.3f}" if tag else '',
                    'tag_y_mm': f"{tag[1]*1000:.3f}" if tag else '',
                    'tag_z_mm': f"{tag[2]*1000:.3f}" if tag else '',
                    'axis_x': f"{axis[0]:.6f}" if axis else '',
                    'axis_y': f"{axis[1]:.6f}" if axis else '',
                    'axis_z': f"{axis[2]:.6f}" if axis else '',
                    'source': bolt.get('source', ''),
                })

    print(f'\n结果保存到:')
    print(f'  {summary_path}')
    print(f'  {csv_path}')


if __name__ == '__main__':
    main()
