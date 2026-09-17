#!/usr/bin/env python3
"""
螺栓法向量 + 位置多方案对比测试

以 AprilTag 方案为真值，记录每个位姿下各方案的法向量方向和位置，
统计和真值的偏差。

真值定义:
  - 法向量: AprilTag PnP 解算的法向
  - 位置: AprilTag 坐标系下螺栓标靶的相对位置
  - Tag-Bolt 距离: 46.1mm (物理常量)

交互指令:
  record [备注]  — 录制当前位姿 (默认50帧)
  show           — 显示所有结果
  save           — 保存并退出
  q              — 退出
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'src', 'panel_detection'))
from panel_detection.concentric_bolt_marker import detect_concentric_bolt_markers
from panel_detection.depth_utils import (
    get_robust_depth, deproject_pixel_to_point, filter_depth,
    estimate_fastener_patch_axis_direction,
    estimate_fastener_group_axis_direction,
    estimate_object_axis_direction,
)
from panel_detection.camera.base import CameraIntrinsics
from panel_detection.fastener_precision_recorder import (
    build_tag_frame, camera_point_to_tag,
)


DEPTH_SCALE = 0.001
GROUND_TRUTH_TAG_BOLT_DIST_MM = 46.1


def parse_args():
    parser = argparse.ArgumentParser(description='法向量+位置多方案对比')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--geometry-topic', default='/objects/geometry')
    parser.add_argument('--output-dir', default='')
    return parser.parse_args()


def angle_between(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return math.degrees(math.acos(abs(dot)))


def collect_pose(node, state, intrin_state, n_frames, timeout=90):
    """采集一个位姿的多帧数据，同时从传感器和话题获取。"""
    import rclpy

    frames = []
    collected = 0
    t0 = time.time()
    last_geom_stamp = [0]

    while collected < n_frames and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)

        color = state.get('color')
        depth_raw = state.get('depth')
        intrin = intrin_state.get('intrin')
        geom = state.get('geom')
        if color is None or depth_raw is None or intrin is None:
            continue

        markers = detect_concentric_bolt_markers(color, {})
        if not markers:
            continue

        collected += 1
        m = markers[0]
        mx, my = int(round(m.center[0])), int(round(m.center[1]))
        filtered = filter_depth(depth_raw, method='bilateral', kernel_size=5)

        frame = {
            'marker_px': (mx, my),
            'marker_radius': m.radius,
        }

        # === Method A: 纯深度 (center 3x3) ===
        d = get_robust_depth(filtered, mx, my, sample_radius=3, depth_scale=DEPTH_SCALE)
        if d > 0:
            xyz = deproject_pixel_to_point(intrin, (mx, my), d)
            frame['depth_position'] = np.array(xyz)
        else:
            frame['depth_position'] = None

        # === Method B: local_patch_plane 法向 ===
        patch_result = estimate_fastener_patch_axis_direction(
            filtered, intrin, m.bbox, depth_scale=DEPTH_SCALE)
        if patch_result is not None:
            frame['patch_axis'] = np.array(patch_result[0])
        else:
            frame['patch_axis'] = None

        # === Method C: local_depth 法向 (bbox 内部) ===
        obj_result = estimate_object_axis_direction(
            filtered, intrin, m.bbox, depth_scale=DEPTH_SCALE,
            object_class='bolt')
        if obj_result is not None:
            frame['local_depth_axis'] = np.array(obj_result[0])
        else:
            frame['local_depth_axis'] = None

        # === Method D: AprilTag (from topic) ===
        frame['tag_axis'] = None
        frame['tag_position'] = None
        frame['tag_bolt_position_tag_frame'] = None
        frame['tag_bolt_dist_mm'] = None
        frame['tag_source'] = None

        if geom is not None:
            stamp = geom.get('stamp', 0)
            if stamp != last_geom_stamp[0]:
                last_geom_stamp[0] = stamp
                ref = geom.get('axis_reference')
                if ref is not None:
                    frame['tag_source'] = ref.get('source')
                    tag_normal = ref.get('normal')
                    if tag_normal is not None:
                        frame['tag_axis'] = np.array(tag_normal, dtype=np.float64)

                    # Tag origin
                    tag_origin = ref.get('origin', ref.get('centroid'))
                    if tag_origin is not None:
                        frame['tag_origin'] = np.array(tag_origin, dtype=np.float64)

                    # AprilTag 平面法: marker 位置
                    for ax in geom.get('axis_directions', []):
                        if 'marker' in ax.get('source', ''):
                            c = ax['centroid']
                            frame['tag_position'] = np.array(c, dtype=np.float64)
                            # Tag 坐标系下的位置
                            tag_pt = camera_point_to_tag(
                                np.array(c, dtype=np.float64), ref)
                            if tag_pt is not None:
                                frame['tag_bolt_position_tag_frame'] = tag_pt
                            # Tag-Bolt 距离
                            if tag_origin is not None:
                                dist = np.linalg.norm(
                                    np.array(c) - np.array(tag_origin)) * 1000
                                frame['tag_bolt_dist_mm'] = dist
                            break

        frames.append(frame)
        if collected % 10 == 0 or collected == n_frames:
            print('\r    %d/%d' % (collected, n_frames), end='', flush=True)

    print()
    return frames


def summarize_pose(frames):
    """统计一个位姿的各方案结果。"""
    result = {'n_frames': len(frames)}

    def axis_stats(key):
        axes = [f[key] for f in frames if f.get(key) is not None]
        if len(axes) < 3:
            return None
        arr = np.array(axes)
        mean = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-9:
            mean = mean / norm
        devs = []
        for a in arr:
            d = float(np.clip(abs(np.dot(a / np.linalg.norm(a), mean)), -1, 1))
            devs.append(math.degrees(math.acos(d)))
        return {
            'mean': mean.tolist(),
            'std_deg': round(float(np.std(devs)), 4),
            'max_deg': round(float(np.max(devs)), 4),
            'samples': len(axes),
        }

    def pos_stats(key):
        pts = [f[key] for f in frames if f.get(key) is not None]
        if len(pts) < 3:
            return None
        arr = np.array(pts)
        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0) * 1000
        res = np.linalg.norm(arr - median, axis=1) * 1000
        rms = math.sqrt(np.mean(res ** 2))
        return {
            'median_m': median.tolist(),
            'std_mm': std.tolist(),
            'rms_mm': round(rms, 4),
            'samples': len(pts),
        }

    result['tag_axis'] = axis_stats('tag_axis')
    result['patch_axis'] = axis_stats('patch_axis')
    result['local_depth_axis'] = axis_stats('local_depth_axis')
    result['tag_position'] = pos_stats('tag_position')
    result['depth_position'] = pos_stats('depth_position')
    result['tag_bolt_tag_frame'] = pos_stats('tag_bolt_position_tag_frame')

    # Tag-Bolt 距离一致性
    dists = [f['tag_bolt_dist_mm'] for f in frames if f.get('tag_bolt_dist_mm') is not None]
    if dists:
        arr = np.array(dists)
        result['tag_bolt_dist'] = {
            'median_mm': round(float(np.median(arr)), 3),
            'std_mm': round(float(np.std(arr)), 3),
            'samples': len(dists),
        }

    # 方案间法向量夹角
    if result['tag_axis'] and result['patch_axis']:
        angle = angle_between(result['tag_axis']['mean'], result['patch_axis']['mean'])
        result['tag_vs_patch_deg'] = round(angle, 3) if angle is not None else None
    if result['tag_axis'] and result['local_depth_axis']:
        angle = angle_between(result['tag_axis']['mean'], result['local_depth_axis']['mean'])
        result['tag_vs_local_depth_deg'] = round(angle, 3) if angle is not None else None

    # 位置偏差 (depth vs tag)
    if result['tag_position'] and result['depth_position']:
        delta = (np.array(result['depth_position']['median_m'])
                 - np.array(result['tag_position']['median_m'])) * 1000
        result['position_delta_mm'] = {
            'dx': round(delta[0], 3),
            'dy': round(delta[1], 3),
            'dz': round(delta[2], 3),
            'dist_3d': round(float(np.linalg.norm(delta)), 3),
        }

    return result


def print_pose(label, summary):
    print('\n  === %s (%d frames) ===' % (label, summary['n_frames']))

    # 法向量
    methods = [
        ('AprilTag PnP (真值)', 'tag_axis'),
        ('local_patch_plane', 'patch_axis'),
        ('local_depth (bbox)', 'local_depth_axis'),
    ]
    print('\n  法向量:')
    print('    %-25s  %-36s  std      max      samples' % ('方案', '均值'))
    for name, key in methods:
        s = summary.get(key)
        if s is None:
            print('    %-25s  (无数据)' % name)
            continue
        m = s['mean']
        print('    %-25s  (% .4f, % .4f, % .4f)  %.3f°   %.3f°   %d' % (
            name, m[0], m[1], m[2], s['std_deg'], s['max_deg'], s['samples']))

    # 法向量偏差
    if summary.get('tag_vs_patch_deg') is not None:
        print('\n    patch vs AprilTag:       %.3f°' % summary['tag_vs_patch_deg'])
    if summary.get('tag_vs_local_depth_deg') is not None:
        print('    local_depth vs AprilTag: %.3f°' % summary['tag_vs_local_depth_deg'])

    # 位置
    print('\n  位置:')
    for name, key in [('AprilTag plane (真值)', 'tag_position'),
                       ('Depth sensor', 'depth_position')]:
        s = summary.get(key)
        if s is None:
            print('    %-22s  (无数据)' % name)
            continue
        m = s['median_m']
        st = s['std_mm']
        print('    %-22s  (%.4f, %.4f, %.4f)m  std=(%.3f,%.3f,%.3f)mm  RMS=%.3fmm' % (
            name, m[0], m[1], m[2], st[0], st[1], st[2], s['rms_mm']))

    if summary.get('position_delta_mm'):
        d = summary['position_delta_mm']
        print('\n    Depth vs Tag 偏差: dx=%.3f dy=%.3f dz=%.3f  3D=%.3f mm' % (
            d['dx'], d['dy'], d['dz'], d['dist_3d']))

    # Tag-Bolt 距离
    if summary.get('tag_bolt_dist'):
        tb = summary['tag_bolt_dist']
        err = abs(tb['median_mm'] - GROUND_TRUTH_TAG_BOLT_DIST_MM)
        print('\n    Tag-Bolt 距离: %.2f mm (真值=%.1f, 偏差=%.2f mm, std=%.3f)' % (
            tb['median_mm'], GROUND_TRUTH_TAG_BOLT_DIST_MM, err, tb['std_mm']))

    # Tag 坐标系下的位置稳定性
    s = summary.get('tag_bolt_tag_frame')
    if s is not None:
        m = s['median_m']
        st = s['std_mm']
        print('\n    Tag坐标系下螺栓位置: (%.3f, %.3f, %.3f)mm  std=(%.3f,%.3f,%.3f)mm' % (
            m[0]*1000, m[1]*1000, m[2]*1000, st[0], st[1], st[2]))


def print_all(poses):
    print('\n' + '=' * 70)
    print('  螺栓法向量+位置 多方案对比 (AprilTag为真值)')
    print('=' * 70)

    for label, summary in poses:
        print_pose(label, summary)

    if len(poses) >= 2:
        print('\n  --- 跨位姿汇总 ---')
        # 法向量偏差
        patch_angles = [s.get('tag_vs_patch_deg') for _, s in poses
                        if s.get('tag_vs_patch_deg') is not None]
        local_angles = [s.get('tag_vs_local_depth_deg') for _, s in poses
                        if s.get('tag_vs_local_depth_deg') is not None]
        if patch_angles:
            arr = np.array(patch_angles)
            print('    patch vs Tag:       mean=%.3f°  max=%.3f°  (%d poses)' % (
                np.mean(arr), np.max(arr), len(arr)))
        if local_angles:
            arr = np.array(local_angles)
            print('    local_depth vs Tag: mean=%.3f°  max=%.3f°  (%d poses)' % (
                np.mean(arr), np.max(arr), len(arr)))

        # 位置偏差
        pos_deltas = [s['position_delta_mm']['dist_3d'] for _, s in poses
                      if s.get('position_delta_mm')]
        if pos_deltas:
            arr = np.array(pos_deltas)
            print('    Position depth vs tag: mean=%.3f mm  max=%.3f mm' % (
                np.mean(arr), np.max(arr)))

        # Tag 坐标系位置一致性
        tag_frame_positions = [
            np.array(s['tag_bolt_tag_frame']['median_m'])
            for _, s in poses if s.get('tag_bolt_tag_frame')]
        if len(tag_frame_positions) >= 2:
            arr = np.array(tag_frame_positions)
            spread = np.ptp(arr, axis=0) * 1000
            print('    Tag坐标系跨位姿 range: (%.3f, %.3f, %.3f) mm' % (
                spread[0], spread[1], spread[2]))

    print('=' * 70)


def main():
    args = parse_args()

    import rclpy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('axis_accuracy')

    state = {}
    intrin_state = {}

    def on_color(msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        state['color'] = img

    def on_depth(msg):
        if msg.encoding in ('16UC1', 'mono16'):
            state['depth'] = np.frombuffer(
                msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def on_info(msg):
        intrin_state['intrin'] = CameraIntrinsics(
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
            width=msg.width, height=msg.height,
            coeffs=list(msg.d) if msg.d else [0.0] * 5)

    def on_geom(msg):
        state['geom'] = json.loads(msg.data)

    node.create_subscription(Image, '/camera/color/image_raw', on_color, 5)
    node.create_subscription(Image, '/camera/depth/image_raw', on_depth, 5)
    node.create_subscription(CameraInfo, '/camera/color/camera_info', on_info, 5)
    node.create_subscription(String, args.geometry_topic, on_geom, 20)

    if not args.output_dir:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = '/tmp/axis_accuracy_%s' % ts

    poses = []
    all_frames = []
    pose_count = 0

    print('螺栓法向量+位置 多方案对比测试')
    print('  geometry: %s' % args.geometry_topic)
    print('  每位姿帧数: %d' % args.frames)
    print('  真值: AprilTag PnP, Tag-Bolt距离=%.1fmm' % GROUND_TRUTH_TAG_BOLT_DIST_MM)
    print()
    print('指令: record [备注] / show / save / q')
    print()

    while True:
        try:
            cmd = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd.lower() in ('q', 'quit', 'exit'):
            break

        elif cmd.lower().startswith('record'):
            parts = cmd.split(maxsplit=1)
            pose_count += 1
            label = parts[1] if len(parts) > 1 else 'pose_%d' % pose_count
            print('  录制 [%s] (%d帧)...' % (label, args.frames))
            frames = collect_pose(node, state, intrin_state, args.frames)
            if not frames:
                print('  无有效帧')
                continue
            summary = summarize_pose(frames)
            poses.append((label, summary))
            all_frames.append((label, frames))
            print_pose(label, summary)

        elif cmd.lower() == 'show':
            print_all(poses)

        elif cmd.lower() == 'save':
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                raise TypeError(str(type(obj)))

            summary_path = out / 'axis_accuracy.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [{'label': l, 'summary': s} for l, s in poses],
                    f, indent=2, ensure_ascii=False, default=convert)

            csv_path = out / 'axis_frames.csv'
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'pose', 'frame', 'marker_px_x', 'marker_px_y',
                    'depth_x', 'depth_y', 'depth_z',
                    'tag_x', 'tag_y', 'tag_z',
                    'tag_axis_x', 'tag_axis_y', 'tag_axis_z',
                    'patch_axis_x', 'patch_axis_y', 'patch_axis_z',
                    'local_axis_x', 'local_axis_y', 'local_axis_z',
                    'tag_bolt_dist_mm', 'tag_source',
                ])
                writer.writeheader()
                for label, frames in all_frames:
                    for idx, f in enumerate(frames):
                        row = {'pose': label, 'frame': idx,
                               'marker_px_x': f['marker_px'][0],
                               'marker_px_y': f['marker_px'][1]}
                        for prefix, key in [
                            ('depth', 'depth_position'),
                            ('tag', 'tag_position'),
                        ]:
                            v = f.get(key)
                            if v is not None:
                                row['%s_x' % prefix] = '%.6f' % v[0]
                                row['%s_y' % prefix] = '%.6f' % v[1]
                                row['%s_z' % prefix] = '%.6f' % v[2]
                        for prefix, key in [
                            ('tag_axis', 'tag_axis'),
                            ('patch_axis', 'patch_axis'),
                            ('local_axis', 'local_depth_axis'),
                        ]:
                            v = f.get(key)
                            if v is not None:
                                row['%s_x' % prefix] = '%.6f' % v[0]
                                row['%s_y' % prefix] = '%.6f' % v[1]
                                row['%s_z' % prefix] = '%.6f' % v[2]
                        row['tag_bolt_dist_mm'] = (
                            '%.3f' % f['tag_bolt_dist_mm']
                            if f.get('tag_bolt_dist_mm') else '')
                        row['tag_source'] = f.get('tag_source', '')
                        writer.writerow(row)

            print('\n保存到:')
            print('  %s' % summary_path)
            print('  %s' % csv_path)
            print_all(poses)
            break

        else:
            print('  指令: record [备注] / show / save / q')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
