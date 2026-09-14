#!/usr/bin/env python3
"""
螺栓识别多视角一致性测试

两阶段：
  1. 参考阶段：在最佳观测位置采集 N 帧，取中位值作为伪真值
  2. 测试阶段：在不同角度/距离采集，和伪真值比较偏差

所有坐标统一到 AprilTag 坐标系，消除相机移动的影响。

用法:
  python3 scripts/accuracy_test.py [--frames 100] [--output-dir /tmp/accuracy_test]

交互指令:
  ref        — 录制参考值（伪真值）
  test       — 录制一个测试位姿
  show       — 显示当前所有结果
  save       — 保存结果并退出
  q / quit   — 退出
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
from panel_detection.fastener_precision_recorder import (
    build_tag_frame, camera_point_to_tag,
)


def load_ref_from_csv(csv_path, source_filter='apriltag_pnp_marker'):
    """从 precision_test.py 输出的 frames.csv 加载参考数据。"""
    frames_by_idx = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get('source', '')
            if source_filter and source_filter not in source:
                continue
            tag_x = row.get('tag_x_mm', '')
            tag_y = row.get('tag_y_mm', '')
            tag_z = row.get('tag_z_mm', '')
            if not tag_x or not tag_y or not tag_z:
                continue
            key = row.get('key', 'unknown')
            # 统一 key 格式
            if 'marker' not in key and 'marker' in source:
                key = f"marker:{key}"
            frame_idx = int(row.get('frame', 0))
            frames_by_idx[frame_idx].append({
                'key': key,
                'tag_m': [float(tag_x) / 1000, float(tag_y) / 1000, float(tag_z) / 1000],
                'camera_m': [float(row.get('cam_x_m', 0)),
                             float(row.get('cam_y_m', 0)),
                             float(row.get('cam_z_m', 0))],
                'axis': ([float(row['axis_x']), float(row['axis_y']), float(row['axis_z'])]
                         if row.get('axis_x') else None),
                'source': source,
            })
    frames = []
    for idx in sorted(frames_by_idx):
        frames.append({
            'stamp': 0.0,
            'tag_source': source_filter,
            'bolts': frames_by_idx[idx],
        })

    tag_points = defaultdict(list)
    axes = defaultdict(list)
    for frame in frames:
        for bolt in frame['bolts']:
            tag_points[bolt['key']].append(np.array(bolt['tag_m']))
            if bolt['axis'] is not None:
                axes[bolt['key']].append(np.array(bolt['axis']))

    positions = {}
    for key, pts in tag_points.items():
        arr = np.array(pts)
        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0) * 1000
        entry = {
            'samples': len(pts),
            'median_tag_m': median.tolist(),
            'std_mm': std.tolist(),
        }
        if axes.get(key):
            ax_arr = np.array(axes[key])
            mean_ax = np.mean(ax_arr, axis=0)
            norm = np.linalg.norm(mean_ax)
            if norm > 1e-9:
                mean_ax = mean_ax / norm
            entry['mean_axis'] = mean_ax.tolist()
        positions[key] = entry
    return frames, positions


def parse_args():
    parser = argparse.ArgumentParser(description='螺栓多视角一致性测试')
    parser.add_argument('--frames', type=int, default=100,
                        help='每个位姿采集帧数 (default: 100)')
    parser.add_argument('--ref-frames', type=int, default=200,
                        help='参考位姿采集帧数 (default: 200)')
    parser.add_argument('--geometry-topic', default='/objects/geometry')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--class-filter', default='bolt')
    parser.add_argument('--load-ref', default='',
                        help='从已有 CSV 加载参考值 (precision_test 输出的 frames.csv)')
    return parser.parse_args()


def collect_frames(node, topic_state, n_frames, timeout=120.0, class_filter='bolt'):
    """采集 n_frames 帧含有目标螺栓的 geometry 数据。"""
    frames = []
    t0 = time.time()
    topic_state['latest'] = None

    while len(frames) < n_frames and time.time() - t0 < timeout:
        import rclpy
        rclpy.spin_once(node, timeout_sec=0.05)

        geom = topic_state.get('latest')
        if geom is None:
            continue
        topic_state['latest'] = None

        ref = geom.get('axis_reference')
        tag_frame = build_tag_frame(ref) if ref else None
        if tag_frame is None:
            continue

        bolt_entries = []
        for ax in geom.get('axis_directions', []):
            if class_filter and ax.get('class') != class_filter:
                continue
            centroid = ax.get('centroid')
            if centroid is None:
                continue
            camera_pt = np.array(centroid, dtype=np.float64)
            if not np.all(np.isfinite(camera_pt)) or camera_pt[2] <= 0:
                continue
            tag_pt = camera_point_to_tag(camera_pt, ref)
            if tag_pt is None:
                continue

            source = ax.get('source', '')
            slot = ax.get('slot', 'unknown')
            bolt_id = ax.get('id', '?')
            group_id = ax.get('group_id', '?')
            # 只保留同心圆标靶检测结果，YOLO 裸检测精度差且无稳定 ID
            if 'marker' in source:
                key = f"marker:G{group_id}:ID{bolt_id}:{slot}"
            elif slot != 'unknown' and bolt_id != '?':
                key = f"G{group_id}:ID{bolt_id}:{slot}"
            else:
                continue
            axis_dir = ax.get('axis_direction')

            bolt_entries.append({
                'key': key,
                'camera_m': camera_pt.tolist(),
                'tag_m': tag_pt.tolist(),
                'axis': axis_dir,
                'source': ax.get('source', ''),
            })

        if not bolt_entries:
            continue

        frames.append({
            'stamp': geom.get('stamp', 0.0),
            'tag_source': ref.get('source') if ref else None,
            'tag_reprojection_px': ref.get('reprojection_error_px') if ref else None,
            'bolts': bolt_entries,
        })

        count = len(frames)
        if count % 10 == 0 or count == n_frames:
            elapsed = time.time() - t0
            hz = count / max(elapsed, 0.01)
            print(f'\r    {count}/{n_frames} ({hz:.1f} Hz, {elapsed:.1f}s)',
                  end='', flush=True)

    print()
    return frames


def compute_median_positions(frames):
    """从帧列表中算出每个螺栓的中位 tag 坐标和轴线。"""
    tag_points = defaultdict(list)
    axes = defaultdict(list)
    for frame in frames:
        for bolt in frame['bolts']:
            tag_points[bolt['key']].append(np.array(bolt['tag_m']))
            if bolt['axis'] is not None:
                axes[bolt['key']].append(np.array(bolt['axis']))

    result = {}
    for key, pts in tag_points.items():
        arr = np.array(pts)
        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0) * 1000
        entry = {
            'samples': len(pts),
            'median_tag_m': median.tolist(),
            'std_mm': std.tolist(),
        }
        if axes.get(key):
            ax_arr = np.array(axes[key])
            mean_ax = np.mean(ax_arr, axis=0)
            norm = np.linalg.norm(mean_ax)
            if norm > 1e-9:
                mean_ax = mean_ax / norm
            entry['mean_axis'] = mean_ax.tolist()
        result[key] = entry
    return result


def compare_to_reference(test_positions, ref_positions):
    """计算测试位姿相对参考值的偏差。"""
    comparisons = {}
    for key in test_positions:
        if key not in ref_positions:
            continue
        ref_pt = np.array(ref_positions[key]['median_tag_m'])
        test_pt = np.array(test_positions[key]['median_tag_m'])
        delta = test_pt - ref_pt
        dist_3d = float(np.linalg.norm(delta)) * 1000

        entry = {
            'ref_tag_m': ref_pt.tolist(),
            'test_tag_m': test_pt.tolist(),
            'delta_mm': (delta * 1000).tolist(),
            'error_3d_mm': round(dist_3d, 3),
            'test_std_mm': test_positions[key]['std_mm'],
            'test_samples': test_positions[key]['samples'],
        }

        ref_axis = ref_positions[key].get('mean_axis')
        test_axis = test_positions[key].get('mean_axis')
        if ref_axis is not None and test_axis is not None:
            ra = np.array(ref_axis)
            ta = np.array(test_axis)
            dot = float(np.clip(abs(np.dot(ra, ta)), -1.0, 1.0))
            entry['axis_error_deg'] = round(math.degrees(math.acos(dot)), 3)

        comparisons[key] = entry
    return comparisons


def print_pose_result(label, positions, comparisons=None):
    print(f'\n  === {label} ===')
    for key, pos in sorted(positions.items()):
        median = pos['median_tag_m']
        std = pos['std_mm']
        print(f'    {key}: tag=({median[0]*1000:.2f}, {median[1]*1000:.2f}, '
              f'{median[2]*1000:.2f})mm  '
              f'std=({std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f})mm  '
              f'({pos["samples"]}帧)')
        if comparisons and key in comparisons:
            c = comparisons[key]
            d = c['delta_mm']
            print(f'           偏差: dx={d[0]:.3f} dy={d[1]:.3f} dz={d[2]:.3f}mm  '
                  f'3D={c["error_3d_mm"]:.3f}mm', end='')
            if 'axis_error_deg' in c:
                print(f'  轴线差={c["axis_error_deg"]:.2f}°', end='')
            print()


def print_all_results(ref_positions, test_results):
    print('\n' + '=' * 70)
    print('  螺栓多视角一致性测试汇总')
    print('=' * 70)

    if ref_positions:
        print_pose_result('参考位姿 (伪真值)', ref_positions)

    for i, (label, positions, comparisons) in enumerate(test_results):
        print_pose_result(f'测试位姿 #{i+1}: {label}', positions, comparisons)

    if test_results:
        all_errors = []
        all_axis_errors = []
        for _, _, comparisons in test_results:
            for c in comparisons.values():
                all_errors.append(c['error_3d_mm'])
                if 'axis_error_deg' in c:
                    all_axis_errors.append(c['axis_error_deg'])

        if all_errors:
            errors = np.array(all_errors)
            print(f'\n  --- 总体偏差统计 ({len(errors)} 组) ---')
            print(f'    3D 偏差均值:  {np.mean(errors):.3f} mm')
            print(f'    3D 偏差中位:  {np.median(errors):.3f} mm')
            print(f'    3D 偏差最大:  {np.max(errors):.3f} mm')
            if len(errors) >= 3:
                print(f'    3D 偏差 std:  {np.std(errors):.3f} mm')
        if all_axis_errors:
            ax_err = np.array(all_axis_errors)
            print(f'    轴线偏差均值: {np.mean(ax_err):.3f}°')
            print(f'    轴线偏差最大: {np.max(ax_err):.3f}°')

    print('=' * 70)


def save_results(output_dir, ref_positions, ref_frames, test_results, all_test_frames):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        'reference': {
            'positions': ref_positions,
            'frame_count': len(ref_frames),
        },
        'tests': [],
    }
    for label, positions, comparisons in test_results:
        data['tests'].append({
            'label': label,
            'positions': positions,
            'comparisons': comparisons,
        })

    # Summary JSON
    summary_path = out / 'accuracy_summary.json'

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f'{type(obj)}')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_convert)

    # Per-frame CSV
    csv_path = out / 'all_frames.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'phase', 'label', 'frame_idx', 'stamp', 'key',
            'tag_x_mm', 'tag_y_mm', 'tag_z_mm',
            'cam_x_m', 'cam_y_m', 'cam_z_m',
            'axis_x', 'axis_y', 'axis_z', 'source',
        ])
        writer.writeheader()

        def write_frames(phase, label, frames):
            for idx, frame in enumerate(frames):
                for bolt in frame['bolts']:
                    tag = bolt['tag_m']
                    cam = bolt['camera_m']
                    axis = bolt.get('axis')
                    writer.writerow({
                        'phase': phase, 'label': label,
                        'frame_idx': idx,
                        'stamp': f"{frame['stamp']:.9f}",
                        'key': bolt['key'],
                        'tag_x_mm': f'{tag[0]*1000:.3f}',
                        'tag_y_mm': f'{tag[1]*1000:.3f}',
                        'tag_z_mm': f'{tag[2]*1000:.3f}',
                        'cam_x_m': f'{cam[0]:.6f}',
                        'cam_y_m': f'{cam[1]:.6f}',
                        'cam_z_m': f'{cam[2]:.6f}',
                        'axis_x': f'{axis[0]:.6f}' if axis else '',
                        'axis_y': f'{axis[1]:.6f}' if axis else '',
                        'axis_z': f'{axis[2]:.6f}' if axis else '',
                        'source': bolt.get('source', ''),
                    })

        write_frames('reference', 'ref', ref_frames)
        for i, (label, frames) in enumerate(all_test_frames):
            write_frames('test', label, frames)

    print(f'\n结果已保存:')
    print(f'  {summary_path}')
    print(f'  {csv_path}')


def main():
    args = parse_args()

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('accuracy_test')

    topic_state = {'latest': None}

    def on_geom(msg):
        topic_state['latest'] = json.loads(msg.data)

    node.create_subscription(String, args.geometry_topic, on_geom, 20)

    if not args.output_dir:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'/tmp/bolt_accuracy_{ts}'

    ref_positions = {}
    ref_frames = []
    test_results = []
    all_test_frames = []
    test_count = 0

    if args.load_ref:
        ref_frames, ref_positions = load_ref_from_csv(args.load_ref)
        if ref_positions:
            print(f'已加载参考值: {args.load_ref}')
            print(f'  {len(ref_frames)} 帧, {len(ref_positions)} 个目标')
            print_pose_result('参考位姿 (从文件加载)', ref_positions)
        else:
            print(f'⚠ 从 {args.load_ref} 加载参考值失败')

    print('\n螺栓多视角一致性测试')
    print(f'  geometry topic: {args.geometry_topic}')
    print(f'  参考帧数: {args.ref_frames}, 测试帧数: {args.frames}')
    print(f'  输出: {args.output_dir}')
    print()
    print('指令:')
    print('  ref   — 录制参考值 (把相机放到最佳位置后输入)')
    print('  test  — 录制测试位姿 (移动相机到新位置后输入)')
    print('  show  — 显示当前结果')
    print('  save  — 保存结果并退出')
    print('  q     — 退出')
    print()

    while True:
        try:
            cmd = input('> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ('q', 'quit', 'exit'):
            break

        elif cmd == 'ref':
            print(f'  录制参考值 ({args.ref_frames} 帧)...')
            print('  确保相机在最佳观测位置，AprilTag 和螺栓清晰可见')
            ref_frames = collect_frames(
                node, topic_state, args.ref_frames,
                class_filter=args.class_filter)
            if not ref_frames:
                print('  ⚠ 未采集到有效帧')
                continue
            ref_positions = compute_median_positions(ref_frames)
            print(f'  参考值录制完成 ({len(ref_frames)} 帧, '
                  f'{len(ref_positions)} 个目标)')
            print_pose_result('参考位姿', ref_positions)

        elif cmd.startswith('test'):
            parts = cmd.split(maxsplit=1)
            test_count += 1
            label = parts[1] if len(parts) > 1 else f'pose_{test_count}'
            print(f'  录制测试位姿 [{label}] ({args.frames} 帧)...')
            frames = collect_frames(
                node, topic_state, args.frames,
                class_filter=args.class_filter)
            if not frames:
                print('  ⚠ 未采集到有效帧')
                continue
            positions = compute_median_positions(frames)
            comparisons = compare_to_reference(positions, ref_positions) if ref_positions else {}
            test_results.append((label, positions, comparisons))
            all_test_frames.append((label, frames))
            print_pose_result(f'测试位姿: {label}', positions, comparisons)

        elif cmd == 'show':
            print_all_results(ref_positions, test_results)

        elif cmd == 'save':
            save_results(args.output_dir, ref_positions, ref_frames,
                         test_results, all_test_frames)
            print_all_results(ref_positions, test_results)
            break

        else:
            print('  未知指令，可用: ref / test / show / save / q')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
