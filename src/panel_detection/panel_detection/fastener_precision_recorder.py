"""Record fastener repeatability in the coordinate frame of an AprilTag."""

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _unit(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm < 1e-9:
        return None
    return array / norm


def build_tag_frame(axis_reference):
    """Return metric origin and camera-frame basis for a decoded Tag."""
    if not isinstance(axis_reference, dict):
        return None
    origin = axis_reference.get('origin', axis_reference.get('centroid'))
    x_axis = _unit(axis_reference.get('x_axis'))
    z_axis = _unit(axis_reference.get('normal'))
    if origin is None or x_axis is None or z_axis is None:
        return None
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        return None

    # The normal is temporally filtered by the detector. Re-project the raw
    # PnP X axis onto that filtered plane so the published frame remains
    # orthonormal.
    x_axis = _unit(x_axis - np.dot(x_axis, z_axis) * z_axis)
    if x_axis is None:
        return None
    y_axis = _unit(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return origin, np.vstack([x_axis, y_axis, z_axis])


def camera_point_to_tag(point, axis_reference):
    frame = build_tag_frame(axis_reference)
    point = np.asarray(point, dtype=np.float64)
    if frame is None or point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    origin, camera_to_tag = frame
    return camera_to_tag @ (point - origin)


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class PrecisionAccumulator:
    """Accumulate Tag-relative positions and pair-distance repeatability."""

    def __init__(self, class_filter='bolt', registered_only=True,
                 key_mode='slot', max_tag_distance_m=0.20):
        self.class_filter = str(class_filter)
        self.registered_only = bool(registered_only)
        self.key_mode = str(key_mode)
        self.max_tag_distance_m = float(max_tag_distance_m)
        self.frames_received = 0
        self.frames_usable = 0
        self.frames_without_tag_frame = 0
        self.frames_tag_geometry_outlier = 0
        self.targets_skipped = 0
        self.positions = defaultdict(list)
        self.raw_positions = defaultdict(list)
        self.pair_distances = defaultdict(list)
        self.target_metadata = {}
        self.tag_sources = Counter()
        self.tag_reprojection_errors = []
        self.tag_plane_rms = []

    def _target_key(self, target):
        target_id = target.get('id', target.get('target_id'))
        if target_id is None:
            return None
        if self.key_mode == 'slot' and target.get('slot'):
            return f"slot:{target['slot']}"
        group_id = target.get('group_id', 0)
        return f'G{group_id}:ID{target_id}'

    def add_frame(self, targets_payload, geometry_payload):
        self.frames_received += 1
        reference = geometry_payload.get('axis_reference') or {}
        frame = build_tag_frame(reference)
        if frame is None:
            self.frames_without_tag_frame += 1
            return None

        self.tag_sources[str(reference.get('source', 'unknown'))] += 1
        reprojection = reference.get('reprojection_error_px')
        if reprojection is not None:
            self.tag_reprojection_errors.append(float(reprojection))
        plane_rms = reference.get('rms_error_m')
        if plane_rms is not None:
            self.tag_plane_rms.append(float(plane_rms))

        stamp = float(targets_payload.get('stamp', geometry_payload.get('stamp', 0.0)))
        frame_record = {
            'stamp': stamp,
            'tag_id': reference.get('tag_id'),
            'tag_source': reference.get('source'),
            'tag_origin_camera_m': reference.get('origin', reference.get('centroid')),
            'tag_x_axis_camera': reference.get('x_axis'),
            'tag_y_axis_camera': reference.get('y_axis'),
            'tag_z_axis_camera': reference.get('normal'),
            'tag_geometry_valid': False,
            'median_tag_distance_m': None,
            'targets': [],
            'pair_distances_m': {},
        }
        camera_positions = {}
        for target in targets_payload.get('targets') or []:
            if self.class_filter and target.get('class') != self.class_filter:
                continue
            if self.registered_only and not target.get('registered', False):
                self.targets_skipped += 1
                continue
            key = self._target_key(target)
            position = target.get('position') or {}
            try:
                camera_point = np.array([
                    float(position['x']), float(position['y']),
                    float(position['z']),
                ], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                self.targets_skipped += 1
                continue
            tag_point = camera_point_to_tag(camera_point, reference)
            if key is None or tag_point is None:
                self.targets_skipped += 1
                continue
            self.raw_positions[key].append(tag_point)
            self.target_metadata[key] = {
                'group_id': target.get('group_id'),
                'id': target.get('id', target.get('target_id')),
                'slot': target.get('slot'),
                'class': target.get('class'),
            }
            camera_positions[key] = camera_point
            frame_record['targets'].append({
                'key': key,
                'group_id': target.get('group_id'),
                'id': target.get('id', target.get('target_id')),
                'slot': target.get('slot'),
                'camera_position_m': camera_point.tolist(),
                'tag_position_m': tag_point.tolist(),
                'confidence': target.get('confidence'),
                'classification_source': target.get('classification_source'),
                'depth_source': target.get('depth_source'),
            })

        tag_distances = [
            float(np.linalg.norm(target['tag_position_m']))
            for target in frame_record['targets']
        ]
        if tag_distances:
            median_distance = float(np.median(tag_distances))
            frame_record['median_tag_distance_m'] = median_distance
            tag_geometry_valid = (
                self.max_tag_distance_m <= 0.0
                or median_distance <= self.max_tag_distance_m)
            frame_record['tag_geometry_valid'] = tag_geometry_valid
            if tag_geometry_valid:
                self.frames_usable += 1
                for target in frame_record['targets']:
                    self.positions[target['key']].append(np.asarray(
                        target['tag_position_m'], dtype=np.float64))
            else:
                self.frames_tag_geometry_outlier += 1

        keys = sorted(camera_positions)
        for first_index, first_key in enumerate(keys):
            first_group = first_key.split(':', 1)[0]
            for second_key in keys[first_index + 1:]:
                if second_key.split(':', 1)[0] != first_group:
                    continue
                pair_key = f'{first_key}|{second_key}'
                distance = float(np.linalg.norm(
                    camera_positions[first_key] - camera_positions[second_key]))
                self.pair_distances[pair_key].append(distance)
                frame_record['pair_distances_m'][pair_key] = distance
        return frame_record

    @staticmethod
    def _position_summary(values):
        points = np.asarray(values, dtype=np.float64)
        median = np.median(points, axis=0)
        mean = np.mean(points, axis=0)
        residuals = np.linalg.norm(points - median, axis=1) * 1000.0
        standard_deviation = np.std(points, axis=0) * 1000.0
        ranges = np.ptp(points, axis=0) * 1000.0
        return {
            'samples': int(len(points)),
            'median_tag_position_m': median.tolist(),
            'mean_tag_position_m': mean.tolist(),
            'std_xyz_mm': standard_deviation.tolist(),
            'range_xyz_mm': ranges.tolist(),
            'rms_3d_mm': float(math.sqrt(np.mean(residuals ** 2))),
            'p95_3d_mm': _percentile(residuals, 95),
            'max_3d_mm': float(np.max(residuals)),
        }

    @staticmethod
    def _distance_summary(values):
        distances_mm = np.asarray(values, dtype=np.float64) * 1000.0
        median = float(np.median(distances_mm))
        deviations = np.abs(distances_mm - median)
        return {
            'samples': int(len(distances_mm)),
            'median_distance_mm': median,
            'mean_distance_mm': float(np.mean(distances_mm)),
            'std_mm': float(np.std(distances_mm)),
            'range_mm': float(np.ptp(distances_mm)),
            'p95_abs_deviation_mm': _percentile(deviations, 95),
            'max_abs_deviation_mm': float(np.max(deviations)),
        }

    @staticmethod
    def _quality_summary(values, scale=1.0):
        if not values:
            return None
        scaled = np.asarray(values, dtype=np.float64) * scale
        return {
            'samples': int(len(scaled)),
            'median': float(np.median(scaled)),
            'p95': _percentile(scaled, 95),
            'max': float(np.max(scaled)),
        }

    def summary(self):
        return {
            'metric_definition': {
                'tag_relative': (
                    'Repeatability of each fastener XYZ in the decoded '
                    'AprilTag coordinate frame; residuals use the per-target median.'),
                'pair_distance': (
                    'Repeatability of distances between fasteners; independent '
                    'of the estimated Tag origin and orientation.'),
                'note': (
                    'These are precision/repeatability metrics. Known physical '
                    'dimensions are required to measure absolute accuracy.'),
            },
            'frames_received': self.frames_received,
            'frames_usable': self.frames_usable,
            'frames_without_tag_frame': self.frames_without_tag_frame,
            'frames_tag_geometry_outlier': self.frames_tag_geometry_outlier,
            'targets_skipped': self.targets_skipped,
            'settings': {
                'class_filter': self.class_filter,
                'registered_only': self.registered_only,
                'key_mode': self.key_mode,
                'max_tag_distance_m': self.max_tag_distance_m,
            },
            'tag_sources': dict(self.tag_sources),
            'tag_quality': {
                'reprojection_error_px': self._quality_summary(
                    self.tag_reprojection_errors),
                'depth_plane_rms_mm': self._quality_summary(
                    self.tag_plane_rms, scale=1000.0),
            },
            'targets': {
                key: {
                    **self.target_metadata.get(key, {}),
                    'accepted': (
                        self._position_summary(self.positions[key])
                        if self.positions.get(key) else None),
                    'raw': self._position_summary(values),
                }
                for key, values in sorted(self.raw_positions.items())
            },
            'pair_distances': {
                key: self._distance_summary(values)
                for key, values in sorted(self.pair_distances.items())
            },
        }


class FastenerPrecisionRecorderNode:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        class _Node(Node):
            pass

        self.node = _Node('fastener_precision_recorder')
        self.node.declare_parameter('targets_topic', '/fasteners/targets')
        self.node.declare_parameter('geometry_topic', '/fasteners/geometry')
        self.node.declare_parameter('output_prefix', 'fastener_precision')
        self.node.declare_parameter('class_filter', 'bolt')
        self.node.declare_parameter('registered_only', True)
        self.node.declare_parameter('key_mode', 'slot')
        self.node.declare_parameter('max_tag_distance_m', 0.20)
        prefix = str(self.node.get_parameter('output_prefix').value)
        if prefix.endswith('.jsonl'):
            prefix = prefix[:-6]
        self.prefix = Path(prefix).expanduser()
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(str(self.prefix) + '.jsonl')
        self.csv_path = Path(str(self.prefix) + '.csv')
        self.summary_path = Path(str(self.prefix) + '.summary.json')
        self.accumulator = PrecisionAccumulator(
            class_filter=self.node.get_parameter('class_filter').value,
            registered_only=self.node.get_parameter('registered_only').value,
            key_mode=self.node.get_parameter('key_mode').value,
            max_tag_distance_m=self.node.get_parameter(
                'max_tag_distance_m').value,
        )
        self.pending_targets = {}
        self.pending_geometry = {}
        self.jsonl_file = self.jsonl_path.open('w', encoding='utf-8')
        self.csv_file = self.csv_path.open('w', encoding='utf-8', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'stamp', 'key', 'group_id', 'id', 'slot',
            'tag_x_mm', 'tag_y_mm', 'tag_z_mm',
            'camera_x_m', 'camera_y_m', 'camera_z_m',
            'confidence', 'classification_source', 'depth_source',
            'tag_geometry_valid',
        ])
        self.csv_writer.writeheader()
        targets_topic = str(self.node.get_parameter('targets_topic').value)
        geometry_topic = str(self.node.get_parameter('geometry_topic').value)
        self.target_sub = self.node.create_subscription(
            String, targets_topic, self._targets_callback, 20)
        self.geometry_sub = self.node.create_subscription(
            String, geometry_topic, self._geometry_callback, 20)
        self.timer = self.node.create_timer(5.0, self.write_summary)
        self.closed = False
        self.node.get_logger().info(
            f'精度记录已启动: {self.prefix}.[jsonl|csv|summary.json]')

    @staticmethod
    def _stamp_key(payload):
        return int(round(float(payload.get('stamp', 0.0)) * 1_000_000.0))

    def _targets_callback(self, message):
        self._receive(message.data, targets=True)

    def _geometry_callback(self, message):
        self._receive(message.data, targets=False)

    def _receive(self, data, targets):
        try:
            payload = json.loads(data)
        except (TypeError, json.JSONDecodeError) as error:
            self.node.get_logger().warning(f'忽略无效 JSON: {error}')
            return
        key = self._stamp_key(payload)
        destination = self.pending_targets if targets else self.pending_geometry
        destination[key] = payload
        if key in self.pending_targets and key in self.pending_geometry:
            targets_payload = self.pending_targets.pop(key)
            geometry_payload = self.pending_geometry.pop(key)
            record = self.accumulator.add_frame(targets_payload, geometry_payload)
            if record is not None:
                self._write_record(record)
        self._prune_pending()

    def _prune_pending(self):
        for pending in (self.pending_targets, self.pending_geometry):
            if len(pending) > 200:
                for key in sorted(pending)[:-100]:
                    pending.pop(key, None)

    def _write_record(self, record):
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.jsonl_file.flush()
        for target in record['targets']:
            tag_point = np.asarray(target['tag_position_m']) * 1000.0
            camera_point = target['camera_position_m']
            self.csv_writer.writerow({
                'stamp': f"{record['stamp']:.9f}",
                'key': target['key'],
                'group_id': target.get('group_id'),
                'id': target.get('id'),
                'slot': target.get('slot'),
                'tag_x_mm': f'{tag_point[0]:.3f}',
                'tag_y_mm': f'{tag_point[1]:.3f}',
                'tag_z_mm': f'{tag_point[2]:.3f}',
                'camera_x_m': f'{camera_point[0]:.6f}',
                'camera_y_m': f'{camera_point[1]:.6f}',
                'camera_z_m': f'{camera_point[2]:.6f}',
                'confidence': target.get('confidence'),
                'classification_source': target.get('classification_source'),
                'depth_source': target.get('depth_source'),
                'tag_geometry_valid': record['tag_geometry_valid'],
            })
        self.csv_file.flush()

    def write_summary(self):
        if self.closed:
            return
        temporary_path = Path(str(self.summary_path) + '.tmp')
        with temporary_path.open('w', encoding='utf-8') as output:
            json.dump(self.accumulator.summary(), output, ensure_ascii=False,
                      indent=2)
            output.write('\n')
        os.replace(temporary_path, self.summary_path)

    def close(self):
        if self.closed:
            return
        self.write_summary()
        self.closed = True
        self.jsonl_file.close()
        self.csv_file.close()
        summary = self.accumulator.summary()
        if self.node.context.ok():
            self.node.get_logger().info(
                f"精度记录完成: usable={summary['frames_usable']}, "
                f"targets={len(summary['targets'])}, summary={self.summary_path}")


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    recorder = FastenerPrecisionRecorderNode()
    try:
        rclpy.spin(recorder.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        recorder.close()
        recorder.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
