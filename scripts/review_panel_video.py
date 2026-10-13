#!/usr/bin/env python3
"""Offline YOLO + AprilTag review; no ROS, depth, or synthetic camera data."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'panel_detection'))
from panel_detection.default_config import DEFAULT_CONFIG
from panel_detection.panel_scene import merge_config
from panel_detection.detector_onnx import YoloV5ORT
from panel_detection.panel_apriltag import (
    PanelAprilTagTracker, apply_panel_tag_rules, detect_panel_tags,
    draw_panel_tag_assignments,
)
from panel_detection.target_registry import FrameDetection

PANEL_CLASSES = {'button', 'door_button', 'knob', 'light'}




class VisualReviewer:
    def __init__(self, detector, tag_config):
        self.detector = detector
        self.tag_config = tag_config
        self.tracker = PanelAprilTagTracker(tag_config)

    def process(self, frame):
        _, class_ids, boxes, confidences = self.detector.detect(frame)
        detections, raw_classes = [], []
        for class_id, box, confidence in zip(class_ids, boxes, confidences):
            name = self.detector.class_names[class_id]
            if name not in PANEL_CLASSES:
                continue
            detections.append(FrameDetection(
                name, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                tuple(box), float(confidence)))
            raw_classes.append(name)
        markers = detect_panel_tags(frame, self.tag_config)
        assignments, _ = apply_panel_tag_rules(detections, markers, self.tracker)
        canvas = frame.copy()
        records = []
        for i, (det, raw_class) in enumerate(zip(detections, raw_classes)):
            assignment = assignments.get(i)
            x1, y1, x2, y2 = [round(v) for v in det.bbox]
            if assignment is None:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 2)
                label = f'{det.class_name} {det.confidence:.2f} [unmatched]'
                cv2.putText(canvas, label, (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            detail = f'YOLO:{raw_class} {det.confidence:.2f}'
            if assignment and assignment.source == 'apriltag_track':
                detail += ' [tracked]'
            cv2.putText(canvas, detail, (max(0, x1), min(frame.shape[0] - 6, y2 + 17)),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            records.append({
                'bbox': list(det.bbox), 'raw_class': raw_class,
                'class': det.class_name, 'confidence': det.confidence,
                'tag_id': assignment.tag_id if assignment else None,
                'target_id': assignment.target_id if assignment else None,
                'source': assignment.source if assignment else 'unmatched',
            })
        draw_panel_tag_assignments(canvas, markers, assignments, detections)
        return canvas, {
            'tags': [{'tag_id': m.tag_id, 'corners': m.corners.tolist()} for m in markers],
            'detections': records,
        }


def review(source, destination, detector, cfg, args):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')
    encoder = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise ValueError(f'Invalid frame rate: {fps}')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y, w, h = args.crop or (0, 0, width, height)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(f'Crop {x,y,w,h} exceeds {width}x{height}')
        destination.mkdir(parents=True, exist_ok=False)
        reviewer = VisualReviewer(detector, cfg['panel_apriltag'])
        # Pad to even dimensions for broadly compatible H.264/yuv420p playback.
        encoder = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h+36}', '-r', str(fps),
            '-i', '-', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(destination / 'annotated.mp4'),
        ], stdin=subprocess.PIPE)
        summary = {
            'input': str(source), 'fps': fps, 'crop_xywh': [x, y, w, h],
            'coordinate_space': 'cropped image', 'frames': 0,
            'frames_with_tags': 0, 'frames_with_direct_match': 0,
            'direct_matches': 0, 'tracked_matches': 0, 'unmatched_detections': 0,
            'reclassified_detections': 0, 'tag_ids': [],
            'scope': 'YOLO + shared AprilTag rules only; no layout fallback, angles or depth',
            'config': cfg,
        }
        tag_ids = set()
        best_score = -1
        started = time.monotonic()
        with (destination / 'frames.jsonl').open('w') as log:
            while args.max_frames is None or summary['frames'] < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                index = summary['frames']
                canvas, record = reviewer.process(frame[y:y+h, x:x+w])
                direct = sum(d['source'] == 'apriltag_36h11' for d in record['detections'])
                tracked = sum(d['source'] == 'apriltag_track' for d in record['detections'])
                summary['frames_with_tags'] += bool(record['tags'])
                summary['frames_with_direct_match'] += direct > 0
                summary['direct_matches'] += direct
                summary['tracked_matches'] += tracked
                summary['unmatched_detections'] += len(record['detections']) - direct - tracked
                summary['reclassified_detections'] += sum(d['raw_class'] != d['class'] for d in record['detections'])
                tag_ids.update(m['tag_id'] for m in record['tags'])
                record.update(frame=index, time_seconds=index/fps)
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                canvas = cv2.copyMakeBorder(canvas, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cv2.putText(canvas, f'{index/fps:.2f}s | tags:{len(record["tags"])} direct:{direct} tracked:{tracked} | 2D REVIEW',
                            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes())
                score = direct * 100 + len(record['tags'])
                if score > best_score:
                    best_score = score
                    if not cv2.imwrite(str(destination / 'preview.jpg'), canvas):
                        raise RuntimeError('Could not save preview')
                    summary['preview_frame'] = index
                if index % max(1, round(fps * 15)) == 0:
                    cv2.imwrite(str(destination / f'frame_{index:06d}.jpg'), canvas)
                summary['frames'] += 1
                if summary['frames'] % 150 == 0:
                    print(f'{source.name}: {summary["frames"]} frames, {summary["frames"]/(time.monotonic()-started):.1f} processing fps', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
        if summary['frames'] == 0:
            raise RuntimeError('No frames decoded')
        summary['tag_ids'] = sorted(tag_ids)
        summary['elapsed_seconds'] = time.monotonic() - started
        (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Completed: {destination} ({summary["frames"]} frames)', flush=True)
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg is required')
    if not hasattr(cv2, 'aruco'):
        parser.error('OpenCV with aruco support is required (opencv-contrib-python)')
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        cfg = merge_config(cfg, yaml.safe_load(args.config.read_text()) or {})
    if not cfg.get('panel_apriltag', {}).get('enable', True):
        parser.error('This review requires panel_apriltag.enable=true')
    package = Path(__file__).resolve().parents[1] / 'src' / 'panel_detection' / 'panel_detection'
    model = args.model or Path(cfg['onnx_model'])
    if not model.is_absolute():
        model = package / model
    cfg['onnx_model'] = str(model.resolve())
    destinations = [args.output_dir / source.stem for source in args.videos]
    if len(set(destinations)) != len(destinations):
        parser.error('Input filenames must have unique stems')
    for source, destination in zip(args.videos, destinations):
        if not source.is_file():
            parser.error(f'Input does not exist: {source}')
        if destination.exists():
            parser.error(f'Refusing to overwrite: {destination}')
    with tempfile.TemporaryDirectory(prefix='panel-review-') as temp:
        config_path = Path(temp) / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        detector = YoloV5ORT(str(model), str(config_path), threads=cfg['onnx_threads'])
    for source, destination in zip(args.videos, destinations):
        review(source.resolve(), destination, detector, cfg, args)


if __name__ == '__main__':
    main()
