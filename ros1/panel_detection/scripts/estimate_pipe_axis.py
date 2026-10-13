#!/usr/bin/env python3
import argparse
import json
import os
import sys

import cv2

from panel_detection.pipe_axis import (
    draw_pipe_axis_result,
    estimate_pipe_axis_from_image,
    result_to_dict,
)


def _parse_point(value):
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('point must be formatted as x,y')
    try:
        return int(round(float(parts[0]))), int(round(float(parts[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('point must contain numeric x,y') from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description='Estimate 2D pipe axis direction around a leak point.'
    )
    parser.add_argument('--image', required=True, help='Input RGB image path.')
    parser.add_argument(
        '--point',
        required=True,
        type=_parse_point,
        help='Leak point in image pixels, formatted as x,y.',
    )
    parser.add_argument(
        '--output',
        default='pipe_axis_result.jpg',
        help='Output visualization image path.',
    )
    parser.add_argument(
        '--json-output',
        default='',
        help='Optional output JSON path. Defaults to output image stem + .json.',
    )
    parser.add_argument('--roi-half-width', type=int, default=170)
    parser.add_argument('--roi-half-height', type=int, default=80)
    parser.add_argument('--canny-low', type=int, default=40)
    parser.add_argument('--canny-high', type=int, default=130)
    parser.add_argument('--hough-threshold', type=int, default=35)
    parser.add_argument('--min-line-length', type=int, default=55)
    parser.add_argument('--max-line-gap', type=int, default=12)
    parser.add_argument('--max-line-distance', type=float, default=50.0)
    parser.add_argument(
        '--angle-prior-deg',
        type=float,
        default=None,
        help='Optional rough pipe angle prior in image degrees.',
    )
    parser.add_argument(
        '--angle-tolerance-deg',
        type=float,
        default=45.0,
        help='Axis tolerance when --angle-prior-deg is provided.',
    )
    parser.add_argument(
        '--consensus-tolerance-deg',
        type=float,
        default=18.0,
        help='Tolerance used to keep the dominant parallel line cluster.',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f'Failed to read image: {args.image}', file=sys.stderr)
        return 2

    result = estimate_pipe_axis_from_image(
        image,
        leak_point=args.point,
        roi_half_size=(args.roi_half_width, args.roi_half_height),
        canny_thresholds=(args.canny_low, args.canny_high),
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        max_line_distance=args.max_line_distance,
        angle_prior_deg=args.angle_prior_deg,
        angle_tolerance_deg=args.angle_tolerance_deg,
        consensus_tolerance_deg=args.consensus_tolerance_deg,
    )
    if result is None:
        print('No reliable pipe axis found near the leak point.', file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    vis = draw_pipe_axis_result(image, result)
    cv2.imwrite(args.output, vis)

    json_output = args.json_output
    if not json_output:
        stem, _ = os.path.splitext(args.output)
        json_output = stem + '.json'
    os.makedirs(os.path.dirname(os.path.abspath(json_output)), exist_ok=True)

    payload = result_to_dict(result)
    payload.update({
        'source_image': os.path.abspath(args.image),
        'output_image': os.path.abspath(args.output),
    })
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
