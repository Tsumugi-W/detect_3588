"""Visual observations of the validation-site white frame, handle and blue socket.

These are image observations, not a mechanical seating/locking measurement.
Occluded or ambiguous features remain None; an absent blue socket alone never
implies an inserted card. The fixed-frame pose is deliberately unchanged.
"""
import cv2
import numpy as np

from .depth_utils import deproject_pixel_to_point


def _ordered_box(rect):
    points = cv2.boxPoints(rect).astype(np.float32)
    # Choose the short edge as frame horizontal, then orient image-upwards.
    centre = points.mean(axis=0)
    vectors = [points[1] - points[0], points[2] - points[1]]
    lengths = [float(np.linalg.norm(v)) for v in vectors]
    right = (min(vectors, key=np.linalg.norm) if max(lengths)/max(min(lengths), 1) > 1.2
             else max(vectors, key=lambda v: abs(v[0])/max(float(np.linalg.norm(v)), 1)))
    right /= max(float(np.linalg.norm(right)), 1e-6)
    if right[0] < 0:
        right = -right
    down = np.array([-right[1], right[0]], dtype=np.float32)
    x = (points - centre) @ right
    y = (points - centre) @ down
    return np.array([centre + right * a + down * b for a, b in
                     [(min(x), min(y)), (max(x), min(y)),
                      (max(x), max(y)), (min(x), max(y))]], np.float32)


def _feature(points, inverse, confidence):
    points = cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(1, -1, 2), inverse)[0]
    centre = points.mean(axis=0)
    return {'pixel': [round(float(v), 2) for v in centre],
            'polygon': [[round(float(v), 2) for v in p] for p in points],
            'confidence': round(float(confidence), 3),
            'position_valid': False, 'position': None}


def detect_board_features(image, bbox, cfg=None):
    cfg = cfg or {}
    result = {'insertion_state': 'unknown', 'state_source': 'visual',
              'handle': None, 'slot': None}
    if image is None or not cfg.get('enable', True):
        return result
    ih, iw = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
    if x2-x1 < 45 or y2-y1 < 70:
        return result
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165)).astype(np.uint8)*255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        if (short < (x2-x1)*0.5 or long < (y2-y1)*0.65 or
                not 1.0 <= long/max(short, 1) < 4.5):
            continue
        if cv2.contourArea(contour) < roi.shape[0]*roi.shape[1]*0.25:
            continue
        candidates.append((cv2.contourArea(contour), rect))
    if not candidates:
        return result
    box = _ordered_box(max(candidates, key=lambda item: item[0])[1])
    box += [x1, y1]
    width, height = 240, 400
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    transform = cv2.getPerspectiveTransform(box, dst)
    inverse = cv2.getPerspectiveTransform(dst, box)
    rectified = cv2.warpPerspective(image, transform, (width, height))
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    region = np.zeros((height, width), np.uint8)
    region[40:360, 48:192] = 255
    # The visible blue connector is positive evidence of an exposed socket.
    blue = ((hue >= cfg.get('slot_hue_min', 85)) &
            (hue <= cfg.get('slot_hue_max', 125)) &
            (sat >= 75) & (val >= 35) & (region > 0)).astype(np.uint8)*255
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sockets = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        area = cv2.contourArea(contour)
        if area >= 150 and long >= 95 and 3.0 <= long/max(short, 1) <= 30:
            sockets.append((area, rect))
    if sockets:
        _, rect = max(sockets, key=lambda item: item[0])
        result['slot'] = _feature(cv2.boxPoints(rect), inverse, .85)

    # A white handle can touch the white frame in the threshold mask. Locate
    # its two long edges instead; require a bright strip between observed
    # edges so the exposed dark/blue socket cannot become a handle.
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 70)
    edges[:65] = 0
    edges[395:] = 0
    edges[:, :24] = 0
    edges[:, 188:] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                            minLineLength=80, maxLineGap=18)
    segments = []
    for line in ([] if lines is None else lines.reshape(-1, 4)):
        xa, ya, xb, yb = map(float, line)
        if ya > yb:
            xa, ya, xb, yb = xb, yb, xa, ya
        if yb-ya < 80 or abs(xb-xa)/(yb-ya) > .3:
            continue
        slope = (xb-xa)/(yb-ya)
        segments.append((slope, xa-slope*ya, ya, yb))
    handles = []
    for i, left in enumerate(segments):
        for right in segments[i+1:]:
            lo, hi = max(left[2], right[2]), min(left[3], right[3])
            if hi-lo < 80 or abs(left[0]-right[0]) > .15:
                continue
            mid = (lo+hi)*.5
            xl, xr = left[0]*mid+left[1], right[0]*mid+right[1]
            if xl > xr:
                left_edge, right_edge = right, left
                xl, xr = xr, xl
            else:
                left_edge, right_edge = left, right
            if not 25 <= xr-xl <= 85 or not 50 <= (xl+xr)*.5 <= 140:
                continue
            polygon = np.array([
                [left_edge[0]*lo+left_edge[1], lo],
                [right_edge[0]*lo+right_edge[1], lo],
                [right_edge[0]*hi+right_edge[1], hi],
                [left_edge[0]*hi+left_edge[1], hi]], np.float32)
            mask = np.zeros_like(gray)
            centre = polygon.mean(axis=0)
            cv2.fillConvexPoly(mask, np.rint(centre+(polygon-centre)*.8).astype(np.int32), 255)
            pixels = gray[mask > 0]
            if len(pixels) < 300 or np.median(pixels) < 170 or np.mean(pixels > 150) < .8:
                continue
            sample_y = np.linspace(lo+10, hi-10, 40).astype(int)
            sample_l = np.rint(left_edge[0]*sample_y+left_edge[1]).astype(int)
            sample_r = np.rint(right_edge[0]*sample_y+right_edge[1]).astype(int)
            # Reject the recessed bevel: it is darker than both bounding
            # white surfaces. A protruding grip face has positive contrast
            # against at least one adjacent shadow/background.
            contrast_l = np.median(gray[sample_y, np.clip(sample_l+5,0,239)].astype(float) -
                                   gray[sample_y, np.clip(sample_l-5,0,239)].astype(float))
            contrast_r = np.median(gray[sample_y, np.clip(sample_r-5,0,239)].astype(float) -
                                   gray[sample_y, np.clip(sample_r+5,0,239)].astype(float))
            if max(contrast_l, contrast_r) < 12:
                continue
            # A grip face is approximately uniform. An edge pair spanning
            # the bright side face plus the recessed bevel is not one part.
            spread = float(np.percentile(pixels, 80)-np.percentile(pixels, 20))
            score = (hi-lo) + min(xr-xl, 65)*2 - spread*4 - abs((xl+xr)*.5-100)*.2
            handles.append((score, polygon))
    if handles and result['slot'] is None:
        _, polygon = max(handles, key=lambda item: item[0])
        result['handle'] = _feature(polygon, inverse, .75)
    if result['slot'] and result['handle']:
        result['insertion_state'] = 'transition'
    elif result['slot']:
        result['insertion_state'] = 'empty'
    elif result['handle']:
        result['insertion_state'] = 'inserted'
    return result


def add_board_feature_depth(features, depth, intrin, depth_scale):
    """Publish observed feature centres only when local depth is available.

    Handle depth comes from its central grip region. Slot depth comes from the
    connector's visible rim, avoiding the dark central opening. No hidden part
    is extrapolated from the fixed frame or from a prior image.
    """
    if depth is None or intrin is None:
        return features
    h, w = depth.shape[:2]
    for name in ('handle', 'slot'):
        item = features.get(name)
        if item is None:
            continue
        points = np.asarray(item['polygon'], np.float32)
        centre = points.mean(axis=0)
        mask = np.zeros((h, w), np.uint8)
        shrink = .45 if name == 'handle' else .9
        cv2.fillConvexPoly(mask, np.rint(centre+(points-centre)*shrink).astype(np.int32), 255)
        if name == 'slot':
            cv2.fillConvexPoly(mask, np.rint(centre+(points-centre)*.45).astype(np.int32), 0)
        samples = np.asarray(depth[mask > 0], np.float64)*float(depth_scale)
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if len(samples) < 8:
            continue
        z = float(np.median(samples))
        if np.percentile(samples, 80)-np.percentile(samples, 20) > max(.025, z*.03):
            continue
        xyz = deproject_pixel_to_point(intrin, item['pixel'], z)
        if not np.isfinite(xyz).all() or xyz[2] <= 0:
            continue
        item['position_valid'] = True
        item['position'] = {axis: round(float(v), 4) for axis, v in zip('xyz', xyz)}
        item['depth_source'] = 'handle_surface' if name == 'handle' else 'socket_rim'
    return features


def draw_board_features(canvas, features, bbox):
    for name, color in [('handle', (0, 255, 255)), ('slot', (255, 180, 0))]:
        item = features.get(name)
        if item is None:
            continue
        points = np.rint(item['polygon']).astype(np.int32)
        centre = tuple(int(round(v)) for v in item['pixel'])
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, centre, color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(canvas, name, (centre[0]+8, centre[1]), 0, .5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'card: '+features['insertion_state'],
                (int(bbox[0]), max(18, int(bbox[1])-22)), 0, .55,
                (0, 255, 255), 2, cv2.LINE_AA)
