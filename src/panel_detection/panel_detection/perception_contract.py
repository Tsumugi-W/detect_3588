"""Camera-independent panel IDs and measurement-only output, shared by ROS1/2."""
import math

PANEL_CLASSES = frozenset(('button', 'door_button', 'knob', 'light', 'board_card'))


def panel_matches(detections, assignments):
    """Keep every panel detection. Negative keys are private, frame-local keys.

    Layout guesses and robot task assignments never become public identities.
    A short-lived tracker assignment retains its explicit identity source.
    """
    result = []
    for index, detection in enumerate(detections):
        assignment = assignments.get(index)
        if assignment is not None:
            detection.class_name = assignment.forced_class
            result.append((int(assignment.tag_id), detection))
        elif detection.class_name in PANEL_CLASSES:
            result.append((-index - 1, detection))
    return sorted(result, key=lambda item: item[0])


def _valid_position(position):
    if not isinstance(position, dict):
        return False
    try:
        values = [float(position[k]) for k in ('x', 'y', 'z')]
        return all(math.isfinite(v) for v in values) and values[2] > 0
    except (KeyError, ValueError, TypeError):
        return False


def normalize_panel_target(target):
    """Report identity/measurement quality; execution decisions belong downstream."""
    result = dict(target)
    if result.get('class') not in PANEL_CLASSES:
        return result
    tag_id = result.get('tag_id')
    result['id'] = int(tag_id) if tag_id is not None else None
    result['identity_valid'] = tag_id is not None
    result['identity_source'] = (result.get('identity_source') or
                                 result.get('classification_source') or
                                 ('tag' if tag_id is not None else 'unassigned'))
    if tag_id is None:
        result['identity_source'] = 'unassigned'
    valid = _valid_position(result.get('position'))
    result['position_valid'] = valid
    if not valid:
        result['position'] = None
    detail = result.get('board_card')
    if isinstance(detail, dict):
        detail = dict(detail)
        known = detail.get('state', detail.get('insertion_state')) in ('inserted', 'empty')
        ready = valid and known and bool(detail.get('stable'))
        if detail.get('state') == 'inserted':
            ready = ready and not detail.get('clipped', True)
        if 'operation_ready' in detail:
            ready = ready and bool(detail.pop('operation_ready'))
        detail.pop('graspable', None)
        detail['measurement_ready'] = bool(ready)
        if 'operation_reason' in detail:
            detail['measurement_reason'] = detail.pop('operation_reason')
        result['board_card'] = detail
    else:
        ready = valid and result.get('class') != 'board_card'
    result.pop('graspable', None)
    result.pop('operation_ready', None)
    result['measurement_ready'] = bool(ready)
    return result


def panel_payload(stamp, frame_id, targets, camera_id=None):
    """Position stays in frame_id. camera_id is source metadata, never an arm ID."""
    return {'schema_version': 2, 'stamp': float(stamp), 'frame_id': frame_id,
            'camera_id': camera_id or frame_id,
            'targets': [normalize_panel_target(t) for t in targets]}
