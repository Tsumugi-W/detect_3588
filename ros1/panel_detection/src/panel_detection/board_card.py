"""Board-card identity and pose payload; YOLO remains the class authority."""
import math


def board_card_target(detection, assignment, xyz, quaternion, orientation_valid):
    position_valid = (xyz is not None and len(xyz) == 3 and
                      all(math.isfinite(float(v)) for v in xyz) and xyz[2] > 0)
    orientation_valid = bool(orientation_valid and quaternion is not None and
                             len(quaternion) == 4 and
                             all(math.isfinite(float(v)) for v in quaternion))
    return {
        'id': assignment.target_id if assignment is not None else None,
        'tag_id': assignment.tag_id if assignment is not None else None,
        'class': 'board_card',
        'classification_source': 'yolo',
        'identity_source': assignment.source if assignment is not None else None,
        'bbox': [round(float(v), 2) for v in detection.bbox],
        'confidence': round(float(detection.confidence), 3),
        'position_valid': bool(position_valid),
        'orientation_valid': orientation_valid,
        'position': ({axis: round(float(v), 4) for axis, v in zip('xyz', xyz)}
                     if position_valid else None),
        'orientation': ({axis: float(v) for axis, v in zip('xyzw', quaternion)}
                        if orientation_valid else None),
        'orientation_source': 'panel_normal' if orientation_valid else None,
    }
