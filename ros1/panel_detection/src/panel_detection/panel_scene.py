"""Scene-specific Tag semantics shared by live and offline detection."""
from copy import deepcopy

DEFAULT_SCENE = 'validation_site'
SCENE_TAG_CLASSES = {
    'validation_site': {i: ('board_card' if i == 7 else
                            'knob' if i in (2, 3) else 'button') for i in range(8)},
    'test_site': {i: ('button' if i in (0, 1, 2, 5, 6) else
                      'knob' if i in (3, 4) else 'light') for i in range(40)},
}


def resolve_tag_classes(cfg=None):
    cfg = cfg or {}
    scene = cfg.get('scene', DEFAULT_SCENE)
    if scene not in SCENE_TAG_CLASSES:
        raise ValueError(f'Unknown panel scene: {scene}; use validation_site or test_site')
    raw = cfg.get('tag_classes', SCENE_TAG_CLASSES[scene])
    if not isinstance(raw, dict) or not raw:
        raise ValueError('panel_apriltag.tag_classes must be a non-empty ID-to-class mapping')
    result = {}
    for key, value in raw.items():
        if isinstance(key, bool) or not (isinstance(key, int) or isinstance(key, str) and key.isdecimal()):
            raise ValueError(f'Invalid Tag ID: {key!r}')
        tag_id = int(key)
        if not 0 <= tag_id <= 39 or tag_id in result:
            raise ValueError(f'Invalid or duplicate Tag ID: {key!r}')
        if value not in ('button', 'knob', 'light', 'board_card'):
            raise ValueError(f'Invalid class for T{tag_id:02d}: {value!r}')
        result[tag_id] = value
    return result


def merge_config(base, override):
    """Merge partial YAML; explicit tag_classes replaces the entire mapping."""
    if not isinstance(override, dict):
        raise ValueError('Configuration must be a YAML mapping')
    result = deepcopy(base)
    for key, value in override.items():
        if key != 'tag_classes' and isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
