from collections import deque

# This synchronization helper is pure Python; load the real function without
# importing ROS transport or camera drivers in a local unit test.
import ast
from pathlib import Path
import panel_detection
_source = Path(panel_detection.__file__).parent / 'panel_detect_node.py'
_tree = ast.parse(_source.read_text('utf-8'))
_function = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pop_synced_frame_pair')
exec(compile(ast.Module(body=[_function], type_ignores=[]), str(_source), 'exec'))


def test_topic_sync_drops_unpaired_frames_and_returns_matching_cycle():
    colors = deque([
        (1.000, 'c0', None),
        (1.033, 'c1', None),
        (1.066, 'c2', None),
    ])
    depths = deque([
        (1.032, 'd1', None),
        (1.067, 'd2', None),
    ])

    color, depth = _pop_synced_frame_pair(colors, depths, max_dt=0.02)

    assert color[1] == 'c1'
    assert depth[1] == 'd1'


def test_topic_sync_waits_when_no_compatible_pair_is_available():
    colors = deque([(2.000, 'c0', None)])
    depths = deque([(2.100, 'd0', None)])

    assert _pop_synced_frame_pair(colors, depths, max_dt=0.02) is None
    assert not colors
    assert len(depths) == 1
