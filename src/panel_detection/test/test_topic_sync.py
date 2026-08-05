from collections import deque

from panel_detection.panel_detect_node import _pop_synced_frame_pair


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
