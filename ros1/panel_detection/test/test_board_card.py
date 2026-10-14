import math
from types import SimpleNamespace
from panel_detection.board_card import board_card_target


def detection():
    return SimpleNamespace(bbox=(10., 20., 100., 160.), confidence=.9)


def test_board_pose_includes_tag_identity_and_yolo_class():
    a = SimpleNamespace(target_id=8, tag_id=7, source='apriltag_36h11')
    p = board_card_target(detection(), a, [.1, .2, 1.], [0., 0., 0., 1.], True)
    assert p['id'] == 8 and p['tag_id'] == 7
    assert p['class'] == 'board_card' and p['classification_source'] == 'yolo'
    assert p['identity_source'] == 'apriltag_36h11'
    assert p['position'] == {'x': .1, 'y': .2, 'z': 1.}
    assert p['orientation'] == {'x': 0., 'y': 0., 'z': 0., 'w': 1.}
    assert p['position_valid'] and p['orientation_valid']


def test_unassigned_board_pose_keeps_class_without_inventing_tag_id():
    p = board_card_target(detection(), None, [.1, .2, 1.], None, False)
    assert p['id'] is None and p['tag_id'] is None
    assert p['class'] == 'board_card' and p['identity_source'] is None
    assert p['position_valid'] and not p['orientation_valid']
    assert p['orientation'] is None


def test_invalid_depth_never_becomes_a_zero_or_nonfinite_pose():
    for xyz in [None, [0., 0., 0.], [0., 0., -1.], [math.nan, 0., 1.]]:
        p = board_card_target(detection(), None, xyz, None, False)
        assert p['position'] is None and not p['position_valid']
        assert p['class'] == 'board_card' and p['bbox'] == [10., 20., 100., 160.]


def test_tag_dropout_keeps_tracked_identity_and_yolo_class():
    a = SimpleNamespace(target_id=8, tag_id=7, source='apriltag_track')
    p = board_card_target(detection(), a, [0., 0., 1.], None, False)
    assert p['id'] == 8 and p['identity_source'] == 'apriltag_track'
    assert p['classification_source'] == 'yolo'
