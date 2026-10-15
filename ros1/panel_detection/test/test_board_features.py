from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from panel_detection.board_features import detect_board_features, add_board_feature_depth


@pytest.mark.parametrize('second', [95, 100, 101, 102, 103, 104, 105, 110, 111, 114])
def test_real_inserted_and_empty_frames(second):
    image = cv2.imread(str(Path(__file__).parent/'fixtures'/'board'/f'board_{second}.png'))
    assert image is not None
    h, w = image.shape[:2]
    features = detect_board_features(image, (0, 0, w, h))
    inserted = second < 106
    assert features['insertion_state'] == ('inserted' if inserted else 'empty')
    present, absent = ('handle', 'slot') if inserted else ('slot', 'handle')
    assert features[absent] is None
    point = features[present]['pixel']
    if inserted:
        # Human-reviewed grip silhouettes in cropped original RGB, including
        # the side face, but excluding the recessed white bevel to the right.
        polygons = {
            95: [(125, 75), (190, 70), (220, 405), (150, 405)],
            100: [(105, 120), (200, 55), (172, 310), (140, 410), (70, 405)],
            101: [(112, 112), (201, 50), (179, 290), (140, 400), (75, 398)],
            102: [(92, 145), (190, 90), (205, 320), (181, 440), (110, 440)],
            103: [(80, 135), (180, 55), (195, 300), (148, 410), (90, 410)],
            104: [(90, 120), (170, 48), (200, 290), (190, 400), (125, 400)],
            105: [(85, 120), (170, 55), (195, 300), (180, 410), (130, 410)],
        }
        assert cv2.pointPolygonTest(np.array(polygons[second], np.float32), tuple(point), False) >= 0
    else:
        # The point must lie on the actual blue connector, not the white frame.
        x, y = map(lambda v: int(round(v)), point)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        patch = hsv[max(0,y-12):y+13, max(0,x-12):x+13]
        blue = (patch[:,:,0] >= 85) & (patch[:,:,0] <= 125) & (patch[:,:,1] >= 75)
        assert blue.mean() > .15
    assert not features[present]['position_valid']
    intrin = SimpleNamespace(fx=500., fy=500., cx=w/2, cy=h/2)
    add_board_feature_depth(features, np.full((h,w), 1000, np.uint16), intrin, .001)
    assert features[present]['position_valid']
    assert features[present]['position']['z'] == 1.


def test_unknown_is_not_assumed_inserted_and_bad_depth_stays_invalid():
    image = np.zeros((400, 240, 3), np.uint8)
    result = detect_board_features(image, (0, 0, 240, 400))
    assert result['insertion_state'] == 'unknown'
    assert result['handle'] is None and result['slot'] is None
    feature = {'handle': {'polygon': [[20,20],[40,20],[40,80],[20,80]],
                         'pixel': [30,50], 'position_valid': False, 'position': None}, 'slot': None}
    add_board_feature_depth(feature, np.zeros((100,100), np.uint16),
                            SimpleNamespace(fx=100., fy=100., cx=50., cy=50.), .001)
    assert not feature['handle']['position_valid'] and feature['handle']['position'] is None
