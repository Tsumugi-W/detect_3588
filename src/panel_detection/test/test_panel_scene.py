from pathlib import Path
import numpy as np
import pytest
import yaml
from panel_detection.default_config import DEFAULT_CONFIG
from panel_detection.panel_scene import merge_config, resolve_tag_classes
from panel_detection.panel_apriltag import PanelAprilTagTracker, PanelTagMarker, apply_panel_tag_rules
from panel_detection.target_registry import FrameDetection


def test_default_validation_and_explicit_scene_files():
    config_dir = Path(__file__).parents[1] / 'config'
    cfg = merge_config(DEFAULT_CONFIG, yaml.safe_load((config_dir/'validation_site.yaml').read_text()))
    assert resolve_tag_classes(DEFAULT_CONFIG['panel_apriltag']) == resolve_tag_classes(cfg['panel_apriltag'])
    assert resolve_tag_classes(cfg['panel_apriltag']) == {0:'button',1:'button',2:'knob',3:'knob',4:'button',5:'button',6:'button',7:'board_card'}
    old = merge_config(DEFAULT_CONFIG, yaml.safe_load((config_dir/'test_site.yaml').read_text()))
    assert old['onnx_model'] == DEFAULT_CONFIG['onnx_model']
    assert resolve_tag_classes(old['panel_apriltag'])[2] == 'button'
    assert resolve_tag_classes(old['panel_apriltag'])[4] == 'knob'
    assert resolve_tag_classes(old['panel_apriltag'])[7] == 'light'
    assert DEFAULT_CONFIG['panel_apriltag']['scene'] == 'validation_site'


@pytest.mark.parametrize('scene, expected', [('validation_site','knob'), ('test_site','button')])
def test_tag02_final_class_and_stable_target_id(scene, expected):
    marker=PanelTagMarker(2,np.array([[80,20],[120,20],[120,60],[80,60]],dtype=np.float32),(100.,40.),1600.)
    det=FrameDetection('knob',100.,130.,(75.,100.,125.,160.),.9)
    assignments,_=apply_panel_tag_rules([det],[marker],PanelAprilTagTracker({'scene':scene}))
    assert det.class_name == expected
    assert assignments[0].target_id == 3


@pytest.mark.parametrize('cfg', [ {'scene':'typo'}, {'tag_classes':{}}, {'tag_classes':{40:'button'}}, {'tag_classes':{2:'bad'}}, {'tag_classes':{2:'knob','02':'button'}} ])
def test_invalid_scene_mapping_fails(cfg):
    with pytest.raises(ValueError):
        PanelAprilTagTracker(cfg)


def test_explicit_mapping_replaces_instead_of_leaking_old_ids():
    merged=merge_config({'panel_apriltag':{'tag_classes':{0:'button',2:'knob'}}},{'panel_apriltag':{'tag_classes':{3:'knob'}}})
    assert resolve_tag_classes(merged['panel_apriltag']) == {3:'knob'}
    assert resolve_tag_classes().get(8) is None
