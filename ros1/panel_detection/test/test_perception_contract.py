"""Public identity invariants for any camera, including partial views."""
from types import SimpleNamespace
import math
import pytest
import numpy as np
from panel_detection.perception_contract import panel_matches, panel_payload, normalize_panel_target
from panel_detection.panel_apriltag import PanelAprilTagTracker, PanelTagMarker
from panel_detection.target_registry import FrameDetection


def target(tag=0, **kw):
    return dict({'id':999,'tag_id':tag,'class':'button','position':{'x':.1,'y':.2,'z':.5}},**kw)


@pytest.mark.parametrize('tag',range(8))
def test_same_physical_tag_has_same_id_on_every_camera(tag):
    results=[panel_payload(12.3,frame,[target(tag)],camera) for frame,camera in
             [('camera_a_optical','camera_a'),('camera_b_optical','camera_b')]]
    assert [r['targets'][0]['id'] for r in results]==[tag,tag]
    assert results[0]['frame_id']!=results[1]['frame_id']
    assert results[0]['camera_id']!=results[1]['camera_id']


def test_tag_zero_is_not_treated_as_unknown():
    t=normalize_panel_target(target(0))
    assert t['id']==0 and t['identity_valid']


def test_layout_guess_never_becomes_a_public_identity():
    t=normalize_panel_target(target(None))
    assert t['id'] is None and not t['identity_valid']
    assert t['identity_source']=='unassigned' and t['measurement_ready']


def test_keep_all_controls_in_a_partial_camera_view():
    ds=[SimpleNamespace(class_name='button'),SimpleNamespace(class_name='knob'),SimpleNamespace(class_name='light')]
    assignments={0:SimpleNamespace(tag_id=0,forced_class='button'),1:SimpleNamespace(tag_id=3,forced_class='knob')}
    matched=panel_matches(ds,assignments)
    assert len(matched)==3 and {id(d) for _,d in matched}=={id(d) for d in ds}
    assert {i for i,_ in matched if i>=0}=={0,3}


def test_identity_follows_tag_under_camera_translation_and_order_change():
    for dx,reverse in [(0,False),(180,True)]:
        detections=[];markers=[]
        for tag,x in [(0,60),(5,180)]:
            x+=dx
            detections.append(FrameDetection('button',x,110,(x-20,90,x+20,130),.9))
            corners=np.float32([[x-10,35],[x+10,35],[x+10,55],[x-10,55]])
            markers.append(PanelTagMarker(tag,corners,(x,45),400))
        if reverse:detections.reverse()
        assignments=PanelAprilTagTracker({'scene':'validation_site'}).update(detections,markers)
        assert {i for i,_ in panel_matches(detections,assignments)}=={0,5}


def test_public_contract_has_measurement_quality_without_execution_permission():
    t=normalize_panel_target(target(3,graspable=True,operation_ready=True))
    assert t['measurement_ready'] and 'graspable' not in t and 'operation_ready' not in t


@pytest.mark.parametrize('state',['empty','inserted'])
def test_known_board_state_is_a_measurement_even_when_empty(state):
    t=target(7,**{'class':'board_card','board_card':{'state':state,'stable':True,'clipped':False,'graspable':state=='inserted'}})
    o=normalize_panel_target(t)
    assert o['id']==7 and o['measurement_ready']
    assert o['board_card']['measurement_ready'] and 'graspable' not in o['board_card']
    assert 'measurement_ready' not in t # no mutation of internal observation


@pytest.mark.parametrize('position',[None,{'x':math.nan,'y':0,'z':1},{'x':0,'y':0,'z':-1}])
def test_invalid_measurement_retains_identity_and_revokes_quality(position):
    t=normalize_panel_target(target(5,position=position))
    assert t['id']==5 and t['position'] is None and not t['measurement_ready']


def test_unknown_board_is_not_a_ready_measurement():
    o=normalize_panel_target(target(7,**{'class':'board_card','board_card':{'state':'unknown','stable':False}}))
    assert not o['measurement_ready']


def test_explicit_board_freshness_rejection_survives_normalization():
    t=target(7,**{'class':'board_card','board_card':{'state':'inserted','stable':True,'clipped':False,'operation_ready':False,'operation_reason':'camera_moving'}})
    o=normalize_panel_target(t)
    assert not o['measurement_ready'] and o['board_card']['measurement_reason']=='camera_moving'


def test_multi_dictionary_input_keeps_same_zero_based_identity():
    import cv2
    from panel_detection.panel_apriltag import detect_panel_tags
    cfg={'scene':'validation_site','dictionary':['DICT_6X6_250','DICT_APRILTAG_36h11']}
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    image=np.full((180,180,3),255,np.uint8)
    marker=cv2.aruco.generateImageMarker(dictionary,0,100)
    image[40:140,40:140]=marker[:,:,None]
    tags=detect_panel_tags(image,cfg)
    assert [t.tag_id for t in tags]==[0]
