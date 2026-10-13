import numpy as np
from panel_detection.camera.base import CameraIntrinsics
from panel_detection.pipe_axis import PipeAxisResult
from panel_detection.pipe_axis_3d import estimate_pipe_axis_3d


def scene(slope=.4):
    intrin=CameraIntrinsics(fx=500.,fy=500.,cx=160.,cy=120.,width=320,height=240)
    xs=(np.arange(320)-160)/500
    depth=np.tile(1/(1-slope*xs),(240,1))/.001
    result=PipeAxisResult((160,120),(20,40,300,200),0.,(1.,0.),[])
    return result,depth,intrin


def test_tilted_axis_has_true_depth_component_and_metric_span():
    result,depth,intrin=scene()
    got=estimate_pipe_axis_3d(result,depth,intrin)
    expected=np.array([1.,0.,.4]);expected/=np.linalg.norm(expected)
    assert got['valid']
    assert np.dot(got['direction_xyz'],expected)>.999
    assert .3<got['span_m']<.5
    assert got['rms_error_m']<1e-6


def test_outlier_depths_and_holes_are_rejected_without_flattening_axis():
    result,depth,intrin=scene()
    depth[:,60:78]=0
    depth[:,200:218]=2700
    got=estimate_pipe_axis_3d(result,depth,intrin)
    assert got['valid']
    assert abs(got['direction_xyz'][2]-.4/np.sqrt(1.16))<.01


def test_invalid_depth_does_not_publish_a_fake_direction():
    result,depth,intrin=scene()
    for fill in [0,np.nan,np.inf]:
        got=estimate_pipe_axis_3d(result,np.full_like(depth,fill),intrin)
        assert not got['valid']
        assert got['direction_xyz'] is None


def test_depth_must_exist_near_leak_and_match_intrinsics():
    result,depth,intrin=scene()
    depth[:,145:176]=0
    assert estimate_pipe_axis_3d(result,depth,intrin)['reason']=='missing_depth_near_leak'
    intrin.width=640
    assert estimate_pipe_axis_3d(result,depth,intrin)['reason']=='intrinsics_size_mismatch'


def test_vertical_axis_and_minimum_span():
    result,depth,intrin=scene(0)
    result.vector_xy=(0.,1.)
    got=estimate_pipe_axis_3d(result,depth,intrin)
    assert got['valid']
    assert np.allclose(got['direction_xyz'],[0,1,0])
    assert not estimate_pipe_axis_3d(result,depth,intrin,cfg={'half_length_px':4})['valid']
