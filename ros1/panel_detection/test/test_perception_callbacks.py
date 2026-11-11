"""Execute actual node callback AST bodies with middleware IO fakes.

Works without ROS installed (Windows or a local container). The callbacks and
payload methods are compiled unchanged from the candidate node source files.
Only ROS transport/camera/detector dependencies are replaced.
"""
import ast, copy, json, sys, threading, time, types, unittest
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
if ROOT.name=='panel_detection':
    ROOT=ROOT.parents[1] if ROOT.parent.name=='ros1' else ROOT.parents[1]
MARKER={'tag_id':7,'corners':np.array([[20,20],[40,20],[40,40],[20,40]],float)}
from panel_detection.perception_contract import panel_matches,panel_payload
from panel_detection.panel_apriltag import PanelAprilTagTracker,PanelTagMarker
from panel_detection.target_registry import FrameDetection

class Message:
    def __init__(self,data=None):self.data=data
class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,msg):self.messages.append(msg)
class FakeLocalizer:
    cfg={'tag_id':7}
    def __init__(self):self.timestamps=[]
    def observe(self,color,depth,info,marker=None,tag_id=None,timestamp_s=None):
        self.timestamps.append(timestamp_s)
        return {'tag_id':7,'state':'inserted' if marker is not None else 'unknown',
                'card_present':marker is not None,'stable':marker is not None,
                'state_confidence':.9,'clipped':False,'position':(.01,.02,.5) if marker is not None else None,
                'center_px':(30.,35.),'quad_px':[(20,20),(40,20),(40,50),(20,50)],
                'pose_locked':True,'pose_samples':5,'position_stable':True}

def compile_node(path, methods, functions, namespace):
    tree=ast.parse(path.read_text(encoding='utf8'))
    clazz=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name in methods for m in n.body))
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in functions:
            exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),namespace)
    body=[copy.deepcopy(m) for m in clazz.body if isinstance(m,ast.FunctionDef) and m.name in methods]
    node=ast.ClassDef(name='TestNode',bases=[],keywords=[],body=body,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['TestNode']

class CallbackTests(unittest.TestCase):
    def common(self):
        self.markers=[types.SimpleNamespace(tag_id=7,corners=MARKER['corners'].copy())]
        return {'np':np,'cv2':cv2,'json':json,'time':time,'String':Message,
                'detect_panel_tags':lambda *args:self.markers,
                'panel_matches':panel_matches,'panel_payload':panel_payload,
                'draw_panel_tag_assignments':lambda *args:None,
                'filter_depth':lambda d,**kwargs:d,'CvBridgeError':RuntimeError,
                'rospy':types.SimpleNamespace(logwarn_throttle=lambda *args:None),
                'FrameDetection':FrameDetection}
    def make_ros1(self):
        ns=self.common();C=compile_node(ROOT/'ros1/panel_detection/src/panel_detection/panel_detect_node.py',
            {'_process_images','_board_card_target','_board_card_payload'},set(),ns)
        n=C();n.lock=threading.Lock();n.latest_camera_info={};n.camera_frame='color'
        n.bridge=types.SimpleNamespace(imgmsg_to_cv2=lambda m,**kwargs:m.array)
        n.detector=types.SimpleNamespace(detect=lambda c:[])
        n.class_pose_pubs={'board_card':Publisher()};n.board_card_localizer=FakeLocalizer()
        n.panel_tag_enabled=True;n.panel_tag_cfg={};n.panel_tag_tracker=types.SimpleNamespace(update=lambda *args:{})
        n._select_panel_row=lambda d:([],None,None)
        n.registry=types.SimpleNamespace(identify=lambda *args,**kwargs:[])
        n.position_stabilizer=types.SimpleNamespace(update=lambda *args:{})
        n.knob_angle_stabilizer=types.SimpleNamespace(prune=lambda *args:None)
        n._publish_debug=lambda *args:None;n._publish_status=lambda *args:None
        n._pose_message=lambda stamp,point:types.SimpleNamespace(stamp=stamp,point=point)
        n.targets_pub=Publisher();n.distance_pub=Publisher();n.knob_angle_pub=Publisher()
        def call(t):
            stamp=types.SimpleNamespace(to_sec=lambda:t)
            c=types.SimpleNamespace(array=np.ones((60,60,3),np.uint8),header=types.SimpleNamespace(stamp=stamp,frame_id='color'))
            d=types.SimpleNamespace(array=np.ones((60,60),np.uint16)*500)
            n._process_images(c,d)
        return n,call,n.targets_pub,n.board_card_localizer
    def make_ros2(self):
        ns=self.common();C=compile_node(ROOT/'src/panel_detection/panel_detection/panel_detect_node.py',
            {'_detection_callback','_publish_board_cards'}, {'_board_card_observation_summary','_stamp_to_sec'},ns)
        n=C();n.cfg={};n._camera_ready=True;n._use_topic=True;n._process_leak=False;n._color_pub=None
        n._detector=types.SimpleNamespace(detect=lambda c:(c.copy(),[],[],[]))
        n._panel_tags_enabled=True;n._panel_tag_cfg={};n._process_fasteners=False;n._process_valves=False
        n._concentric_marker_enabled=False;n._panel_normal_cache=None;n._frame_count=0;n._normal_interval=30
        n._active_classes=None;n._depth_scale=.001;n._board_card_localizer=FakeLocalizer()
        n._board_cards_pub=Publisher();n._targets_pub=Publisher();n._topic_color_frame_id='color'
        n._panel_tag_tracker=types.SimpleNamespace(update=lambda *args:{})
        n._publish_status=lambda *args:None;n._update_display_frame=lambda *args:None
        n._select_deprojection_intrin=lambda *args:None
        n.get_logger=lambda:types.SimpleNamespace(info=lambda *args:None)
        def call(t):
            stamp=types.SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9)))
            n._get_synced_topic_frames=lambda:(None,None,np.ones((60,60,3),np.uint8),np.ones((60,60),np.uint16)*500,stamp,None)
            n._detection_callback()
        return n,call,n._targets_pub,n._board_card_localizer
    def check_callback(self,factory):
        n,call,pub,loc=factory()
        call(10.)
        payload=json.loads(pub.messages[-1].data)
        assert payload['frame_id']=='color'
        assert payload['camera_id']=='color'
        board=next(t for t in payload['targets'] if t['class']=='board_card')
        assert board['id']==7 and board['measurement_ready']
        assert 'graspable' not in board
        assert 'graspable' not in board['board_card']
        self.markers=[];call(10.1)
        # ROS1 publishes an explicit unknown observation; ROS2 main currently
        # only publishes stable known T07 state. Neither may retain permission.
        if len(pub.messages)>1:
            payload=json.loads(pub.messages[-1].data)
            for t in payload['targets']:
                assert not t['measurement_ready']
    def test_ros1_callback_emits_measurement_contract(self):self.check_callback(self.make_ros1)
    def test_ros2_callback_emits_measurement_contract(self):self.check_callback(self.make_ros2)
    def test_ros1_keeps_tag_zero_and_untagged_target_without_depth(self):
        n,call,pub,_=self.make_ros1()
        self.markers=[PanelTagMarker(0,np.float32([[20,20],[40,20],[40,40],[20,40]]),(30.,30.),400.)]
        raw=[types.SimpleNamespace(class_name='button',center=(30.,95.),bbox=(10.,75.,50.,115.),confidence=.9),
             types.SimpleNamespace(class_name='knob',center=(150.,95.),bbox=(130.,75.,170.,115.),confidence=.9)]
        n.detector=types.SimpleNamespace(detect=lambda c:raw)
        n.panel_tag_tracker=PanelAprilTagTracker({'scene':'validation_site'})
        n._point_from_detection=lambda *args:None
        n._should_estimate_angle=lambda *args:False
        n.registry_target_ids=[5,6] # obsolete caller state must not hide Tag 0
        n.camera_id='sensor_a'
        call(10.)
        msg=json.loads(pub.messages[-1].data)
        control=next(t for t in msg['targets'] if t['id']==0)
        assert control['position'] is None and not control['measurement_ready']
        assert any(t['id'] is None and t['class']=='knob' for t in msg['targets'])
        assert msg['camera_id']=='sensor_a'


if __name__=='__main__':unittest.main()
