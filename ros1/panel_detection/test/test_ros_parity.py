"""Prevent algorithm drift between the ROS1 and ROS2 distributions."""
import ast
from pathlib import Path


def roots():
    root=next(p for p in Path(__file__).resolve().parents if (p/'ros1/panel_detection').is_dir())
    return root/'src/panel_detection/panel_detection',root/'ros1/panel_detection/src/panel_detection'


def test_all_non_ros_modules_identical():
    a,b=roots()
    adapters={'panel_detect_node.py','fastener_precision_recorder.py','rgb_depth_viewer.py'}
    for p in a.rglob('*.py'):
        if p.name not in adapters and not p.name.endswith('.launch.py'):
            assert p.read_bytes()==(b/p.relative_to(a)).read_bytes(),str(p)


def normalized_method(path, name):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PanelDetectionNode')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for n in ast.walk(method):
        if isinstance(n,ast.Attribute):
            n.attr={'secs':'sec','nsecs':'nanosec','K':'k','D':'d'}.get(n.attr,n.attr)
    return ast.dump(method)


def test_entire_detection_and_leak_algorithms_identical():
    a,b=roots()
    for name in ['_detection_callback','_process_leak_frame','_publish_camera_topics','_publish_status','_publish_board_cards']:
        assert normalized_method(a/'panel_detect_node.py',name)==normalized_method(b/'panel_detect_node.py',name),name


def test_recorder_numeric_implementation_identical():
    a,b=roots()
    def numeric(p):
        t=ast.parse((p/'fastener_precision_recorder.py').read_text())
        return [ast.dump(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in ('main','FastenerPrecisionRecorderNode')]
    assert numeric(a)==numeric(b)
