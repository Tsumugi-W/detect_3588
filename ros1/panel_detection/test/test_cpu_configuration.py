"""Check launch parameter propagation and lightweight ROS callback boundaries."""
import ast
from pathlib import Path
import re
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


SOURCE_ROOT = Path(__file__).resolve().parents[2]
LAUNCHES = [
    "visual_servo/launch/persistent_camera_mission.launch",
    "visual_servo/launch/dual_arm_panel_pipeline.launch",
    "visual_servo/launch/aruco_pose.launch",
    "panel_detection/launch/d435_panel_detection.launch",
    "panel_detection/launch/panel_detection.launch",
]


def resolve(value, arguments):
    return re.sub(r"\$\(arg ([^)]+)\)", lambda match: arguments[match.group(1)], value)


def launch_arguments(root, overrides=None):
    args = {}
    for element in root.findall("arg"):
        name = element.attrib["name"]
        if overrides and name in overrides:
            args[name] = overrides[name]
        elif "default" in element.attrib:
            args[name] = resolve(element.attrib["default"], args)
    return args


class CpuConfigurationTest(unittest.TestCase):
    def test_all_internal_include_arguments_exist_and_are_unique(self):
        for relative in LAUNCHES:
            root = ET.parse(str(SOURCE_ROOT / relative)).getroot()
            names = [arg.attrib["name"] for arg in root.findall("arg")]
            self.assertEqual(len(names), len(set(names)), relative)
            for include in root.iter("include"):
                match = re.fullmatch(r"\$\(find ([^)]+)\)/(.*)", include.attrib["file"])
                if not match:
                    continue
                target = SOURCE_ROOT / match.group(1) / match.group(2)
                if not target.exists():
                    continue  # Driver packages are external to this src tree.
                declared = {arg.attrib["name"] for arg in ET.parse(str(target)).getroot().findall("arg")}
                passed = [arg.attrib["name"] for arg in include.findall("arg")]
                self.assertTrue(set(passed).issubset(declared), (relative, target, set(passed) - declared))
                self.assertEqual(len(passed), len(set(passed)))

    def test_low_cpu_values_reach_both_detectors_and_camera_drivers(self):
        overrides = {"camera_fps": "15", "usb_camera_fps": "10", "detection_max_fps": "5",
                     "detection_idle_fps": "1", "onnx_threads": "1", "opencv_threads": "1",
                     "aruco_max_fps": "10", "publish_perception_debug": "false"}
        seen = {}

        def walk(relative, supplied):
            root = ET.parse(str(SOURCE_ROOT / relative)).getroot()
            args = launch_arguments(root, supplied)
            for node in root.findall("node"):
                params = {param.attrib["name"]: resolve(param.attrib["value"], args)
                          for param in node.findall("param") if "value" in param.attrib}
                seen[resolve(node.attrib["name"], args)] = params
            for include in root.findall("include"):
                passed = {arg.attrib["name"]: resolve(arg.attrib["value"], args) for arg in include.findall("arg")}
                file = include.attrib["file"]
                match = re.fullmatch(r"\$\(find ([^)]+)\)/(.*)", file)
                target = match.group(1) + "/" + match.group(2)
                if target in LAUNCHES:
                    walk(target, passed)
                elif match.group(1) == "realsense2_camera":
                    seen[passed["camera"]] = passed

        walk(LAUNCHES[0], overrides)
        for name, side in (("left_panel_detection", "left"), ("right_panel_detection", "right")):
            params = seen[name]
            self.assertEqual(params["max_processing_fps"], "5")
            self.assertEqual(params["idle_processing_fps"], "1")
            self.assertEqual(params["onnx_threads"], "1")
            self.assertEqual(params["opencv_threads"], "1")
            self.assertEqual(params["activity_topic"], "/visual_servo/" + side + "_sequence_status")
            self.assertEqual(params["targets_topic"], "/panel/" + side + "/targets")
            self.assertEqual(params["publish_debug_image"], "false")
        for name in ("camera_d435_0", "camera_d435_1"):
            self.assertEqual(seen[name]["color_fps"], "15")
            self.assertEqual(seen[name]["depth_fps"], "15")
            self.assertEqual(seen[name]["align_depth"], "true")
        self.assertEqual(seen["camera_d435_2"]["color_fps"], "15")
        self.assertEqual(seen["persistent_usb_cameras"]["frame_rate"], "10")
        self.assertEqual(seen["aruco_pose"]["max_processing_fps"], "10")

    def test_ros_callbacks_only_enqueue_and_preserve_message_objects(self):
        cases = [
            ("panel_detection/src/panel_detection/panel_detect_node.py", "PanelDetectionNode", 2),
            ("visual_servo/src/visual_servo/aruco_pose_node.py", "ArucoPoseNode", 1),
        ]
        for relative, class_name, arity in cases:
            tree = ast.parse((SOURCE_ROOT / relative).read_text(encoding="utf-8"))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_image_callback")
            namespace = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), relative, "exec"), namespace)
            processor = mock.Mock()
            instance = types.SimpleNamespace(processing=processor)
            messages = tuple(object() for _ in range(arity))
            namespace["_image_callback"](instance, *messages)
            processor.submit.assert_called_once_with(*messages)

    def test_step2_keeps_confirmation_and_defaults_to_low_cpu(self):
        script = (SOURCE_ROOT / "runtime_workflow/02_start_d435_task_runtime.sh").read_text(encoding="utf-8")
        self.assertIn('"${1:-}" != "--confirm-motion"', script)
        self.assertIn('RUNTIME_PROFILE="${RUNTIME_PROFILE:-low_cpu}"', script)
        self.assertIn('ONNX_THREADS="${ONNX_THREADS:-1}"', script)
        self.assertIn('pipeline_use_rviz:=false', script)
        self.assertIn('mission_motion_confirmed:=true', script)
        self.assertIn('start_head_camera:=true', script)
        self.assertIn('start_usb_cameras:=true', script)


if __name__ == "__main__":
    unittest.main()
