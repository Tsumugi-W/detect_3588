"""Validate ONNX session configuration without loading a model or ROS."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy  # noqa: F401 - do not import numpy for the first time inside patch.dict


class DetectorThreadsTest(unittest.TestCase):
    def test_cpu_session_is_bounded_and_does_not_spin(self):
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                ort = types.ModuleType("onnxruntime")
                options = mock.Mock()
                ort.SessionOptions = mock.Mock(return_value=options)
                ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
                ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
                session = mock.Mock()
                session.get_inputs.return_value = [types.SimpleNamespace(name="images")]
                ort.InferenceSession = mock.Mock(return_value=session)
                path = Path(__file__).resolve().parents[1] / "src/panel_detection/detector_onnx.py"
                name = "panel_detection._thread_test_detector"
                spec = importlib.util.spec_from_file_location(name, str(path))
                module = importlib.util.module_from_spec(spec)
                with mock.patch.dict(sys.modules, {"onnxruntime": ort, "cv2": types.ModuleType("cv2"), name: module}):
                    spec.loader.exec_module(module)
                    module.YoloV5Onnx("model.onnx", ["button"], threads=threads)
                self.assertEqual(options.intra_op_num_threads, threads)
                self.assertEqual(options.inter_op_num_threads, 1)
                self.assertEqual(options.execution_mode, "sequential")
                options.add_session_config_entry.assert_any_call("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry.assert_any_call("session.inter_op.allow_spinning", "0")
                self.assertEqual(ort.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])
                self.assertEqual(session.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
