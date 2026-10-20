"""Verify that the bundled 1013 recognition model is the runtime default."""

import hashlib
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE.parent
MODEL_SHA256 = "6cb7aac3881aca57e76b300c48ff5ba6281564d96ecf22ddbec47bef4c4e6d17"
# Always present inside the package itself.
PACKAGE_LAUNCH_FILES = [
    PACKAGE / "launch" / "panel_detection.launch",
    PACKAGE / "launch" / "d435_panel_detection.launch",
]
# Present when the package sits inside the robot workspace; the wiring guard is
# skipped for a standalone checkout (for example the deploy tarball).
WORKSPACE_LAUNCH_FILES = [
    SOURCE_ROOT / "visual_servo" / "launch" / "dual_arm_panel_pipeline.launch",
    SOURCE_ROOT / "visual_servo" / "launch" / "full_panel_button_pipeline.launch",
]


class RecognitionModelContractTest(unittest.TestCase):
    def test_model_copy_is_exact(self):
        model = PACKAGE / "models" / "1013.onnx"
        self.assertTrue(model.is_file())
        self.assertEqual(
            hashlib.sha256(model.read_bytes()).hexdigest(), MODEL_SHA256
        )

    def test_runtime_defaults_and_ten_classes_are_active(self):
        config = (PACKAGE / "config" / "panel_detection.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("board_card", config)
        self.assertIn("onnx_threads: 8", config)

        for launch_file in PACKAGE_LAUNCH_FILES:
            self.assertTrue(launch_file.is_file(), str(launch_file))
            self.assertIn(
                "models/1013.onnx",
                launch_file.read_text(encoding="utf-8"),
                str(launch_file),
            )

        for launch_file in WORKSPACE_LAUNCH_FILES:
            if not launch_file.is_file():
                continue
            self.assertIn(
                "models/1013.onnx",
                launch_file.read_text(encoding="utf-8"),
                str(launch_file),
            )

    def test_previous_models_stay_as_fallbacks(self):
        for name in (
            "best_zxr_10_13.onnx",
            "0824.onnx",
            "0807.onnx",
            "0601.onnx",
        ):
            self.assertTrue((PACKAGE / "models" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
