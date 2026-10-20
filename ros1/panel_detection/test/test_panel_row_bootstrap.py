#!/usr/bin/env python3
"""Regression tests for the knob-free panel-row bootstrap.

The right wrist camera only ever sees the ID5/ID6 buttons, so the two-knob row
fit can never run there and the detector used to fall back to sorting by image
x.  These tests pin the button-based bootstrap that gives that view a real
panel axis.
"""

import ast
import types
from pathlib import Path
import unittest

import numpy as np

from panel_detection.target_registry import (
    FrameDetection,
    PersistentPanelAxis,
)


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "panel_detection"
    / "panel_detect_node.py"
)
WANTED = (
    "PanelDetectionNode",
    "_fit_axis_from_points",
    "_is_button_like_detection",
    "_select_panel_row",
    "_filter_row_detections",
    "_publish_panel_axis",
)


def _load_namespace():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    namespace = {"np": np}
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in WANTED
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<node>", "exec"), namespace)
        elif isinstance(node, ast.ClassDef) and node.name == "PanelDetectionNode":
            for method in node.body:
                if (
                    isinstance(method, ast.FunctionDef)
                    and method.name in WANTED
                ):
                    exec(
                        compile(
                            ast.Module(body=[method], type_ignores=[]),
                            "<node>",
                            "exec",
                        ),
                        namespace,
                    )
    return namespace


NAMESPACE = _load_namespace()


def _detection(class_name, center_x, center_y=40.0, size=30.0):
    return FrameDetection(
        class_name=class_name,
        center_x=center_x,
        center_y=center_y,
        bbox=(
            center_x - size / 2.0,
            center_y - size / 2.0,
            center_x + size / 2.0,
            center_y + size / 2.0,
        ),
        confidence=0.9,
    )


def _node(allowed_ids, allow_button_axis):
    node = types.SimpleNamespace(
        panel_line={
            "initial_dist_ratio": 1.1,
            "dist_ratio": 0.85,
            "min_dist": 45.0,
            "max_dist": 110.0,
            "proj_margin_ratio": 3.0,
            "min_proj_margin": 450.0,
        },
        panel_axis=PersistentPanelAxis(min_proj_margin=450.0),
        registry_target_ids=list(allowed_ids),
        allow_button_axis=allow_button_axis,
        panel_axis_pub=None,
    )
    for name in (
        "_select_panel_row",
        "_filter_row_detections",
        "_publish_panel_axis",
    ):
        setattr(
            node, name, types.MethodType(NAMESPACE[name], node)
        )
    return node


class KnoblessRowBootstrapTest(unittest.TestCase):
    def test_two_buttons_alone_produce_a_panel_axis(self):
        node = _node([5, 6], allow_button_axis=True)
        red = _detection("button", 60.0)
        green = _detection("button", 120.0)

        selected, origin, axis = node._select_panel_row([red, green])

        self.assertIsNotNone(origin)
        self.assertIsNotNone(axis)
        self.assertEqual([detection.center_x for detection in selected], [60.0, 120.0])
        self.assertGreater(abs(axis[0]), abs(axis[1]))
        self.assertTrue(node.panel_axis.has_axis)

    def test_axis_is_ordered_left_to_right_in_the_image(self):
        node = _node([5, 6], allow_button_axis=True)
        selected, _, axis = node._select_panel_row(
            [_detection("button", 120.0), _detection("button", 60.0)]
        )
        self.assertEqual(len(selected), 2)
        self.assertGreater(axis[0], 0.0)

    def test_button_like_light_is_promoted_on_the_row(self):
        node = _node([5, 6], allow_button_axis=True)
        light = _detection("light", 60.0)
        green = _detection("button", 120.0)

        selected, _, _ = node._select_panel_row([light, green])

        self.assertEqual([detection.center_x for detection in selected], [60.0, 120.0])
        self.assertEqual(light.class_name, "button")

    def test_off_row_detection_is_dropped(self):
        """The row filter drops detections away from the axis normal."""

        node = _node([5, 6], allow_button_axis=True)
        on_row = _detection("button", 120.0, center_y=40.0)
        stray = _detection("button", 120.0, center_y=240.0)

        kept = node._filter_row_detections(
            [on_row, stray],
            np.asarray([60.0, 40.0], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
        )

        self.assertEqual([detection.center_x for detection in kept], [120.0])
        self.assertIs(kept[0], on_row)

    def test_disabled_bootstrap_keeps_the_cold_start_fallback(self):
        node = _node([], allow_button_axis=False)
        detections = [_detection("button", 60.0), _detection("button", 120.0)]

        selected, origin, axis = node._select_panel_row(detections)

        self.assertIsNone(origin)
        self.assertIsNone(axis)
        self.assertEqual(len(selected), 2)
        self.assertFalse(node.panel_axis.has_axis)

    def test_two_knobs_still_bootstrap_the_row(self):
        node = _node([], allow_button_axis=False)
        selected, _, axis = node._select_panel_row(
            [_detection("knob", 60.0), _detection("knob", 120.0)]
        )
        self.assertEqual(len(selected), 2)
        self.assertIsNotNone(axis)
        self.assertTrue(node.panel_axis.has_axis)

    def test_shared_axis_is_adopted_and_then_reused(self):
        node = _node([5, 6], allow_button_axis=False)
        node.panel_axis.update((0.0, 0.0), (1.0, 0.0))

        selected, origin, axis = node._select_panel_row(
            [_detection("button", 60.0), _detection("button", 120.0)]
        )

        self.assertIsNotNone(axis)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
