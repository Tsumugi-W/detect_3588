"""Regression coverage for the shared ROS/offline Tag postprocessing."""
from copy import deepcopy
import numpy as np
from panel_detection.panel_apriltag import (
    PanelAprilTagTracker, PanelTagMarker, apply_panel_tag_rules,
    reclassify_buttons_without_tags,
)
from panel_detection.target_registry import FrameDetection


def test_shared_rules_match_previous_node_sequence_across_tag_dropout():
    old_tracker, shared_tracker = PanelAprilTagTracker(), PanelAprilTagTracker()
    marker = PanelTagMarker(0, np.array([[80, 20], [120, 20], [120, 60], [80, 60]], dtype=np.float32), (100., 40.), 1600.)
    for markers in [[marker], [], [], [marker], *([[]] * 10)]:
        original = [FrameDetection('button', 100., 130., (75., 100., 125., 160.), .9)]
        shared = deepcopy(original)
        old_assignments = old_tracker.update(original, markers)
        for index, assignment in old_assignments.items():
            original[index].class_name = assignment.forced_class
        old_changed = reclassify_buttons_without_tags(original, old_assignments)
        changed_ids = {id(d) for d in old_changed}
        old_assignments = {i: a for i, a in old_assignments.items() if id(original[i]) not in changed_ids}
        assignments, changed = apply_panel_tag_rules(shared, markers, shared_tracker)
        assert [d.class_name for d in shared] == [d.class_name for d in original]
        assert [(i, a.tag_id, a.source) for i, a in assignments.items()] == [(i, a.tag_id, a.source) for i, a in old_assignments.items()]
        assert len(changed) == len(old_changed)
        if not markers:
            assert shared[0].class_name == 'light'
            assert not assignments
        else:
            assert assignments[0].target_id == 1


def test_empty_detections_and_unmatched_button():
    tracker = PanelAprilTagTracker()
    assert apply_panel_tag_rules([], [], tracker) == ({}, [])
    detections = [FrameDetection('button', 100., 130., (75., 100., 125., 160.), .9)]
    assignments, changed = apply_panel_tag_rules(detections, [], tracker)
    assert assignments == {}
    assert changed == detections
    assert detections[0].class_name == 'light'
