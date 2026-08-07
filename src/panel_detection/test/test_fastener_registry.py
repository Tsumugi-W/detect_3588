from panel_detection.fastener_registry import (
    FastenerGroupRegistry,
    FastenerObservation,
)


def _obs(det_idx, xy, point, class_name='bolt', normal=(0.0, 0.0, -1.0)):
    x, y = xy
    return FastenerObservation(
        det_idx=det_idx,
        class_name=class_name,
        center_xy=(float(x), float(y)),
        bbox=(x - 5.0, y - 5.0, x + 5.0, y + 5.0),
        confidence=0.9,
        point_3d=point,
        axis_direction=normal,
    )


def test_fastener_registry_initializes_from_three_and_tracks_single_target():
    registry = FastenerGroupRegistry(max_slot_distance_m=0.08)
    initial = [
        _obs(0, (100, 100), (-0.05, -0.05, 0.5)),
        _obs(1, (200, 100), (0.05, -0.05, 0.5)),
        _obs(2, (200, 200), (0.05, 0.05, 0.5)),
    ]

    matched = registry.update(initial, frame_index=1)

    assert {matched[i].target_id for i in matched} == {1, 2, 3}
    assert all(item.group_id == 1 for item in matched.values())
    assert all(item.registered for item in matched.values())

    later = [_obs(10, (203, 202), (0.052, 0.049, 0.501))]
    matched = registry.update(later, frame_index=2)

    assert list(matched) == [10]
    assert matched[10].group_id == 1
    assert matched[10].target_id == 3
    assert matched[10].slot == 'bottom_right'


def test_fastener_registry_publishes_provisional_ids_from_two_cold_start_points():
    registry = FastenerGroupRegistry()
    matched = registry.update([
        _obs(0, (100, 100), (-0.05, -0.05, 0.5)),
        _obs(1, (200, 100), (0.05, -0.05, 0.5)),
    ], frame_index=1)

    assert set(matched) == {0, 1}
    assert {item.group_id for item in matched.values()} == {1}
    assert len({item.target_id for item in matched.values()}) == 2
    assert all(not item.registered for item in matched.values())


def test_fastener_registry_promotes_two_point_group_when_third_appears():
    registry = FastenerGroupRegistry(max_slot_distance_m=0.08)
    first = [
        _obs(0, (100, 100), (-0.05, -0.05, 0.5)),
        _obs(1, (200, 100), (0.05, -0.05, 0.5)),
    ]
    provisional = registry.update(first, frame_index=1)

    promoted = registry.update(first + [
        _obs(2, (200, 200), (0.05, 0.05, 0.5)),
    ], frame_index=2)

    assert {item.group_id for item in promoted.values()} == {
        provisional[0].group_id
    }
    assert {item.target_id for item in promoted.values()} == {1, 2, 3}
    assert all(item.registered for item in promoted.values())


def test_fastener_registry_keeps_spatially_separate_groups_apart():
    registry = FastenerGroupRegistry(max_group_distance_m=0.25)
    group_a = [
        _obs(0, (100, 100), (-0.05, -0.05, 0.5)),
        _obs(1, (200, 100), (0.05, -0.05, 0.5)),
        _obs(2, (200, 200), (0.05, 0.05, 0.5)),
    ]
    group_b = [
        _obs(3, (400, 100), (0.7, -0.05, 0.5)),
        _obs(4, (500, 100), (0.8, -0.05, 0.5)),
        _obs(5, (500, 200), (0.8, 0.05, 0.5)),
    ]

    matched = registry.update(group_a + group_b, frame_index=1)

    group_ids = {item.group_id for item in matched.values()}
    assert group_ids == {1, 2}
    assert {matched[i].group_id for i in (0, 1, 2)} == {1}
    assert {matched[i].group_id for i in (3, 4, 5)} == {2}


def test_fastener_registry_rejects_normal_mismatch_for_existing_group():
    registry = FastenerGroupRegistry(max_slot_distance_m=0.08,
                                     normal_angle_thresh_deg=15.0)
    initial = [
        _obs(0, (100, 100), (-0.05, -0.05, 0.5)),
        _obs(1, (200, 100), (0.05, -0.05, 0.5)),
        _obs(2, (200, 200), (0.05, 0.05, 0.5)),
    ]
    registry.update(initial, frame_index=1)

    mismatched = [_obs(10, (202, 200), (0.05, 0.05, 0.5),
                       normal=(1.0, 0.0, 0.0))]

    assert registry.update(mismatched, frame_index=2) == {}
