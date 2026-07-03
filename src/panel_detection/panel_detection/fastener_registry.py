from dataclasses import dataclass, field
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SLOT_NAMES = {
    1: 'top_left',
    2: 'top_right',
    3: 'bottom_right',
    4: 'bottom_left',
}


@dataclass
class FastenerObservation:
    det_idx: int
    class_name: str
    center_xy: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    point_3d: Sequence[float]
    axis_direction: Optional[Sequence[float]] = None


@dataclass
class FastenerAssignment:
    det_idx: int
    group_id: int
    target_id: int
    slot: str
    registered: bool
    distance_m: float


@dataclass
class _FastenerGroup:
    group_id: int
    slots: Dict[int, np.ndarray]
    origin: np.ndarray
    normal: Optional[np.ndarray]
    last_seen_frame: int
    class_by_slot: Dict[int, str] = field(default_factory=dict)
    confidence_by_slot: Dict[int, float] = field(default_factory=dict)


def _normalize(vec) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return None
    arr = arr / norm
    if arr.shape[0] >= 3 and arr[2] > 0:
        arr = -arr
    return arr


def _angle_between_deg(a, b) -> Optional[float]:
    na = _normalize(a)
    nb = _normalize(b)
    if na is None or nb is None:
        return None
    dot = abs(float(np.dot(na, nb)))
    return math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))


def _fit_normal(points: List[np.ndarray],
                fallback_normals: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(points) >= 3:
        pts = np.asarray(points, dtype=np.float64)
        centered = pts - np.mean(pts, axis=0)
        try:
            _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = None
            s_vals = []
        if vt is not None and len(s_vals) >= 2 and s_vals[1] > 1e-6:
            normal = vt[-1]
            if normal[2] > 0:
                normal = -normal
            return normal / np.linalg.norm(normal)

    if fallback_normals:
        normal = np.mean(np.asarray(fallback_normals, dtype=np.float64), axis=0)
        return _normalize(normal)
    return None


def _infer_missing_slot(slots: Dict[int, np.ndarray]) -> None:
    if len(slots) != 3:
        return
    missing = ({1, 2, 3, 4} - set(slots)).pop()
    if missing == 1 and 2 in slots and 4 in slots and 3 in slots:
        slots[1] = slots[2] + slots[4] - slots[3]
    elif missing == 2 and 1 in slots and 3 in slots and 4 in slots:
        slots[2] = slots[1] + slots[3] - slots[4]
    elif missing == 3 and 2 in slots and 4 in slots and 1 in slots:
        slots[3] = slots[2] + slots[4] - slots[1]
    elif missing == 4 and 1 in slots and 3 in slots and 2 in slots:
        slots[4] = slots[1] + slots[3] - slots[2]


def _assign_slots_by_image(observations: List[FastenerObservation]) -> Dict[int, int]:
    observations = sorted(
        observations, key=lambda obs: obs.confidence, reverse=True)[:4]
    centers = np.asarray([obs.center_xy for obs in observations], dtype=np.float64)
    cx, cy = np.mean(centers, axis=0)
    desired = {
        1: np.array([-1.0, -1.0]),
        2: np.array([1.0, -1.0]),
        3: np.array([1.0, 1.0]),
        4: np.array([-1.0, 1.0]),
    }
    obs_quadrants = []
    for obs in observations:
        x, y = obs.center_xy
        vec = np.array([
            -1.0 if x < cx else 1.0,
            -1.0 if y < cy else 1.0,
        ])
        obs_quadrants.append(vec)

    best_perm = None
    best_score = float('inf')
    for slots in itertools.permutations([1, 2, 3, 4], len(observations)):
        score = 0.0
        for idx, slot_id in enumerate(slots):
            score += float(np.linalg.norm(obs_quadrants[idx] - desired[slot_id]))
        if score < best_score:
            best_score = score
            best_perm = slots

    return {
        obs.det_idx: int(slot_id)
        for obs, slot_id in zip(observations, best_perm or [])
    }


class FastenerGroupRegistry:
    """Track and number bolt/nut groups mounted on local four-fastener planes."""

    def __init__(self, min_init_observations=3, max_group_distance_m=0.35,
                 max_slot_distance_m=0.12, slot_match_ratio=0.45,
                 normal_angle_thresh_deg=25.0, ema_alpha=0.35,
                 stale_frames=120):
        self.min_init_observations = int(min_init_observations)
        self.max_group_distance_m = float(max_group_distance_m)
        self.max_slot_distance_m = float(max_slot_distance_m)
        self.slot_match_ratio = float(slot_match_ratio)
        self.normal_angle_thresh_deg = float(normal_angle_thresh_deg)
        self.ema_alpha = float(ema_alpha)
        self.stale_frames = int(stale_frames)
        self._groups: List[_FastenerGroup] = []
        self._next_group_id = 1

    def reset(self):
        self._groups.clear()
        self._next_group_id = 1

    def update(self, observations: List[FastenerObservation],
               frame_index: int) -> Dict[int, FastenerAssignment]:
        valid = [
            obs for obs in observations
            if obs.point_3d is not None and np.asarray(obs.point_3d).shape[0] == 3
        ]
        self._prune(frame_index)
        assignments: Dict[int, FastenerAssignment] = {}
        if not valid:
            return assignments

        remaining = set(range(len(valid)))
        for group in sorted(self._groups, key=lambda g: g.last_seen_frame, reverse=True):
            group_obs_indices = self._select_observations_for_group(
                group, valid, remaining)
            if not group_obs_indices:
                continue
            group_obs = [valid[i] for i in group_obs_indices]
            group_assignments = self._update_group(group, group_obs, frame_index)
            assignments.update(group_assignments)
            assigned_det_indices = set(group_assignments)
            remaining.difference_update(
                idx for idx in group_obs_indices
                if valid[idx].det_idx in assigned_det_indices)

        for cluster_indices in self._cluster_unassigned(valid, remaining):
            if len(cluster_indices) < self.min_init_observations:
                continue
            cluster_obs = [valid[i] for i in cluster_indices]
            group = self._create_group(cluster_obs, frame_index)
            if group is None:
                continue
            self._groups.append(group)
            assignments.update(self._update_group(group, cluster_obs, frame_index))

        return assignments

    def _prune(self, frame_index: int) -> None:
        self._groups = [
            group for group in self._groups
            if frame_index - group.last_seen_frame <= self.stale_frames
        ]

    def _slot_spacing(self, group: _FastenerGroup) -> float:
        points = list(group.slots.values())
        if len(points) < 2:
            return self.max_slot_distance_m
        distances = [
            float(np.linalg.norm(a - b))
            for i, a in enumerate(points)
            for b in points[i + 1:]
        ]
        if not distances:
            return self.max_slot_distance_m
        return float(np.median(distances))

    def _match_gate(self, group: _FastenerGroup) -> float:
        return max(self.max_slot_distance_m,
                   self.slot_match_ratio * self._slot_spacing(group))

    def _normal_ok(self, group: _FastenerGroup, obs: FastenerObservation) -> bool:
        if group.normal is None or obs.axis_direction is None:
            return True
        angle = _angle_between_deg(group.normal, obs.axis_direction)
        return angle is None or angle <= self.normal_angle_thresh_deg

    def _select_observations_for_group(
            self, group: _FastenerGroup, observations: List[FastenerObservation],
            remaining: set) -> List[int]:
        selected = []
        gate = self._match_gate(group)
        for idx in list(remaining):
            obs = observations[idx]
            point = np.asarray(obs.point_3d, dtype=np.float64)
            slot_dist = min(
                float(np.linalg.norm(point - slot))
                for slot in group.slots.values()
            )
            origin_dist = float(np.linalg.norm(point - group.origin))
            if (slot_dist <= gate or origin_dist <= self.max_group_distance_m):
                if self._normal_ok(group, obs):
                    selected.append(idx)
        return selected

    def _cluster_unassigned(self, observations: List[FastenerObservation],
                            remaining: set) -> List[List[int]]:
        clusters = []
        unused = set(remaining)
        while unused:
            seed = unused.pop()
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                for idx in list(unused):
                    if self._can_join_cluster(observations[idx],
                                              [observations[i] for i in cluster]):
                        cluster.append(idx)
                        unused.remove(idx)
                        changed = True
            clusters.append(cluster)
        return clusters

    def _can_join_cluster(self, obs: FastenerObservation,
                          cluster: List[FastenerObservation]) -> bool:
        point = np.asarray(obs.point_3d, dtype=np.float64)
        cluster_points = [np.asarray(item.point_3d, dtype=np.float64)
                          for item in cluster]
        if min(float(np.linalg.norm(point - other))
               for other in cluster_points) > self.max_group_distance_m:
            return False

        if obs.axis_direction is None:
            return True
        for item in cluster:
            if item.axis_direction is None:
                continue
            angle = _angle_between_deg(obs.axis_direction, item.axis_direction)
            if angle is not None and angle <= self.normal_angle_thresh_deg:
                return True
        return not any(item.axis_direction is not None for item in cluster)

    def _create_group(self, observations: List[FastenerObservation],
                      frame_index: int) -> Optional[_FastenerGroup]:
        slot_observations = sorted(
            observations, key=lambda obs: obs.confidence, reverse=True)[:4]
        slot_by_det_idx = _assign_slots_by_image(slot_observations)
        slots = {}
        class_by_slot = {}
        confidence_by_slot = {}
        for obs in slot_observations:
            slot_id = slot_by_det_idx.get(obs.det_idx)
            if slot_id is None:
                continue
            slots[slot_id] = np.asarray(obs.point_3d, dtype=np.float64)
            class_by_slot[slot_id] = obs.class_name
            confidence_by_slot[slot_id] = float(obs.confidence)
        if len(slots) < self.min_init_observations:
            return None
        _infer_missing_slot(slots)

        points = [np.asarray(obs.point_3d, dtype=np.float64) for obs in observations]
        normals = [
            _normalize(obs.axis_direction)
            for obs in observations if obs.axis_direction is not None
        ]
        normals = [normal for normal in normals if normal is not None]
        normal = _fit_normal(points, normals)
        origin = np.mean(np.asarray(list(slots.values()), dtype=np.float64), axis=0)
        group = _FastenerGroup(
            group_id=self._next_group_id,
            slots=slots,
            origin=origin,
            normal=normal,
            last_seen_frame=frame_index,
            class_by_slot=class_by_slot,
            confidence_by_slot=confidence_by_slot,
        )
        self._next_group_id += 1
        return group

    def _update_group(self, group: _FastenerGroup,
                      observations: List[FastenerObservation],
                      frame_index: int) -> Dict[int, FastenerAssignment]:
        matches = self._match_observations_to_slots(group, observations)
        if not matches:
            return {}

        observed_normals = []
        assignments = {}
        alpha = np.clip(self.ema_alpha, 0.0, 1.0)
        for obs, slot_id, distance in matches:
            point = np.asarray(obs.point_3d, dtype=np.float64)
            if slot_id in group.slots:
                group.slots[slot_id] = (
                    (1.0 - alpha) * group.slots[slot_id] + alpha * point)
            else:
                group.slots[slot_id] = point
            group.class_by_slot[slot_id] = obs.class_name
            group.confidence_by_slot[slot_id] = float(obs.confidence)
            normal = _normalize(obs.axis_direction) if obs.axis_direction is not None else None
            if normal is not None:
                observed_normals.append(normal)
            assignments[obs.det_idx] = FastenerAssignment(
                det_idx=obs.det_idx,
                group_id=group.group_id,
                target_id=slot_id,
                slot=SLOT_NAMES.get(slot_id, f'slot_{slot_id}'),
                registered=len(group.slots) >= 4,
                distance_m=float(distance),
            )

        if len(group.slots) == 3:
            _infer_missing_slot(group.slots)
        group.origin = np.mean(np.asarray(list(group.slots.values()),
                                          dtype=np.float64), axis=0)
        normal = _fit_normal(
            [np.asarray(p, dtype=np.float64) for p in group.slots.values()],
            observed_normals)
        if normal is not None:
            if group.normal is not None and np.dot(group.normal, normal) < 0:
                normal = -normal
            group.normal = (
                normal if group.normal is None else
                _normalize((1.0 - alpha) * group.normal + alpha * normal))
        group.last_seen_frame = frame_index
        return assignments

    def _match_observations_to_slots(
            self, group: _FastenerGroup,
            observations: List[FastenerObservation]):
        slot_ids = sorted(group.slots)
        if not slot_ids:
            return []
        gate = self._match_gate(group)
        best = None
        best_score = float('inf')
        match_count = min(len(observations), len(slot_ids))
        observation_subsets = itertools.combinations(observations, match_count)
        for chosen_observations in observation_subsets:
            for chosen_slots in itertools.permutations(slot_ids, match_count):
                score = 0.0
                candidate = []
                ok = True
                for obs, slot_id in zip(chosen_observations, chosen_slots):
                    point = np.asarray(obs.point_3d, dtype=np.float64)
                    distance = float(np.linalg.norm(point - group.slots[slot_id]))
                    if distance > gate:
                        ok = False
                        break
                    class_name = group.class_by_slot.get(slot_id)
                    if class_name is not None and class_name != obs.class_name:
                        score += gate * 0.25
                    score += distance
                    candidate.append((obs, slot_id, distance))
                if ok and score < best_score:
                    best_score = score
                    best = candidate
        return best or []
