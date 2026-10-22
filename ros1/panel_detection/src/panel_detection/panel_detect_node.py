#!/usr/bin/env python3
"""ROS1 Noetic panel detector.

The node consumes a color image, depth registered to that color image and the
color CameraInfo. It publishes the same JSON contract used by the ROS2
visual-servo package on ``/panel/targets``.
"""
import json
import os
import threading
import time
from dataclasses import replace

import cv2
import message_filters
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from panel_detection.detector_onnx import YoloV5Onnx
from panel_detection.frame_processing import LatestFrameProcessor
from panel_detection.image_message import bgr8_image_message
from panel_detection.knob_angle import (
    DiscreteAngleStabilizer,
    draw_knob_angle,
    estimate_knob_angle,
)
from panel_detection.position_stabilizer import PositionStabilizer
from panel_detection.target_registry import (
    FrameDetection,
    PersistentPanelAxis,
    TargetRegistry,
)
from panel_detection.panel_apriltag import (
    PanelAprilTagTracker,
    detect_panel_tags,
    draw_panel_tag_assignments,
)
from panel_detection.board_card_localizer import (
    BoardCardLocalizer,
    draw_board_card_observation,
)


CLASS_TOPICS = {
    "light": "/panel/lights",
    "knob": "/panel/knobs",
    "bolt": "/panel/bolts",
    "nut": "/panel/nuts",
    "valve": "/panel/valves",
    "pump": "/panel/pumps",
    "button": "/panel/buttons",
}


def _is_button_like_detection(detection):
    """Allow square/circular light detections to recover a missed button."""

    width = max(1.0, detection.bbox[2] - detection.bbox[0])
    height = max(1.0, detection.bbox[3] - detection.bbox[1])
    aspect = width / height
    return 0.55 <= aspect <= 1.8


def _fit_axis_from_points(points, fallback_axis):
    """Fit the panel row with PCA and orient it from image-left to right."""

    point_array = np.asarray(points, dtype=np.float64)
    if len(point_array) < 2:
        axis = np.asarray(fallback_axis, dtype=np.float64)
    else:
        centered = point_array - np.mean(point_array, axis=0)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        axis = right_vectors[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        axis = np.asarray(fallback_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
    axis = axis / max(norm, 1e-6)
    if axis[0] < 0.0:
        axis = -axis
    return axis


def _parse_angle_constraint_mode(value):
    """Parse the three ROS2-compatible knob-angle modes.

    1: white marker line (ID3 red handle), stabilized discrete 0/90 output.
    2: red-handle/white-tape detection with physical angle constraints.
    3: red-handle/white-tape detection without physical constraints.
    """

    if isinstance(value, bool):
        return 2 if value else 3
    if isinstance(value, (int, float)):
        mode = int(value)
    else:
        text = str(value).strip().lower()
        if text in ("true", "yes", "on"):
            return 2
        if text in ("false", "no", "off"):
            return 3
        try:
            mode = int(text)
        except ValueError:
            mode = 1
    return mode if mode in (1, 2, 3) else 1


class PanelDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_camera_info = None
        self.last_status = None
        opencv_threads = int(rospy.get_param("~opencv_threads", 0))
        if opencv_threads > 0:
            cv2.setNumThreads(opencv_threads)

        self.color_topic = rospy.get_param(
            "~color_topic", "/camera_d435_0/color/image_raw"
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera_d435_0/aligned_depth_to_color/image_raw"
        )
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera_d435_0/color/camera_info"
        )
        self.camera_frame = rospy.get_param(
            "~camera_frame", "camera_d435_0_color_optical_frame"
        )
        self.depth_scale = float(rospy.get_param("~depth_scale", 0.001))
        position_config = rospy.get_param("~position_stabilizer", {})
        if not isinstance(position_config, dict):
            raise rospy.ROSInitException(
                "position_stabilizer must be a YAML mapping"
            )
        self.position_stabilizer = PositionStabilizer(
            enabled=position_config.get("enable", True),
            still_time=position_config.get("still_time", 3.0),
            pixel_thresh=position_config.get("pixel_thresh", 5.0),
            window_size=position_config.get("window_size", 45),
            ema_alpha=position_config.get(
                "ema_alpha",
                rospy.get_param("~position_filter_alpha", 0.25),
            ),
            depth_std_thresh=position_config.get(
                "depth_std_thresh", 0.01
            ),
        )

        class_names = rospy.get_param(
            "~class_names",
            [
                "light",
                "knob",
                "bolt",
                "nut",
                "valve",
                "pump",
                "button",
                "door_button",
                "air_switch",
                "board_card",
            ],
        )
        model_path = self._resolve_model_path(rospy.get_param("~model_path", ""))
        self.detector = YoloV5Onnx(
            model_path=model_path,
            class_names=class_names,
            input_size=int(rospy.get_param("~input_size", 640)),
            confidence_threshold=float(
                rospy.get_param("~confidence_threshold", 0.30)
            ),
            iou_threshold=float(rospy.get_param("~iou_threshold", 0.01)),
            threads=int(rospy.get_param("~onnx_threads", 8)),
        )
        registry_target_ids = rospy.get_param("~registry_target_ids", [])
        if not isinstance(registry_target_ids, (list, tuple)):
            raise rospy.ROSInitException(
                "registry_target_ids must be a YAML list"
        )
        self.registry_target_ids = [
            int(value) for value in registry_target_ids
        ]
        try:
            self.registry = TargetRegistry(
                stable_frames=int(
                    rospy.get_param("~registry_stable_frames", 15)
                ),
                green_hue_range=tuple(
                    rospy.get_param("~registry_green_hue_range", [35, 85])
                ),
                match_distance_thresh=float(
                    rospy.get_param("~registry_match_distance_px", 80.0)
                ),
                color_minimum_ratio=float(
                    rospy.get_param("~registry_color_minimum_ratio", 0.15)
                ),
                allowed_ids=self.registry_target_ids,
            )
        except ValueError as error:
            raise rospy.ROSInitException(str(error))
        board_card_config = rospy.get_param("~board_card_localizer", {})
        if not isinstance(board_card_config, dict):
            raise rospy.ROSInitException(
                "board_card_localizer must be a YAML mapping"
            )
        board_card_config = dict(board_card_config)
        board_card_config.setdefault("depth_scale", self.depth_scale)
        self.board_card_localizer = (
            BoardCardLocalizer(board_card_config)
            if board_card_config.get("enable", True)
            else None
        )
        self.panel_line = {
            "initial_dist_ratio": float(
                rospy.get_param("~panel_line_initial_dist_ratio", 1.1)
            ),
            "dist_ratio": float(
                rospy.get_param("~panel_line_dist_ratio", 0.85)
            ),
            "min_dist": float(
                rospy.get_param("~panel_line_min_dist_px", 45.0)
            ),
            "max_dist": float(
                rospy.get_param("~panel_line_max_dist_px", 110.0)
            ),
            "proj_margin_ratio": float(
                rospy.get_param("~panel_line_proj_margin_ratio", 3.0)
            ),
            "min_proj_margin": float(
                rospy.get_param("~panel_line_min_projection_margin_px", 450.0)
            ),
        }
        self.panel_axis = PersistentPanelAxis(
            dist_ratio=self.panel_line["dist_ratio"],
            min_dist=self.panel_line["min_dist"],
            max_dist=self.panel_line["max_dist"],
            min_proj_margin=self.panel_line["min_proj_margin"],
        )

        angle_config = rospy.get_param("~knob_angle", {})
        if not isinstance(angle_config, dict):
            raise rospy.ROSInitException("knob_angle must be a YAML mapping")
        self.knob_angle_enabled = bool(angle_config.get("enable", True))
        self.knob_angle_binary_threshold = int(
            angle_config.get("binary_thresh", 180)
        )
        self.knob_angle_circle_mask_ratio = float(
            angle_config.get("circle_mask_ratio", 0.85)
        )
        if not 0.1 <= self.knob_angle_circle_mask_ratio <= 1.0:
            raise rospy.ROSInitException(
                "knob_angle/circle_mask_ratio must be in [0.1, 1.0]"
            )
        configured_mode = angle_config.get("use_constraint")
        if configured_mode is None:
            # Compatibility with the first ROS1 angle port.
            configured_mode = (
                3
                if str(angle_config.get("output_mode", "discrete")).lower()
                == "continuous"
                else 1
            )
        constraint_alias = rospy.get_param("~use_constrain", "")
        constraint_param = constraint_alias or rospy.get_param(
            "~use_constraint", configured_mode
        )
        self.knob_angle_constraint_mode = _parse_angle_constraint_mode(
            constraint_param
        )
        # ROS2 gates angle recognition by detector class only.  In particular,
        # ID4 is a button and must never enter the knob-angle path.
        self.knob_angle_class = str(
            angle_config.get("knob_class", "knob")
        )
        self.knob_angle_stabilizer = DiscreteAngleStabilizer(
            switch_margin=float(angle_config.get("discrete_switch_margin", 8.0)),
            confirm_frames=int(angle_config.get("discrete_confirm_frames", 3)),
        )

        self.targets_pub = rospy.Publisher(
            rospy.get_param("~targets_topic", "/panel/targets"),
            String,
            queue_size=10,
        )
        self.status_pub = rospy.Publisher(
            rospy.get_param("~status_topic", "/panel/status"),
            String,
            queue_size=10,
            latch=True,
        )
        self.distance_pub = rospy.Publisher(
            rospy.get_param("~distance_topic", "/panel/distance"),
            String,
            queue_size=10,
        )
        self.knob_angle_pub = rospy.Publisher(
            rospy.get_param("~knob_angle_topic", "/panel/knob_angles"),
            String,
            queue_size=10,
        )
        class_topic_prefix = str(
            rospy.get_param("~class_topic_prefix", "")
        ).rstrip("/")
        self.class_pose_pubs = {
            class_name: rospy.Publisher(
                rospy.get_param(
                    "~{}_topic".format(class_name),
                    (
                        "{}/{}s".format(class_topic_prefix, class_name)
                        if class_topic_prefix
                        else default_topic
                    ),
                ),
                PoseStamped,
                queue_size=10,
            )
            for class_name, default_topic in CLASS_TOPICS.items()
        }
        self.publish_debug_image = bool(
            rospy.get_param("~publish_debug_image", True)
        )
        self.debug_pub = rospy.Publisher(
            rospy.get_param("~debug_image_topic", "/panel/debug_image"),
            Image,
            queue_size=1,
        )

        # AprilTag overlay.  Keep the existing registry/numbering behavior
        # untouched and only add the decoded Tag boxes/assignments to the
        # debug canvas (plus tag_id in the target JSON).
        raw_panel_tags = rospy.get_param("~use_panel_tags", True)
        if isinstance(raw_panel_tags, str):
            raw_panel_tags = raw_panel_tags.strip().lower() in (
                "1", "true", "yes", "on")
        self.panel_tag_cfg = rospy.get_param("~panel_apriltag", {})
        if not isinstance(self.panel_tag_cfg, dict):
            self.panel_tag_cfg = {}
        self.panel_tag_cfg = dict(self.panel_tag_cfg)
        self.panel_tag_cfg.setdefault("scene", "validation_site")
        self.panel_tag_cfg.setdefault("enable", True)
        self.panel_tag_enabled = (
            bool(raw_panel_tags)
            and bool(self.panel_tag_cfg.get("enable", True))
        )
        self.panel_tag_tracker = (
            PanelAprilTagTracker(self.panel_tag_cfg)
            if self.panel_tag_enabled else None
        )
        if self.panel_tag_enabled:
            if hasattr(cv2, "aruco"):
                rospy.loginfo(
                    "Panel AprilTag overlay enabled (scene=%s)",
                    self.panel_tag_cfg.get("scene", "validation_site"),
                )
            else:
                rospy.logwarn(
                    "OpenCV has no aruco; panel AprilTag overlay disabled"
                )
                self.panel_tag_enabled = False
                self.panel_tag_tracker = None

        # Panel-row axis sharing.  The full-panel detector is the authority and
        # publishes the fitted image-space row direction; a restricted wrist
        # detector can consume it with external_panel_axis:=true when both wrist
        # cameras are known to share an image orientation.
        self.panel_axis_topic = str(
            rospy.get_param("~panel_axis_topic", "/panel/panel_axis")
        )
        self.publish_panel_axis = bool(
            rospy.get_param("~publish_panel_axis", True)
        ) and not self.registry_target_ids
        self.panel_axis_pub = (
            rospy.Publisher(
                self.panel_axis_topic,
                Vector3Stamped,
                queue_size=1,
                latch=True,
            )
            if self.publish_panel_axis
            else None
        )
        # A restricted wrist camera (for example the right arm, whose valid
        # targets are the ID5/ID6 buttons) may never see a knob, so the
        # knob-based row fit cannot run.  It then fits the row from its own
        # button-like detections instead.
        self.allow_button_axis = bool(
            rospy.get_param(
                "~button_axis_bootstrap", bool(self.registry_target_ids)
            )
        )
        self.external_panel_axis = bool(
            rospy.get_param("~external_panel_axis", False)
        )
        self.panel_axis_sub = (
            rospy.Subscriber(
                self.panel_axis_topic,
                Vector3Stamped,
                self._external_panel_axis_callback,
                queue_size=1,
            )
            if self.external_panel_axis
            else None
        )

        self.info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._camera_info_callback,
            queue_size=1,
        )
        self.processing = LatestFrameProcessor(
            self._process_images,
            max_fps=float(rospy.get_param("~max_processing_fps", 0.0)),
            idle_fps=float(rospy.get_param("~idle_processing_fps", rospy.get_param("~max_processing_fps", 0.0))),
            max_age_seconds=float(rospy.get_param("~max_pending_age", 0.4)),
            on_error=lambda error: rospy.logerr_throttle(5.0, "Panel image processing failed: %s", error),
        )
        rospy.on_shutdown(self.processing.close)
        activity_topic = str(rospy.get_param("~activity_topic", ""))
        self.activity_sub = None
        if activity_topic:
            self.activity_sub = rospy.Subscriber(
                activity_topic, String, self._activity_callback, queue_size=10
            )
        color_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=1, buff_size=8388608)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=8388608)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=int(rospy.get_param("~sync_queue_size", 8)),
            slop=float(rospy.get_param("~sync_slop", 0.08)),
            allow_headerless=False,
        )
        sync.registerCallback(self._image_callback)
        self._sync = sync
        self._color_sub = color_sub
        self._depth_sub = depth_sub

        self._publish_status("waiting_camera")
        rospy.loginfo("Panel processing: active<=%s fps idle<=%s fps ONNX threads=%s",
                      self.processing.max_fps, self.processing.idle_fps,
                      rospy.get_param("~onnx_threads", 8))
        rospy.loginfo(
            "panel_detection ready: color=%s depth=%s info=%s model=%s",
            self.color_topic,
            self.depth_topic,
            self.camera_info_topic,
            model_path,
        )
        rospy.loginfo(
            "panel target-ID restriction: %s",
            self.registry_target_ids or "full layout 0..6",
        )
        rospy.loginfo(
            "knob angle detection: enabled=%s use_constraint=%s "
            "class=%s",
            self.knob_angle_enabled,
            self.knob_angle_constraint_mode,
            self.knob_angle_class,
        )

    @staticmethod
    def _resolve_model_path(configured_path):
        if configured_path:
            path = os.path.abspath(os.path.expanduser(configured_path))
            if os.path.isfile(path):
                return path
            raise rospy.ROSInitException("ONNX model does not exist: " + path)

        package_path = rospkg.RosPack().get_path("panel_detection")
        candidates = [
            os.path.join(package_path, "models", "1020.onnx"),
            os.path.join(package_path, "models", "best_zxr_10_13.onnx"),
            os.path.join(package_path, "models", "0824.onnx"),
            os.path.join(package_path, "models", "0807.onnx"),
            os.path.join(package_path, "models", "0601.onnx"),
            os.path.normpath(
                os.path.join(
                    package_path,
                    "..",
                    "panel_detection",
                    "panel_detection",
                    "panel_detection",
                    "1020.onnx",
                )
            ),
            os.path.normpath(
                os.path.join(
                    package_path,
                    "..",
                    "panel_detection",
                    "panel_detection",
                    "panel_detection",
                    "best_zxr_10_13.onnx",
                )
            ),
            os.path.normpath(
                os.path.join(
                    package_path,
                    "..",
                    "panel_detection",
                    "panel_detection",
                    "panel_detection",
                    "0824.onnx",
                )
            ),
            os.path.normpath(
                os.path.join(
                    package_path,
                    "..",
                    "panel_detection",
                    "panel_detection",
                    "panel_detection",
                    "0807.onnx",
                )
            ),
            os.path.normpath(
                os.path.join(
                    package_path,
                    "..",
                    "panel_detection",
                    "panel_detection",
                    "panel_detection",
                    "0601.onnx",
                )
            ),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise rospy.ROSInitException(
            "No ONNX model found. Pass model_path:=/absolute/path/to/model.onnx"
        )

    def _camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def _image_callback(self, color_msg, depth_msg):
        self.processing.submit(color_msg, depth_msg)

    def _activity_callback(self, message):
        self.processing.set_activity(str(message.data))

    def _process_images(self, color_msg, depth_msg):
        started_at = time.monotonic()
        with self.lock:
            info = self.latest_camera_info
        if info is None:
            self._publish_status("waiting_camera_info")
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except CvBridgeError as error:
            rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", error)
            return
        if depth.ndim != 2:
            rospy.logwarn_throttle(2.0, "Depth image must be single-channel")
            return
        if color.shape[:2] != depth.shape[:2]:
            rospy.logwarn_throttle(
                2.0,
                "Color/depth size mismatch; use aligned_depth_to_color: %s vs %s",
                color.shape[:2],
                depth.shape[:2],
            )
            return

        raw_detections = self.detector.detect(color)
        if time.monotonic() - started_at > 0.4:
            rospy.logwarn_throttle(10.0, "Panel inference exceeds 0.4s; inspect CPU load / ONNX_THREADS. Freshness safety limits are unchanged.")
        for detection in raw_detections:
            point = self._point_from_detection(detection, depth, info)
            publisher = self.class_pose_pubs.get(detection.class_name)
            if point is not None and publisher is not None:
                publisher.publish(
                    self._pose_message(color_msg.header.stamp, point)
                )

        # Keep detector_onnx.Detection unchanged at the inference boundary and
        # adapt it to the richer ROS2-compatible registry representation here.
        detections = [
            FrameDetection(
                class_name=detection.class_name,
                center_x=float(detection.center[0]),
                center_y=float(detection.center[1]),
                bbox=tuple(float(value) for value in detection.bbox),
                confidence=float(detection.confidence),
            )
            for detection in raw_detections
        ]
        panel_tag_markers = []
        panel_tag_assignments = {}
        if self.panel_tag_enabled and self.panel_tag_tracker is not None:
            panel_tag_markers = detect_panel_tags(color, self.panel_tag_cfg)
            panel_tag_assignments = self.panel_tag_tracker.update(
                detections, panel_tag_markers)
        tag_id_by_object = {
            id(detections[index]): assignment.tag_id
            for index, assignment in panel_tag_assignments.items()
            if 0 <= index < len(detections)
        }
        board_card_observations = {}
        board_card_observation = None
        board_card_tag_id = 7
        if self.board_card_localizer is not None:
            board_card_tag_id = int(
                self.board_card_localizer.cfg.get("tag_id", 7)
            )
            board_card_marker = next(
                (
                    marker
                    for marker in panel_tag_markers
                    if int(marker.tag_id) == board_card_tag_id
                ),
                None,
            )
            # T07 drives board-card state even when YOLO missed the slot.  The
            # detector class is trained on both empty and inserted slots, so it
            # must not be the gate for running the localizer.
            board_card_observation = self.board_card_localizer.observe(
                color,
                depth,
                info,
                marker=board_card_marker,
                tag_id=board_card_tag_id,
            )

            for index, assignment in panel_tag_assignments.items():
                if not 0 <= index < len(detections):
                    continue
                detection = detections[index]
                if detection.class_name != "board_card":
                    continue
                if (
                    int(assignment.tag_id) == board_card_tag_id
                    and board_card_observation is not None
                ):
                    observation = board_card_observation
                else:
                    observation = self.board_card_localizer.observe(
                        color,
                        depth,
                        info,
                        marker=assignment.marker,
                        tag_id=assignment.tag_id,
                    )
                if observation is None:
                    continue
                board_card_observations[id(detection)] = observation
                # Refine the detector's loose white-frame box to the measured
                # outer card silhouette.  The detection centre follows the
                # outer-card centre, not the top anchor.
                if observation.get("card_present") and observation.get("bbox"):
                    detection.bbox = tuple(observation["bbox"])
                    center = observation.get("center_px") or observation.get(
                        "anchor_px"
                    )
                    if center is not None:
                        detection.center_x, detection.center_y = center
        (
            panel_detections,
            panel_axis_origin,
            panel_axis_vector,
        ) = self._select_panel_row(detections)
        matched = dict(
            self.registry.identify(
                panel_detections,
                color,
                axis_origin=panel_axis_origin,
                axis_vector=panel_axis_vector,
            )
        )
        # Lower row (IDs 8..10).  Only detections the main row did NOT use are
        # candidates, so main-row IDs 0..6 can never be stolen.  "light" is
        # included because round panel buttons are frequently detected as light.
        main_row_used = {id(detection) for detection in matched.values()}
        second_row_candidates = [
            detection
            for detection in detections
            if id(detection) not in main_row_used
            and detection.class_name in ("button", "knob", "light")
        ]
        if second_row_candidates:
            diag = self.registry.classify_lower_row(second_row_candidates, color)
            rospy.loginfo_throttle(
                5.0,
                "lower-row candidates (class,colour): %s",
                [(class_name, colour) for _, class_name, colour in diag],
            )
            for target_id, detection in self.registry.identify_second_row(
                second_row_candidates, color
            ):
                if target_id not in matched:
                    matched[target_id] = detection
            lower_ids = sorted(
                target_id for target_id in (8, 9, 10) if target_id in matched
            )
            if lower_ids:
                rospy.loginfo_throttle(
                    2.0, "Lower-row targets matched: %s", lower_ids
                )

        targets = []
        valid_depths = []
        knob_angles = []
        knob_angles_by_target = {}
        matched_points = {}
        position_measurements = []
        for target_id, detection in sorted(matched.items()):
            observation = board_card_observations.get(id(detection))
            if (
                observation is not None
                and not observation.get("card_present")
                and observation.get("position") is None
            ):
                # Empty slot without a measured slot centre: never fall back to
                # the YOLO white-frame box.
                continue
            if observation is not None and observation.get("position") is not None:
                point = np.asarray(observation["position"], dtype=np.float64)
            else:
                point = self._point_from_detection(detection, depth, info)
            if point is None:
                continue
            matched_points[target_id] = point
            position_measurements.append(
                (target_id, detection.center, point)
            )
        stable_points = self.position_stabilizer.update(
            position_measurements, time.time()
        )
        active_angle_ids = [
            target_id
            for target_id, detection in matched.items()
            if self._should_estimate_angle(target_id, detection)
        ]
        self.knob_angle_stabilizer.prune(active_angle_ids)
        for target_id, detection in sorted(matched.items()):
            if target_id not in matched_points:
                continue
            point = np.asarray(
                stable_points.get(target_id, matched_points[target_id]),
                dtype=np.float64,
            )
            valid_depths.append(point[2])
            tag_id = tag_id_by_object.get(id(detection))
            output_id = (int(tag_id)) if tag_id is not None else None
            target = {
                "id": output_id,
                "class": detection.class_name,
                "tag_id": tag_id,
                "position": {
                    "x": round(float(point[0]), 5),
                    "y": round(float(point[1]), 5),
                    "z": round(float(point[2]), 5),
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
                "confidence": round(float(detection.confidence), 4),
            }
            observation = board_card_observations.get(id(detection))
            if observation is not None:
                target["graspable"] = bool(observation.get("card_present"))
                target["board_card"] = self._board_card_payload(observation)
            else:
                target["graspable"] = True
            targets.append(target)

            if self._should_estimate_angle(target_id, detection):
                raw_angle = self._estimate_detection_angle(
                    color, target_id, detection
                )
                mode_0_90 = self.knob_angle_constraint_mode == 1
                if mode_0_90:
                    output_angle = self.knob_angle_stabilizer.update(
                        target_id, raw_angle
                    )
                else:
                    output_angle = raw_angle
                if output_angle is not None:
                    if mode_0_90:
                        output_angle = int(round(output_angle))
                    else:
                        output_angle = round(float(output_angle), 1)
                    angle_entry = {
                        "id": output_id,
                        "class": detection.class_name,
                        "position": dict(target["position"]),
                        "angle": output_angle,
                        "confidence": round(float(detection.confidence), 4),
                    }
                    knob_angles.append(angle_entry)
                    if output_id is not None:
                        knob_angles_by_target[output_id] = output_angle

        # Tag-only IDs: detections without a tag are still published with
        # id=null so downstream can see them without inventing a layout ID.
        matched_object_ids = {id(detection) for detection in matched.values()}
        for detection in detections:
            if id(detection) in matched_object_ids:
                continue
            if detection.class_name not in (
                    "button", "door_button", "knob", "light", "board_card"):
                continue
            observation = board_card_observations.get(id(detection))
            if (
                observation is not None
                and not observation.get("card_present")
                and observation.get("position") is None
            ):
                continue
            if observation is not None and observation.get("position") is not None:
                point = np.asarray(observation["position"], dtype=np.float64)
            else:
                point = self._point_from_detection(detection, depth, info)
            if point is None:
                continue
            tag_id = tag_id_by_object.get(id(detection))
            output_id = (int(tag_id)) if tag_id is not None else None
            target = {
                "id": output_id,
                "class": detection.class_name,
                "tag_id": tag_id,
                "position": {
                    "x": round(float(point[0]), 5),
                    "y": round(float(point[1]), 5),
                    "z": round(float(point[2]), 5),
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
                "confidence": round(float(detection.confidence), 4),
            }
            if observation is not None:
                target["graspable"] = bool(observation.get("card_present"))
                target["board_card"] = self._board_card_payload(observation)
            else:
                target["graspable"] = True
            targets.append(target)
            valid_depths.append(point[2])

        # Board-card state is published once per frame, keyed by T07.  Empty
        # slots carry the measured slot centre but are never graspable;
        # inserted cards carry the measured outer-card centre.  T07 drives the
        # localizer even when YOLO missed the slot.
        if (
            board_card_observation is not None
            and board_card_observation.get("stable")
            and board_card_observation.get("state") in ("empty", "inserted")
        ):
            existing = next(
                (
                    target
                    for target in targets
                    if target.get("class") == "board_card"
                    and (
                        target.get("tag_id") == board_card_tag_id
                        or target.get("id") == board_card_tag_id
                    )
                ),
                None,
            )
            if existing is None:
                targets.append(
                    self._board_card_target(
                        board_card_observation, board_card_tag_id
                    )
                )
            else:
                existing["id"] = int(board_card_tag_id)
                existing["tag_id"] = int(board_card_tag_id)
                existing["board_card"] = self._board_card_payload(
                    board_card_observation
                )
                existing["graspable"] = bool(
                    board_card_observation.get("card_present")
                )
                if board_card_observation.get("position") is not None:
                    px, py, pz = board_card_observation["position"]
                    existing["position"] = {
                        "x": round(float(px), 5),
                        "y": round(float(py), 5),
                        "z": round(float(pz), 5),
                    }
                else:
                    existing["position"] = None

        # The empty-slot reference must not enter the distance median.
        valid_depths = [
            float(target["position"]["z"])
            for target in targets
            if target.get("position") is not None
            and not (
                target.get("class") == "board_card"
                and not target.get("graspable", True)
            )
        ]

        if targets:
            payload = {
                "stamp": color_msg.header.stamp.to_sec(),
                "frame_id": color_msg.header.frame_id or self.camera_frame,
                "targets": targets,
            }
            self.targets_pub.publish(
                String(data=json.dumps(payload, ensure_ascii=False))
            )
            if valid_depths:
                self.distance_pub.publish(
                    String(
                        data=json.dumps(
                            {
                                "stamp": color_msg.header.stamp.to_sec(),
                                "distance_m": round(
                                    float(np.median(valid_depths)), 5
                                ),
                            }
                        )
                    )
                )
            self._publish_status("registered")
        else:
            self._publish_status("no_targets")
        if knob_angles:
            self.knob_angle_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "stamp": color_msg.header.stamp.to_sec(),
                            "frame_id": color_msg.header.frame_id
                            or self.camera_frame,
                            "knob_angles": knob_angles,
                        },
                        ensure_ascii=False,
                    )
                )
            )
        self._publish_debug(
            color_msg,
            color,
            detections,
            matched,
            knob_angles_by_target,
            panel_tag_markers,
            panel_tag_assignments,
            board_card_observations,
        )

    def _should_estimate_angle(self, target_id, detection):
        del target_id
        if not self.knob_angle_enabled:
            return False
        return detection.class_name == self.knob_angle_class

    def _estimate_detection_angle(self, color, target_id, detection):
        height, width = color.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        mode_0_90 = self.knob_angle_constraint_mode == 1
        # ID 4 is the red long-handle knob in every mode.  In discrete mode
        # its directional red-handle estimate is subsequently reduced to the
        # stable, unoriented 0/90 state by DiscreteAngleStabilizer.  This keeps
        # the ROS2 red-handle detector active while preserving the ROS1
        # visual-servo direction contract.
        if int(target_id) == 3:
            pointer_color = "red_handle"
        elif mode_0_90:
            pointer_color = "white_line"
        else:
            pointer_color = "white_tape"
        angle_range = None
        if self.knob_angle_constraint_mode == 2:
            angle_range = (
                (180.0, 270.0)
                if int(target_id) == 3
                else (0.0, 90.0)
            )
        return estimate_knob_angle(
            color[y1:y2, x1:x2],
            binary_thresh=self.knob_angle_binary_threshold,
            circle_mask_ratio=self.knob_angle_circle_mask_ratio,
            angle_range=angle_range,
            pointer_color=pointer_color,
        )

    def _select_panel_row(self, detections):
        """Select and order the physical button/knob row.

        Two visible knobs bootstrap the row.  When the view is narrower, the
        cached direction is reused and its parallel line is relocated from the
        currently visible detections.

        A restricted wrist view (``registry_target_ids`` set, for example the
        right arm with the ID5/ID6 buttons) may never contain a knob at all, so
        ``button_axis_bootstrap`` lets it fit the row from its own button-like
        detections instead.  Without an axis that view would be ordered by raw
        image x, which is what made its IDs unstable.
        """

        knobs = [
            detection
            for detection in detections
            if detection.class_name == "knob"
        ]
        if len(knobs) >= 2:
            knob_1, knob_2 = sorted(
                knobs, key=lambda detection: detection.center_x
            )[:2]
            line_vector = np.asarray(
                [
                    knob_2.center_x - knob_1.center_x,
                    knob_2.center_y - knob_1.center_y,
                ],
                dtype=np.float64,
            )
            line_length = float(np.linalg.norm(line_vector))
            if line_length > 1.0:
                line_axis = line_vector / line_length
                line_normal = np.asarray(
                    [-line_axis[1], line_axis[0]], dtype=np.float64
                )
                panel_sizes = [
                    max(1.0, detection.bbox[3] - detection.bbox[1])
                    for detection in detections
                    if detection.class_name in ("button", "knob")
                ]
                median_height = (
                    float(np.median(panel_sizes)) if panel_sizes else 80.0
                )
                initial_threshold = float(
                    np.clip(
                        median_height
                        * self.panel_line["initial_dist_ratio"],
                        self.panel_line["min_dist"],
                        self.panel_line["max_dist"],
                    )
                )
                projection_margin = max(
                    self.panel_line["min_proj_margin"],
                    line_length * self.panel_line["proj_margin_ratio"],
                )

                rough_row = []
                knob_origin = np.asarray(
                    [knob_1.center_x, knob_1.center_y], dtype=np.float64
                )
                for detection in detections:
                    can_join = (
                        detection.class_name in ("button", "knob")
                        or (
                            detection.class_name == "light"
                            and _is_button_like_detection(detection)
                        )
                    )
                    if not can_join:
                        continue
                    point = (
                        np.asarray(
                            [detection.center_x, detection.center_y],
                            dtype=np.float64,
                        )
                        - knob_origin
                    )
                    distance = abs(float(np.dot(point, line_normal)))
                    projection = float(np.dot(point, line_axis))
                    if (
                        distance <= initial_threshold
                        and -projection_margin
                        <= projection
                        <= line_length + projection_margin
                    ):
                        rough_row.append(detection)

                if len(rough_row) >= 3:
                    fitted_axis = _fit_axis_from_points(
                        [
                            (detection.center_x, detection.center_y)
                            for detection in rough_row
                        ],
                        line_axis,
                    )
                    axis_origin = np.mean(
                        np.asarray(
                            [
                                (
                                    detection.center_x,
                                    detection.center_y,
                                )
                                for detection in rough_row
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                else:
                    fitted_axis = line_axis
                    axis_origin = knob_origin

                fitted_normal = np.asarray(
                    [-fitted_axis[1], fitted_axis[0]], dtype=np.float64
                )
                rough_projections = [
                    float(
                        np.dot(
                            np.asarray(
                                [
                                    detection.center_x,
                                    detection.center_y,
                                ],
                                dtype=np.float64,
                            )
                            - axis_origin,
                            fitted_axis,
                        )
                    )
                    for detection in rough_row
                ]
                if rough_projections:
                    projection_min = (
                        min(rough_projections)
                        - self.panel_line["min_proj_margin"]
                    )
                    projection_max = (
                        max(rough_projections)
                        + self.panel_line["min_proj_margin"]
                    )
                else:
                    projection_min = -projection_margin
                    projection_max = line_length + projection_margin

                row_threshold = float(
                    np.clip(
                        median_height * self.panel_line["dist_ratio"],
                        self.panel_line["min_dist"],
                        self.panel_line["max_dist"],
                    )
                )
                origin_tuple = (
                    float(axis_origin[0]),
                    float(axis_origin[1]),
                )
                vector_tuple = (
                    float(fitted_axis[0]),
                    float(fitted_axis[1]),
                )
                self.panel_axis.update(origin_tuple, vector_tuple)
                self._publish_panel_axis(vector_tuple)

                panel_detections = []
                for detection in detections:
                    can_join = (
                        detection.class_name in ("button", "knob")
                        or (
                            detection.class_name == "light"
                            and _is_button_like_detection(detection)
                        )
                    )
                    if not can_join:
                        continue
                    point = (
                        np.asarray(
                            [detection.center_x, detection.center_y],
                            dtype=np.float64,
                        )
                        - axis_origin
                    )
                    distance = abs(float(np.dot(point, fitted_normal)))
                    projection = float(np.dot(point, fitted_axis))
                    on_row = (
                        distance <= row_threshold
                        and projection_min
                        <= projection
                        <= projection_max
                    )
                    if (
                        detection.class_name == "light"
                        and on_row
                    ):
                        detection.class_name = "button"
                    if (
                        on_row
                        and detection.class_name in ("button", "knob")
                    ):
                        panel_detections.append(detection)
                return panel_detections, origin_tuple, vector_tuple

        cached = self.panel_axis.select(detections)
        if cached is not None:
            return cached

        # A restricted wrist camera such as the right arm may only ever see the
        # ID5/ID6 buttons, so the two-knob bootstrap above can never run and
        # sorted-by-image-x ordering would decide the IDs.  Fit the row from the
        # button-like detections themselves: with the axis known, detections are
        # ordered along the real panel row and a "light" reading of a square
        # button is still promoted to "button".
        if self.allow_button_axis:
            row_points = [
                detection
                for detection in detections
                if detection.class_name in ("button", "knob")
                or (
                    detection.class_name == "light"
                    and _is_button_like_detection(detection)
                )
            ]
            if len(row_points) >= 2:
                fitted_axis = _fit_axis_from_points(
                    [
                        (detection.center_x, detection.center_y)
                        for detection in row_points
                    ],
                    (1.0, 0.0),
                )
                axis_origin = np.mean(
                    np.asarray(
                        [
                            (detection.center_x, detection.center_y)
                            for detection in row_points
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                )
                vector_tuple = (
                    float(fitted_axis[0]),
                    float(fitted_axis[1]),
                )
                origin_tuple = (float(axis_origin[0]), float(axis_origin[1]))
                self.panel_axis.update(origin_tuple, vector_tuple)
                self._publish_panel_axis(vector_tuple)
                return (
                    self._filter_row_detections(
                        detections, axis_origin, fitted_axis
                    ),
                    origin_tuple,
                    vector_tuple,
                )

        # Cold-start fallback: class/colour matching still supplies absolute
        # IDs even before a reliable panel axis has been observed.
        panel_detections = [
            detection
            for detection in detections
            if detection.class_name in ("button", "knob")
        ]
        return panel_detections, None, None

    def _filter_row_detections(self, detections, axis_origin, fitted_axis):
        """Keep the detections that sit on the row described by the axis."""

        axis = np.asarray(fitted_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return []
        axis = axis / norm
        origin = np.asarray(axis_origin, dtype=np.float64)
        normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

        panel_sizes = [
            max(1.0, detection.bbox[3] - detection.bbox[1])
            for detection in detections
            if detection.class_name in ("button", "knob")
        ]
        median_height = (
            float(np.median(panel_sizes)) if panel_sizes else 80.0
        )
        row_threshold = float(
            np.clip(
                median_height * self.panel_line["dist_ratio"],
                self.panel_line["min_dist"],
                self.panel_line["max_dist"],
            )
        )

        projections = [
            float(
                np.dot(
                    np.asarray(
                        [detection.center_x, detection.center_y],
                        dtype=np.float64,
                    )
                    - origin,
                    axis,
                )
            )
            for detection in detections
        ]
        if projections:
            projection_min = (
                min(projections) - self.panel_line["min_proj_margin"]
            )
            projection_max = (
                max(projections) + self.panel_line["min_proj_margin"]
            )
        else:
            projection_min = -np.inf
            projection_max = np.inf

        panel_detections = []
        for detection in detections:
            can_join = detection.class_name in ("button", "knob") or (
                detection.class_name == "light"
                and _is_button_like_detection(detection)
            )
            if not can_join:
                continue
            point = (
                np.asarray(
                    [detection.center_x, detection.center_y],
                    dtype=np.float64,
                )
                - origin
            )
            distance = abs(float(np.dot(point, normal)))
            projection = float(np.dot(point, axis))
            on_row = (
                distance <= row_threshold
                and projection_min <= projection <= projection_max
            )
            if detection.class_name == "light" and on_row:
                detection.class_name = "button"
            if on_row and detection.class_name in ("button", "knob"):
                panel_detections.append(detection)
        return panel_detections

    def _publish_panel_axis(self, vector_tuple):
        if self.panel_axis_pub is None:
            return
        message = Vector3Stamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.camera_frame
        message.vector.x = float(vector_tuple[0])
        message.vector.y = float(vector_tuple[1])
        message.vector.z = 0.0
        self.panel_axis_pub.publish(message)

    def _external_panel_axis_callback(self, message):
        vector = np.asarray(
            [message.vector.x, message.vector.y], dtype=np.float64
        )
        if float(np.linalg.norm(vector)) < 1e-6:
            return
        if not self.panel_axis.has_axis:
            rospy.loginfo(
                "adopting shared panel axis (%.3f, %.3f) from %s",
                float(vector[0]),
                float(vector[1]),
                self.panel_axis_topic,
            )
        self.panel_axis.update(
            (0.0, 0.0), (float(vector[0]), float(vector[1]))
        )

    def _board_card_target(self, observation, tag_id):
        """Synthetic id=T07 target used when YOLO misses the slot."""

        position = observation.get("position")
        payload_position = None
        if position is not None:
            payload_position = {
                "x": round(float(position[0]), 5),
                "y": round(float(position[1]), 5),
                "z": round(float(position[2]), 5),
            }
        return {
            "id": int(tag_id),
            "class": "board_card",
            "tag_id": int(tag_id),
            "position": payload_position,
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "confidence": round(float(observation.get("state_confidence", 0.0)), 4),
            "graspable": bool(observation.get("card_present")),
            "board_card": self._board_card_payload(observation),
        }

    def _board_card_payload(self, observation):
        """JSON-safe board-card geometry, without non-finite values."""

        def point(value):
            if value is None or len(value) != 2:
                return None
            return [round(float(value[0]), 3), round(float(value[1]), 3)]

        def points(values):
            if values is None:
                return None
            return [
                [round(float(value[0]), 3), round(float(value[1]), 3)]
                for value in values
            ]

        def position(value):
            if value is None:
                return None
            return {
                "x": round(float(value[0]), 5),
                "y": round(float(value[1]), 5),
                "z": round(float(value[2]), 5),
            }

        return {
            "card_present": bool(observation.get("card_present")),
            "state": observation.get("state", observation.get("insertion_state", "unknown")),
            "insertion_state": observation.get("insertion_state", "unknown"),
            "stable": bool(observation.get("stable", False)),
            "graspable": bool(observation.get("card_present")),
            "pose_locked": bool(observation.get("pose_locked", False)),
            "pose_samples": int(observation.get("pose_samples", 0)),
            "pose_spread_tag": (
                round(float(observation["pose_spread_tag"]), 4)
                if observation.get("pose_spread_tag") is not None else None
            ),
            "position_stable": bool(observation.get("position_stable", False)),
            "anchor_source": observation.get("anchor_source"),
            "anchor_px": point(observation.get("anchor_px")),
            "top_px": point(observation.get("top_px")),
            "bottom_px": point(observation.get("bottom_px")),
            "center_px": point(observation.get("center_px")),
            "axis_px": point(observation.get("axis_px")),
            "slot_px": point(observation.get("slot_px")),
            "quad_px": points(observation.get("quad_px")),
            "slot_position": position(observation.get("slot_position")),
            "width_px": (
                round(float(observation["width_px"]), 2)
                if observation.get("width_px") is not None else None
            ),
            "width_tag": (
                round(float(observation["width_tag"]), 4)
                if observation.get("width_tag") is not None else None
            ),
            "length_tag": (
                round(float(observation["length_tag"]), 4)
                if observation.get("length_tag") is not None else None
            ),
            "visible_length_px": round(
                float(observation.get("visible_length_px", 0.0)), 2
            ),
            "nominal_length_px": (
                round(float(observation["nominal_length_px"]), 2)
                if observation.get("nominal_length_px") is not None else None
            ),
            "length_estimated": bool(observation.get("length_estimated")),
            "clipped": bool(observation.get("clipped")),
            "confidence": round(float(observation.get("confidence", 0.0)), 4),
            "position_source": observation.get(
                "position_source", observation.get("source")
            ),
            "depth_source": observation.get("depth_source"),
            "plane_rms_m": (
                round(float(observation["plane_rms_m"]), 5)
                if observation.get("plane_rms_m") is not None else None
            ),
            "panel_plane_rms_m": (
                round(float(observation["panel_plane_rms_m"]), 5)
                if observation.get("panel_plane_rms_m") is not None else None
            ),
            "top_position": position(observation.get("top_position")),
            "anchor_position": position(observation.get("anchor_position")),
            "bottom_position": position(observation.get("bottom_position")),
            "geometry_source": observation.get("geometry_source"),
            "card_depth_m": (
                round(float(observation["card_depth_m"]), 5)
                if observation.get("card_depth_m") is not None else None
            ),
            "card_plane_rms_m": (
                round(float(observation["card_plane_rms_m"]), 5)
                if observation.get("card_plane_rms_m") is not None else None
            ),
        }

    def _point_from_detection(self, detection, depth, info):
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        width = max(2, x2 - x1)
        height = max(2, y2 - y1)
        margin_x = int(width * 0.28)
        margin_y = int(height * 0.28)
        x1 = max(0, x1 + margin_x)
        x2 = min(depth.shape[1], x2 - margin_x)
        y1 = max(0, y1 + margin_y)
        y2 = min(depth.shape[0], y2 - margin_y)
        if x2 <= x1 or y2 <= y1:
            return None
        values = np.asarray(depth[y1:y2, x1:x2], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size < 8:
            return None
        z = float(np.median(values))
        if depth.dtype != np.float32 and depth.dtype != np.float64:
            z *= self.depth_scale
        if not 0.05 <= z <= 2.0:
            return None
        u, v = detection.center
        fx, fy, cx, cy = info.K[0], info.K[4], info.K[2], info.K[5]
        if fx <= 0.0 or fy <= 0.0:
            return None
        return np.array(
            [(float(u) - cx) * z / fx, (float(v) - cy) * z / fy, z],
            dtype=np.float64,
        )

    def _pose_message(self, stamp, point):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.pose.position.x = float(point[0])
        msg.pose.position.y = float(point[1])
        msg.pose.position.z = float(point[2])
        msg.pose.orientation.w = 1.0
        return msg

    def _publish_status(self, status):
        if status == self.last_status:
            return
        self.last_status = status
        self.status_pub.publish(String(data=status))
        rospy.loginfo("Panel status: %s", status)

    def _publish_debug(
        self,
        source_msg,
        color,
        detections,
        matched,
        knob_angles_by_target,
        panel_tag_markers=None,
        panel_tag_assignments=None,
        board_card_observations=None,
    ):
        if not self.publish_debug_image or self.debug_pub.get_num_connections() == 0:
            return
        canvas = color.copy()
        tag_id_by_object = {
            id(detections[index]): assignment.tag_id
            for index, assignment in (panel_tag_assignments or {}).items()
            if 0 <= index < len(detections)
        }
        id_by_object = {
            object_id: (int(tag_id))
            for object_id, tag_id in tag_id_by_object.items()
            if tag_id is not None
        }
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            target_id = id_by_object.get(id(detection))
            label = detection.class_name
            if target_id is not None:
                label = "#{} {}".format(target_id, label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                "{} {:.2f}".format(label, detection.confidence),
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if target_id in knob_angles_by_target:
                draw_knob_angle(
                    canvas,
                    detection.bbox,
                    knob_angles_by_target[target_id],
                )
        if panel_tag_markers:
            display_assignments = {
                index: replace(assignment, target_id=int(assignment.tag_id))
                for index, assignment in (panel_tag_assignments or {}).items()
            }
            draw_panel_tag_assignments(
                canvas,
                panel_tag_markers,
                display_assignments,
                detections,
            )
        for observation in (board_card_observations or {}).values():
            draw_board_card_observation(canvas, observation)
        try:
            debug = bgr8_image_message(canvas, Image, source_msg.header)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(2.0, "Cannot pack panel debug image: %s", error)
            return
        self.debug_pub.publish(debug)


def main():
    rospy.init_node("panel_detection")
    try:
        PanelDetectionNode()
        rospy.spin()
    except (rospy.ROSInitException, RuntimeError, ValueError) as error:
        rospy.logfatal("%s", error)
        raise
