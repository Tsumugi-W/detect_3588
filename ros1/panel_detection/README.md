# panel_detection (ROS1 Noetic)

This package ports the current ROS2 detector to ROS1 Noetic while keeping the
same YOLO, target numbering, angle, 3D position and axis-estimation algorithms.
It consumes color-aligned D435 depth and uses `0813.onnx` by default.

## Build

```bash
cd ~/panel_ros1_ws_fd7aaef
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Inputs

- `/camera_d435_0/color/image_raw`
- `/camera_d435_0/aligned_depth_to_color/image_raw`
- `/camera_d435_0/color/camera_info`

The depth image must be registered to the color image. Override
`camera_name:=...` when the D435 namespace differs.

## Launch

Run against an already running camera:

```bash
roslaunch panel_detection panel_detection.launch
roslaunch panel_detection panel_controls.launch
roslaunch panel_detection valve_detection.launch
roslaunch panel_detection fastener_detection.launch
```

Start one D435 and the detector together:

```bash
roslaunch panel_detection d435_panel_detection.launch serial_no:=<serial>
```

GUI is disabled by default. View the annotated Canvas with:

```bash
rqt_image_view /panel/debug_image
```

## Outputs

Panel mode preserves the existing formats:

- `/panel/targets`, `/panel/knob_angles`, `/panel/status`, `/panel/distance`

Dedicated modes publish only their own object family:

- `/valve/targets`, `/valve/geometry`, `/valve/status`
- `/fasteners/targets`, `/fasteners/geometry`, `/fasteners/status`

In panel mode, every final `button` must be directly associated with its own
AprilTag in the current frame. A YOLO `button` without a direct assignment is
treated as `light`, even when another tag is visible or a stale tracked tag ID
exists. Layout fallback cannot promote an untagged `light` back to `button`.
`door_button`, knobs and non-panel modes are unaffected. Set
`use_panel_tags:=false` to retain the legacy no-tag classification behavior.

All structured results use `std_msgs/String` JSON. Debug images use
`sensor_msgs/Image`. Legacy per-class `PoseStamped` topics remain available via
`publish_legacy_topics:=true`, but are disabled by default.

## Model

The bundled model has nine classes:

`light, knob, bolt, nut, valve, pump, button, door_button, air_switch`

Override it with an absolute ONNX path:

```bash
roslaunch panel_detection panel_detection.launch model_path:=/path/model.onnx
```
