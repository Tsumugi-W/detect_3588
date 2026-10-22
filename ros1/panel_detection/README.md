# panel_detection

`panel_detection` is the ROS1 Noetic port of the ROS2
`panel_detection` package. It keeps the `/panel/targets` JSON contract used by
the visual-servo pipeline.

## Inputs

- Color image
- Depth image registered to that color image
- Color `sensor_msgs/CameraInfo`

The recommended input is the official ROS1 `realsense2_camera` wrapper with
`align_depth:=true`. The older `embodied_camera` publisher is not used by
default because it does not publish `CameraInfo` and its depth image is not
aligned to color.

## Outputs

- `/panel/targets` (`std_msgs/String`, JSON)
- `/panel/knob_angles` (`std_msgs/String`, JSON; marked knob angle in degrees)
- `/panel/status` (`std_msgs/String`)
- `/panel/distance` (`std_msgs/String`, JSON)
- `/panel/buttons` (`geometry_msgs/PoseStamped`)
- `/panel/lights`, `/panel/knobs`, `/panel/bolts`, `/panel/nuts`,
  `/panel/valves`, `/panel/pumps` (`geometry_msgs/PoseStamped`)
- `/panel/debug_image` (`sensor_msgs/Image`)

## Knob angle detection

Angle recognition is enabled by default and uses the complete ROS2 traditional
CV implementation, including white marker-line, white-tape, red-handle,
bright-pointer and dark-pointer fallbacks. `use_constraint` selects the same
three modes as ROS2:

- `1`: ID3 uses the red long-handle detector and other configured IDs use the
  white line detector, followed by hysteresis and three-frame confirmation;
  output is stable `0` or `90` (vertical is `0`, horizontal is `90`).
- `2`: ID 3 uses red-handle detection constrained to `180..270` degrees;
  other knobs use white-tape detection constrained to `0..90` degrees.
- `3`: the mode-2 red-handle/white-tape pipeline without physical constraints.

```yaml
knob_angle:
  enable: true
  use_constraint: 1
  knob_class: knob
knob_angle_topic: /panel/knob_angles
```

The launch argument overrides the YAML value, matching ROS2 usage:

```bash
roslaunch panel_detection d435_panel_detection.launch \
  serial_no:=420222072543 use_constraint:=2
```

The JSON payload is compatible with the ROS2 detector:

```json
{
  "stamp": 1720000000.0,
  "frame_id": "camera_d435_0_color_optical_frame",
  "knob_angles": [
    {"id": 3, "class": "knob", "position": {"x": 0.1, "y": 0.0, "z": 0.5}, "angle": 90, "confidence": 0.94}
  ]
}
```

As in ROS2, the angle path only accepts the detector class configured by
`knob_class` (normally `knob`). ID4 is a `button`, so it is published only as a
button target and never appears on `/panel/knob_angles`. The visual-servo task
can still rotate ID4 using its configured fallback direction.

## Board-card localization

The detector class `board_card` is intentionally trained on **both** an empty
slot and a fully inserted black card.  It only localises the slot region; it is
not the state source.  T07 (or the configured `board_card_localizer.tag_id`)
drives the state machine and the geometry.

The node warps a fixed tag-relative slot ROI and measures the black card's
**full outer silhouette**: a low-value/low-saturation black mask, largest
card-body component, and small card fragments merged across the card's white
label.  Components are also checked against the panel depth, so the dark gap
between the card and the recess wall is not merged into the card.  The white
recess frame and the blue backplane are excluded.  A frame is `inserted` only
when the silhouette passes the measured length, black-body fraction and
fill-ratio gates.  The empty slot's blue backplane, white walls and recess
shadow are therefore classified as `empty`, not as a card.

Production has only two stable states:

- `empty`: `card_present=false`; `position` is the measured blue-backplane
  slot centre on the panel plane, and `graspable=false`;
- `inserted`: `card_present=true`; `position` is the measured centre of the
  black card's outer silhouette.

There is no `pulled`/`transition` state; insertion motion is out of scope.

After `inserted` is confirmed, the localizer locks the robust median
tag-relative outer **centre/axis/width/length**.  The per-frame silhouette is
used to feed that lock, never as a fixed prior.  There is **no fixed
tag-relative top/slot anchor and no fixed card length**: the tag may be mounted
at a different offset on another panel.  When the silhouette is temporarily
unavailable, the locked outer pose is projected through the current T07
homography; the colour strip is only a short-lived fallback and is never
locked.

3D uses the **measured card-front depth** inside the outer quad: the white
label and illuminated card edges usually provide valid depth even when the
black body is a depth hole.  The panel plane plus `panel_depth_offset_m` is a
fallback only.  When T07 is stationary for several frames, the final
camera-frame 3D position is locked to a median and stops publishing residual
depth/tag jitter; camera motion unlocks it again.

`board_card` payload exposes `state`, `stable`, `pose_locked`, `position_stable`,
`anchor_px`, `center_px`, `slot_px`, `quad_px`, `width_tag`, `length_tag`,
`anchor_position`, `position`, `slot_position`, `geometry_source`,
`card_depth_m`, `card_plane_rms_m`, `panel_plane_rms_m` and `depth_source`.
`insertion_state` is kept as a one-release compatibility alias.  The orange
debug quad is the measured outer card, not the YOLO white-frame box.

Configuration lives under `board_card_localizer:` in
`config/panel_detection.yaml`.  The default slot ROI is
`x=[-3.5,3.5]`, `y=[1,11]` tag-side units and assumes a 20 mm tag; all pixel
gates scale with that ROI.  Set `enable: false` to fall back to the YOLO box.

## Model

The package uses the supplied `models/1020.onnx` model by default. It contains
ten classes: `light`, `knob`, `bolt`, `nut`, `valve`, `pump`, `button`,
`door_button`, `air_switch`, and `board_card`. The `best_zxr_10_13.onnx`,
`0824.onnx`, `0807.onnx` and `0601.onnx` exports remain bundled as fallbacks.
You can still override the model explicitly:

```bash
model_path:=/absolute/path/to/model.onnx
```

`models/1020.onnx` is the same file shipped by the upstream `detect_3588`
repository, so the robot and the RK3588 detector use identical weights. The
1013 export remains available as a fallback.

ONNX Runtime is not provided by ROS Noetic:

```bash
python3 -m pip install --user onnxruntime
```

## Absolute target IDs

The ROS1 node now uses the same panel-layout logic as the ROS2 package:

| ID | Class | Colour |
|---:|---|---|
| 1 | button | green |
| 2 | button | red |
| 3 | button | red |
| 4 | knob | red |
| 5 | knob | black |
| 6 | button | red |
| 7 | button | green |

Visible button/knob detections are sorted along the physical panel axis and
matched to a contiguous window of this layout. This means `/panel/targets`
contains absolute IDs even when only part of the panel is visible. Show the
camera the complete seven-target row once after startup whenever possible:
that seeds a position cache which disambiguates a single knob or other narrow
views. The panel-axis direction is also cached, so the row can be relocated
after camera translation.

Square/circular `light` detections on the panel row are treated as buttons to
recover a missed button classification. Elongated indicator lights are
excluded by their aspect ratio. Tune `registry_*` and `panel_line_*` in
`config/panel_detection.yaml` if the camera resolution or panel scale changes.

For a camera dedicated to a known physical subset, set
`registry_target_ids`. The dual-arm launch uses `[5, 6]` for the right wrist,
so a solitary red right-side button is ID5 rather than being matched to the
first red button ID1. Ambiguous unknown-colour detections are not assigned.

### Knob-free wrist views

The row axis is normally bootstrapped from two visible knobs. The right wrist
only ever sees the ID5/ID6 buttons and therefore contains no knob, so that
bootstrap can never run there. Previously every frame of that view fell through
to the cold-start path: detections were ordered by raw image `x` and matched by
colour alone, so a colour misread, a `light` reading of a square button, or a
mirrored camera image changed the IDs between the two arms.

`button_axis_bootstrap` (default `1` whenever `registry_target_ids` is not
empty) makes such a view fit the row from its own button-like detections. The
same axis then drives the row filtering, the image-order sorting and the
`light` -> `button` promotion, so both arms number the same physical targets.

The full-panel detector additionally publishes its fitted row direction on
`panel_axis_topic` (`geometry_msgs/Vector3Stamped`, latched). A restricted
detector can adopt it with `external_panel_axis:=true`, but the published
vector is expressed in the publishing camera's image plane, so only enable it
when both wrist cameras share an image orientation. The local bootstrap is the
default and needs no cross-camera assumption.

Two instances can run in one ROS master by giving them different
`detection_node_name`, output topics and `class_topic_prefix`. The visual-servo
dual launch does this automatically, publishing `/panel/left/*` and
`/panel/right/*` without mixing measurements from the two optical frames.

## Run

Start the camera and detector together:

```bash
roslaunch panel_detection d435_panel_detection.launch \
  serial_no:=420222072543 \
  camera_name:=camera_d435_0
```

Inspect:

```bash
rostopic echo /panel/status
rostopic echo /panel/targets
rostopic echo /panel/knob_angles
rqt_image_view /panel/debug_image
```

Verify the absolute IDs on the debug image before enabling robot motion. The
`registry_stable_frames` parameter is retained for ROS2/config compatibility;
the current layout matcher identifies each frame without a registration wait.
