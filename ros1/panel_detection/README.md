# panel_detection (ROS1 Noetic)

## 快速启动：面板 / 螺栓检测

**每个终端先加载 ROS1 工作空间**（路径按实际部署修改）：

```bash
cd ~/panel_ros1_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

**面板检测（相机已单独启动）**：

```bash
roslaunch panel_detection panel_controls.launch
```

**螺栓 / 螺母检测（另开终端）**：

```bash
roslaunch panel_detection fastener_detection.launch
```

默认订阅 `/camera_d435_0` 的彩色图和对齐深度；相机命名不同可加 `camera_name:=名称`。
ROS1 默认通过图像话题查看检测画面：

```bash
rqt_image_view /panel/debug_image
rqt_image_view /fasteners/debug_image
```

有桌面时可给检测 launch 加 `show_gui:=true` 弹出 OpenCV 窗口。
面板结果包含 `/panel/targets`、`/panel/board_cards`；紧固件结果见 `/fasteners/targets`、`/fasteners/geometry`。

This package ports the current ROS2 detector to ROS1 Noetic while keeping the
same YOLO, target numbering, angle, 3D position and axis-estimation algorithms.
It consumes color-aligned D435 depth and uses `1013.onnx` by default.

## Build

```bash
cd ~/panel_ros1_ws
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

- `/panel/targets`, `/panel/board_cards`, `/panel/knob_angles`, `/panel/status`, `/panel/distance`

Dedicated modes publish only their own object family:

- `/valve/targets`, `/valve/geometry`, `/valve/status`
- `/fasteners/targets`, `/fasteners/geometry`, `/fasteners/status`

In panel mode, every final `button` must be directly associated with its own
AprilTag in the current frame. A YOLO `button` without a direct assignment is
treated as `light`, even when another tag is visible or a stale tracked tag ID
exists. Layout fallback cannot promote an untagged `light` back to `button`.
`door_button`, knobs, YOLO board cards and non-panel modes are unaffected. Set
`use_panel_tags:=false` to retain the legacy no-tag classification behavior.

All structured results use `std_msgs/String` JSON. Debug images use
`sensor_msgs/Image`. Legacy per-class `PoseStamped` topics are disabled by default.
For panel mode with legacy topics, use the general launch:

```bash
roslaunch panel_detection panel_detection.launch detection_mode:=panel_controls publish_legacy_topics:=true
```

The task-specific ROS1 wrapper does not expose `publish_legacy_topics`.

## Model

The bundled model has ten classes:

`light, knob, bolt, nut, valve, pump, button, door_button, air_switch, board_card`

Override it with an absolute ONNX path:

```bash
roslaunch panel_detection panel_detection.launch model_path:=/path/model.onnx
```

## 面板场景配置（与 ROS2 同步）

默认 `validation_site`：T02/T03 为旋钮，T00/T01/T04/T05/T06 为按钮，T07 对应板卡 `board_card`。
T07 仅给 YOLO 板卡分配编号 8，有无 Tag 均不修改板卡类别；不把其他类别改为板卡。
其他 Tag 不参与关联；目标编号仍为 Tag ID+1。
`test_site` 保留旧场景：T03/T04 为旋钮，T00/T01/T02/T05/T06 为按钮，T07–39 为灯。

```bash
# 默认验证场，保留 ROS1 相机配置
roslaunch panel_detection panel_controls.launch
# 测试场
roslaunch panel_detection panel_controls.launch config_file:=$(rospack find panel_detection)/config/test_site.yaml
# 显示深度采样区域（通过已有 debug_image 话题查看）
roslaunch panel_detection panel_controls.launch debug_depth_roi:=true
```

`config_file` 也可指定自定义 YAML。配置递归合并，`panel_apriltag.tag_classes`
整体替换，规则与 ROS2 一致。验证场关闭旧布局回退，测试场开启。
本帧没有直接 Tag 对应的 button 仍改为 light；knob 不因缺少 Tag 被改类。
ROS1 的图像话题参数、Noetic 时间戳、模型查找路径和 debug_image 发布方式保留。

完整检测功能现已与 ROS2 对齐：包括同心紧固件标记、Tag 几何参考、稳定器、
精度记录器、RGB/深度查看器、RKNN 后端、模型导出与二维管道复核工具。
算法模块由跨版本测试检查一致性，节点保留 ROS1 时间戳、话题和模型路径适配。

新增入口：`leak_detection.launch`、`panel_controls_debug.launch`、
`rgb_depth_viewer.launch`、`fastener_precision_recorder.launch`。
支持 `use_topic:=false` 直接连接已安装 SDK 的相机；相机驱动仍使用各 ROS 版本原生包。
ROS1 默认 YAML 现在启用 AprilTag 几何参考，与 ROS2 相同。

漏点模式默认开启完整漏点/管道检测和三维轴线拟合。仅检测螺栓时使用
`config_file:=$(rospack find panel_detection)/config/leak_bolt_only.yaml`。
三维输出协议和深度要求见仓库根 README 的“管道三维轴线”。


## 板卡检测与位姿发布

- `/panel/board_cards`（`std_msgs/String` JSON）默认发布在 panel_controls/all 模式，
  包含时间戳、相机坐标系 `frame_id` 和所有 YOLO 板卡的 `targets`。
  每个目标含 `class: board_card`、框、置信度、`classification_source: yolo`、
  `id`、`tag_id`、`identity_source`、`position`（米）和 `orientation`（四元数）。
- 验证场 T07 仅关联已检测为板卡的目标，编号仍为 8；没有 Tag 关联时 id/tag_id 为 null，
  板卡类别和有效深度位姿仍发布。短暂丢 Tag 时沿用原有跟踪期限。
- `position_valid` / `orientation_valid` 标明有效性。缺少有效深度时 position 为 null；
  缺少面板法向时 orientation 为 null。姿态沿用面板法向，不估计板卡面内旋转角。
  无板卡的帧发布空 targets，避免下游保留旧检测。
- 有 Tag 编号且深度有效的板卡同时进入 `/panel/targets`。
  通用入口 `panel_detection.launch publish_legacy_topics:=true` 时另发 `/panel/board_card_pose`（`geometry_msgs/PoseStamped`），
  沿用旧类别位姿接口；区分多个板卡和识别无效姿态时应使用 JSON 话题。


```bash
# 默认面板入口已发布板卡 JSON；无需开启旧话题
rostopic echo /panel/board_cards
rostopic echo /panel/targets
# 如需同时发布板卡 PoseStamped，使用通用入口（相机命名保留默认 camera_d435_0）
roslaunch panel_detection panel_detection.launch detection_mode:=panel_controls publish_legacy_topics:=true
rostopic echo /panel/board_card_pose
```

完整 JSON 示例、字段含义和无效位姿处理见[根 README 的板卡话题说明](../../README.md#panelboard_cards-string-json--板卡检测与位姿)。
板卡只框固定白色矩形外框；位置由检测框中心深度反投影，姿态沿用面板法向，不估计面内旋转。
`position_valid` 和 `orientation_valid` 需分别检查；目标消失时收到空 `targets`。

当前中心深度采样可能受到插入件、插槽背景和手部遮挡影响；`position_valid` 不保证采样点属于固定外框。
插拔前后的几何位置一致性尚未验证，限制与改进方向见根 README 板卡话题一节。
