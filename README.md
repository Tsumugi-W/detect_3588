# ROS2 面板检测功能包 (RK3588 + Orbbec Gemini 336)

基于 YOLOv5 + 深度相机的操作面板实时 3D 检测系统，封装为标准 ROS2 Humble 功能包。

检测 8 类目标：指示灯(light)、旋钮(knob)、螺栓(bolt)、螺母(nut)、阀门(valve)、泵(pump)、按钮(button)、门按钮(door_button)。支持目标注册编号、旋钮角度估计和螺栓/螺母/阀门轴线方向估计。

## 系统架构

```
Orbbec 官方驱动 (独立 launch)
    ├── /camera/color/image_raw
    ├── /camera/depth/image_raw
    ├── /camera/color/camera_info
    └── /camera/depth/camera_info
            │
            ▼
面板检测节点 (话题订阅模式)
    ├── /panel/targets       (带编号的检测结果)
    ├── /panel/knob_angles   (旋钮角度)
    ├── /panel/distance      (相机到面板平面的垂直距离)
    ├── /panel/status        (注册状态)
    └── /panel/buttons, /panel/knobs ...  (兼容旧话题)
```

## 平台要求

| 项目 | 要求 |
|------|------|
| 系统 | Ubuntu 22.04 (aarch64 或 x86_64) |
| ROS2 | Humble |
| Python | 3.10+ |
| 相机 | 奥比中光 Gemini 336 (默认) 或 Intel RealSense D435i |
| 推理 | ONNX Runtime CPU (默认) 或 RKNN NPU |

## 安装教程

### 第一步：安装 ROS2 Humble

如果系统已安装 ROS2 Humble 则跳过此步。

```bash
sudo apt update
sudo apt install ros-humble-ros-base ros-humble-vision-msgs python3-colcon-common-extensions -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 第二步：克隆仓库

```bash
mkdir -p ~/ros2_ws
cd ~/ros2_ws
git clone --recursive git@github.com:Tsumugi-W/detect_3588.git .
```

> 如果忘记 `--recursive`，需要手动初始化 submodule：
> ```bash
> git submodule update --init --recursive
> ```

### 第三步：安装系统依赖

```bash
# OrbbecSDK_ROS2 编译所需的 ROS2 包
sudo apt install -y \
  ros-humble-image-transport \
  ros-humble-image-publisher \
  ros-humble-cv-bridge \
  ros-humble-camera-info-manager \
  ros-humble-tf2-ros \
  ros-humble-tf2-sensor-msgs \
  ros-humble-backward-ros \
  ros-humble-message-filters \
  ros-humble-rclcpp-components \
  ros-humble-statistics-msgs

# libusb（相机驱动需要）
sudo apt install -y libusb-1.0-0-dev libudev-dev
```

### 第四步：安装 Python 依赖（系统级）

```bash
# 清华镜像加速
sudo pip3 install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple

# opencv、pyyaml 通常系统已自带，如果没有：
# sudo pip3 install opencv-python pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第五步：配置相机权限

```bash
# 安装 Orbbec udev 规则
sudo bash src/OrbbecSDK_ROS2/orbbec_camera/scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

配置完成后**重新插拔相机** USB 线。

### 第六步：编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash

# 先编译相机驱动
colcon build --packages-up-to orbbec_camera
source install/setup.bash

# 再编译检测节点
colcon build --packages-select panel_detection
source install/setup.bash
```

> 也可以一次编译全部：`colcon build`，但如果遇到依赖问题建议分步。

### 第七步：运行

```bash
# 终端1：启动相机（发布 /camera/color 与 /camera/depth 话题）
ros2 launch panel_detection camera.launch.py

# 终端2：启动检测节点（推荐：订阅相机话题）
ros2 launch panel_detection panel_detection.launch.py use_topic:=true registered_depth:=true
```

启动后检测节点会弹出可视化窗口。启动后先进入**注册阶段**（积累稳定帧为目标分配编号），注册完成后进入**跟踪阶段**（发布带编号的检测结果）。按 `q` 或 `ESC` 退出。

## 启动方式与参数配置

### 常用启动命令

**推荐方式：相机节点 + 检测节点分开启动**

```bash
# 终端1：启动 Orbbec Gemini 336，相机话题深度已注册到彩色图
ros2 launch panel_detection camera.launch.py

# 终端2：检测节点订阅相机话题
ros2 launch panel_detection panel_detection.launch.py use_topic:=true registered_depth:=true
```

**直连模式：检测节点直接打开相机**

```bash
ros2 launch panel_detection panel_detection.launch.py
```

直连模式会由检测节点自己连接相机，并转发 `/camera/color/image_raw`、`/camera/depth/image_raw` 等话题。现场调试更推荐使用 `use_topic:=true`，相机和检测解耦，bag 回放也使用同一条路径。

**Bag 回放验证**

```bash
# 终端1：启动检测节点
ros2 launch panel_detection panel_detection.launch.py use_topic:=true registered_depth:=true

# 终端2：回放已录制的相机话题
ros2 bag play panel_test_bag_2 \
  --topics /camera/color/image_raw /camera/depth/image_raw /camera/color/camera_info /camera/depth/camera_info \
  --start-offset 20 \
  -r 0.5
```

**旋钮角度模式**

```bash
ros2 launch panel_detection panel_detection.launch.py use_topic:=true use_constraint:=1
```

`use_constraint:=1` 为默认模式：只根据旋钮白色手柄线与竖直线的夹角，稳定输出 `0` 或 `90`。旧写法仍兼容：`true` 等价于 `2`，`false` 等价于 `3`。

### Launch 参数

| 参数 | 默认值 | 作用 | 何时修改 |
|------|--------|------|----------|
| `use_topic` | `false` | `true` 时订阅 `/camera/*` 话题；`false` 时检测节点直连相机 | 使用 `camera.launch.py` 或 bag 回放时设为 `true` |
| `registered_depth` | `true` | topic 模式下深度图是否已对齐到彩色图 | Orbbec `camera.launch.py` 已设置 `depth_registration=true`，保持 `true` |
| `use_constraint` | `1` | 旋钮角度模式：`1`=0/90 稳定输出，`2`=旧物理范围约束，`3`=旧无约束；`use_constrain` 也可作为别名 | 默认保持 `1`；需要旧行为时设为 `2` 或 `3` |
| `config_path` | 空字符串 | 外部 YAML 配置文件路径；为空使用默认配置 | 需要换模型、阈值、相机后端、推理后端时使用 |

### 指定配置文件

```bash
ros2 launch panel_detection panel_detection.launch.py \
  use_topic:=true \
  registered_depth:=true \
  config_path:=/home/ztl/project/panel_ws/my_panel_config.yaml
```

注意：`config_path` 是检测节点配置，不是 launch 文件配置。配置文件中未写的字段不会自动和默认配置递归合并，因此建议从下面示例复制完整配置后修改。

### 相机启动参数

`camera.launch.py` 固定使用 Orbbec 官方 `gemini_330_series.launch.py`，并传入：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `depth_registration` | `true` | 深度图注册到彩色图坐标系 |
| `color_width` / `color_height` | `1280` / `720` | 彩色图分辨率 |
| `depth_width` / `depth_height` | `1280` / `720` | 深度图分辨率 |
| `color_fps` / `depth_fps` | `30` / `30` | 帧率 |
| `color_format` | `ANY` | 由驱动选择彩色格式 |
| `interleave_ae_mode` | `none` | 关闭交替曝光模式 |

如果要改相机分辨率或帧率，优先修改 `src/panel_detection/launch/camera.launch.py` 中的这些参数，并保持 `registered_depth:=true`。

### 可选：写入 bashrc 简化日常使用

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## 两阶段工作流

### 注册阶段

节点启动后自动进入注册阶段：
1. 等待连续 N 帧同时检测到 5 个按钮 + 2 个旋钮
2. 按 x 坐标从左到右排序
3. 验证布局：[button, button, button, knob, knob, button, button]
4. 验证第 1 个按钮为绿色
5. 验证通过后分配编号 1-7

### 跟踪阶段

注册完成后，每帧检测结果与注册表匹配，输出带编号的目标信息。

## 发布话题

### /panel/targets (String, JSON) — 主话题

```json
{
  "stamp": 1716192000.123,
  "targets": [
    {
      "id": 1,
      "class": "button",
      "position": {"x": 0.123, "y": -0.045, "z": 0.850},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.92
    },
    {
      "id": 4,
      "class": "knob",
      "position": {"x": 0.320, "y": -0.040, "z": 0.845},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.88
    }
  ]
}
```

位置单位：**米 (m)**，相机坐标系。

### /panel/knob_angles (String, JSON) — 旋钮角度、六边形角度、轴线方向

```json
{
  "stamp": 1716192000.123,
  "knob_angles": [
    {
      "id": 4,
      "position": {"x": 0.32, "y": -0.04, "z": 0.85},
      "angle": 171.5,
      "confidence": 0.92
    }
  ],
  "hex_angles": [
    {
      "class": "nut",
      "bbox": [510.0, 260.0, 560.0, 310.0],
      "hex_angle": 28.4,
      "nut_refined_conf": 0.81
    }
  ],
  "axis_directions": [
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "source": "fastener_current",
      "axis_direction": [0.012, -0.034, -0.999],
      "centroid": [0.18, -0.03, 0.82],
      "point_count": 742
    }
  ]
}
```

角度以 12 点钟方向为 0°，顺时针增加，范围 [0, 360)。`axis_direction` 是相机坐标系下的单位方向向量，指向相机方向时 z 为负。螺栓、螺母、阀门优先使用当前帧附近同平面器件的 3D 点拟合安装平面，`source` 为 `fastener_current`；如果只有两个邻近点，则用两点连线约束局部安装面法向量，`source` 为 `fastener_line`；再往后依次回退到目标周边局部安装面 `local_mount_plane`、全局面板平面 `panel_plane` 和局部目标深度 `local_depth`。

### /panel/distance (String, JSON) — 相机到面板平面的垂直距离

```json
{
  "stamp": 1716192000.123,
  "distance_m": 0.8123,
  "normal": [0.01, -0.02, -0.9997],
  "centroid": [0.02, -0.01, 0.812]
}
```

`distance_m` 单位为米，表示相机坐标系原点到操作面板拟合平面的垂直距离。

### /panel/status (String) — 注册状态

- `"registering"` — 注册中
- `"registered"` — 注册完成

### 兼容旧话题 (PoseStamped)

| 话题 | 类别 |
|------|------|
| `/panel/lights` | 指示灯 |
| `/panel/knobs` | 旋钮 |
| `/panel/buttons` | 按钮 |
| `/panel/bolts` | 螺栓 |
| `/panel/nuts` | 螺母 |
| `/panel/valves` | 阀门 |
| `/panel/pumps` | 泵 |

`/panel/nuts` 发布的是 refined 后的螺母操作点：像素中心来自六角螺母外轮廓拟合，深度来自外圈 mask 采样并排除中心螺丝/螺杆。refined 失败或外圈深度无效时不会发布 `/panel/nuts`，避免机械臂拿到错误点。

## 验证话题

```bash
# 另开终端
ros2 topic list
ros2 topic echo /panel/targets
ros2 topic echo /panel/knob_angles
ros2 topic echo /panel/distance
ros2 topic echo /panel/status
```

## 在其他节点中订阅

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class PanelSubscriber(Node):
    def __init__(self):
        super().__init__('panel_subscriber')
        self.create_subscription(String, '/panel/targets', self.targets_cb, 10)
        self.create_subscription(String, '/panel/knob_angles', self.angle_cb, 10)

    def targets_cb(self, msg):
        data = json.loads(msg.data)
        for t in data['targets']:
            p = t['position']
            self.get_logger().info(
                f"ID={t['id']} {t['class']} 位置: ({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})")

    def angle_cb(self, msg):
        data = json.loads(msg.data)
        for k in data['knob_angles']:
            self.get_logger().info(f"ID={k['id']} 角度: {k['angle']:.1f}°")
```

## 配置文件说明

创建 YAML 文件后通过 `config_path` 参数传入：

```bash
ros2 launch panel_detection panel_detection.launch.py use_topic:=true config_path:=/path/to/config.yaml
```

配置示例：

```yaml
camera_backend: 'orbbec'       # 'orbbec' | 'realsense'
camera:
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30

inference_backend: 'onnx'      # 'onnx' | 'rknn'
onnx_model: '0624.onnx'        # 相对 panel_detection 包目录，或填写绝对路径
onnx_threads: 8
# rknn_model: '0624.rknn'      # inference_backend='rknn' 时使用

class_num: 8
class_name: ['light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button', 'door_button']
threshold:
  confidence: 0.3
  iou: 0.01

knob_angle:
  enable: true
  binary_thresh: 180
  circle_mask_ratio: 0.85
  knob_class: 'knob'
  use_constraint: 1

depth_scale: 0.001             # 深度图原始值 × depth_scale = 米

topic_sync:
  max_dt: 0.05                 # color/depth 时间戳最大允许差值，秒
  registered_depth: true       # 深度图是否已注册到彩色图

position_stabilizer:
  enable: true
  still_time: 3.0
  pixel_thresh: 5.0
  window_size: 45
  ema_alpha: 0.25
  depth_std_thresh: 0.01

panel_line:
  initial_dist_ratio: 1.1
  dist_ratio: 0.85
  min_dist: 45.0
  max_dist: 110.0
  proj_margin_ratio: 3.0
  min_proj_margin: 450.0

panel_normal_interval: 10
```

## 目录结构

```
ros2_ws/
├── README.md
├── .gitignore
└── src/
    ├── OrbbecSDK_ROS2/          ← 官方相机驱动
    └── panel_detection/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   ├── camera.launch.py           ← 启动相机
        │   └── panel_detection.launch.py  ← 启动检测节点
        └── panel_detection/
            ├── __init__.py
            ├── panel_detect_node.py       ← 检测主节点
            ├── target_registry.py         ← 目标注册与跟踪
            ├── nut_localizer.py           ← nut 外六角 refined 定位
            ├── camera/
            │   ├── base.py
            │   ├── orbbec.py
            │   └── realsense.py
            ├── depth_utils.py
            ├── detector_onnx.py
            ├── detector_rknn.py
            ├── knob_angle.py
            ├── 0624.onnx
            └── 0624.pt
```

## 常见问题

**Q: 启动报 `No device found`**

相机未连接或权限未配置。检查 USB 连接和 udev 规则。

**Q: 启动报 `No module named 'onnxruntime'`**

```bash
sudo pip3 install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: numpy 版本冲突 (`_ARRAY_API not found`)**

```bash
sudo pip3 install "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 检测节点报等待相机话题**

确保先启动相机：`ros2 launch panel_detection camera.launch.py`

**Q: 注册一直无法完成**

检查是否同时检测到 5 个 button + 2 个 knob，可能需要调整相机角度或检测阈值。

## 许可证

Apache 2.0
