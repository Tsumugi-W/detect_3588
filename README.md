# ROS2 面板检测功能包 (RK3588 + Orbbec Gemini 336)

基于 YOLOv5 + 深度相机的操作面板实时 3D 检测系统，封装为标准 ROS2 Humble 功能包。

检测 7 类目标：指示灯(light)、旋钮(knob)、螺栓(bolt)、螺母(nut)、阀门(valve)、泵(pump)、按钮(button)。支持目标注册编号和旋钮角度估计。

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
# 终端1：启动相机
ros2 launch panel_detection camera.launch.py

# 终端2：启动检测节点
ros2 launch panel_detection panel_detection.launch.py
```

启动后检测节点会弹出可视化窗口。启动后先进入**注册阶段**（积累稳定帧为目标分配编号），注册完成后进入**跟踪阶段**（发布带编号的检测结果）。按 `q` 或 `ESC` 退出。

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

### /panel/knob_angles (String, JSON) — 旋钮角度

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
  ]
}
```

角度以 12 点钟方向为 0°，顺时针增加，范围 [0, 360)。

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

## 配置说明

创建 yaml 文件通过参数传入：

```bash
ros2 launch panel_detection panel_detection.launch.py config_path:=/path/to/config.yaml
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
onnx_model: '0520.onnx'
onnx_threads: 4

class_num: 7
class_name: ['light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button']
threshold:
  confidence: 0.3
  iou: 0.01

knob_angle:
  enable: true

depth_scale: 0.001             # 深度图原始值 × depth_scale = 米

registry:
  stable_frames: 15            # 注册需要的连续稳定帧数
  green_hue_range: [35, 85]    # 绿色按钮 HSV 色调范围
  match_distance_thresh: 80    # 匹配最大像素距离
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
            ├── camera/
            │   ├── base.py
            │   ├── orbbec.py
            │   └── realsense.py
            ├── depth_utils.py
            ├── detector_onnx.py
            ├── detector_rknn.py
            ├── knob_angle.py
            ├── 0520.onnx
            └── 0520.pt
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
