# ROS2 面板检测功能包 (RK3588 + Orbbec Gemini 336)

基于 YOLOv5 + 深度相机的操作面板实时 3D 检测系统，封装为标准 ROS2 Humble 功能包。

检测 7 类目标：指示灯(light)、旋钮(knob)、螺栓(bolt)、螺母(nut)、阀门(valve)、泵(pump)、按钮(button)，输出每个目标的 3D 位姿(PoseStamped)和旋钮角度。

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
# 设置 locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# 添加 ROS2 源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装
sudo apt update
sudo apt install ros-humble-ros-base ros-humble-vision-msgs python3-colcon-common-extensions -y

# 写入 bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 第二步：克隆仓库

```bash
mkdir -p ~/ros2_ws
cd ~/ros2_ws
git clone git@github.com:Tsumugi-W/detect_3588.git .
```

或者如果已有代码，直接把 `src/panel_detection` 放入你的 ROS2 工作空间的 `src/` 下。

### 第三步：安装 Python 依赖

```bash
cd ~/ros2_ws
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 核心依赖
pip install onnxruntime
pip install opencv-python
pip install numpy
pip install pyyaml

# 相机 SDK (Orbbec)
pip install pyorbbecsdk2

# 如果使用 RealSense 相机则安装:
# pip install pyrealsense2

# 模型导出时需要 (运行时不需要):
# pip install torch==2.7.0 torchvision==0.22.0 onnx
```

### 第四步：配置相机权限

**Orbbec 相机：**

```bash
source .venv/bin/activate
sudo bash $(python3 -c "import pyorbbecsdk,os; print(os.path.dirname(pyorbbecsdk.__file__))")/shared/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

配置完成后需要**重新插拔相机** USB 线。

**RealSense 相机：**

```bash
sudo apt install ros-humble-librealsense2* -y
```

### 第五步：编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source .venv/bin/activate
colcon build --packages-select panel_detection
source install/setup.bash
```

### 第六步：运行

```bash
# 确保环境已 source (每次新终端都需要)
source /opt/ros/humble/setup.bash
source ~/ros2_ws/.venv/bin/activate
source ~/ros2_ws/install/setup.bash

# 启动节点
python3 -m panel_detection.panel_detect_node
```

启动后会弹出可视化窗口，显示检测框、3D 坐标、旋钮角度。按 `q` 或 `ESC` 退出。

### 可选：写入 bashrc 简化启动

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/.venv/bin/activate" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

之后每次打开终端直接运行：

```bash
python3 -m panel_detection.panel_detect_node
```

## 发布话题

每检测到一个目标，在对应类别话题上发布一条 `geometry_msgs/PoseStamped`。

| 话题 | 类别 | 消息类型 |
|------|------|----------|
| `/panel/lights` | 指示灯 | PoseStamped |
| `/panel/knobs` | 旋钮 | PoseStamped |
| `/panel/buttons` | 按钮 | PoseStamped |
| `/panel/bolts` | 螺栓 | PoseStamped |
| `/panel/nuts` | 螺母 | PoseStamped |
| `/panel/valves` | 阀门 | PoseStamped |
| `/panel/pumps` | 泵 | PoseStamped |
| `/panel/knob_angles` | 旋钮角度 | String (JSON) |

### PoseStamped 格式

```
header:
  stamp: {sec: ..., nanosec: ...}
  frame_id: "camera_color_optical_frame"
pose:
  position:
    x: 0.123    # 相机坐标系 X (米)
    y: -0.045   # 相机坐标系 Y (米)
    z: 0.850    # 深度 (米)
  orientation:
    x: 0.01     # 面板法向量四元数
    y: 0.02
    z: 0.0
    w: 0.99
```

### 旋钮角度格式 (/panel/knob_angles)

```json
{
  "stamp": 1716192000.123,
  "knob_angles": [
    {
      "position": {"x": 0.12, "y": -0.05, "z": 0.83},
      "angle": 171.5,
      "confidence": 0.92
    }
  ]
}
```

角度以 12 点钟方向为 0 度，顺时针增加，范围 [0, 360)。

## 验证话题

```bash
# 另开终端
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 查看所有活跃话题
ros2 topic list

# 查看旋钮位姿
ros2 topic echo /panel/knobs

# 查看旋钮角度
ros2 topic echo /panel/knob_angles

# 查看发布频率
ros2 topic hz /panel/knobs
```

## 在其他节点中订阅

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import json


class PanelSubscriber(Node):
    def __init__(self):
        super().__init__('panel_subscriber')
        self.create_subscription(PoseStamped, '/panel/knobs', self.knob_cb, 10)
        self.create_subscription(String, '/panel/knob_angles', self.angle_cb, 10)

    def knob_cb(self, msg):
        p = msg.pose.position
        self.get_logger().info(f'旋钮位置: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

    def angle_cb(self, msg):
        data = json.loads(msg.data)
        for k in data['knob_angles']:
            self.get_logger().info(f"角度: {k['angle']}°, 置信度: {k['confidence']}")
```

## 配置说明

节点启动时默认使用内置配置。如需自定义，创建 yaml 文件并通过参数传入：

```bash
python3 -m panel_detection.panel_detect_node --ros-args -p config_path:=/path/to/config.yaml
```

配置文件示例：

```yaml
# 相机
camera_backend: 'orbbec'       # 'orbbec' | 'realsense'
camera:
  color_width: 1280
  color_height: 720
  depth_width: 1280
  depth_height: 720
  fps: 30

# 推理
inference_backend: 'onnx'      # 'onnx' | 'rknn'
onnx_model: '0520.onnx'
onnx_threads: 4

# 模型
weight: '0520.pt'
input_size: 640
class_num: 7
class_name: ['light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button']
threshold:
  confidence: 0.3
  iou: 0.01

# 旋钮角度
knob_angle:
  enable: true
  binary_thresh: 180
  circle_mask_ratio: 0.85
  knob_class: 'knob'
```

## 目录结构

```
ros2_ws/
├── README.md
├── .gitignore
└── src/
    └── panel_detection/            # ROS2 功能包
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── resource/panel_detection
        └── panel_detection/        # 源码 + 模型 (自包含)
            ├── __init__.py
            ├── panel_detect_node.py    # 主节点
            ├── camera/                 # 相机抽象层
            │   ├── __init__.py
            │   ├── base.py
            │   ├── orbbec.py
            │   └── realsense.py
            ├── depth_utils.py          # 深度处理工具
            ├── detector_onnx.py        # ONNX 推理器
            ├── detector_rknn.py        # RKNN 推理器
            ├── knob_angle.py           # 旋钮角度估计
            ├── 0520.onnx               # ONNX 模型 (部署用)
            └── 0520.pt                 # PyTorch 权重 (导出用)
```

## 常见问题

**Q: 启动报 `No device found`**

相机未连接或权限未配置。检查：
- USB 线是否插好
- 是否执行了 udev 规则安装
- 是否重新插拔了相机

**Q: 启动报 `No module named 'onnxruntime'`**

Python 环境未正确激活。确保：
```bash
source ~/ros2_ws/.venv/bin/activate
```

**Q: 检测帧率太低**

- 降低相机分辨率（640x480 比 1280x720 快一倍）
- 调整 `onnx_threads` 参数
- 未来可切换到 RKNN NPU 推理（~30FPS）

**Q: 如何切换到 RealSense 相机**

在配置文件中修改：
```yaml
camera_backend: 'realsense'
```

## 许可证

Apache 2.0
