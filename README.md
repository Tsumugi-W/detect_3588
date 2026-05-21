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
sudo apt update
sudo apt install ros-humble-ros-base ros-humble-vision-msgs python3-colcon-common-extensions -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 第二步：克隆仓库

```bash
mkdir -p ~/ros2_ws
cd ~/ros2_ws
git clone git@github.com:Tsumugi-W/detect_3588.git .
```

### 第三步：安装 Python 依赖（系统级）

```bash
# 清华镜像加速
sudo pip3 install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install pyorbbecsdk2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# opencv、pyyaml 通常系统已自带，如果没有：
# sudo pip3 install opencv-python pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果使用 RealSense 相机：
# sudo pip3 install pyrealsense2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第四步：配置相机权限

**Orbbec 相机：**

```bash
sudo bash $(python3 -c "import pyorbbecsdk,os; print(os.path.dirname(pyorbbecsdk.__file__))")/shared/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

配置完成后**重新插拔相机** USB 线。

### 第五步：编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select panel_detection
source install/setup.bash
```

### 第六步：运行

```bash
ros2 launch panel_detection panel_detection.launch.py
```

启动后会弹出可视化窗口，显示检测框、3D 坐标、旋钮角度。按 `q` 或 `ESC` 退出。

### 可选：写入 bashrc 简化日常使用

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

之后每次打开终端直接：

```bash
ros2 launch panel_detection panel_detection.launch.py
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
ros2 topic list
ros2 topic echo /panel/knobs
ros2 topic echo /panel/knob_angles
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
```

## 目录结构

```
ros2_ws/
├── README.md
├── .gitignore
└── src/
    └── panel_detection/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── launch/
        │   └── panel_detection.launch.py
        ├── resource/panel_detection
        └── panel_detection/
            ├── __init__.py
            ├── panel_detect_node.py
            ├── camera/
            │   ├── __init__.py
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

**Q: 如何切换到 RealSense 相机**

通过配置文件或直接改 `panel_detect_node.py` 中 `DEFAULT_CONFIG` 的 `camera_backend` 为 `'realsense'`。

## 许可证

Apache 2.0
