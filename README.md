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
检测节点按任务 launch 拆分
    ├── panel_controls.launch.py   → /panel/targets, /panel/knob_angles
    ├── valve_detection.launch.py  → /valve/targets, /valve/geometry
    └── fastener_detection.launch.py → /fasteners/targets, /fasteners/geometry
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

# 铭牌 OCR（可选，不装则 OCR 功能自动跳过）
sudo pip3 install rapidocr-onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

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

# 终端2：按任务启动检测节点
ros2 launch panel_detection panel_controls.launch.py
```

启动后检测节点会弹出可视化窗口。`panel_controls.launch.py` 会先进入**注册阶段**（积累稳定帧为按钮/旋钮分配编号），注册完成后进入**跟踪阶段**（发布带编号的检测结果）。`valve_detection.launch.py` 直接发布阀门结果；`fastener_detection.launch.py` 会对同一安装面的螺栓/螺母建立局部 4 槽位编号。按 `q` 或 `ESC` 退出。

## 启动方式与参数配置

### 常用启动命令

**推荐方式：相机节点 + 检测节点分开启动**

```bash
# 终端1：启动 Orbbec Gemini 336，相机话题深度已注册到彩色图
ros2 launch panel_detection camera.launch.py

# 终端2：面板旋钮/按钮
ros2 launch panel_detection panel_controls.launch.py

# 或：阀门
ros2 launch panel_detection valve_detection.launch.py

# 或：螺栓/螺母
ros2 launch panel_detection fastener_detection.launch.py
```

**直连模式：检测节点直接打开相机**

```bash
ros2 launch panel_detection panel_detection.launch.py
```

直连模式会由检测节点自己连接相机，并转发 `/camera/color/image_raw`、`/camera/depth/image_raw` 等话题。现场调试更推荐使用 `use_topic:=true`，相机和检测解耦，bag 回放也使用同一条路径。

`panel_detection.launch.py` 保留为兼容入口，默认 `detection_mode:=all`。新流程推荐使用三个任务 launch，避免不同器件的话题混在一起。

旧版本常用启动指令仍可使用：

```bash
ros2 launch panel_detection panel_detection.launch.py use_topic:=true use_constraint:=1
```

该指令使用兼容 all 模式，会同时处理面板、阀门、螺栓和螺母；新下游建议改用上面的任务 launch。

**Bag 回放验证**

```bash
# 终端1：启动检测节点
ros2 launch panel_detection valve_detection.launch.py

# 终端2：回放已录制的相机话题
ros2 bag play panel_test_bag_2 \
  --topics /camera/color/image_raw /camera/depth/image_raw /camera/color/camera_info /camera/depth/camera_info \
  --start-offset 20 \
  -r 0.5
```

当画面中同时出现完整阀门和 AprilTag 参考板时，检测节点会把阀门轴线与参考板深度平面法向量的对比结果写入 `axis_reference_log.jsonl`。每行包含 ROS 时间戳、两组法向量、夹角误差、参考板 RANSAC 内点比例和残差，可用来回到 bag 对应时间点排查误差帧。阀门 bbox 贴近图像边缘时视为不完整，不参与轴线精度统计。

**Bag 彩色图/深度图同步查看与截图**

```bash
# 终端1：启动同步可视化，S 保存当前对比图，Q 或 Esc 退出
ros2 run panel_detection rgb_depth_viewer

# 终端2：播放 bag
ros2 bag play /path/to/bag \
  --topics /camera/color/image_raw /camera/depth/image_raw
```

窗口左侧是彩色图，右侧是同一时间戳附近的深度伪彩色图，顶部显示两帧的时间差。默认只配对时间差不超过 20 ms 的图像，深度显示范围为 0.2-3.0 m，截图保存到当前目录的 `rgb_depth_captures/`。需要自动保存第一组同步帧时：

```bash
ros2 run panel_detection rgb_depth_viewer --ros-args \
  -p save_once:=true \
  -p output_dir:=/home/ztl/project/panel_ws/rgb_depth_captures \
  -p min_depth_m:=0.2 \
  -p max_depth_m:=2.0
```

`16UC1`/`mono16` 深度默认按毫米解释（`depth_scale:=0.001`），`32FC1` 按米解释。彩色与深度分辨率不一致时仅缩放深度显示；要逐像素对照，录包和回放时应使用已注册到彩色相机坐标系的深度话题。
无桌面环境只需要自动截图时可再传入 `-p show_window:=false`。

**旋钮角度模式**

```bash
ros2 launch panel_detection panel_controls.launch.py use_constraint:=1
```

`use_constraint:=1` 为默认模式：只根据旋钮白色手柄线与竖直线的夹角，稳定输出 `0` 或 `90`。旧写法仍兼容：`true` 等价于 `2`，`false` 等价于 `3`。

**离线管道轴线估计脚本**

给定一张图像和一个模拟/实测漏点像素坐标，可以用独立脚本估计漏点附近的 2D 管道轴线方向。该脚本不在任何 launch 中启动。

```bash
PYTHONPATH=src/panel_detection \
src/panel_detection/scripts/estimate_pipe_axis.py \
  --image bag9_valve_angle_review_after2/frames/050_1782718665566346318.jpg \
  --point 660,322 \
  --output pipe_axis_demo/script_pipe_axis_demo.jpg \
  --angle-prior-deg 0 \
  --angle-tolerance-deg 35 \
  --consensus-tolerance-deg 18
```

输出角度定义为相对图像 x 轴的角度，图像坐标系 y 轴向下；轴线本身有 180 度二义性，因此 `v` 和 `-v` 表示同一条管道轴线。

当前脚本在 `bag9_valve_angle_review_after2/frames` 的 100 帧样本上做过离线验证：根据阀门 bbox 推出旁侧管道模拟漏点，97/100 帧能输出轴线，复核图中箭头基本贴合漏点附近可见管道。3 帧未输出是局部 ROI 内没有足够可靠的管道边缘线段。复核输出示例位于本地调试目录 `pipe_axis_demo/batch100_consensus/review_contact.jpg`。

### Launch 参数

| 参数 | 默认值 | 作用 | 何时修改 |
|------|--------|------|----------|
| `use_topic` | 任务 launch 为 `true`，兼容 launch 为 `false` | `true` 时订阅 `/camera/*` 话题；`false` 时检测节点直连相机 | 使用 `camera.launch.py` 或 bag 回放时保持 `true` |
| `registered_depth` | `true` | topic 模式下深度图是否已对齐到彩色图 | Orbbec `camera.launch.py` 已设置 `depth_registration=true`，保持 `true` |
| `use_constraint` | `1` | 面板 launch 的旋钮角度模式：`1`=0/90 稳定输出，`2`=旧物理范围约束，`3`=旧无约束；`use_constrain` 也可作为别名 | 默认保持 `1`；需要旧行为时设为 `2` 或 `3` |
| `config_path` | 空字符串 | 外部 YAML 配置文件路径；为空使用默认配置 | 需要换模型、阈值、相机后端、推理后端时使用 |
| `detection_mode` | `all` | `panel_detection.launch.py` 的兼容模式选择：`all` / `panel_controls` / `valve` / `fastener` | 新流程通常不用手动设置，三个任务 launch 已固定 |
| `publish_legacy_topics` | `false` | 是否发布 `/panel/valves`、`/panel/bolts` 等旧 PoseStamped 兼容话题 | 旧下游仍订阅这些话题时才打开 |
| `capture_dir` | 空字符串 | 非空时按 ROS 图像时间戳保存完整检测 canvas | bag 批量复核或制作检测截图时设置 |
| `capture_hz` | `1.0` | `capture_dir` 启用时的输入采样和画面保存频率 | 需要其他采样频率时修改 |
| `show_gui` | `true` | 是否显示 OpenCV 检测窗口 | 无桌面批处理时设为 `false` |

### 指定配置文件

```bash
ros2 launch panel_detection valve_detection.launch.py \
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

本节只适用于 `panel_controls.launch.py`。阀门和螺栓/螺母 launch 不做面板编号注册。

### 注册阶段

面板检测节点启动后自动进入注册阶段：
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
      "label": "自锁按钮2",
      "position": {"x": 0.123, "y": -0.045, "z": 0.850},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.92
    },
    {
      "id": 4,
      "class": "knob",
      "label": "旋钮开关",
      "position": {"x": 0.320, "y": -0.040, "z": 0.845},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.88
    }
  ]
}
```

位置单位：**米 (m)**，相机坐标系。`label` 字段在铭牌 OCR 确认后出现，表示器件上方金属铭牌的识别文字。

### /panel/knob_angles (String, JSON) — 面板旋钮角度

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

面板定义中只有按钮和旋钮；该话题只发布旋钮角度，`knob_angles` 内每个条目的格式保持不变。旋钮 `angle` 以 12 点钟方向为 0°，顺时针增加，范围 [0, 360)。

### /valve/targets (String, JSON) — 阀门位置

`valve_detection.launch.py` 只发布阀门目标：

```json
{
  "stamp": 1716192000.123,
  "targets": [
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "position": {"x": 0.18, "y": -0.03, "z": 0.82},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.88
    }
  ]
}
```

### /valve/geometry (String, JSON) — 阀门角度与轴线方向

`valve_detection.launch.py` 只发布阀门几何，字段结构如下：

```json
{
  "stamp": 1716192000.123,
  "object_angles": [
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "hex_angle": 12.5,
      "valve_angle": 12.5
    }
  ],
  "axis_directions": [
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "source": "valve_wheel",
      "axis_direction": [0.012, -0.034, -0.999],
      "centroid": [0.18, -0.03, 0.82],
      "point_count": 742
    }
  ]
}
```

调试时如果画面中有 AprilTag 参考板，该话题会额外带 `axis_reference` 字段。

### /fasteners/targets (String, JSON) — 螺栓/螺母位置

`fastener_detection.launch.py` 只发布螺栓和螺母目标，`class` 为 `bolt` 或 `nut`。当同一安装面至少 3 个 fastener 被可靠检测后，节点会建立 4 槽位模板，并在后续帧给已检测到的目标附加稳定编号：

```json
{
  "stamp": 1716192000.123,
  "targets": [
    {
      "id": 2,
      "group_id": 1,
      "slot": "top_right",
      "registered": true,
      "class": "bolt",
      "bbox": [510.0, 260.0, 560.0, 310.0],
      "position": {"x": 0.08, "y": -0.04, "z": 0.33},
      "confidence": 0.74
    }
  ]
}
```

编号定义为每个 `group_id` 内的局部槽位：`1=top_left`、`2=top_right`、`3=bottom_right`、`4=bottom_left`。机械臂指定目标时应使用 `group_id + id`，不要只使用单独的 `id`。首次冷启动只有 1-2 个 fastener 可见时不会强行猜最终编号；模板注册后，即使后续只检测到部分目标，也会按历史槽位继续编号。

### /fasteners/geometry (String, JSON) — 螺栓/螺母角度与轴线方向

`fastener_detection.launch.py` 只发布 `class=bolt` / `class=nut` 的条目：

```json
{
  "stamp": 1716192000.123,
  "object_angles": [
    {
      "id": 3,
      "group_id": 1,
      "slot": "bottom_right",
      "class": "nut",
      "bbox": [510.0, 260.0, 560.0, 310.0],
      "hex_angle": 28.4,
      "nut_refined_conf": 0.81
    }
  ],
  "axis_directions": [
    {
      "id": 3,
      "group_id": 1,
      "slot": "bottom_right",
      "class": "nut",
      "bbox": [510.0, 260.0, 560.0, 310.0],
      "source": "fastener_current",
      "axis_direction": [0.012, -0.034, -0.999],
      "centroid": [0.18, -0.03, 0.82],
      "point_count": 4
    }
  ]
}
```

### /objects/geometry (String, JSON) — 兼容 all 模式对象几何

```json
{
  "stamp": 1716192000.123,
  "object_angles": [
    {
      "class": "nut",
      "bbox": [510.0, 260.0, 560.0, 310.0],
      "hex_angle": 28.4,
      "nut_refined_conf": 0.81
    },
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "hex_angle": 12.5,
      "valve_angle": 12.5
    }
  ],
  "axis_directions": [
    {
      "class": "valve",
      "bbox": [620.0, 240.0, 700.0, 320.0],
      "source": "valve_wheel",
      "axis_direction": [0.012, -0.034, -0.999],
      "centroid": [0.18, -0.03, 0.82],
      "point_count": 742
    }
  ]
}
```

`/objects/geometry` 只在兼容入口 `panel_detection.launch.py detection_mode:=all` 中发布。新流程中，阀门使用 `/valve/geometry`，螺栓/螺母使用 `/fasteners/geometry`。

螺栓/螺母 `hex_angle` 是六边形相对画面水平线的对称角，范围 [0, 60)。阀门 `valve_angle` 使用红色十字/外八边形角点方向估计，是相对画面水平线的对称角，范围 [0, 45)；为兼容旧字段，同一数值也放在 `hex_angle` 中。`axis_direction` 是相机坐标系下的单位方向向量，指向相机方向时 z 为负。阀门只使用手轮自身环形深度点拟合轴线，`source` 为 `valve_wheel`；质量门控失败或阀门贴边不完整时不输出阀门轴线，不使用周围安装平面或 bbox 整体深度回退。螺栓、螺母优先使用当前帧附近同平面器件的 3D 点拟合安装平面，`source` 为 `fastener_current`；如果只有两个邻近点，则用两点连线约束局部安装面法向量，`source` 为 `fastener_line`；再往后使用目标外侧深度连续 patch 拟合局部安装面，`source` 为 `local_patch_plane`，该结果会按 z 分量、RANSAC 内点比例和残差做质量门控；最后才回退到全局面板平面 `panel_plane` 和局部目标深度 `local_depth`。调试时如果画面中有 AprilTag 板，节点会额外输出 `axis_reference`，用 tag 内部深度平面法向量和阀门轴线计算 `reference_angle_deg`，该参考只用于精度评估，不参与正式阀门轴线估计。

### /panel/labels (String, JSON) — 铭牌文字标签

`panel_controls.launch.py` 在 OCR 确认铭牌文字后发布：

```json
{
  "stamp": 1716192000.123,
  "labels": [
    {"id": 1, "label": "自锁按钮2"},
    {"id": 4, "label": "旋钮开关"}
  ]
}
```

铭牌 OCR 使用 RapidOCR (ONNX Runtime) 识别器件上方的金属铭牌文字，多帧投票确认后输出。确认后低频发布（~1Hz），不影响检测帧率。需要安装 `rapidocr-onnxruntime`；未安装时该功能静默跳过。可通过配置 `nameplate_ocr.enable: false` 关闭。

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

### 状态话题 (String)

| Launch | 状态话题 |
|--------|----------|
| `panel_controls.launch.py` | `/panel/status` |
| `valve_detection.launch.py` | `/valve/status` |
| `fastener_detection.launch.py` | `/fasteners/status` |

常见状态：

- `"waiting_camera"` — 等待相机
- `"waiting_topic_frames"` — 等待同步后的相机话题
- `"no_detection"` — 当前帧没有对应模式的检测结果
- `"no_targets"` — 有检测但没有可发布目标
- `"registered"` — 当前模式已有可发布目标；面板模式也表示编号结果可用

### 兼容旧话题 (PoseStamped)

旧 PoseStamped 话题默认不发布。需要兼容旧下游时，在任一检测 launch 中加入 `publish_legacy_topics:=true`。

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
ros2 topic echo /panel/labels
ros2 topic echo /valve/geometry
ros2 topic echo /valve/targets
ros2 topic echo /valve/status
ros2 topic echo /fasteners/geometry
ros2 topic echo /fasteners/targets
ros2 topic echo /fasteners/status
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
        self.create_subscription(String, '/valve/geometry', self.geometry_cb, 10)
        self.create_subscription(String, '/fasteners/geometry', self.geometry_cb, 10)

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

    def geometry_cb(self, msg):
        data = json.loads(msg.data)
        for axis in data['axis_directions']:
            n = axis['axis_direction']
            self.get_logger().info(
                f"{axis['class']} 轴线: ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})")
```

## 配置文件说明

创建 YAML 文件后通过 `config_path` 参数传入：

```bash
ros2 launch panel_detection panel_controls.launch.py config_path:=/path/to/config.yaml
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
onnx_model: '0807.onnx'        # 相对 panel_detection 包目录，或填写绝对路径
onnx_threads: 8
# rknn_model: '0630.rknn'      # inference_backend='rknn' 时使用

detection_mode: 'all'          # panel_detection.launch.py 兼容模式使用
publish_legacy_topics: false   # 是否发布旧 PoseStamped 兼容话题

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

fastener_registry:
  min_init_observations: 3      # 首次建立 4 槽位模板所需的最少可见 fastener 数
  max_group_distance_m: 0.35    # 同组聚类的 3D 距离门限
  max_slot_distance_m: 0.12     # 历史槽位匹配的最小距离门限
  slot_match_ratio: 0.45        # 槽距比例门限，和 max_slot_distance_m 取较大值
  normal_angle_thresh_deg: 25.0 # 同组/同槽位法向量夹角门限
  ema_alpha: 0.35               # 槽位 3D 位置平滑系数
  stale_frames: 120             # 多久未观测后删除该 fastener 组

nameplate_ocr:
  enable: true                 # 铭牌文字识别总开关
  confirm_frames: 3            # 同一文字连续出现 N 帧后确认
  ocr_interval: 10             # 每隔 N 帧执行一次 OCR
  roi_above_ratio: 0.6         # 铭牌 ROI 高度 = 器件高度 × ratio
  roi_gap_ratio: 0.1           # 铭牌与器件间隙 = 器件高度 × ratio
  roi_width_ratio: 2.8         # 铭牌 ROI 宽度 = 器件宽度 × ratio
  roi_min_height: 15           # ROI 最小高度（像素）
  use_gpu: false
  lang: 'ch'

panel_normal_interval: 10
```

## 目录结构

```
ros2_ws/
├── README.md
├── .gitignore
└── src/
    ├── OrbbecSDK_ROS2/          ← 官方相机驱动 (submodule)
    └── panel_detection/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   ├── camera.launch.py           ← 启动相机
        │   ├── panel_controls.launch.py   ← 面板旋钮/按钮
        │   ├── valve_detection.launch.py  ← 阀门
        │   ├── fastener_detection.launch.py ← 螺栓/螺母
        │   └── panel_detection.launch.py  ← 兼容 all 模式
        ├── scripts/
        │   ├── estimate_pipe_axis.py      ← 离线管道轴线估计
        │   └── export_model.py            ← 模型转换 (.pt → .onnx)
        ├── test/
        │   ├── test_apriltag_reference.py
        │   └── test_topic_sync.py
        └── panel_detection/
            ├── __init__.py
            ├── panel_detect_node.py       ← 检测主节点
            ├── target_registry.py         ← 目标注册与跟踪
            ├── fastener_registry.py       ← 螺栓/螺母分组与槽位编号
            ├── nameplate_ocr.py           ← 铭牌文字识别 (OCR)
            ├── nut_localizer.py           ← nut 外六角 refined 定位
            ├── pipe_axis.py               ← 管道轴线方向估计
            ├── apriltag_reference.py      ← AprilTag 参考轴线
            ├── camera/
            │   ├── base.py
            │   ├── orbbec.py
            │   └── realsense.py
            ├── depth_utils.py
            ├── detector_onnx.py
            ├── detector_rknn.py
            ├── knob_angle.py
            ├── rgb_depth_viewer.py        ← 彩色/深度同步查看工具
            ├── 0630.onnx
            ├── 0630.pt
            ├── 0727.onnx                  ← 备用权重
            ├── 0727.pt
            ├── 0807.onnx                  ← 默认权重
            └── 0807.pt
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
