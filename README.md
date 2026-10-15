# ROS2 面板检测功能包 (RK3588 + Orbbec Gemini 336)

## 快速启动：面板 / 螺栓检测

**每个终端先加载环境**（当前机器工作空间为 `~/project/panel_ws`）：

```bash
cd ~/project/panel_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

**终端 1：启动相机**。相机话题已经运行时可跳过。

```bash
ros2 launch panel_detection camera.launch.py
```

**终端 2：启动面板检测（按钮、旋钮、灯、板卡）**：

```bash
ros2 launch panel_detection panel_controls.launch.py
```

**终端 3：启动螺栓 / 螺母检测**：

```bash
ros2 launch panel_detection fastener_detection.launch.py show_gui:=true
```

两个检测节点可订阅同一套相机话题。面板默认弹出 OpenCV 窗口；上面的紧固件命令通过 `show_gui:=true` 开启窗口，紧固件 launch 默认不显示窗口。按 `q` 或 `Esc` 退出。
SSH 启动并希望窗口出现在这台机器已登录的显示器上时，先在该终端执行：

```bash
export DISPLAY=:0
export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
```

上述显示设置适用于当前机器的 GNOME/X11 会话；其他桌面需使用对应的 DISPLAY 和授权文件。
无桌面时可使用通用入口关闭窗口：

```bash
ros2 launch panel_detection panel_detection.launch.py use_topic:=true detection_mode:=panel_controls show_gui:=false
```

面板结果：`/panel/targets`、`/panel/board_cards`；紧固件结果：`/fasteners/targets`、`/fasteners/geometry`。
ROS1 命令见 [ROS1 README](ros1/panel_detection/README.md#快速启动面板--螺栓检测)。

基于 YOLOv5 + 深度相机的操作面板实时 3D 检测系统，封装为标准 ROS2 Humble 功能包。

检测 10 类目标：指示灯(light)、旋钮(knob)、螺栓(bolt)、螺母(nut)、阀门(valve)、泵(pump)、按钮(button)、门按钮(door_button)、空气开关(air_switch)、板卡(board_card)。支持目标注册编号、旋钮角度估计、螺栓/螺母/阀门轴线方向估计，以及管道漏点检测、管道轴线方向估计和 leak_bolt（加固螺栓）定位与位姿估计。

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
    ├── panel_controls.launch.py     → /panel/targets, /panel/board_cards, /panel/knob_angles
    ├── valve_detection.launch.py    → /valve/targets, /valve/geometry
    ├── fastener_detection.launch.py → /fasteners/targets, /fasteners/geometry
    └── leak_detection.launch.py     → /leak/targets, /leak/pipe_axis
```

## 平台要求

| 项目 | 要求 |
|------|------|
| 系统 | Ubuntu 22.04 (aarch64 或 x86_64) |
| ROS2 | Humble |
| Python | 3.10+ |
| 相机 | 奥比中光 Gemini 336 (默认) 或 Intel RealSense D435i |
| 推理 | ONNX Runtime CPU (默认) 或 RKNN NPU |

## ROS1 Noetic 版本

仓库中的 `ros1/panel_detection` 是当前算法的 ROS1 Noetic catkin 版本，默认面向
Intel RealSense D435 的彩色图、对齐深度图和彩色相机内参，使用同一份
`1013.onnx` 十类别权重和 `leak_bolt.onnx` 漏点螺栓权重。该版本保留面板、阀门、
螺栓/螺母和漏点管道四个独立检测模式及对应 JSON 话题，并通过 `/panel/debug_image`
等图像话题发布 Canvas；无显示环境默认不创建 OpenCV 窗口。

```bash
mkdir -p ~/panel_ros1_ws/src
cp -a ros1/panel_detection ~/panel_ros1_ws/src/
# 更新已有工作区时清除旧版本遗留的忽略标记
rm -f ~/panel_ros1_ws/src/panel_detection/COLCON_IGNORE
cd ~/panel_ros1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# 相机已单独启动时，按任务选择一个 launch
roslaunch panel_detection panel_controls.launch
roslaunch panel_detection valve_detection.launch
roslaunch panel_detection fastener_detection.launch
roslaunch panel_detection leak_detection.launch
```

完整的 ROS1 输入话题、输出话题和 D435 联合启动方式见
`ros1/panel_detection/README.md`。

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

源码压缩包已包含 Orbbec 驱动及 SDK，解压后可直接从安装系统依赖开始，
无需执行 Git 子模块初始化。

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

# 终端2：按任务启动检测节点
ros2 launch panel_detection panel_controls.launch.py
```

启动后检测节点会弹出可视化窗口。`panel_controls.launch.py` 同时运行 YOLO 和 tag36h11 检测，默认使用验证场配置：T02/T03 对应旋钮，T00/T01/T04/T05/T06 对应按钮，T07 仅给 YOLO 板卡关联编号 8。板卡类别始终来自 YOLO；验证场默认关闭旧布局编号回退。`valve_detection.launch.py` 直接发布阀门结果；`fastener_detection.launch.py` 会对同一安装面的螺栓/螺母建立局部 4 槽位编号。按 `q` 或 `ESC` 退出。

## 启动方式与参数配置

### 快捷启动脚本

仓库根目录提供 `run.sh` 快捷脚本，省去记忆 launch 文件名：

```bash
./run.sh camera              # 启动相机
./run.sh panel               # 面板检测 (light/knob/button/board_card)
./run.sh fastener            # 螺栓螺帽检测
./run.sh valve               # 阀门检测
./run.sh leak                # 漏点+管道+leak_bolt 检测
./run.sh all                 # 全部模式
./run.sh leak --config x.yaml  # 带自定义配置
./run.sh fastener --direct   # 直连相机模式
./run.sh record my_bag       # 录制 rosbag
./run.sh play bags/xx --rate 0.5  # 回放
./run.sh build               # 编译
./run.sh --help              # 查看全部命令
```

### 常用启动命令

**推荐方式：相机节点 + 检测节点分开启动**

```bash
# 终端1：启动 Orbbec Gemini 336，相机话题深度已注册到彩色图
ros2 launch panel_detection camera.launch.py

# 终端2：面板按钮/旋钮/指示灯
ros2 launch panel_detection panel_controls.launch.py

# 或：阀门
ros2 launch panel_detection valve_detection.launch.py

# 或：螺栓/螺母
ros2 launch panel_detection fastener_detection.launch.py

# 或：管道漏点
ros2 launch panel_detection leak_detection.launch.py
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

**管道漏点检测**

```bash
# 终端2：管道漏点检测
ros2 launch panel_detection leak_detection.launch.py
```

Leak 模式工作流程：
1. 优先使用上游节点通过 `/leak/upstream_point` (PointStamped) 发布的 3D 漏点坐标，投影到图像平面
2. 上游漏点超过 2 秒未更新时，自动回退到黑色标记检测（在亮色管道上检测暗色方块）
3. 以漏点为中心，用 Canny + HoughLinesP 提取管道边缘线段，共识聚类后估计管道轴线方向
4. 使用专用 `leak_bolt.onnx` 模型检测加固螺栓（leak_bolt），计算 3D 位置和表面法向量，发布位姿信息供下游机械臂使用

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
| `detection_mode` | `all` | `panel_detection.launch.py` 的兼容模式选择：`all` / `panel_controls` / `valve` / `fastener` / `leak` | 新流程通常不用手动设置，各任务 launch 已固定 |
| `publish_legacy_topics` | `false` | 是否发布 `/panel/valves`、`/panel/bolts` 等旧 PoseStamped 兼容话题 | 旧下游仍订阅这些话题时才打开 |
| `capture_dir` | 空字符串 | 非空时按 ROS 图像时间戳保存完整检测 canvas | bag 批量复核或制作检测截图时设置 |
| `capture_hz` | `1.0` | `capture_dir` 启用时的输入采样和画面保存频率 | 需要其他采样频率时修改 |
| `show_gui` | 通用入口 `true`；紧固件入口 `false` | 是否显示 OpenCV 检测窗口 | 无桌面批处理时设为 `false` |
| `use_panel_tags` | `true` | 按场景关联 tag36h11；按钮/旋钮/灯可修正类别，板卡只关联编号 | 临时使用无 Tag 的旧面板时设为 `false` |

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

## 面板 AprilTag 分类与编号

本节只适用于 `panel_controls.launch.py`。阀门和螺栓/螺母 launch 不做面板编号注册。

面板元器件上方使用边长 20 mm 的 tag36h11，**默认场景是验证场**：

| Tag ID | 验证场类别 | 关联规则 | 下游目标编号 |
|--------|------------|----------|--------------|
| 00、01、04、05、06 | `button` | 成功关联后修正类别 | Tag ID + 1 |
| 02、03 | `knob` | 成功关联后修正类别 | 3、4 |
| 07 | `board_card` | 只关联 YOLO 板卡，不修改任何目标类别 | 8 |
| 08–39 | — | 不参与验证场关联 | — |

测试场 `test_site` 保留旧映射：00/01/02/05/06 为按钮，03/04 为旋钮，07–39 为灯。
通过 `config_path:=配置文件路径` 切换场景；详见“场景配置：测试场与验证场”。

每帧先运行 YOLO，再解码 Tag。关联使用 Tag 解码后的局部横、竖轴，
避免相机倾斜超过 45° 时交换行列；Tag 必须位于元件的局部上方，并通过收紧的横向距离、纵向间距和
一对一关联门控。板卡 Tag 只能关联同为 `board_card` 的 YOLO 结果；其他类别的 Tag 不能
给板卡改类或编号。短暂漏 Tag 可沿用有限帧跟踪，超过 `stale_frames` 后关联失效。

面板模式启用 Tag 识别时，每个最终分类为 `button` 的目标都必须在当前帧直接
关联到自己的 AprilTag。没有直接关联 Tag 的 YOLO `button` 会强制改为 `light`，
即使画面中其他位置存在 Tag，或者历史跟踪仍有该目标的 Tag ID，也不能保留为
`button`；面板行布局同样不能把未关联 Tag 的 `light` 提升为 `button`。该规则
不影响 `door_button`、`knob`、`board_card` 或其他任务模式；`use_panel_tags:=false` 时仍保留
旧的无 Tag 分类和布局回退行为。

验证场默认关闭布局编号回退；测试场开启。关闭 Tag 识别时可使用旧布局编号流程。板卡不参与按钮/旋钮的行布局拟合和回退编号。可视化中 `Txx` 是原始 Tag ID，`#n` 是发布给下游的目标 ID，连线表示 Tag 与元器件的关联。使用无 Tag 的旧面板时可启动：

```bash
ros2 launch panel_detection panel_controls.launch.py use_panel_tags:=false
```

## 发布话题

### /panel/targets (String, JSON) — 主话题

```json
{
  "stamp": 1716192000.123,
  "targets": [
    {
      "id": 1,
      "class": "button",
      "tag_id": 0,
      "classification_source": "apriltag_36h11",
      "label": "自锁按钮2",
      "position": {"x": 0.123, "y": -0.045, "z": 0.850},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.92
    },
    {
      "id": 4,
      "class": "knob",
      "tag_id": 3,
      "classification_source": "apriltag_track",
      "label": "旋钮开关",
      "position": {"x": 0.320, "y": -0.040, "z": 0.845},
      "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.99},
      "confidence": 0.88
    }
  ]
}
```

位置单位：**米 (m)**，相机坐标系。`tag_id` 为成功关联的原始 Tag 编号，无 Tag 回退时为 `null`。`classification_source` 对原有元件为 `apriltag_36h11`、`apriltag_track` 或 `yolo_layout`。板卡固定为 `yolo`，关联来源另见 `identity_source`；只有已关联编号且深度有效的板卡进入此话题。板卡位姿有效性字段与 `/panel/board_cards` 一致。

### /panel/board_cards (String, JSON) — 板卡检测与位姿

`panel_controls` / `all` 模式默认发布所有 YOLO 板卡，包括没有 Tag 关联的板卡。
板卡只框固定的白色矩形外框，不扩展到拉手或被抽出的面板。

```json
{
  "stamp": 1716192000.123,
  "frame_id": "camera_color_optical_frame",
  "targets": [
    {
      "id": 8,
      "tag_id": 7,
      "class": "board_card",
      "classification_source": "yolo",
      "identity_source": "apriltag_36h11",
      "bbox": [510.0, 92.0, 892.0, 607.0],
      "confidence": 0.91,
      "position_valid": true,
      "orientation_valid": true,
      "position": {"x": 0.068, "y": -0.057, "z": 1.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
      "orientation_source": "panel_normal"
    }
  ]
}
```

| 字段 / 情况 | 含义 |
|-------------|------|
| `id` / `tag_id` | 验证场 T07 对应编号 8；无有效关联时均为 null |
| `classification_source` | 固定 `yolo`；有无 Tag 均不改板卡类别 |
| `identity_source` | 直接 Tag 为 `apriltag_36h11`，短暂丢失跟踪为 `apriltag_track`，无关联为 null |
| `bbox` | 原图像素坐标 `[x1, y1, x2, y2]` |
| `position` | 深度反投影得到的检测框中心位置，单位米，坐标系由顶层 `frame_id` 指定 |
| `position_valid: false` | 有效深度缺失，`position` 为 null，不能作为有效位置使用 |
| `orientation` | 沿用面板法向形成的四元数，不估计板卡面内旋转角 |
| `orientation_valid: false` | 面板法向不可用，`orientation` / `orientation_source` 为 null |
| `targets: []` | 当前帧没有板卡，下游应清空旧检测 |

没有 Tag 关联仍发布板卡框和有效位姿；超过跟踪期限后编号清空，不会改为按钮或灯。
位置与姿态独立标明有效性：有效深度缺失时，之前获得的面板法向仍可能有效。
有编号且深度有效的板卡同时进入 `/panel/targets`。
开启 `publish_legacy_topics:=true` 后另发 `/panel/board_card_pose`（PoseStamped）；
该兼容话题没有身份和有效性字段，多目标或姿态有效性判断应使用 JSON 话题。


**插拔状态与操作部位（ROS1/ROS2 同步）**：`board_card` 仍表示固定白色外框；
新增字段同时出现在 `/panel/board_cards` 和已编号板卡的 `/panel/targets`：

| 字段 | 含义 |
|------|------|
| `insertion_state` | `inserted`：检测到白色拉手；`empty`：蓝色槽口暴露；`unknown`：证据不足或遮挡 |
| `state_source` | `visual`，表示视觉状态；不代表已检测机械锁紧或电气接通 |
| `handle` | 插入时可见拉手上的夹取区域；不可见时为 null |
| `slot` | 拔出时可见蓝色连接器的槽口区域；被插入件遮住时为 null |
| `handle.pixel` / `slot.pixel` | 操作部位的原图像素中心 `[u, v]`；`polygon` 是观察到的区域四角 |
| `handle.position` / `slot.position` | 对应部位的相机坐标（米），不能用外框中心的 `position` 代替 |
| `position_valid`（部位内） | 该部位有效深度是否充足；无深度或深度离散过大时为 false，位置为 null |

拉手深度从夹取区域内部采样；槽口深度从可见连接器边缘采样，避开中间孔洞。
部位 `confidence` 是几何规则分数，未做概率校准。算法针对验证场白色外框、白色拉手、
蓝色槽口；可通过 `board_features.enable: false` 关闭，蓝色阈值由
`board_features.slot_hue_min/slot_hue_max` 设置（OpenCV HSV，默认 85/125）。
拉手退出外框视野后不再追踪；遮挡时不推测隐藏操作点。旧外框位姿字段的取样方式保持不变。

**插拔状态对位置的影响**：当前深度取检测框中心附近 7×7 像素的有效值中位数，
没有单独提取白色外框。插上时可能采到插入件，拔出后可能采到更深的插槽/背景；
即使框中心不变，深度及反投影位置也可能改变。手部遮挡和框抖动还会改变采样区域。
`position_valid` 只检查数值有效，不保证采样表面属于固定外框；Tag 07 关联不消除这一影响。
需要插拔前后位置一致时，应从固定白色外框拟合平面，再求外框中心射线与平面的交点，
排除中央插槽、插入件及遮挡；当前版本尚未实现该几何定位。

```bash
ros2 topic echo /panel/board_cards
ros2 launch panel_detection panel_controls.launch.py publish_legacy_topics:=true
```

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

该话题只发布有编号的旋钮角度，条目字段保持兼容。默认模式 1 输出 `0`（竖直）或 `90`（水平），
采用白色条纹的无向轴及连续帧确认；过渡方向存在迟滞区。模式 2/3 才输出连续有向角，
以图像 12 点钟方向为 0°，顺时针增加。角度不是经过面板透视校正的机械关节角。
条纹拟合排除外圈金属反光，并支持水平/竖直的简单矩形轮廓。

### /panel/knob_observations (String, JSON) — 包含无编号旋钮的视觉角度

新增话题形如 `{"stamp": 1716192000.123, "knobs": [...]}`，每个旋钮包括
`id`、`tag_id`、`bbox`、`angle`、`raw_angle`、`angle_valid`、`angle_source`、
`confidence`、`position_valid` 和 `position`。未关联 Tag 的旋钮仍计算并显示角度，
`id/tag_id` 为 null，不虚构编号，也不写入旧 `/panel/knob_angles`。
`raw_angle` 是离散化前的图像角度，`angle_source: held` 表示短时保留已有档位。
深度无效时 `position` 为 null，二维角度不依赖深度。

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

`fastener_detection.launch.py` 只发布螺栓和螺母目标，`class` 为 `bolt` 或 `nut`。同一安装面检测到 2 个 fastener 时会先输出临时编号（`registered=false`）；至少 3 个被可靠检测后，节点会建立 4 槽位模板，并在后续帧给已检测到的目标附加稳定编号：

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

编号定义为每个 `group_id` 内的局部槽位：`1=top_left`、`2=top_right`、`3=bottom_right`、`4=bottom_left`。机械臂指定目标时应使用 `group_id + id`，不要只使用单独的 `id`。冷启动只有 2 个 fastener 可见时会输出稳定的临时 ID，但两点无法唯一确定四槽位布局，因此 `registered=false`，第三个目标出现后 ID 可能校正一次；模板注册后，即使后续只检测到部分目标，也会按历史槽位继续编号。

螺母外轮廓精定位失败时，bbox 中心的有效深度只用于分组编号，不会作为机械臂操作点发布；无效深度或 `(0,0,0)` 坐标不会进入目标话题和注册器。

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

### /leak/targets (String, JSON) — 漏点位置与 leak_bolt 位姿

`leak_detection.launch.py` 发布漏点三维坐标和加固螺栓位姿：

```json
{
  "stamp": 1716192000.123,
  "leak_point": {"x": 0.15, "y": -0.02, "z": 0.65},
  "leak_source": "upstream",
  "bolt": {
    "class": "leak_bolt",
    "bbox": [603, 369, 637, 401],
    "position": {"x": 0.16, "y": -0.01, "z": 0.64},
    "orientation": {"x": 0.05, "y": 0.44, "z": 0.0, "w": 0.90},
    "confidence": 0.85,
    "axis_direction": [-0.791, 0.098, -0.604],
    "axis_source": "object_plane"
  }
}
```

`leak_source` 为 `upstream`（来自上游节点发布的 `/leak/upstream_point`）或 `black_marker`（回退到黑色标记检测）。上游漏点超过 2 秒未更新时自动切换到黑色标记检测。

`bolt` 字段在检测到加固螺栓时出现，格式与 fastener 模式的 bolt 输出一致。`position` 为相机坐标系下的三维坐标（米），`orientation` 为表面法向量对应的四元数，`axis_direction` 为表面法向量（指向相机时 z 为负），`axis_source` 为 `object_plane`（目标表面深度拟合）或 `patch_plane`（周围安装面拟合）。未检测到 bolt 时无此字段。

### /leak/pipe_axis (String, JSON) — 管道轴线方向

```json
{
  "stamp": 1716192000.123,
  "leak_point_px": [640, 360],
  "leak_source": "black_marker",
  "pipe_axis_angle_deg": -25.9,
  "pipe_axis_vector_xy": [0.900, -0.437],
  "num_segments": 11
}
```

管道轴线方向为图像平面 2D 方向，`pipe_axis_angle_deg` 相对图像 x 轴，有 180° 二义性。`num_segments` 为参与估计的 Hough 线段数量。

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

铭牌 OCR 当前已停用。节点不会加载 OCR 后端、绘制铭牌 ROI 或创建
`/panel/labels` 话题；保留 `nameplate_ocr.py` 仅用于后续可能的恢复。

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
| `leak_detection.launch.py` | `/leak/status` |

常见状态：

- `"waiting_camera"` — 等待相机
- `"waiting_topic_frames"` — 等待同步后的相机话题
- `"no_detection"` — 当前帧没有对应模式的检测结果
- `"no_targets"` — 有检测但没有可发布目标
- `"registered"` — 当前模式已有可发布目标；面板模式也表示编号结果可用
- `"no_leak_point"` — leak 模式：既没有上游漏点，也没有检测到黑色标记
- `"leak_no_axis"` — leak 模式：漏点已定位，但附近未能估计管道轴线

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
| `/panel/board_card_pose` | 板卡（有效深度，PoseStamped，不携带 Tag ID 或有效性字段） |

`/panel/nuts` 发布的是 refined 后的螺母操作点：像素中心来自六角螺母外轮廓拟合，深度来自外圈 mask 采样并排除中心螺丝/螺杆。refined 失败或外圈深度无效时不会发布 `/panel/nuts`，避免机械臂拿到错误点。

## 验证话题

```bash
# 另开终端
ros2 topic list
ros2 topic echo /panel/targets
ros2 topic echo /panel/knob_angles
ros2 topic echo /panel/board_cards
ros2 topic echo /valve/geometry
ros2 topic echo /valve/targets
ros2 topic echo /valve/status
ros2 topic echo /fasteners/geometry
ros2 topic echo /fasteners/targets
ros2 topic echo /fasteners/status
ros2 topic echo /panel/distance
ros2 topic echo /panel/status
ros2 topic echo /leak/targets
ros2 topic echo /leak/pipe_axis
ros2 topic echo /leak/status
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

## 默认权重与重新转换

默认主检测权重为 `src/panel_detection/panel_detection/1013.pt` 及由它导出的
`1013.onnx`；ROS1 使用 `ros1/panel_detection/models/1013.onnx` 的同一份模型。
ONNX 为 FP32、opset 12、固定输入 `[1, 3, 640, 640]`，输出为已解码的
`[1, 25200, 15]`，NMS 由检测器执行。十类别顺序与下方配置一致，
新增 `board_card` 的模型类别索引为 9。`panel_controls` 和 `all` 模式都显示板卡，
默认发布 `/panel/board_cards`；阀门、紧固件和漏点模式仍保持各自类别筛选。
人工标注数据集的类别编号可能重新排列，评估时应按类别名称映射。

运行检测只需 ONNX Runtime；若要重新转换 PT，在转换环境准备 YOLOv5 源码
及导出依赖。`1013.pt` 训练时的 YOLOv5 提交为
`781b9d57649da18aa1127707a371a13c5ef8a9ec`：

```bash
git clone https://github.com/ultralytics/yolov5.git /path/to/yolov5
git -C /path/to/yolov5 checkout 781b9d57649da18aa1127707a371a13c5ef8a9ec
python3 -m pip install -r /path/to/yolov5/requirements.txt \
  'torch>=2.6' onnx==1.17.0 onnxruntime==1.20.1 onnxsim==0.4.36

# 在本仓库根目录执行
python3 src/panel_detection/scripts/export_model.py \
  --pt src/panel_detection/panel_detection/1013.pt \
  --yolov5-dir /path/to/yolov5 --format onnx --img-size 640
cp src/panel_detection/panel_detection/1013.onnx ros1/panel_detection/models/1013.onnx
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
onnx_model: '1013.onnx'        # 相对 panel_detection 包目录，或填写绝对路径
onnx_threads: 8
# rknn_model: '0630.rknn'      # inference_backend='rknn' 时使用

detection_mode: 'all'          # panel_detection.launch.py 兼容模式使用
publish_legacy_topics: false   # 是否发布旧 PoseStamped 兼容话题

class_num: 10
class_name: ['light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button', 'door_button', 'air_switch', 'board_card']
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

panel_apriltag:
  enable: true
  dictionary: 'DICT_APRILTAG_36h11'
  min_id: 0
  max_id: 39
  min_area_px: 64.0
  max_horizontal_ratio: 0.65
  max_vertical_ratio: 3.0
  track_distance_px: 90.0
  stale_frames: 8

fastener_registry:
  min_init_observations: 2      # 两个目标先建立临时编号，三个目标升级为 4 槽位模板
  max_group_distance_m: 0.35    # 同组聚类的 3D 距离门限
  max_slot_distance_m: 0.12     # 历史槽位匹配的最小距离门限
  slot_match_ratio: 0.45        # 槽距比例门限，和 max_slot_distance_m 取较大值
  normal_angle_thresh_deg: 25.0 # 同组/同槽位法向量夹角门限
  ema_alpha: 0.35               # 槽位 3D 位置平滑系数
  stale_frames: 120             # 多久未观测后删除该 fastener 组

fastener_axis_stabilizer:
  enable: true                  # 螺栓/螺母法向量跨帧稳定
  ema_alpha: 0.18               # 小幅变化的单位向量 EMA 系数
  max_jump_deg: 12.0            # 超过该角度视为候选跳变
  confirm_frames: 5             # 大角度变化连续出现多少帧后才接受
  max_output_step_deg: 3.0      # 确认变化后每帧允许的最大渐进转角
  match_distance_px: 60.0       # 静止相机下的目标中心关联距离
  match_size_ratio: 1.2         # bbox 尺寸相关的关联距离倍率
  stale_frames: 60              # 轨迹丢失多少帧后删除

fastener_position_stabilizer:
  enable: true                  # 按 group_id + id 稳定螺栓/螺母三维坐标
  still_time: 1.0               # 画面静止多久后启用窗口滤波
  pixel_thresh: 5.0             # 判断目标静止的像素位移门限
  window_size: 30               # 三维坐标中值滤波窗口
  ema_alpha: 0.20               # 稳定坐标 EMA 系数
  depth_std_thresh: 0.008       # 深度标准差门限，单位 m

panel_normal_interval: 10
```

## 目录结构

```
ros2_ws/
├── README.md
├── .gitignore
├── run.sh                         ← 快捷启动脚本
└── src/
    ├── OrbbecSDK_ROS2/          ← 官方相机驱动 (submodule)
    └── panel_detection/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   ├── camera.launch.py             ← 启动相机
        │   ├── panel_controls.launch.py     ← 面板按钮/旋钮/指示灯/板卡
        │   ├── valve_detection.launch.py    ← 阀门
        │   ├── fastener_detection.launch.py ← 螺栓/螺母
        │   ├── leak_detection.launch.py     ← 管道漏点
        │   └── panel_detection.launch.py    ← 兼容 all 模式
        ├── scripts/
        │   ├── estimate_pipe_axis.py      ← 离线管道轴线估计
        │   └── export_model.py            ← 模型转换 (.pt → .onnx)
        ├── test/
        │   ├── test_apriltag_reference.py
        │   ├── test_panel_apriltag.py
        │   └── test_topic_sync.py
        └── panel_detection/
            ├── __init__.py
            ├── panel_detect_node.py       ← 检测主节点
            ├── target_registry.py         ← 目标注册与跟踪
            ├── fastener_registry.py       ← 螺栓/螺母分组与槽位编号
            ├── nameplate_ocr.py           ← 已停用的铭牌 OCR 保留实现
            ├── nut_localizer.py           ← nut 外六角 refined 定位
            ├── pipe_axis.py               ← 管道轴线方向估计 + 黑色标记检测
            ├── panel_apriltag.py           ← 面板 Tag 分类、编号和短时跟踪
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
            ├── 1013.onnx                  ← 默认十类别权重
            ├── leak_bolt.onnx             ← leak_bolt 专用权重
            └── 1013.pt
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

## 离线视频：面板 Tag 与元件对应复核

无需 ROS、相机、深度或内参，直接读取视频。使用现有 `1013.onnx`、
`detect_panel_tags`、`PanelAprilTagTracker` 和与 ROS 节点共享的
`apply_panel_tag_rules`。默认配置也与节点共享。原有 launch、话题和深度流程不变。

依赖 Python 3、NumPy、PyYAML、ONNX Runtime、带 aruco 的 OpenCV 以及含 libx264 的 ffmpeg。
在仓库根目录执行：

```bash
python3 scripts/review_panel_video.py \
  /data/little8/1013/*_无损.mp4 \
  --output-dir /data/little8/1013_panel_review \
  --crop 176 50 1000 748
```

`--crop X Y WIDTH HEIGHT` 是这批录屏的相机画面区域；其他视频可省略以处理全图，
或重新指定裁剪区域。不会修改原视频。已有同名输出目录时拒绝覆盖。
`--max-frames 30` 用于短片段试跑；`--config FILE` 合并覆盖默认 YAML 配置；
`--model FILE` 指定 ONNX 权重。每个视频独立重置 Tag 跟踪状态，逐帧处理、不主动跳帧，
输出保留源视频帧率（无音轨）。

每段视频生成一个子目录：

- `annotated.mp4`：H.264/yuv420p 标注视频，顶部增加 36 像素状态栏。
- `frames.jsonl`：每帧的时间、Tag 四角、检测框、YOLO 原始类别、最终类别、
  Tag ID、目标 ID、匹配来源。像素坐标相对于裁剪后的原图，不包含状态栏偏移。
- `summary.json`：输入、裁剪、实际模型配置、帧数、匹配计数和预览帧号。
- `preview.jpg`：优先选择直接关联数量较多的帧；每 15 秒另存一张抽帧图。

画面中 `T00` 表示 Tag ID=0，`#1` 表示目标 ID=1（现有规则为 Tag ID+1）。
连线表示本帧直接识别到的 Tag 与元件的关联；橙色 Tag 框表示没有匹配元件；
灰色元件框表示没有有效 Tag 关联；`[tracked]` 表示短时跟踪关联，当前帧没有直接连线。
`YOLO:...` 保留原始分类，主标签显示应用 Tag 规则后的分类。

本工具复核二维关联、旋钮角度及板卡插拔状态/拉手/槽口位置；使用与 ROS 节点相同的算法。
不运行布局回退编号、深度或三维位置发布。
保持现有节点规则：启用 Tag 时，本帧没有直接 Tag 关联的 button 会改判为 light。
Tag 识别到了但 YOLO 未检出对应元件时，只画 Tag，不虚构元件框。
统计是检测/关联次数，不是准确率；真实准确率需要人工标注。

## 场景配置：测试场与验证场

默认场景现为 **验证场 `validation_site`**：T00/T01/T04/T05/T06 为按钮，
T02/T03 为旋钮，T07 对应板卡 `board_card`；仅 T00–T07 有效，其他 Tag 不参与关联。目标编号仍为 Tag ID+1。
测试场 `test_site` 保留旧规则：T00/T01/T02/T05/T06 为按钮，T03/T04 为旋钮，
T07–T39 为指示灯。

配置文件位于 `src/panel_detection/config/`，构建安装时也会进入包的 share/config。

```bash
# 验证场：不传配置也使用此场景
./run.sh panel --config src/panel_detection/config/validation_site.yaml
# 切回测试场
./run.sh panel --config src/panel_detection/config/test_site.yaml
# 离线视频采用同一份场景配置
python3 scripts/review_panel_video.py "/data/little8/1013/Screencast 2026-09-28 203939_无损.mp4" \
  --config src/panel_detection/config/validation_site.yaml \
  --output-dir /data/little8/1013_panel_review_validation --crop 176 50 1000 748
```

ROS launch 也可以使用 `config_path:=配置文件路径`。ROS 使用安装包运行时，
修改源码后需在现有 ROS 构建环境中重新构建/加载工作空间。

配置逻辑：

- `panel_apriltag.scene` 选择预设，缺省为 `validation_site`；未知名称立即报错。
- 可选 `panel_apriltag.tag_classes` 为完整 ID→类别映射，**整体替换**预设映射，
  未列出的 Tag 忽略；支持 `button`、`knob`、`light`、`board_card`，ID 范围 0–39。
  请用整数键（例如 `2`），不要混用 `2` 与 `"02"`。
- 部分 YAML 配置递归覆盖默认配置，无需复制模型、阈值等无关参数。
- `layout_fallback` 控制 ROS 节点的旧七元件布局回退编号：测试场默认开启，
  验证场默认关闭，避免漏 Tag 时按旧场景分配 ID。离线工具仅复核 Tag 关联，始终不运行布局回退。
- 没有直接 Tag 关联的 button 改判为 light 的原有规则保持不变；
  `tag_classes` 只对成功关联到 Tag 的元件生效，不会将所有未匹配目标改成按钮。
- 板卡以 YOLO 类别为准：T07 仅关联已经检测为 `board_card` 的目标并分配编号 8，
  不把其他类别改为板卡，也不把板卡改为按钮/旋钮/灯；无 Tag 时仍保留板卡框，
  没有有效编号的板卡不发布为已编号目标。短暂丢 Tag 可沿用原有跟踪关联。
- 离线复核包含旋钮角度和板卡操作部位；无深度视频只输出二维像素位置。

新增其他 Tag 时，在验证场配置的 `tag_classes` 中显式添加其 ID 与类别。
历史 `1013_panel_review` 输出仍保留旧规则，新默认结果保存到
`1013_panel_review_validation`，避免覆盖对照结果。


## ROS1 / ROS2 功能同步

两端具备相同的面板、阀门、紧固件、漏点和 all 检测流程，包含同心标记检测及其位姿/角度稳定、
Tag 几何参考、紧固件组编号、旋钮深度修正、场景配置和深度采样调试。
公共算法文件保持一致，`test_ros_parity.py` 检查模块内容与核心回调 AST，防止再次遗漏同步。
ROS1 另外保留原生时间戳、可配置相机话题及 models/ 权重查找方式；两端都支持 debug_image 输出，
ROS2 可传 `-p publish_debug_image:=true -p debug_image_topic:=/panel/debug_image`。

ROS1 新增精度记录器、同步 RGB/深度查看器和对应 launch，安装时包含 camera 子包与工具脚本。
录屏离线复核使用根目录的 `scripts/review_panel_video.py`，不依赖 ROS 运行时。
相机驱动使用平台原生驱动，RKNN/相机 SDK/OCR 的可选依赖需要按目标机器安装。

初次功能同步验证环境：ROS1 Noetic、ROS2 Galactic。两端均通过完整单元测试和五种模式的合成 RGB-D
运行验证；查看器完成同步图像解码与截图，记录器完成初始化及输出文件关闭。
真实相机及 RK3588 NPU 未接入本轮测试。构建在独立目录验证，不覆盖现有部署。
LAB 的用户级 setuptools 与系统 packaging 存在版本冲突，ROS2 构建时使用临时
`packaging==24.2` 路径覆盖；未修改机器全局依赖。

板卡关联/话题更新已在 ROS1 Noetic 与 ROS2 Humble 验证：两端各 92 项测试通过，
catkin/colcon 构建通过；ROS2 实际话题测试使用已有录屏图片和模拟深度，覆盖直接 Tag、
Tag 丢失、无效深度和空检测。尚未验证真实板卡深度精度及插拔前后的位姿一致性。

## 管道三维轴线

原 `/leak/pipe_axis` 只有二维角度和图像向量。现在保留这些字段并新增 `pipe_axis_3d`。
默认开启完整漏点检测；使用 `config/leak_bolt_only.yaml` 可切回仅螺栓模式，
`config/leak_full.yaml` 显式开启完整模式。

算法沿二维轴线在局部 ROI 内按像素步长取样，对窄条内有效深度取中位数，
剔除远离漏点局部深度的样本，再用相机内参反投影为三维点，经过 RANSAC 和 SVD 拟合。
必须使用与彩色图对齐、尺寸一致的深度和对应的针孔/校正图像内参；未注册深度拒绝输出三维方向。
深度单位由 `depth_scale` 指定，输出距离统一为米。

`pipe_axis_3d` 字段：

- `valid` / `reason`：是否有效及失败原因；缺少二维轴线或深度不足时也发布无效状态。
- `frame_id`：彩色光学坐标系（x 向右、y 向下、z 向前）；话题模式采用输入彩色图 frame_id。
- `point_xyz_m`：拟合点云的中心点；这是可见管道表面上的局部轴线参考点，**不是圆柱中心**。
- `direction_xyz`：三维单位方向向量；有正负二义性，符号按二维轴线方向统一，不代表流向。
- `endpoints_xyz_m`：有数据支持的拟合线段端点。
- `point_count`、`inlier_count`、`inlier_ratio`、`rms_error_m`、`span_m`：拟合质量指标。

无效时位置、方向和端点为 null，消费者必须检查 valid，不能沿用上一次结果。
顶层仍有 `pipe_axis_angle_deg`、`pipe_axis_vector_xy`、`num_segments`，
没有二维结果时角度/向量为 null、段数为 0。消息仍为 std_msgs/String JSON，未新增自定义消息依赖。

`pipe_axis_3d` 配置可调整采样长度、条带宽度、有效深度范围、离群阈值、最少点数、
最小支持长度及内点比例。默认半长 100 像素、步长 2 像素、条带半宽 2 像素、
内点距离阈值 8 mm、至少 12 点、内点比例 0.65、支持长度至少 4 cm。
本批 MP4 没有原始深度，无法据此验证真实三维精度；已通过已知倾斜方向、异常深度、空洞、
无效深度和丢失三维结果的合成测试。

## 原始 YOLO 图片复核

```bash
python3 scripts/annotate_yolo_images.py "/data/little8/1013_frames_1hz/Screencast 2026-09-28 203939_无损" --crop 176 50 1000 748
```

结果放在输入目录的 `yolo_raw/`（可通过 `--output-dir` 指定新目录），保留原图。
仅使用 YOLO 原始框、类别和置信度，不经过 Tag、场景类别映射或布局修正。
`predictions.jsonl` 保存每张图片的预测；坐标相对于原图，输出 PNG 顶部另加 32 像素标题栏。
`summary.json` 记录模型、阈值、裁剪区域和预测计数；这些不是人工真值，不计算准确率。
