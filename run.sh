#!/bin/bash
# detect_3588 快捷启动脚本
# 用法: ./run.sh <命令> [选项]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/humble/setup.bash
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
fi

_print_help() {
    cat << 'EOF'
detect_3588 快捷启动脚本

用法: ./run.sh <命令> [选项]

─── 相机 ───
  camera                启动 Orbbec Gemini 336 相机

─── 检测模式 ───
  panel [选项]          面板检测 (light/knob/button)
  fastener [选项]       螺栓螺帽检测 (bolt/nut)
  valve [选项]          阀门检测
  leak [选项]           漏点+管道+leak_bolt 检测
  all [选项]            全部模式

─── 检测选项 ───
  --config FILE         自定义 YAML 配置文件
  --direct              直连相机模式 (默认订阅话题)

─── 录制与回放 ───
  record [NAME]         录制 rosbag (默认名: detect_bag)
  record --topics T...  只录指定话题
  play BAG_PATH         回放 rosbag
  play BAG_PATH --rate R  指定回放速率 (默认 1.0)

─── 工具 ───
  topics                列出当前活跃话题
  echo TOPIC            打印话题消息
  build                 编译 panel_detection 包

─── 示例 ───
  ./run.sh camera                       # 启动相机
  ./run.sh panel                        # 面板检测
  ./run.sh leak --config leak.yaml      # 带配置的漏点检测
  ./run.sh record my_test               # 录制 rosbag
  ./run.sh play bags/my_test --rate 0.5 # 半速回放
  ./run.sh fastener --direct            # 直连相机做螺栓检测
EOF
}

# ─── 命令实现 ───

cmd_camera() {
    ros2 launch panel_detection camera.launch.py
}

cmd_detect() {
    local mode="$1"; shift
    local config_arg=""
    local use_topic="true"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config) config_arg="config_path:=$2"; shift 2 ;;
            --direct) use_topic="false"; shift ;;
            *) echo "未知选项: $1"; exit 1 ;;
        esac
    done

    local launch_file
    case "$mode" in
        panel)    launch_file="panel_controls.launch.py" ;;
        fastener) launch_file="fastener_detection.launch.py" ;;
        valve)    launch_file="valve_detection.launch.py" ;;
        leak)     launch_file="leak_detection.launch.py" ;;
        all)      launch_file="panel_detection.launch.py" ;;
        *)        echo "未知检测模式: $mode"; exit 1 ;;
    esac

    ros2 launch panel_detection "$launch_file" use_topic:="$use_topic" $config_arg
}

cmd_record() {
    local bag_name="detect_bag"
    local extra_args=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --topics) shift; extra_args="--topics $*"; break ;;
            *) bag_name="$1"; shift ;;
        esac
    done

    local bag_dir="$SCRIPT_DIR/bags"
    mkdir -p "$bag_dir"

    echo "录制到 $bag_dir/$bag_name ..."
    echo "按 Ctrl+C 停止"
    ros2 bag record -o "$bag_dir/$bag_name" \
        /camera/color/image_raw \
        /camera/depth/image_raw \
        /camera/color/camera_info \
        /camera/depth/camera_info \
        $extra_args
}

cmd_play() {
    local bag_path="$1"; shift
    local rate="1.0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rate) rate="$2"; shift 2 ;;
            *) echo "未知选项: $1"; exit 1 ;;
        esac
    done

    ros2 bag play "$bag_path" --rate "$rate"
}

# ─── 入口 ───

if [ $# -eq 0 ]; then
    _print_help
    exit 0
fi

case "$1" in
    -h|--help|help)    _print_help ;;
    camera)            shift; cmd_camera "$@" ;;
    panel|fastener|valve|leak|all)
                       cmd_detect "$@" ;;
    record)            shift; cmd_record "$@" ;;
    play)              shift; cmd_play "$@" ;;
    topics)            ros2 topic list ;;
    echo)              shift; ros2 topic echo "$@" ;;
    build)             cd "$SCRIPT_DIR" && colcon build --packages-select panel_detection ;;
    *)                 echo "未知命令: $1"; echo ""; _print_help; exit 1 ;;
esac
