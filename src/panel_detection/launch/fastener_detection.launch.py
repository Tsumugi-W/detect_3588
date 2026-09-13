"""
启动螺栓/螺母检测节点。

发布话题:
  /fasteners/targets
  /fasteners/geometry
  /fasteners/status
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_topic_arg = DeclareLaunchArgument(
        'use_topic', default_value='true',
        description='是否通过话题订阅获取图像（true=订阅, false=直连相机）'
    )
    registered_depth_arg = DeclareLaunchArgument(
        'registered_depth', default_value='true',
        description='topic 模式下深度图是否已注册到彩色图'
    )
    config_path_arg = DeclareLaunchArgument(
        'config_path', default_value='',
        description='检测节点 YAML 配置文件路径；为空时使用默认配置'
    )
    publish_legacy_topics_arg = DeclareLaunchArgument(
        'publish_legacy_topics', default_value='false',
        description='是否发布旧 PoseStamped 兼容话题'
    )
    show_gui_arg = DeclareLaunchArgument(
        'show_gui', default_value='false',
        description='是否显示 OpenCV 窗口；无桌面环境必须为 false'
    )
    capture_dir_arg = DeclareLaunchArgument(
        'capture_dir', default_value='',
        description='保存带检测叠加的可视化帧目录；为空时不保存'
    )
    capture_hz_arg = DeclareLaunchArgument(
        'capture_hz', default_value='1.0',
        description='可视化帧保存频率（按 bag 时间戳）'
    )

    detect_node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='fastener_detection_node',
        output='screen',
        parameters=[{
            'use_topic': ParameterValue(
                LaunchConfiguration('use_topic'), value_type=bool),
            'registered_depth': ParameterValue(
                LaunchConfiguration('registered_depth'), value_type=bool),
            'config_path': LaunchConfiguration('config_path'),
            'detection_mode': 'fastener',
            'publish_legacy_topics': ParameterValue(
                LaunchConfiguration('publish_legacy_topics'), value_type=bool),
            'show_gui': ParameterValue(
                LaunchConfiguration('show_gui'), value_type=bool),
            'capture_dir': LaunchConfiguration('capture_dir'),
            'capture_hz': ParameterValue(
                LaunchConfiguration('capture_hz'), value_type=float),
        }],
    )

    return LaunchDescription([
        use_topic_arg, registered_depth_arg, config_path_arg,
        publish_legacy_topics_arg, show_gui_arg, capture_dir_arg,
        capture_hz_arg, detect_node])
