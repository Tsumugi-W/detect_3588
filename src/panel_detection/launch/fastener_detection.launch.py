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
        }],
    )

    return LaunchDescription([
        use_topic_arg, registered_depth_arg, config_path_arg,
        publish_legacy_topics_arg, detect_node])

