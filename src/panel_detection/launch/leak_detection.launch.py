"""
启动漏点管道检测节点。

默认订阅 /leak/upstream_point (PointStamped) 接收上游漏点三维坐标。
超时未收到上游漏点时，回退到黑色标记检测。

发布话题:
  /leak/targets    — 漏点三维坐标
  /leak/pipe_axis  — 管道轴线方向
  /leak/status     — 检测状态
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

    detect_node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='leak_detection_node',
        output='screen',
        parameters=[{
            'use_topic': ParameterValue(
                LaunchConfiguration('use_topic'), value_type=bool),
            'registered_depth': ParameterValue(
                LaunchConfiguration('registered_depth'), value_type=bool),
            'config_path': LaunchConfiguration('config_path'),
            'detection_mode': 'leak',
            'publish_legacy_topics': False,
        }],
    )

    return LaunchDescription([
        use_topic_arg, registered_depth_arg, config_path_arg,
        detect_node])
