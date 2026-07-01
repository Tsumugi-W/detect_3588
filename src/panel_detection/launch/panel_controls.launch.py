"""
启动面板旋钮/按钮检测节点。

发布话题:
  /panel/targets
  /panel/knob_angles
  /panel/status
  /panel/distance
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_topic_arg = DeclareLaunchArgument(
        'use_topic', default_value='true',
        description='是否通过话题订阅获取图像（true=订阅, false=直连相机）'
    )
    use_constraint_arg = DeclareLaunchArgument(
        'use_constraint', default_value='1',
        description='旋钮角度模式：1=0/90稳定输出，2=旧约束，3=旧无约束'
    )
    use_constrain_arg = DeclareLaunchArgument(
        'use_constrain', default_value='',
        description='use_constraint 的兼容别名；非空时优先使用'
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
        name='panel_controls_node',
        output='screen',
        parameters=[{
            'use_topic': ParameterValue(
                LaunchConfiguration('use_topic'), value_type=bool),
            'use_constraint': ParameterValue(
                PythonExpression([
                    "'", LaunchConfiguration('use_constrain'), "' or '",
                    LaunchConfiguration('use_constraint'), "'"
                ]),
                value_type=str),
            'registered_depth': ParameterValue(
                LaunchConfiguration('registered_depth'), value_type=bool),
            'config_path': LaunchConfiguration('config_path'),
            'detection_mode': 'panel_controls',
            'publish_legacy_topics': ParameterValue(
                LaunchConfiguration('publish_legacy_topics'), value_type=bool),
        }],
    )

    return LaunchDescription([
        use_topic_arg, use_constraint_arg, use_constrain_arg,
        registered_depth_arg, config_path_arg, publish_legacy_topics_arg,
        detect_node])

