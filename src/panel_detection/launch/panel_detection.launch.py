"""
启动面板检测节点

用法:
  ros2 launch panel_detection panel_detection.launch.py
  # 使用话题模式（需先启动 camera.launch.py）:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true
  # 旋钮角度模式：1=默认0/90稳定输出，2=旧物理范围约束，3=旧无约束
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true use_constraint:=3
  # topic 深度图已注册到彩色图时（默认）:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true registered_depth:=true
  # 指定配置文件:
  ros2 launch panel_detection panel_detection.launch.py config_path:=/path/to/config.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_topic_arg = DeclareLaunchArgument(
        'use_topic', default_value='false',
        description='是否通过话题订阅获取图像（true=订阅, false=直连相机）'
    )
    use_constraint_arg = DeclareLaunchArgument(
        'use_constraint', default_value='1',
        description='旋钮角度模式：1=0/90稳定输出，2=旧物理范围约束，3=旧无约束；true/false 兼容旧写法'
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
    detection_mode_arg = DeclareLaunchArgument(
        'detection_mode', default_value='all',
        description='检测模式：all / panel_controls / valve / fastener'
    )
    publish_legacy_topics_arg = DeclareLaunchArgument(
        'publish_legacy_topics', default_value='false',
        description='是否发布 /panel/valves 等旧 PoseStamped 兼容话题'
    )
    capture_dir_arg = DeclareLaunchArgument(
        'capture_dir', default_value='',
        description='检测 canvas 保存目录；为空时不保存'
    )
    capture_hz_arg = DeclareLaunchArgument(
        'capture_hz', default_value='1.0',
        description='检测 canvas 按 ROS 时间戳保存的频率'
    )
    show_gui_arg = DeclareLaunchArgument(
        'show_gui', default_value='true',
        description='是否显示 OpenCV 检测窗口'
    )
    use_panel_tags_arg = DeclareLaunchArgument(
        'use_panel_tags', default_value='true',
        description='是否使用 tag36h11 ID 00-39 强制面板元器件分类和编号'
    )

    detect_node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='panel_detection_node',
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
            'detection_mode': LaunchConfiguration('detection_mode'),
            'publish_legacy_topics': ParameterValue(
                LaunchConfiguration('publish_legacy_topics'), value_type=bool),
            'capture_dir': LaunchConfiguration('capture_dir'),
            'capture_hz': ParameterValue(
                LaunchConfiguration('capture_hz'), value_type=float),
            'show_gui': ParameterValue(
                LaunchConfiguration('show_gui'), value_type=bool),
            'use_panel_tags': ParameterValue(
                LaunchConfiguration('use_panel_tags'), value_type=bool),
        }],
    )

    return LaunchDescription([
        use_topic_arg, use_constraint_arg, use_constrain_arg, registered_depth_arg,
        config_path_arg, detection_mode_arg, publish_legacy_topics_arg,
        capture_dir_arg, capture_hz_arg, show_gui_arg, use_panel_tags_arg,
        detect_node])
