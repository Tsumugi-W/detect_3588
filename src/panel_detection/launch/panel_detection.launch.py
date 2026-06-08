"""
启动面板检测节点

用法:
  ros2 launch panel_detection panel_detection.launch.py
  # 使用话题模式（需先启动 camera.launch.py）:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true
  # 关闭旋钮角度物理范围约束:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true use_constraint:=false
  # topic 深度图已注册到彩色图时（默认）:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true registered_depth:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_topic_arg = DeclareLaunchArgument(
        'use_topic', default_value='false',
        description='是否通过话题订阅获取图像（true=订阅, false=直连相机）'
    )
    use_constraint_arg = DeclareLaunchArgument(
        'use_constraint', default_value='true',
        description='是否启用旋钮角度物理范围约束'
    )
    registered_depth_arg = DeclareLaunchArgument(
        'registered_depth', default_value='true',
        description='topic 模式下深度图是否已注册到彩色图'
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
                LaunchConfiguration('use_constraint'), value_type=bool),
            'registered_depth': ParameterValue(
                LaunchConfiguration('registered_depth'), value_type=bool),
        }],
    )

    return LaunchDescription([
        use_topic_arg, use_constraint_arg, registered_depth_arg, detect_node])
