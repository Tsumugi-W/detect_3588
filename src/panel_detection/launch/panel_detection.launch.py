"""
启动面板检测节点

用法:
  ros2 launch panel_detection panel_detection.launch.py
  # 使用话题模式（需先启动 camera.launch.py）:
  ros2 launch panel_detection panel_detection.launch.py use_topic:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_topic_arg = DeclareLaunchArgument(
        'use_topic', default_value='false',
        description='是否通过话题订阅获取图像（true=订阅, false=直连相机）'
    )

    detect_node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='panel_detection_node',
        output='screen',
        parameters=[{'use_topic': LaunchConfiguration('use_topic')}],
    )

    return LaunchDescription([use_topic_arg, detect_node])
