"""
启动面板检测节点（需先启动相机）

用法:
  # 先启动相机
  ros2 launch panel_detection camera.launch.py
  # 再启动检测
  ros2 launch panel_detection panel_detection.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    detect_node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='panel_detection_node',
        output='screen',
        parameters=[{'use_topic': True}],
    )

    return LaunchDescription([detect_node])
