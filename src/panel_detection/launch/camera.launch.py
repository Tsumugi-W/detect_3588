"""
启动 Orbbec Gemini 336 相机（官方驱动）

发布话题:
  /camera/color/image_raw
  /camera/depth/image_raw
  /camera/color/camera_info
  /camera/depth/camera_info
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    orbbec_launch = os.path.join(
        get_package_share_directory('orbbec_camera'),
        'launch', 'gemini_330_series.launch.py'
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch),
        launch_arguments={
            'depth_registration': 'true',
            'color_width': '1280',
            'color_height': '720',
            'color_fps': '30',
            'color_format': 'ANY',
            'depth_width': '1280',
            'depth_height': '720',
            'depth_fps': '30',
            'interleave_ae_mode': 'none',
            # Disable Orbbec diagnostics updater to avoid unsupported
            # OB_STRUCT_DEVICE_TEMPERATURE queries on some firmware versions.
            'diagnostic_period': '0.0',
        }.items(),
    )

    return LaunchDescription([camera])
