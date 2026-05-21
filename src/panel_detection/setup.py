import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'panel_detection'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['*.onnx', '*.pt'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zile',
    maintainer_email='zile@todo.todo',
    description='Panel pose detection with YOLOv5 + depth camera (RK3588)',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'panel_detect_node = panel_detection.panel_detect_node:main',
        ],
    },
)
