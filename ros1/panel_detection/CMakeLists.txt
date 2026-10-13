cmake_minimum_required(VERSION 3.0.2)
project(panel_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  message_filters
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/panel_detect_node.py
  scripts/fastener_precision_recorder.py
  scripts/rgb_depth_viewer.py
  scripts/estimate_pipe_axis.py
  scripts/export_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
