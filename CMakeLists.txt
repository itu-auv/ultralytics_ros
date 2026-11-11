cmake_minimum_required(VERSION 3.16)
project(ultralytics_ros)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(PCL REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/YoloResult.msg"
  DEPENDENCIES
  sensor_msgs
  std_msgs
  vision_msgs
)

include_directories(include)

add_executable(tracker_with_cloud_node src/tracker_with_cloud_node.cpp)
target_link_libraries(tracker_with_cloud_node
  rclcpp::rclcpp
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${vision_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  cv_bridge::cv_bridge
  image_geometry::image_geometry
  message_filters::message_filters
  pcl_ros::pcl_ros_tf
  tf2_eigen::tf2_eigen
  tf2_ros::tf2_ros
  opencv_core
  ${PCL_LIBRARIES}
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(tracker_with_cloud_node "${cpp_typesupport_target}")

install(TARGETS
  tracker_with_cloud_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  script/tracker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz2
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
