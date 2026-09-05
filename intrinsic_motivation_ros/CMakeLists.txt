cmake_minimum_required(VERSION 3.8)
project(intrinsic_motivation_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(intrinsic_core REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ros_gz_interfaces REQUIRED)
find_package(visualization_msgs REQUIRED)

# Shared by the monitor and the climbers: the agent's only source of object
# positions, and deliberately the only one.
add_library(object_source SHARED src/object_source.cpp)
target_include_directories(object_source PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(object_source rclcpp sensor_msgs intrinsic_core Eigen3)
install(TARGETS object_source
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

set(node_deps
  rclcpp sensor_msgs std_msgs geometry_msgs trajectory_msgs
  ros_gz_interfaces visualization_msgs intrinsic_core Eigen3)

foreach(node
    empowerment_node
    greedy_climber
    grasp_climber
    drifter_driver
    detector_eval
    empowerment_node_planar
    greedy_climber_planar)
  add_executable(${node} src/${node}.cpp)
  target_include_directories(${node} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  ament_target_dependencies(${node} ${node_deps})
  target_link_libraries(${node} object_source)
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

install(DIRECTORY launch config worlds urdf rviz gui
  DESTINATION share/${PROJECT_NAME})

ament_package()
