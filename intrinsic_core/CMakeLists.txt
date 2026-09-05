cmake_minimum_required(VERSION 3.8)
project(intrinsic_core)

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

add_library(intrinsic_core SHARED
  src/discretise.cpp
  src/empowerment.cpp
  src/arm2d.cpp
  src/ur5.cpp
  src/sensors.cpp
  src/batched.cpp
  src/grasp.cpp
  src/ur5_grasp.cpp
  src/perception.cpp
)
target_include_directories(intrinsic_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(intrinsic_core Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS intrinsic_core
  EXPORT export_intrinsic_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Command-line tools. They write CSV rather than figures: there is no plotting
# library here, and the numbers were always the artefact.
foreach(tool run_ur5_map run_arm_map sweep_parameters)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} intrinsic_core Eigen3::Eigen)
  install(TARGETS ${tool} DESTINATION lib/${PROJECT_NAME})
endforeach()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_empowerment test/test_empowerment.cpp)
  target_link_libraries(test_empowerment intrinsic_core Eigen3::Eigen)
  ament_add_gtest(test_ur5 test/test_ur5.cpp)
  target_link_libraries(test_ur5 intrinsic_core Eigen3::Eigen)
  ament_add_gtest(test_ur5_grasp test/test_ur5_grasp.cpp TIMEOUT 300)
  target_link_libraries(test_ur5_grasp intrinsic_core Eigen3::Eigen)
  ament_add_gtest(test_perception test/test_perception.cpp TIMEOUT 300)
  target_link_libraries(test_perception intrinsic_core Eigen3::Eigen)
endif()

ament_export_include_directories(include)
ament_export_libraries(intrinsic_core)
ament_export_dependencies(Eigen3)
ament_export_targets(export_intrinsic_core HAS_LIBRARY_TARGET)
ament_package()
