cmake_minimum_required(VERSION 3.8)
project(formation_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "CMAKE_BUILD_TYPE not set")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Default build type: RelWithDebInfo" FORCE)
endif()

message("Build type: ${CMAKE_BUILD_TYPE}")

# find dependencies
find_package(ament_cmake REQUIRED)

# general dependencies
find_package(Eigen3 3.4 REQUIRED)

# ROS dependencies
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(arrc_interfaces REQUIRED)
find_package(qpOASES_e REQUIRED)

set(dependencies
  "rclcpp"
  "nav_msgs"
  "tf2"
  "tf2_ros"
  "arrc_interfaces"
)


# let the compiler search for headers in include folder
include_directories(include)

# define a library target
add_library(${PROJECT_NAME}
    src/formation_controller.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})
target_link_libraries(${PROJECT_NAME} qpOASES_e fmt::fmt)


add_executable(${PROJECT_NAME}_node src/ros_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_node ${dependencies})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} fmt::fmt)

add_executable(target_node src/target_node.cpp)
ament_target_dependencies(target_node ${dependencies})
target_link_libraries(target_node ${PROJECT_NAME} fmt::fmt)

add_executable(obstacles_node src/obstacles_node.cpp)
ament_target_dependencies(obstacles_node ${dependencies})
target_link_libraries(obstacles_node ${PROJECT_NAME} fmt::fmt)


ament_export_targets(${PROJECT_NAME}_TARGETS HAS_LIBRARY_TARGET)

install(TARGETS ${PROJECT_NAME}_node target_node obstacles_node DESTINATION lib/${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_TARGETS
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
# the above command does NOT install includes
install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)
install(PROGRAMS
    scripts/sim_neighbors_sensing_node.py
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
#ament_export_dependencies(${dependencies})
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()
