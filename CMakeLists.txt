cmake_minimum_required(VERSION 3.8)
project(formation_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(arrc_interfaces REQUIRED)
find_package(OsqpEigen REQUIRED)

set(dependencies
  "rclcpp"
  "nav_msgs"
  "tf2"
  "tf2_ros"
  "arrc_interfaces"
  "OsqpEigen"
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

# let the compiler search for headers in include folder
include_directories(include)

# define a library target
add_library(formation_control
  SHARED src/FormationController.cpp
)

ament_target_dependencies(formation_control ${dependencies})
target_link_libraries(formation_control OsqpEigen::OsqpEigen)


# export library
ament_export_targets(formation_control HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})



# install
install(
  DIRECTORY include/formation_control
  DESTINATION include
)

install(
  TARGETS formation_control
  EXPORT formation_control
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


add_executable(formation_node src/formation_node.cpp)
target_include_directories(formation_node PUBLIC formation_control)
# target_compile_features(formation_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(formation_node ${dependencies})
target_link_libraries(formation_node formation_control)
install(TARGETS formation_node DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rclcpp)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(arrc_interfaces)


ament_package()
