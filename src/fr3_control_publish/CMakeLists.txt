cmake_minimum_required(VERSION 3.8)
project(fr3_control_publish)

###############################################################################
# 0. 手动查找 MuJoCo
###############################################################################
if(NOT DEFINED ENV{MUJOCO_PATH})
  message(FATAL_ERROR "请先设置 MUJOCO_PATH，例如：export MUJOCO_PATH=/path/to/mujoco-install")
endif()
set(MUJOCO_PATH $ENV{MUJOCO_PATH})

include_directories(${MUJOCO_PATH}/include)
find_library(MUJOCO_LIB
  NAMES mujoco mujoco2* mujoco210
  PATHS ${MUJOCO_PATH}/lib ${MUJOCO_PATH}/bin
)
if(NOT MUJOCO_LIB)
  message(FATAL_ERROR "未在 ${MUJOCO_PATH}/lib 或 bin 找到 libmujoco.so")
endif()

###############################################################################
# 1. 编译选项
###############################################################################
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

###############################################################################
# 2. ROS 2 & 其他依赖
###############################################################################
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# GLFW
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
add_definitions(${GLFW_CFLAGS_OTHER})

###############################################################################
# 3. 生成自定义消息
###############################################################################
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/TouchTipWrench.msg
  DEPENDENCIES geometry_msgs builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)

###############################################################################
# 4. 可执行节点
###############################################################################
# ---- 4.1 simple_control_node ----------------------------------------------
add_executable(ee_pose_bridge src/ee_pose_bridge.cpp)
ament_target_dependencies(ee_pose_bridge
  rclcpp
  geometry_msgs
  tf2_ros
)

# ---- 4.2 fr3_monitor -------------------------------------------------------
add_executable(fr3_monitor src/fr3_monitor_node.cpp)
ament_target_dependencies(fr3_monitor rclcpp control_msgs diagnostic_msgs)

# ---- 4.3 jacobian_publisher ------------------------------------------------
add_executable(jacobian_publisher src/fr3_jacobian_publisher.cpp)
ament_target_dependencies(jacobian_publisher rclcpp sensor_msgs std_msgs)

# typesupport
rosidl_get_typesupport_target(jac_ts ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(jacobian_publisher
    ${MUJOCO_LIB}
    ${jac_ts}
)

# ---- 4.4 force_position_controller ----------------------------------------
add_executable(force_position_controller src/force_position_controller.cpp)
ament_target_dependencies(force_position_controller
  rclcpp sensor_msgs std_msgs trajectory_msgs control_msgs)
rosidl_get_typesupport_target(ctrl_ts ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(force_position_controller ${ctrl_ts})

add_executable(tangential_velocity_controller src/TangentialVelocityController.cpp)
ament_target_dependencies(tangential_velocity_controller
rclcpp sensor_msgs std_msgs geometry_msgs)

add_executable(hybridforcetangentialcontroller src/HybridForceTangentialController.cpp)
ament_target_dependencies(hybridforcetangentialcontroller
rclcpp sensor_msgs std_msgs geometry_msgs)


add_executable(mujoco_sim_node src/mujoco_sim_node.cpp)
ament_target_dependencies(mujoco_sim_node rclcpp sensor_msgs std_msgs geometry_msgs)
# 链接 MuJoCo 库
target_link_libraries(mujoco_sim_node
  ${MUJOCO_LIB}
  ${GLFW_LIBRARIES}
)
###############################################################################
# 5. 安装
###############################################################################
install(TARGETS
    ee_pose_bridge
    fr3_monitor
    jacobian_publisher
    force_position_controller
    tangential_velocity_controller
    hybridforcetangentialcontroller
    mujoco_sim_node
  DESTINATION lib/${PROJECT_NAME}
)

###############################################################################
# 6. Lint / 测试（可选）
###############################################################################
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

###############################################################################
# 7. 导出
###############################################################################
ament_package()
