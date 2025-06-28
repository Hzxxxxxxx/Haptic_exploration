cmake_minimum_required(VERSION 3.8)
project(fr3_control_publish)

# ---- MuJoCo 手动配置 ----
if(NOT DEFINED ENV{MUJOCO_PATH})
  message(FATAL_ERROR "请先设置 MUJOCO_PATH 环境变量，例如：export MUJOCO_PATH=/home/your_user/mujoco/mujoco-install")
endif()
set(MUJOCO_PATH $ENV{MUJOCO_PATH})

# 包含 MuJoCo 头文件
include_directories(
  ${MUJOCO_PATH}/include
)

# 自动查找 MuJoCo 动态库（.so 或 .dylib）
find_library(MUJOCO_LIB
  NAMES mujoco mujoco210 mujoco2*
  PATHS ${MUJOCO_PATH}/lib ${MUJOCO_PATH}/bin
)
if(NOT MUJOCO_LIB)
  message(FATAL_ERROR "在 ${MUJOCO_PATH}/lib 或 ${MUJOCO_PATH}/bin 下未找到 MuJoCo 库，请检查安装路径")
endif()

###############################################################################
# 1. 编译选项
###############################################################################
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

###############################################################################
# 2. 依赖包
###############################################################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)      # simple_control_node
find_package(control_msgs REQUIRED)         # fr3_monitor
find_package(diagnostic_msgs REQUIRED)      # fr3_monitor
find_package(sensor_msgs REQUIRED)          # jacobian_publisher
find_package(std_msgs REQUIRED)             # jacobian_publisher

# 添加 Eigen 库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

###############################################################################
# 3. 可执行文件
###############################################################################
# simple_control_node: 发布轨迹命令
add_executable(simple_control_node src/fr3_publish_node.cpp)
ament_target_dependencies(simple_control_node
  rclcpp
  trajectory_msgs
)

# fr3_monitor: 监控 JointTrajectoryControllerState
add_executable(fr3_monitor src/fr3_monitor_node.cpp)
ament_target_dependencies(fr3_monitor
  rclcpp
  control_msgs
  diagnostic_msgs
)

# jacobian_publisher: 计算并发布雅各比矩阵
add_executable(jacobian_publisher src/fr3_jacobian_publisher.cpp)
ament_target_dependencies(jacobian_publisher
  rclcpp
  sensor_msgs
  std_msgs
)
# 链接 MuJoCo 库
target_link_libraries(jacobian_publisher
  ${MUJOCO_LIB}
)

# force_position_controller: 获取力误差并输出位置控制命令
add_executable(force_position_controller src/force_position_controller.cpp)
ament_target_dependencies(force_position_controller
  rclcpp
  sensor_msgs
  std_msgs
  trajectory_msgs
)
# Eigen 是头文件库，无需链接

###############################################################################
# 4. 安装
###############################################################################
install(
  TARGETS
    simple_control_node
    fr3_monitor
    jacobian_publisher
    force_position_controller
  DESTINATION lib/${PROJECT_NAME}
)

###############################################################################
# 5. Lint / 测试（可选)
###############################################################################
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
