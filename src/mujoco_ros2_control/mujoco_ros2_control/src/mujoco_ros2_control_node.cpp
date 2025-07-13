#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "mujoco_ros2_control/mujoco_rendering.hpp"
#include "mujoco_ros2_control/mujoco_ros2_control.hpp"

#include <array>
#include <cstring>

// MuJoCo data structures
static mjModel *mujoco_model = nullptr;
static mjData *mujoco_data = nullptr;

int main(int argc, const char **argv)
{
  // ----------------------------- ROS 2 初始化 -----------------------------
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "mujoco_ros2_control_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  RCLCPP_INFO(node->get_logger(), "Initializing mujoco_ros2_control node ...");
  const auto model_path = node->get_parameter("mujoco_model_path").as_string();

  // ----------------------------- 加载 MuJoCo 模型 -----------------------------
  char error[1000] = "Could not load binary model";
  if (model_path.size() > 4 &&
      !std::strcmp(model_path.c_str() + model_path.size() - 4, ".mjb"))
  {
    mujoco_model = mj_loadModel(model_path.c_str(), /*vfs=*/nullptr);
  }
  else
  {
    mujoco_model = mj_loadXML(model_path.c_str(), /*vfs=*/nullptr, error, sizeof(error));
  }
  if (!mujoco_model)
  {
    mju_error("Load model error: %s", error);
  }
  RCLCPP_INFO(node->get_logger(), "MuJoCo model successfully loaded");

  // 创建 simulation data
  mujoco_data = mj_makeData(mujoco_model);

  // ----------------------------- 查找触摸头力传感器 -----------------------------
  const int id_force   = mj_name2id(mujoco_model, mjOBJ_SENSOR, "touch_tip_force");
  const int id_torque  = mj_name2id(mujoco_model, mjOBJ_SENSOR, "touch_tip_torque");
  if (id_force == -1 || id_torque == -1) {
    mju_error("[mujoco_ros2_control_node] Cannot find touch_tip_force / torque sensors in the model");
  }
  const int adr_force  = mujoco_model->sensor_adr[id_force];   // 3 doubles
  const int adr_torque = mujoco_model->sensor_adr[id_torque];  // 3 doubles

  // ----------------------------- ROS 2 Publisher -----------------------------
  auto wrench_pub = node->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "touch_tip/wrench", rclcpp::SensorDataQoS());

  std::array<double,3> f0 {0.0, 0.0, 0.0};
  std::array<double,3> t0 {0.0, 0.0, 0.0};
  bool zero_calibrated = false;   // baseline for gravity etc.

  // ----------------------------- 初始化控制 & 渲染 -----------------------------
  mujoco_ros2_control::MujocoRos2Control control(node, mujoco_model, mujoco_data);
  control.init();
  RCLCPP_INFO(node->get_logger(), "Mujoco ros2 controller initialized");

  auto rendering = mujoco_ros2_control::MujocoRendering::get_instance();
  rendering->init(node, mujoco_model, mujoco_data);
  RCLCPP_INFO(node->get_logger(), "Mujoco rendering initialized");

  // ----------------------------- 主循环 (60 FPS 渲染, 实时仿真) -----------------------------
  while (rclcpp::ok() && !rendering->is_close_flag_raised())
  {
    const mjtNum simstart = mujoco_data->time;          // 当前渲染帧开始时刻
    while (mujoco_data->time - simstart < 1.0 / 60.0)   // 填满 1/60 s 仿真
    {
      control.update();                // 这一步内部会调用 mj_step()

      // ------------------ 读取传感器并发布 ------------------
      const double *f_raw = mujoco_data->sensordata + adr_force;   // Fx, Fy, Fz (world frame)
      const double *t_raw = mujoco_data->sensordata + adr_torque;  // Tx, Ty, Tz (world frame)

      // 第一次有数据时记录零点 (静止、无接触)
      if (!zero_calibrated && mujoco_data->time > 0.0) {
        std::memcpy(f0.data(), f_raw, 3 * sizeof(double));
        std::memcpy(t0.data(), t_raw, 3 * sizeof(double));
        zero_calibrated = true;
      }

      geometry_msgs::msg::WrenchStamped msg;
      msg.header.stamp = node->get_clock()->now();
      msg.header.frame_id = "touch_tip";           // 或 world，看你想要哪系

      msg.wrench.force.x  = f_raw[0] - f0[0];
      msg.wrench.force.y  = f_raw[1] - f0[1];
      msg.wrench.force.z  = f_raw[2] - f0[2];
      msg.wrench.torque.x = t_raw[0] - t0[0];
      msg.wrench.torque.y = t_raw[1] - t0[1];
      msg.wrench.torque.z = t_raw[2] - t0[2];

      wrench_pub->publish(msg);
    }

    rendering->update();   // 刷新 GUI
  }

  // ----------------------------- 清理 -----------------------------
  rendering->close();
  mj_deleteData(mujoco_data);
  mj_deleteModel(mujoco_model);

  RCLCPP_INFO(node->get_logger(), "MuJoCo simulation terminated. Bye!");
  return 0;
}
