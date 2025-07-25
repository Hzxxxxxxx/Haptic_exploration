#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "mujoco_ros2_control/mujoco_rendering.hpp"
#include "mujoco_ros2_control/mujoco_ros2_control.hpp"

#include "mujoco/mjvisualize.h"

#include <cstring>

// MuJoCo data structures
static mjModel* mujoco_model = nullptr;
static mjData*  mujoco_data  = nullptr;

int main(int argc, const char** argv)
{
  // ----------------------------- ROS 2 初始化 -----------------------------
  rclcpp::init(argc, argv);
  auto node   = rclcpp::Node::make_shared(
    "mujoco_ros2_control_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  auto logger = node->get_logger();

  RCLCPP_INFO(logger, "Initializing mujoco_ros2_control node ...");
  std::string model_path = node->get_parameter("mujoco_model_path").as_string();

  // ----------------------------- 加载 MuJoCo 模型 -----------------------------
  char error[1000] = "Could not load binary model";
  if (model_path.size() > 4 &&
      model_path.compare(model_path.size()-4, 4, ".mjb") == 0)
  {
    mujoco_model = mj_loadModel(model_path.c_str(), nullptr);
  }
  else
  {
    mujoco_model = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
  }
  if (!mujoco_model)
    mju_error("Load model error: %s", error);

  RCLCPP_INFO(logger, "MuJoCo model successfully loaded");
  mujoco_data = mj_makeData(mujoco_model);

  // ----------------------------- 获取碰撞 geom ID -----------------------------
  int id_ee  = mj_name2id(mujoco_model, mjOBJ_GEOM, "touch_tip_collision");
  int id_box = mj_name2id(mujoco_model, mjOBJ_GEOM, "exploration_object_collision");
  if (id_ee < 0 || id_box < 0) {
    mju_error("Cannot find touch_tip_collision or exploration_object_collision geom");
  }

  // ----------------------------- ROS 2 发布器 -----------------------------
  auto wrench_pub = node->create_publisher<geometry_msgs::msg::WrenchStamped>(
    "touch_tip/wrench", rclcpp::SensorDataQoS());

  // ----------------------------- 初始化控制 & 渲染 -----------------------------
  mujoco_ros2_control::MujocoRos2Control control(node, mujoco_model, mujoco_data);
  control.init();
  RCLCPP_INFO(logger, "Mujoco ros2 controller initialized");

  auto rendering = mujoco_ros2_control::MujocoRendering::get_instance();
  rendering->init(node, mujoco_model, mujoco_data);
  RCLCPP_INFO(logger, "Mujoco rendering initialized");
  rendering->mjv_opt_.flags[mjVIS_CONTACTPOINT] = 1;
  rendering->mjv_opt_.flags[mjVIS_CONTACTFORCE] = 1;

  // ----------------------------- 主循环 -----------------------------
  while (rclcpp::ok() && !rendering->is_close_flag_raised())
  {
    const mjtNum simstart = mujoco_data->time;
    while (mujoco_data->time - simstart < 1.0/60.0)
    {
      // 一步仿真（包含 mj_step1 + 控制器 + mj_step2）
      control.update();

      // --- 调试打印： contact 对信息 ---
      RCLCPP_INFO(logger, "DEBUG: ncon=%d nefc=%d", mujoco_data->ncon, mujoco_data->nefc);
      for (int i = 0; i < mujoco_data->ncon; ++i) {
        auto& con = mujoco_data->contact[i];
        RCLCPP_INFO(logger,
          "  contact %d: geom1=%d geom2=%d dist=%.6f exclude=%d dim=%d efc_address=%d friction=[%.3g %.3g %.3g]",
          i, con.geom1, con.geom2, con.dist,
          con.exclude, con.dim, con.efc_address,
          con.friction[0], con.friction[1], con.friction[2]
        );
      }

      // --- 1) 提取局部接触力 & 拷贝 contact frame ---
      mjtNum ft_local[6] = {0};  // Contact frame 下的 6D 力/力矩
      mjtNum cf[9];              // Contact frame→world 旋转矩阵
      bool have_contact = false;

      for (int i = 0; i < mujoco_data->ncon; ++i) {
        auto& con = mujoco_data->contact[i];
        if ((con.geom1==id_ee && con.geom2==id_box) ||
            (con.geom1==id_box && con.geom2==id_ee))
        {
          mj_contactForce(mujoco_model, mujoco_data, i, ft_local);
          // 清零切向分量，只保留法向
          ft_local[0] = 0.0;
          ft_local[1] = 0.0;
          std::memcpy(cf, con.frame, sizeof(cf));
          RCLCPP_INFO(logger,
            "DEBUG: local normal-only force=[%.6g,%.6g,%.6g], torque=[%.6g,%.6g,%.6g]",
            ft_local[0], ft_local[1], ft_local[2],
            ft_local[3], ft_local[4], ft_local[5]
          );
          have_contact = true;
          break;
        }
      }

      // --- 2) 正确映射到世界系：用 cf 的第一列 ---
      mjtNum force_world[3]  = {0}, torque_world[3] = {0};
      if (have_contact) {
        for (int r = 0; r < 3; ++r) {
          force_world[r]  = cf[3*r + 2] * ft_local[2];
          torque_world[r] = cf[3*r + 0] * ft_local[3];
        }
        RCLCPP_INFO(logger, "Contact normal (world) = [%.3f, %.3f, %.3f]",
            cf[0], cf[1], cf[2]);

        RCLCPP_INFO(logger,
          "DEBUG: world normal force=[%.6g,%.6g,%.6g], torque=[%.6g,%.6g,%.6g]",
          force_world[0], force_world[1], force_world[2],
          torque_world[0], torque_world[1], torque_world[2]
        );
      }

      // --- 3) 发布 WrenchStamped ---
      geometry_msgs::msg::WrenchStamped msg;
      msg.header.stamp    = node->get_clock()->now();
      msg.header.frame_id = "world";
      msg.wrench.force.x  = force_world[0];
      msg.wrench.force.y  = force_world[1];
      msg.wrench.force.z  = force_world[2];
      msg.wrench.torque.x = torque_world[0];
      msg.wrench.torque.y = torque_world[1];
      msg.wrench.torque.z = torque_world[2];
      wrench_pub->publish(msg);
    }

    rendering->update();  // 刷新渲染
  }

  // ----------------------------- 清理 -----------------------------
  rendering->close();
  mj_deleteData(mujoco_data);
  mj_deleteModel(mujoco_model);
  RCLCPP_INFO(logger, "MuJoCo simulation terminated. Bye!");
  return 0;
}
