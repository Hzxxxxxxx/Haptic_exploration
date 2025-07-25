#include "mujoco/mujoco.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "mujoco_ros2_control/mujoco_rendering.hpp"
#include "mujoco_ros2_control/mujoco_ros2_control.hpp"

#include "mujoco/mjvisualize.h"

#include <array>
#include <cstring>

// MuJoCo data structures
static mjModel *mujoco_model = nullptr;
static mjData  *mujoco_data  = nullptr;

int main(int argc, const char **argv)
{
  // ----------------------------- ROS 2 初始化 -----------------------------
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "mujoco_ros2_control_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // 读取模型路径
  const auto model_path = node->get_parameter("mujoco_model_path").as_string();

  // ----------------------------- 加载 MuJoCo 模型 -----------------------------
  char error[1000] = "Could not load binary model";
  if (model_path.size()>4 &&
      !std::strcmp(model_path.c_str()+model_path.size()-4, ".mjb"))
  {
    mujoco_model = mj_loadModel(model_path.c_str(), nullptr);
  }
  else
  {
    mujoco_model = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
  }
  if (!mujoco_model) {
    mju_error("Load model error: %s", error);
  }
  mujoco_data = mj_makeData(mujoco_model);

  // ----------------------------- 获取必要的 ID -----------------------------
  // 碰撞几何体
  int id_ee  = mj_name2id(mujoco_model, mjOBJ_GEOM, "touch_tip_collision");
  int id_box = mj_name2id(mujoco_model, mjOBJ_GEOM, "exploration_object_collision");
  if (id_ee<0 || id_box<0) {
    mju_error("找不到 touch_tip_collision 或 exploration_object_collision geom");
  }
  // 工具坐标系 site，用于后续投影
  int site_id = mj_name2id(mujoco_model, mjOBJ_SITE, "touch_tip");
  if (site_id<0) {
    mju_error("找不到 touch_tip site");
  }

  // ----------------------------- ROS 2 发布器 -----------------------------
  auto wrench_pub = node->create_publisher<geometry_msgs::msg::WrenchStamped>(
    "touch_tip/wrench", rclcpp::SensorDataQoS());

  // ----------------------------- 初始化控制 & 渲染 -----------------------------
  mujoco_ros2_control::MujocoRos2Control control(node, mujoco_model, mujoco_data);
  control.init();

  auto rendering = mujoco_ros2_control::MujocoRendering::get_instance();
  rendering->init(node, mujoco_model, mujoco_data);

  // 可视化接触点与力
  rendering->mjv_opt_.flags[mjVIS_CONTACTPOINT] = 1;
  rendering->mjv_opt_.flags[mjVIS_CONTACTFORCE] = 1;

  // ----------------------------- 主循环 -----------------------------
  while (rclcpp::ok() && !rendering->is_close_flag_raised())
  {
    const mjtNum simstart = mujoco_data->time;
    while (mujoco_data->time - simstart < 1.0/60.0)
    {
      // 完成一步仿真（mj_step1 + controllers + mj_step2）
      control.update();

      // —— 1) 抽取第一个 EE–Box 接触的局部力，并拷贝 contact frame —— 
      mjtNum ft_local[6] = {0};   // 局部 6D 接触力/力矩
      mjtNum cf[9];               // contact frame 3×3 旋转矩阵
      bool have_contact = false;

      for (int i = 0; i < mujoco_data->ncon; ++i) {
        mjContact& con = mujoco_data->contact[i];
        if ((con.geom1==id_ee && con.geom2==id_box) ||
            (con.geom1==id_box && con.geom2==id_ee))
        {
          // 取局部接触力/力矩（contact frame）
          mj_contactForce(mujoco_model, mujoco_data, i, ft_local);
          // 拷贝旋转矩阵（row-major）
          std::memcpy(cf, con.frame, 9*sizeof(mjtNum));
          have_contact = true;
          break;
        }
      }

      // —— 2) 计算完整接触力在世界系下的向量 —— 
      mjtNum f_world_full[3]  = {0}, tau_world_full[3] = {0};
      if (have_contact) {
        // row-major 矩阵乘法： world = cf * local
        for (int r = 0; r < 3; ++r) {
          f_world_full[r]   = cf[3*r+0]*ft_local[0]
                            + cf[3*r+1]*ft_local[1]
                            + cf[3*r+2]*ft_local[2];
          tau_world_full[r] = cf[3*r+0]*ft_local[3]
                            + cf[3*r+1]*ft_local[4]
                            + cf[3*r+2]*ft_local[5];
        }
      }

      // —— 3) 计算工具 Z 轴在世界系下的单位向量 —— 
      const mjtNum* xmat = mujoco_data->site_xmat + 9*site_id; 
      // xmat row-major: xmat[3*i+j] is row i, col j
      // col 2 (index j=2) is local Z axis in world:
      mjtNum tool_z[3] = { xmat[2], xmat[5], xmat[8] };

      // —— 4) 将完整世界力投影到工具 Z 轴 —— 
      mjtNum f_proj = 0.0, tau_proj = 0.0;
      if (have_contact) {
        for (int i = 0; i < 3; ++i) {
          f_proj   += f_world_full[i]   * tool_z[i];
          tau_proj += tau_world_full[i] * tool_z[i];
        }
      }

      // —— 5) 构建沿工具法向的世界系力向量 —— 
      mjtNum force_world[3]  = {0}, torque_world[3] = {0};
      for (int i = 0; i < 3; ++i) {
        force_world[i]  = tool_z[i] * f_proj;
        torque_world[i] = tool_z[i] * tau_proj;
      }

      // —— 6) 发布 WrenchStamped —— 
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

    rendering->update();
  }

  // ----------------------------- 清理 -----------------------------
  rendering->close();
  mj_deleteData(mujoco_data);
  mj_deleteModel(mujoco_model);

  return 0;
}
