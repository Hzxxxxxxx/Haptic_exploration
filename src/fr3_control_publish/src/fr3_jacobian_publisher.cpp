// src/fr3_jacobian_publisher.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <mujoco/mujoco.h>

#include <vector>
#include <string>
#include <stdexcept>

class JacobianPublisher : public rclcpp::Node {
public:
  JacobianPublisher()
  : Node("jacobian_publisher")
  {
    // 1. 声明并读取参数
    this->declare_parameter<std::string>(
      "model_xml",
      "/home/mscrobotics2425laptop16/mujoco_ros_ws/src/mujoco_ros2_control/mujoco_ros2_control_demos/mujoco_models/fr3.xml"  // 修改为你的 fr3.xml 路径
    );
    this->declare_parameter<std::string>(
      "end_effector_site",
      "attachment_site"  // 默认为 fr3.xml 中定义的 <site name="attachment_site"/>
    );
    model_path_ = this->get_parameter("model_xml").as_string();
    site_name_  = this->get_parameter("end_effector_site").as_string();

    // 2. 加载 MuJoCo 模型
    char error[1000] = "Could not load model";
    m_ = mj_loadXML(model_path_.c_str(), nullptr, error, sizeof(error));
    if (!m_) {
      RCLCPP_FATAL(this->get_logger(),
                   "加载模型失败 '%s': %s", model_path_.c_str(), error);
      throw std::runtime_error("Failed to load MuJoCo model");
    }
    d_ = mj_makeData(m_);
    nv_ = m_->nv;

    // 3. 查 site_id
    site_id_ = mj_name2id(m_, mjOBJ_SITE, site_name_.c_str());
    if (site_id_ < 0) {
      RCLCPP_FATAL(this->get_logger(),
                   "Site '%s' not found in model", site_name_.c_str());
      throw std::runtime_error("Invalid end_effector_site");
    }

    // 4. 预分配雅各比缓存
    jacp_.resize(3 * nv_);
    jacr_.resize(3 * nv_);

    // 5. 订阅 joint_states，发布 jacobian
    sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&JacobianPublisher::on_joint_state, this, std::placeholders::_1)
    );
    pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10
    );

    RCLCPP_INFO(this->get_logger(),
                "JacobianPublisher 启动，模型='%s'，site='%s'",
                model_path_.c_str(), site_name_.c_str());
  }

  ~JacobianPublisher() override {
    mj_deleteData(d_);
    mj_deleteModel(m_);
  }

private:
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg) {
    // 更新 qpos
    for (size_t i = 0; i < msg->name.size(); ++i) {
      int jid = mj_name2id(m_, mjOBJ_JOINT, msg->name[i].c_str());
      if (jid < 0) continue;
      int qadr = m_->jnt_qposadr[jid];
      d_->qpos[qadr] = msg->position[i];
    }

    // 前向运动学
    mj_forward(m_, d_);

    // 计算雅各比：线速度 jacp_，角速度 jacr_
    mj_jacSite(m_, d_, jacp_.data(), jacr_.data(), site_id_);

    // 拼接 6×nv
    std_msgs::msg::Float64MultiArray out;
    out.layout.dim.resize(2);
    out.layout.dim[0].label = "rows";
    out.layout.dim[0].size  = 6;
    out.layout.dim[1].label = "cols";
    out.layout.dim[1].size  = nv_;
    out.data.reserve(6 * nv_);
    // 线速度部分
    for (double v : jacp_) out.data.push_back(v);
    // 角速度部分
    for (double v : jacr_) out.data.push_back(v);

    pub_->publish(std::move(out));
  }

  // MuJoCo
  mjModel* m_{nullptr};
  mjData*  d_{nullptr};
  int      nv_{0};
  int      site_id_{-1};
  std::string model_path_, site_name_;
  std::vector<double> jacp_, jacr_;

  // ROS 2
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<JacobianPublisher>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"),
                 "节点启动失败: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
