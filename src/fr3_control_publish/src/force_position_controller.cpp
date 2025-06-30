#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <Eigen/Dense>
#include <algorithm>

using std::placeholders::_1;

class ForcePositionController : public rclcpp::Node {
public:
  ForcePositionController()
  : Node("force_position_controller")
  {
    // — 参数配置 —
    lambda_base_   = 0.1;
    alpha_f_       = 0.2;
    max_cart_vel_  = 0.05;
    max_joint_vel_ = 0.5;

    // 期望力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
    f_des_.setZero();
    f_des_(1) = -5.0;

    // 阻尼矩阵的逆
    Eigen::Vector<double,6> diag;
    diag << 1.0/50, 1.0/50, 1.0/50, 1.0/5, 1.0/5, 1.0/5;
    B_inv_ = diag.asDiagonal();

    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    // — 初始化 —
    q_curr_.setZero();
    q_des_.setZero();
    f_ext_sensor_.setZero();
    f_ext_filtered_.setZero();
    initialized_   = false;
    have_jacobian_ = false;
    have_wrench_   = false;
    loop_count_    = 0;

    RCLCPP_INFO(get_logger(), "ForcePositionController 初始化完成");

    // 1) 订阅 /joint_states（位置）
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&ForcePositionController::on_joint_state, this, _1));

    // 2) 订阅 /jacobian
    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10,
      std::bind(&ForcePositionController::on_jacobian, this, _1));

    // 3) 订阅 /mujoco/wrench（外力）
    sub_wrench_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/mujoco/wrench", 10,
      std::bind(&ForcePositionController::on_wrench, this, _1));

    // 4) 发布到正确的 topic
    pub_traj_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_position_controller/joint_trajectory", 10);

    // 100 Hz 控制循环
    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&ForcePositionController::control_loop, this));
  }

private:
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg) {
    initialized_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = std::find(joint_names_.begin(), joint_names_.end(),
                          msg->name[i]);
      if (it == joint_names_.end()) continue;
      size_t j = std::distance(joint_names_.begin(), it);
      q_curr_[j] = msg->position[i];
    }
    RCLCPP_DEBUG_STREAM(get_logger(),
      "q_curr: " << q_curr_.transpose());
  }

  void on_jacobian(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->layout.dim.size()!=2 ||
        msg->layout.dim[0].size!=6 ||
        msg->layout.dim[1].size!=7) {
      RCLCPP_WARN(get_logger(), "收到无效维度的雅可比");
      return;
    }
    for (size_t i=0; i<6; ++i)
      for (size_t j=0; j<7; ++j)
        J_(i,j) = msg->data[i*7 + j];
    have_jacobian_ = true;
    RCLCPP_DEBUG_STREAM(get_logger(), "J: \n" << J_);
  }

  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    f_ext_sensor_ << msg->wrench.force.x,
                     msg->wrench.force.y,
                     msg->wrench.force.z,
                     msg->wrench.torque.x,
                     msg->wrench.torque.y,
                     msg->wrench.torque.z;
    have_wrench_ = true;
    RCLCPP_DEBUG_STREAM(get_logger(),
      "Received wrench: " << f_ext_sensor_.transpose());
  }

  void control_loop() {
    if (!initialized_ || !have_jacobian_ || !have_wrench_) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
        "等待初始化 (pos=%d, jac=%d, wrench=%d)",
        initialized_, have_jacobian_, have_wrench_);
      return;
    }
    ++loop_count_;

    // 1) 构造并求逆
    Eigen::Matrix<double,6,6> M = J_ * J_.transpose();
    M += lambda_base_*lambda_base_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> M_inv = M.inverse();

    // 2) 外力估计 + 低通
    Eigen::Vector<double,6> f_ext   = f_ext_sensor_;
    f_ext_filtered_                = alpha_f_ * f_ext + (1.0 - alpha_f_) * f_ext_filtered_;
    Eigen::Vector<double,6> delta_f = f_ext_filtered_ - f_des_;

    // —— 打印所有诊断信息 —— //
    RCLCPP_INFO_STREAM(get_logger(),
      "f_ext_sensor: " << f_ext_sensor_.transpose());
    RCLCPP_INFO_STREAM(get_logger(),
      "M.matrix():\n" << M);
    RCLCPP_INFO_STREAM(get_logger(),
      "det(M):    " << M.determinant());
    RCLCPP_INFO_STREAM(get_logger(),
      "M_inv:\n" << M_inv);
    RCLCPP_INFO_STREAM(get_logger(),
      "f_ext:      " << f_ext.transpose());
    RCLCPP_INFO_STREAM(get_logger(),
      "f_ext_filt: " << f_ext_filtered_.transpose());
    RCLCPP_INFO_STREAM(get_logger(),
      "delta_f:    " << delta_f.transpose());

    // 3) 笛卡尔速度限幅
    Eigen::Vector<double,6> x_dot = B_inv_ * delta_f;
    for (int i=0; i<6; ++i)
      x_dot[i] = std::clamp(x_dot[i], -max_cart_vel_, max_cart_vel_);
    if (!x_dot.allFinite()) {
      RCLCPP_ERROR(get_logger(),
        "===== x_dot contains NaN/Inf, bailing out =====");
      return;
    }

    // 4) 阻尼伪逆 & q_dot 限幅
    Eigen::Matrix<double,7,6> J_pinv = J_.transpose() * M_inv;
    Eigen::Vector<double,7> q_dot    = J_pinv * x_dot;
    for (int i=0; i<7; ++i)
      q_dot[i] = std::clamp(q_dot[i], -max_joint_vel_, max_joint_vel_);
    RCLCPP_DEBUG_STREAM(get_logger(),
      "q_dot: " << q_dot.transpose());
    if (!q_dot.allFinite()) {
      RCLCPP_ERROR(get_logger(),
        "q_dot 包含 NaN/Inf，退出本次循环");
      return;
    }

    // 5) 积分 & 发布
    double dt = 0.01;
    q_des_ += q_dot * dt;
    RCLCPP_DEBUG_STREAM(get_logger(),
      "q_des: " << q_des_.transpose());

    trajectory_msgs::msg::JointTrajectory traj;
    traj.joint_names = joint_names_;
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions       = std::vector<double>(q_des_.data(), q_des_.data()+7);
    pt.time_from_start = rclcpp::Duration::from_seconds(dt);
    traj.points.push_back(pt);
    pub_traj_->publish(traj);
  }

  // — 成员变量 —
  double lambda_base_, alpha_f_, max_cart_vel_, max_joint_vel_;
  Eigen::Vector<double,6> f_des_, f_ext_sensor_, f_ext_filtered_;
  Eigen::DiagonalMatrix<double,6> B_inv_;
  std::vector<std::string> joint_names_;
  bool initialized_, have_jacobian_, have_wrench_;
  size_t loop_count_;
  Eigen::Vector<double,7> q_curr_, q_des_;
  Eigen::Matrix<double,6,7> J_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr      sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_traj_;
  rclcpp::TimerBase::SharedPtr                                       timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ForcePositionController>());
  rclcpp::shutdown();
  return 0;
}
