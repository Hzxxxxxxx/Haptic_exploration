#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <Eigen/Dense>
#include <algorithm>

using std::placeholders::_1;

class ForcePositionController : public rclcpp::Node {
public:
  ForcePositionController()
  : Node("force_position_controller")
  {
    // —————— 参数配置 —————— //
    lambda_base_    = 0.1;      // 初始阻尼
    alpha_f_        = 0.2;      // 力低通滤波系数
    max_cart_vel_   = 0.05;     // 笛卡尔速度限幅
    max_joint_vel_  = 0.5;      // 关节速度限幅

    // 期望力/力矩 [Fx, Fy, Fz, Mx, My, Mz]
    f_des_.setZero();
    f_des_(2) = -5.0;

    // 阻尼矩阵的逆
    Eigen::Vector<double,6> diag;
    diag << 1.0/50.0, 1.0/50.0, 1.0/50.0,
            1.0/5.0,  1.0/5.0,  1.0/5.0;
    B_inv_ = diag.asDiagonal();

    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    // —————— 初始化 —————— //
    q_curr_.setZero();
    q_des_.setZero();
    tau_.setZero();
    f_ext_filtered_.setZero();
    initialized_   = false;
    have_jacobian_ = false;
    loop_count_    = 0;

    RCLCPP_INFO(get_logger(), "ForcePositionController 初始化完成");

    // 订阅 joint_states
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&ForcePositionController::on_joint_state, this, _1));
    // 订阅 jacobian
    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10,
      std::bind(&ForcePositionController::on_jacobian, this, _1));
    // 发布轨迹
    pub_traj_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_position_controller/commands", 10);

    // 100 Hz 控制回路
    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&ForcePositionController::control_loop, this));
  }

private:
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg) {
    initialized_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      for (size_t j = 0; j < joint_names_.size(); ++j) {
        if (msg->name[i] == joint_names_[j]) {
          q_curr_[j] = msg->position[i];
          tau_[j]    = msg->effort[i];
        }
      }
    }
    RCLCPP_DEBUG_STREAM(get_logger(), "q_curr: " << q_curr_.transpose());
    RCLCPP_DEBUG_STREAM(get_logger(), "tau: "    << tau_.transpose());
  }

  void on_jacobian(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->layout.dim.size() != 2 ||
        msg->layout.dim[0].size != 6 ||
        msg->layout.dim[1].size != 7) {
      RCLCPP_WARN(get_logger(), "收到无效维度的雅可比");
      return;
    }
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < 7; ++j)
        J_(i,j) = msg->data[i*7 + j];

    have_jacobian_ = true;
    RCLCPP_DEBUG_STREAM(get_logger(), "J: \n" << J_);
  }

  void control_loop() {
    if (!initialized_ || !have_jacobian_) {
      RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
        "等待初始化 (pos=%d, jac=%d)", initialized_, have_jacobian_);
      return;
    }
    ++loop_count_;

    // 1) 构造并求逆
    Eigen::Matrix<double,6,6> M = J_ * J_.transpose();
    M += lambda_base_*lambda_base_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> M_inv = M.inverse();
    RCLCPP_DEBUG_STREAM(get_logger(), "M: \n" << M);
    RCLCPP_DEBUG_STREAM(get_logger(), "M_inv: \n" << M_inv);

    // 2) 外力估计 + 低通
    Eigen::Vector<double,6> f_ext = M_inv * (J_ * tau_);
    RCLCPP_DEBUG_STREAM(get_logger(), "f_ext (raw): " << f_ext.transpose());

    f_ext_filtered_ = alpha_f_ * f_ext + (1.0 - alpha_f_) * f_ext_filtered_;
    RCLCPP_DEBUG_STREAM(get_logger(), "f_ext_filtered: " << f_ext_filtered_.transpose());

    Eigen::Vector<double,6> delta_f = f_ext_filtered_ - f_des_;
    RCLCPP_DEBUG_STREAM(get_logger(), "delta_f: " << delta_f.transpose());

    // 3) 笛卡尔速度 & 限幅
    Eigen::Vector<double,6> x_dot = B_inv_ * delta_f;
    for (int i = 0; i < 6; ++i)
      x_dot[i] = std::clamp(x_dot[i], -max_cart_vel_, max_cart_vel_);

    if (!x_dot.allFinite()) {
    RCLCPP_ERROR(get_logger(), "===== x_dot contains NaN/Inf, bailing out =====");
    // **新**：先看 J*τ
    Eigen::Vector<double,6> Jtau = J_ * tau_;
    RCLCPP_ERROR_STREAM(get_logger(), "J*τ:       " << Jtau.transpose());
    RCLCPP_ERROR_STREAM(get_logger(), "tau:       " << tau_.transpose());
    RCLCPP_ERROR_STREAM(get_logger(), "M.matrix():\n" << M);
    RCLCPP_ERROR_STREAM(get_logger(), "det(M):    " << M.determinant());
    RCLCPP_ERROR_STREAM(get_logger(), "M_inv:\n"   << M_inv);
    RCLCPP_ERROR_STREAM(get_logger(), "f_ext:     " << f_ext.transpose());
    RCLCPP_ERROR_STREAM(get_logger(), "f_ext_filt:" << f_ext_filtered_.transpose());
    RCLCPP_ERROR_STREAM(get_logger(), "delta_f:   " << delta_f.transpose());
    return;
    }


    // 4) 逆雅可比 (阻尼伪逆) & 限幅
    Eigen::Matrix<double,7,6> J_pinv = J_.transpose() * M_inv;
    Eigen::Vector<double,7> q_dot = J_pinv * x_dot;
    for (int i = 0; i < 7; ++i)
      q_dot[i] = std::clamp(q_dot[i], -max_joint_vel_, max_joint_vel_);
    RCLCPP_DEBUG_STREAM(get_logger(), "q_dot: " << q_dot.transpose());
    if (!q_dot.allFinite()) {
      RCLCPP_ERROR(get_logger(), "q_dot 包含 NaN 或 Inf");
      return;
    }

    // 5) 积分 & 发布
    double dt = 0.01;
    q_des_ += q_dot * dt;
    RCLCPP_DEBUG_STREAM(get_logger(), "q_des: " << q_des_.transpose());

    trajectory_msgs::msg::JointTrajectory traj;
    traj.joint_names = joint_names_;
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = std::vector<double>(q_des_.data(), q_des_.data()+7);
    pt.time_from_start = rclcpp::Duration::from_seconds(dt);
    traj.points.push_back(pt);

    pub_traj_->publish(traj);
  }

  // —————— 成员变量 —————— //
  double lambda_base_, alpha_f_;
  double max_cart_vel_, max_joint_vel_;
  Eigen::Vector<double,6> f_des_, f_ext_filtered_;
  Eigen::DiagonalMatrix<double,6> B_inv_;
  std::vector<std::string> joint_names_;

  bool initialized_, have_jacobian_;
  size_t loop_count_;

  Eigen::Vector<double,7> q_curr_, q_des_, tau_;
  Eigen::Matrix<double,6,7> J_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_traj_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ForcePositionController>());
  rclcpp::shutdown();
  return 0;
}
