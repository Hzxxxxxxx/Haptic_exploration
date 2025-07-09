#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>                 // ← 仍需
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>

using std::placeholders::_1;

class ForcePositionController : public rclcpp::Node
{
public:
  ForcePositionController()
  : Node("force_position_controller")
  {
    /* ---------- 1. 参数 ---------- */
    lambda_base_   = 0.1;
    max_cart_vel_  = 0.05;
    max_joint_vel_ = 0.5;

    f_des_.setZero();
    f_des_(1) = -5.0;

    Eigen::Vector<double,6> diag;
    diag << 1.0/50, 1.0/50, 1.0/50, 1.0/5, 1.0/5, 1.0/5;
    B_inv_ = diag.asDiagonal();

    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    /* ---------- 2. 运行时状态 ---------- */
    q_curr_.setZero();
    q_des_.setZero();
    f_ext_sensor_.setZero();
    initialized_   = false;
    have_jacobian_ = false;
    have_wrench_   = false;
    loop_count_    = 0;

    RCLCPP_INFO(get_logger(), "ForcePositionController 启动完成");

    /* ---------- 3. 订阅 ---------- */
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&ForcePositionController::on_joint_state, this, _1));

    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10,
      std::bind(&ForcePositionController::on_jacobian, this, _1));

    sub_wrench_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
  "/touch_tip/wrench", rclcpp::SensorDataQoS(),
  std::bind(&ForcePositionController::on_wrench, this, _1));


    /* ---------- 4. 发布 & 定时器 ---------- */
    pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10);                // ← 新话题

    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),                             // 100 Hz
      std::bind(&ForcePositionController::control_loop, this));

    /* ---------- 5. 打开 CSV 日志 ---------- */
    log_file_.open("q_des_log.csv", std::ios::out);
    if (!log_file_.is_open()) {
      RCLCPP_ERROR(get_logger(), "无法创建 q_des_log.csv，结束节点");
      rclcpp::shutdown();
    }
    log_file_ << "time,q1,q2,q3,q4,q5,q6,q7\n";
  }

  ~ForcePositionController() override
  {
    if (log_file_.is_open())
      log_file_.close();
  }

private:
  /* ---------- 关节状态回调 ---------- */
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    initialized_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[i]);
      if (it == joint_names_.end()) continue;
      q_curr_[std::distance(joint_names_.begin(), it)] = msg->position[i];
    }
  }

  /* ---------- 雅可比回调 ---------- */
  void on_jacobian(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->layout.dim.size()!=2 ||
        msg->layout.dim[0].size!=6 || msg->layout.dim[1].size!=7) {
      RCLCPP_WARN(get_logger(), "收到无效维度的雅可比");
      return;
    }
    for (size_t i=0; i<6; ++i)
      for (size_t j=0; j<7; ++j)
        J_(i,j) = msg->data[i*7 + j];
    have_jacobian_ = true;
  }

  /* ---------- 外力回调 ---------- */
  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != "touch_tip") return;
    f_ext_sensor_ << msg->wrench.force.x,  msg->wrench.force.y,  msg->wrench.force.z,
                     msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;
    have_wrench_ = true;
  }

  /* ---------- 主控制循环 ---------- */
  void control_loop()
  {
    if (!initialized_ || !have_jacobian_ || !have_wrench_) return;
    ++loop_count_;

    // 1) 阻尼正则化逆
    Eigen::Matrix<double,6,6> M = J_ * J_.transpose();
    M += lambda_base_ * lambda_base_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> M_inv = M.inverse();

    // 2) 力错差
    Eigen::Vector<double,6> delta_f = f_ext_sensor_ - f_des_;

    // 3) 篞卡尔速度
    Eigen::Vector<double,6> x_dot = B_inv_ * delta_f;

    // 4) 关节速度
    Eigen::Vector<double,7> q_dot = (J_.transpose() * M_inv) * x_dot;

    // 5) 数值积分
    constexpr double dt = 0.01;
    q_des_ += q_dot * dt;

    // 6) 发布到 /joint_position_controller/commands
    std_msgs::msg::Float64MultiArray cmd_msg;
    cmd_msg.data.assign(q_des_.data(), q_des_.data() + 7);
    pub_cmd_->publish(cmd_msg);

    // 7) 记录 CSV
    double t_now = loop_count_ * dt;
    log_file_ << t_now;
    for (int i = 0; i < 7; ++i)
      log_file_ << ',' << q_des_[i];
    log_file_ << '\n';
    log_file_.flush();

    /* —— 诊断信息 —— */
    RCLCPP_INFO_STREAM(get_logger(), "\n===== LOOP " << loop_count_ << " =====");
    RCLCPP_INFO_STREAM(get_logger(), "f_ext_sensor : " << f_ext_sensor_.transpose());
    RCLCPP_INFO_STREAM(get_logger(), "delta_f      : " << delta_f.transpose());
    RCLCPP_INFO_STREAM(get_logger(), "q_dot        : " << q_dot.transpose());
    RCLCPP_INFO_STREAM(get_logger(), "q_des        : " << q_des_.transpose());
  }

  /* ---------- 成员 ---------- */
  double lambda_base_, max_cart_vel_, max_joint_vel_;
  Eigen::Vector<double,6> f_des_, f_ext_sensor_;
  Eigen::DiagonalMatrix<double,6> B_inv_;
  std::vector<std::string> joint_names_;
  bool initialized_, have_jacobian_, have_wrench_;
  size_t loop_count_;
  Eigen::Vector<double,7> q_curr_, q_des_;
  Eigen::Matrix<double,6,7> J_;
  std::ofstream log_file_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr            sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr        sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr       sub_wrench_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr           pub_cmd_;   // ← 新 publisher
  rclcpp::TimerBase::SharedPtr                                             timer_;
};

/* ---------- main ---------- */
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ForcePositionController>());
  rclcpp::shutdown();
  return 0;
}
