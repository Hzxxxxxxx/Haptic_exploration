#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>                 
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <Eigen/Dense>
#include <algorithm>

using std::placeholders::_1;

class TangentialVelocityController : public rclcpp::Node
{
public:
  TangentialVelocityController()
  : Node("tangential_velocity_controller")
  {
    // 1) 参数声明与获取
    this->declare_parameter<double>("v_t", 0.005);               // 切向速度 (m/s)
    this->declare_parameter<std::vector<double>>(
      "reference_axis", std::vector<double>{1.0, 0.0, 0.0});     // 参考方向
    this->declare_parameter<double>("damping_lambda", 0.1);      // 阻尼伪逆正则化
    this->declare_parameter<double>("min_force", 0.05);          // 法向力阈值 (N)

    this->get_parameter("v_t",            v_t_);
    this->get_parameter("damping_lambda", lambda_);
    this->get_parameter("min_force",      min_force_);
    std::vector<double> a;
    this->get_parameter("reference_axis", a);
    ref_axis_ = Eigen::Vector3d(a[0], a[1], a[2]).normalized();

    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    have_js_       = false;
    have_jacobian_ = false;
    have_wrench_   = false;
    loop_count_    = 0;

    RCLCPP_INFO(get_logger(), "TangentialVelocityController 启动完成");

    // 2) 订阅
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&TangentialVelocityController::on_joint_state, this, _1));

    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10,
      std::bind(&TangentialVelocityController::on_jacobian, this, _1));

    sub_wrench_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/touch_tip/wrench", rclcpp::SensorDataQoS(),
      std::bind(&TangentialVelocityController::on_wrench, this, _1));

    // 3) 发布到 /joint_position_controller/commands
    pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10);

    // 4) 定时器 (100 Hz)
    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&TangentialVelocityController::control_loop, this));
  }

private:
  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    have_js_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[i]);
      if (it == joint_names_.end()) continue;
      q_curr_[std::distance(joint_names_.begin(), it)] = msg->position[i];
    }
  }

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

  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != "touch_tip") return;
    f_ext_ << msg->wrench.force.x,
              msg->wrench.force.y,
              msg->wrench.force.z;
    have_wrench_ = true;
  }

  void control_loop()
  {
    if (!have_js_ || !have_jacobian_ || !have_wrench_) return;
    ++loop_count_;

    // —— 新增：法向力阈值检测 —— 
    double f_norm = f_ext_.norm();
    if (f_norm < min_force_) {
      RCLCPP_INFO_STREAM(get_logger(),
        "[Loop " << loop_count_ << "] 法向力 " << f_norm 
        << "N 低于阈值 " << min_force_ << "N，跳过切向运动");
      return;
    }

    // 1) 计算接触面法向
    Eigen::Vector3d n = f_ext_.normalized();

    // 2) 切平面投影
    Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - n * n.transpose();
    Eigen::Vector3d t_unnorm = P * ref_axis_;
    if (t_unnorm.norm() < 1e-6) {
      RCLCPP_WARN(get_logger(), "参考向量与法向平行，跳过本次循环");
      return;
    }
    Eigen::Vector3d t = t_unnorm.normalized();

    // 3) 生成笛卡尔切向速度
    Eigen::Matrix<double,6,1> x_dot = Eigen::Matrix<double,6,1>::Zero();
    x_dot.template head<3>() = v_t_ * t;

    // 4) 阻尼伪逆映射到关节速度
    Eigen::Matrix<double,6,6> M = J_ * J_.transpose()
      + lambda_ * lambda_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,6> M_inv = M.inverse();
    Eigen::Vector<double,7> q_dot = J_.transpose() * M_inv * x_dot;

    // 5) 数值积分生成目标位置
    constexpr double dt = 0.01;
    q_des_ += q_dot * dt;

    // 6) 发布
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.assign(q_des_.data(), q_des_.data() + 7);
    pub_cmd_->publish(cmd);

    // 7) 日志打印
    RCLCPP_INFO_STREAM(get_logger(),
      "[Loop " << loop_count_ << "] f_norm=" << f_norm
      << " t_dir=" << t.transpose()
      << " q_dot=" << q_dot.transpose());
  }

  // 参数
  double v_t_, lambda_, min_force_;
  Eigen::Vector3d ref_axis_, f_ext_;
  std::vector<std::string> joint_names_;

  // 状态
  bool have_js_, have_jacobian_, have_wrench_;
  size_t loop_count_;
  Eigen::Vector<double,7> q_curr_, q_des_;
  Eigen::Matrix<double,6,7> J_;

  // ROS 接口
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr    sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr  pub_cmd_;
  rclcpp::TimerBase::SharedPtr                                    timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TangentialVelocityController>());
  rclcpp::shutdown();
  return 0;
}
