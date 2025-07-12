#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>

using std::placeholders::_1;

class HybridForceTangentialController : public rclcpp::Node
{
public:
  HybridForceTangentialController()
  : Node("hybrid_force_tangential_controller"),
    initialized_(false)
  {
    // — 参数声明 —
    this->declare_parameter<double>("lambda_force", 0.1);
    this->declare_parameter<std::vector<double>>(
      "B_inv_diag", std::vector<double>{1.0/50,1.0/50,1.0/50,1.0/5,1.0/5,1.0/5});
    this->declare_parameter<std::vector<double>>(
      "f_des", std::vector<double>{0.0, -5.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<double>("v_t", 0.005);
    this->declare_parameter<std::vector<double>>(
      "reference_axis", std::vector<double>{1.0,0.0,0.0});
    this->declare_parameter<double>("min_force", 0.05);
    this->declare_parameter<double>("lambda_tangent", 0.1);

    // — 参数获取 —
    this->get_parameter("lambda_force", lambda_f_);
    std::vector<double> bdiag, fdes_vec, a;
    this->get_parameter("B_inv_diag", bdiag);
    this->get_parameter("f_des",       fdes_vec);
    this->get_parameter("v_t",         v_t_);
    this->get_parameter("reference_axis", a);
    this->get_parameter("min_force",   min_force_);
    this->get_parameter("lambda_tangent", lambda_t_);

    // 构造 B_inv 和 f_des 向量
    B_inv_ = Eigen::DiagonalMatrix<double,6>(
      bdiag[0], bdiag[1], bdiag[2],
      bdiag[3], bdiag[4], bdiag[5]);
    for(int i=0; i<6; ++i) {
      f_des_[i] = fdes_vec[i];
    }
    ref_axis_ = Eigen::Vector3d(a[0],a[1],a[2]).normalized();

    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    // — 状态初始化 —
    have_js_ = have_jacobian_ = have_wrench_ = false;
    q_curr_.setZero();
    q_des_.setZero();
    loop_count_ = 0;

    // 切向方向初始
    tangent_dir_ = ref_axis_;
    
    // — 构造时打印参数 —
    RCLCPP_INFO(get_logger(), "HybridForceTangentialController 启动");
    RCLCPP_INFO_STREAM(get_logger(),
      " params: lambda_force=" << lambda_f_
      << ", v_t=" << v_t_
      << ", min_force=" << min_force_
      << ", lambda_tangent=" << lambda_t_);
    RCLCPP_INFO_STREAM(get_logger(),
      " B_inv diag = [" << B_inv_.diagonal().transpose() << "]");
    RCLCPP_INFO_STREAM(get_logger(),
      " f_des = [" << f_des_.transpose() << "]");
    RCLCPP_INFO_STREAM(get_logger(),
      " ref_axis = [" << ref_axis_.transpose() << "]");

    // — 订阅 —
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states",  10, std::bind(&HybridForceTangentialController::on_js, this, _1));
    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian",      10, std::bind(&HybridForceTangentialController::on_jac, this, _1));
    sub_wrench_   = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/touch_tip/wrench", rclcpp::SensorDataQoS(),
      std::bind(&HybridForceTangentialController::on_wrench, this, _1));

    // — 发布 & 定时器 — 
    pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10);
    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&HybridForceTangentialController::control_loop, this));
  }

private:
  // JointState 回调
  void on_js(const sensor_msgs::msg::JointState::SharedPtr msg) {
    have_js_ = true;
    for(size_t i=0;i<msg->name.size();++i){
      auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[i]);
      if(it==joint_names_.end()) continue;
      size_t idx = std::distance(joint_names_.begin(), it);
      q_curr_[idx] = msg->position[i];
    }
    if (!initialized_) {
      q_des_ = q_curr_;
      initialized_ = true;
      RCLCPP_INFO(get_logger(), "q_des_ 已初始化为当前关节位置");
    }
    RCLCPP_DEBUG(get_logger(), "Got joint_states");
  }

  // Jacobian 回调
  void on_jac(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if(msg->layout.dim.size()!=2 ||
       msg->layout.dim[0].size!=6 || msg->layout.dim[1].size!=7) {
      RCLCPP_WARN(get_logger(),"雅可比维度错误");
      return;
    }
    for(int i=0;i<6;++i)
      for(int j=0;j<7;++j)
        J_(i,j) = msg->data[i*7+j];
    have_jacobian_ = true;
    RCLCPP_DEBUG(get_logger(), "Got Jacobian");
  }

  // Wrench 回调
  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    if(msg->header.frame_id!="touch_tip") return;
    f_ext3_ << msg->wrench.force.x,
               msg->wrench.force.y,
               msg->wrench.force.z;
    f_ext6_ << msg->wrench.force.x,
               msg->wrench.force.y,
               msg->wrench.force.z,
               msg->wrench.torque.x,
               msg->wrench.torque.y,
               msg->wrench.torque.z;
    have_wrench_ = true;
    RCLCPP_DEBUG_STREAM(get_logger(),
      "Got wrench: f=" << f_ext3_.transpose()
      << ", tau=[" << msg->wrench.torque.x << ","
      << msg->wrench.torque.y << ","
      << msg->wrench.torque.z << "]");
  }

  // 主控制循环
  void control_loop() {
    if (!(have_js_ && have_jacobian_ && have_wrench_ && initialized_)) {
      RCLCPP_DEBUG(get_logger(), "等待所有消息就绪...");
      return;
    }
    ++loop_count_;

    // — 法向导纳速度 x_dot_n —
    Eigen::Matrix<double,6,6> W = J_ * J_.transpose()
      + lambda_f_*lambda_f_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,1> delta_f = f_ext6_ - f_des_;
    Eigen::Matrix<double,6,1> x_dot_n = B_inv_ * delta_f;
    Eigen::Matrix<double,6,6> W_inv = W.inverse();

    // — 切向方向更新（无平滑） & 恒定速度 x_dot_t —
    double f_norm = f_ext3_.norm();
    if (f_norm >= min_force_) {
      Eigen::Vector3d n = f_ext3_.normalized();
      Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - n * n.transpose();
      Eigen::Vector3d cand = P * ref_axis_;
      if (cand.norm() > 1e-6) {
        tangent_dir_ = cand.normalized();
      }
    }
    Eigen::Matrix<double,6,1> x_dot_t = Eigen::Matrix<double,6,1>::Zero();
    x_dot_t.template head<3>() = v_t_ * tangent_dir_;

    // — 合成 & 伪逆映射 to q_dot —
    Eigen::Matrix<double,6,1> x_dot = x_dot_n + x_dot_t;
    Eigen::Matrix<double,7,1> q_dot = J_.transpose() * W_inv * x_dot;

    // — 积分 & 发布 q_des_ — 
    constexpr double dt = 0.01;
    q_des_ += q_dot * dt;
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.assign(q_des_.data(), q_des_.data() + 7);
    pub_cmd_->publish(cmd);

    // — 诊断信息 —
    RCLCPP_INFO_STREAM(get_logger(),
      "[Loop " << loop_count_ << "]\n"
      << "f_ext=[" << f_ext3_.transpose() << "]\n"
      << "|f|=" << f_norm << "\n"
      << "tangent_dir=[" << tangent_dir_.transpose() << "]\n"
      << "ẋ_n=[" << x_dot_n.transpose() << "]\n"
      << "ẋ_t=[" << x_dot_t.transpose() << "]\n"
      << "q̇=[" << q_dot.transpose() << "]");
  }

  // 成员变量
  double lambda_f_, v_t_, min_force_, lambda_t_;
  Eigen::DiagonalMatrix<double,6> B_inv_;
  Eigen::Matrix<double,6,7>       J_;
  Eigen::Matrix<double,7,1>       q_curr_, q_des_;
  Eigen::Matrix<double,6,1>       f_ext6_;
  Eigen::Vector3d                 f_ext3_, ref_axis_, tangent_dir_;
  Eigen::Matrix<double,6,1>       f_des_;
  std::vector<std::string>        joint_names_;

  bool have_js_, have_jacobian_, have_wrench_, initialized_;
  size_t loop_count_;

  // ROS 接口
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr    sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr   pub_cmd_;
  rclcpp::TimerBase::SharedPtr                                     timer_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HybridForceTangentialController>());
  rclcpp::shutdown();
  return 0;
}
