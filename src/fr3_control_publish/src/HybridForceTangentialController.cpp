#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <algorithm>

using std::placeholders::_1;

// 扫描状态机各阶段枚举
enum class ScanState {
  INIT_SCAN,       // 初始化：沿目标中心方向的法向力导纳运动
  PAUSE_MEASURE,   // 停顿测量：冻结导纳，采集静态力
  ALIGN_ORIENT,    // 姿态对齐：围绕接触点纯旋转与法向对齐
  UNFREEZE,        // 解冻导纳：恢复B_inv并设定新目标力
  HORIZONTAL_SCAN, // 横向离散扫描：左右两侧边界记录
  ADVANCE_ON_AXIS, // 主轴离散推进：沿主轴步进
  DONE             // 扫描完成
};

class HybridForceTangentialController : public rclcpp::Node
{
public:
  HybridForceTangentialController()
  : Node("hybrid_force_tangential_controller"),
    state_(ScanState::INIT_SCAN), admittance_frozen_(false), loop_count_(0)
  {
    // ------------ 参数声明并注释 ------------
    declare_parameter<double>("f_high", 5.0);      // 接触阈值, 法向力 ≥ f_high 判定接触 (单位: N)
    declare_parameter<double>("f_low", 2.0);       // 离开阈值, 法向力 ≤ f_low 判定离开 (单位: N)
    declare_parameter<double>("step_length", 0.02); // 主轴离散步长 (单位: m)
    declare_parameter<double>("lateral_step", 0.01);// 横向离散步长 (单位: m)
    declare_parameter<double>("axis_speed", 0.01);  // 主轴导纳限速 (单位: m/s)
    declare_parameter<double>("lateral_speed", 0.01);// 横向导纳限速 (单位: m/s)
    declare_parameter<double>("T_pause", 0.15);    // 静态测力窗口时长 (单位: s)
    declare_parameter<int>("N_min", 10);           // 窗口内最少采样帧数
    declare_parameter<double>("f_eps", 0.2);       // 力波动阈值, max-min<f_eps 认为稳定
    declare_parameter<double>("theta_tol_deg", 2.0);// 姿态对齐容差 (单位: deg)
    declare_parameter<double>("k_theta", 0.2);     // 姿态微旋系数, δθ=k_theta·θ
    declare_parameter<double>("object_center_x", 0.7);// 物体中心X坐标 (单位: m)
    declare_parameter<double>("object_center_y", 0.0);// 物体中心Y坐标 (单位: m)
    declare_parameter<std::string>("ee_pose_topic", "/ee_pose"); // 末端位姿话题名

    // ------------ 参数获取 ------------
    get_parameter("f_high", f_high_);
    get_parameter("f_low", f_low_);
    get_parameter("step_length", step_length_);
    get_parameter("lateral_step", lateral_step_);
    get_parameter("axis_speed", axis_speed_);
    get_parameter("lateral_speed", lateral_speed_);
    get_parameter("T_pause", T_pause_);
    get_parameter("N_min", N_min_);
    get_parameter("f_eps", f_eps_);
    double theta_tol_deg;
    get_parameter("theta_tol_deg", theta_tol_deg);
    theta_tol_ = theta_tol_deg * M_PI / 180.0; // 转换为弧度
    get_parameter("k_theta", k_theta_);
    double obj_x, obj_y;
    get_parameter("object_center_x", obj_x);
    get_parameter("object_center_y", obj_y);
    get_parameter("ee_pose_topic", ee_pose_topic_);

    RCLCPP_INFO(get_logger(), "参数: f_high=%.2f, f_low=%.2f, step_length=%.3f, lateral_step=%.3f", 
                f_high_, f_low_, step_length_, lateral_step_);
    RCLCPP_INFO(get_logger(), "参数: axis_speed=%.3f, lateral_speed=%.3f, T_pause=%.3f", 
                axis_speed_, lateral_speed_, T_pause_);
    RCLCPP_INFO(get_logger(), "参数: theta_tol=%.3f rad, k_theta=%.3f", theta_tol_, k_theta_);
    RCLCPP_INFO(get_logger(), "物体中心: (%.3f, %.3f)", obj_x, obj_y);

    // 初始化虚拟阻尼矩阵 (B_inv) 和期望力 f_des
    B_inv_ = Eigen::DiagonalMatrix<double,6>(
      1/5000.0,1/5000.0,1/5000.0,  // XYZ 方向阻尼倒数
      1/500.0,1/500.0,1/500.0      // RPY 方向阻尼倒数
    );
    f_des_.setZero();

    // 计算主轴与横向轴（XY 平面）
    Eigen::Vector2d base(0.0, 0.0);
    Eigen::Vector2d center(obj_x, obj_y);
    main_axis_    = (center - base).normalized();
    lateral_axis_ = Eigen::Vector2d(-main_axis_.y(), main_axis_.x());
    object_center_.x() = obj_x;
    object_center_.y() = obj_y;

    RCLCPP_INFO_STREAM(get_logger(), "main_axis = [" << main_axis_.transpose() << "]");
    RCLCPP_INFO_STREAM(get_logger(), "lateral_axis = [" << lateral_axis_.transpose() << "]");

    // 初始化变量
    q_curr_.setZero();  q_des_.setZero();
    J_.setZero();      ee_pos_.setZero();
    R_t_.setIdentity(); contact_pt_.setZero();

    // 订阅: 关节状态, 雅可比, 触觉, 末端位姿
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, std::bind(&HybridForceTangentialController::on_js, this, _1));
    sub_jacobian_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10, std::bind(&HybridForceTangentialController::on_jac, this, _1));
    sub_wrench_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/touch_tip/wrench", rclcpp::SensorDataQoS(), std::bind(&HybridForceTangentialController::on_wrench, this, _1));
    sub_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ee_pose_topic_, 10, std::bind(&HybridForceTangentialController::on_pose, this, _1));

    // 发布: 期望关节位置
    pub_cmd_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10);
    timer_ = create_wall_timer(
      std::chrono::milliseconds(10), std::bind(&HybridForceTangentialController::control_loop, this));

    RCLCPP_INFO(get_logger(), "HybridForceTangentialController 初始化完成");
  }

private:

  // --- 回调函数 ---
  void on_js(const sensor_msgs::msg::JointState::SharedPtr msg) {
    have_js_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[i]);
      if (it == joint_names_.end()) continue;
      size_t idx = std::distance(joint_names_.begin(), it);
      q_curr_(idx) = msg->position[i];
    }
    if (!initialized_) {
      q_des_ = q_curr_;
      initialized_ = true;
      RCLCPP_INFO(get_logger(), "首次初始化期望关节位置");
    }
  }

  void on_jac(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // 确保维度 6×7
    if (msg->layout.dim.size()==2 && msg->layout.dim[0].size==6 && msg->layout.dim[1].size==7) {
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 7; ++j)
          J_(i,j) = msg->data[i*7 + j];
      have_jacobian_ = true;
    } else {
      RCLCPP_WARN(get_logger(), "收到错误维度的雅可比: [%zu, %zu]", 
                  msg->layout.dim[0].size, msg->layout.dim[1].size);
    }
  }

  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    if (msg->header.frame_id != "touch_tip") return;
    f_ext6_.head<3>() = Eigen::Vector3d(
      msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z);
    f_ext6_.tail<3>() = Eigen::Vector3d(
      msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z);
    have_wrench_ = true;
  }

  void on_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    ee_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    Eigen::Quaterniond q(msg->pose.orientation.w,
                         msg->pose.orientation.x,
                         msg->pose.orientation.y,
                         msg->pose.orientation.z);
    R_t_ = q.normalized().toRotationMatrix();
    have_pose_ = true;
  }

  // --- 主循环 ---
  void control_loop() {
    // 打印状态以便调试
    RCLCPP_DEBUG(get_logger(), "[Loop %zu] state=%d js=%d jac=%d wrench=%d pose=%d init=%d",
                  loop_count_, static_cast<int>(state_), have_js_, have_jacobian_, have_wrench_, have_pose_, initialized_);
    if (!(have_js_ && have_jacobian_ && have_wrench_ && have_pose_ && initialized_)) return;

    double f_norm = f_ext6_.head<3>().norm();
    RCLCPP_INFO(get_logger(), "当前法向力 f_norm=%.3f", f_norm);

    switch (state_) {

    case ScanState::INIT_SCAN: {
      // 1 MOVE: 法向力导纳向目标中心拉动
      // 只在第一次进入 INIT_SCAN 时算一次
      if (!init_scan_dir_computed_) {
        Eigen::Vector3d dir2d(object_center_.x() - ee_pos_.x(),
                          object_center_.y() - ee_pos_.y(), 0);
        dir2d.normalize();
        init_scan_dir_ = dir2d;
        init_scan_dir_computed_ = true;
      }
      f_des_.head<3>() = init_scan_dir_ * (-200);
      compute_admittance();
      RCLCPP_DEBUG(get_logger(), "INIT_SCAN: f_des=[%.3f,%.3f,%.3f]", 
                   f_des_.x(), f_des_.y(), f_des_.z());
      if (f_norm >= f_high_) {
        contact_pt_ = ee_pos_;
        RCLCPP_INFO(get_logger(), "首次接触, contact_pt=(%.3f,%.3f,%.3f)", 
                     contact_pt_.x(), contact_pt_.y(), contact_pt_.z());
        state_ = ScanState::PAUSE_MEASURE;
        pause_start_ = now();
        f_window_.clear();
      }
      break;
    }

    case ScanState::PAUSE_MEASURE:
      // 2 PAUSE_MEASURE: 冻结导纳 & 收集静态力
      if (!admittance_frozen_) {
        B_inv_saved_ = B_inv_;
        f_des_saved_ = f_des_;
        B_inv_.setZero();
        admittance_frozen_ = true;
        RCLCPP_INFO(get_logger(), "冻结导纳, 进入静态测力窗口");
      }
      f_window_.push_back(f_ext6_);
      if ((now() - pause_start_).seconds() >= T_pause_ && f_window_.size() >= (size_t)N_min_) {
        auto f_est = median6(f_window_);
        n_hat_   = f_est.head<3>().normalized();
        RCLCPP_INFO(get_logger(), "测力完成: f_est=[%.3f,%.3f,%.3f], n_hat=[%.3f,%.3f,%.3f]",
                     f_est.x(), f_est.y(), f_est.z(), n_hat_.x(), n_hat_.y(), n_hat_.z());
        state_ = ScanState::ALIGN_ORIENT;
      }
      break;

    case ScanState::ALIGN_ORIENT: {
      // 3 ALIGN_ORIENT: 围绕contact_pt_纯旋对齐工具Z轴与n_hat_
      Eigen::Vector3d z_tcp = R_t_.col(2);
      Eigen::Vector3d r = ee_pos_ - contact_pt_;
      double theta = std::acos(z_tcp.dot(n_hat_));
      RCLCPP_DEBUG(get_logger(), "ALIGN_ORIENT: theta=%.3f deg", theta * 180.0/M_PI);
      if (theta <= theta_tol_) {
        RCLCPP_INFO(get_logger(), "姿态对齐完成, theta=%.3f deg", theta * 180.0/M_PI);
        state_ = ScanState::UNFREEZE;
        break;
      }
      Eigen::Vector3d omega_hat = z_tcp.cross(n_hat_).normalized();
      double delta = k_theta_ * theta;
      double w_mag = delta / dt_;
      Eigen::Vector3d omega = omega_hat * w_mag;
      Eigen::Vector3d v     = -omega.cross(r);
      Eigen::Matrix<double,6,1> xd;
      xd.head<3>() = v;
      xd.tail<3>() = omega;
      // 计算并发布关节命令
      Eigen::Matrix<double,7,1> qdot = J_.transpose() 
        * (J_*J_.transpose() + lambda_*lambda_*Eigen::Matrix<double,6,6>::Identity()).inverse()
        * xd;
      q_des_ += qdot * dt_;
      publish_cmd();
      break;
    }

    case ScanState::UNFREEZE:
      // 4 UNFREEZE: 恢复导纳, 设定下一步期望力
      f_des_.head<3>() = n_hat_ * (f_high_ - 0.5);
      B_inv_ = B_inv_saved_;
      admittance_frozen_ = false;
      RCLCPP_INFO(get_logger(), "解冻导纳, f_des更新=[%.3f,%.3f,%.3f]", 
                   f_des_.x(), f_des_.y(), f_des_.z());
      // 初始化横向扫描
      state_ = ScanState::HORIZONTAL_SCAN;
      bounds_found_[0] = bounds_found_[1] = false;
      dir_lat_ = +1;
      break;

    case ScanState::HORIZONTAL_SCAN: {
      // 5 HORIZONTAL_SCAN: 力导纳沿 lateral_axis_
      Eigen::Vector3d d(lateral_axis_.x()*dir_lat_, lateral_axis_.y()*dir_lat_, 0.0);
      d.normalize();
      f_des_.head<3>() = d * (f_high_ - 0.5);
      compute_admittance();
      RCLCPP_DEBUG(get_logger(), "HORIZONTAL_SCAN dir=%d, f_des=[%.3f,%.3f,%.3f]",
                   dir_lat_, f_des_.x(), f_des_.y(), f_des_.z());
      if (f_norm <= f_low_ || reached_lateral_limit(dir_lat_)) {
        bounds_found_[dir_lat_>0?0:1] = true;
        RCLCPP_INFO(get_logger(), "记录横向边界 dir=%d", dir_lat_);
        if (bounds_found_[0] && bounds_found_[1]) {
          state_ = ScanState::ADVANCE_ON_AXIS;
        } else {
          dir_lat_ *= -1;
          RCLCPP_INFO(get_logger(), "翻转横向方向 dir=%d", dir_lat_);
          state_ = ScanState::PAUSE_MEASURE;
          pause_start_ = now();
          f_window_.clear();
        }
      }
      break;
    }

    case ScanState::ADVANCE_ON_AXIS: {
      // 6 ADVANCE_ON_AXIS: 力导纳沿 main_axis_
      Eigen::Vector3d d3d(main_axis_.x(), main_axis_.y(), 0.0);
      f_des_.head<3>() = d3d * (f_high_ - 0.5);
      compute_admittance();
      if ((ee_pos_.head<2>() - contact_pt_.head<2>()).dot(main_axis_) >= step_length_) {
        contact_pt_.head<2>() += main_axis_ * step_length_;
        RCLCPP_INFO(get_logger(), "主轴推进到下一点=(%.3f,%.3f)", 
                     contact_pt_.x(), contact_pt_.y());
        state_ = ScanState::PAUSE_MEASURE;
        pause_start_ = now();
        f_window_.clear();
      }
      break;
    }

    case ScanState::DONE:
      // 7 DONE: 扫描完成
      RCLCPP_INFO(get_logger(), "扫描完成, 总循环次数=%zu", loop_count_);
      break;
    }

    loop_count_++;
  }

  // 力导纳计算函数
  void compute_admittance() {
    Eigen::Matrix<double,6,1> delta = f_ext6_ - f_des_;
    Eigen::Matrix<double,6,1> xdn   = B_inv_ * delta;
    Eigen::Matrix<double,6,6> W     = J_*J_.transpose() + lambda_*lambda_*Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,1> xdot  = xdn;
    Eigen::Matrix<double,6,6> Winv  = W.inverse();
    Eigen::Matrix<double,7,1> qdot  = J_.transpose() * (Winv * xdot);
    q_des_ += qdot * dt_;
    publish_cmd();

    RCLCPP_INFO(get_logger(), "delta=[%f, %f, %f, %f, %f, %f]", 
                delta(0), delta(1), delta(2), delta(3), delta(4), delta(5));
  }

  // 发布关节命令
  void publish_cmd() {
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.assign(q_des_.data(), q_des_.data()+7);
    pub_cmd_->publish(cmd);
  }

  // 横向边界判定
  bool reached_lateral_limit(int dir) const {
    double d = (ee_pos_.head<2>() - contact_pt_.head<2>()).dot(lateral_axis_);
    return dir>0 ? (d > lateral_step_*10) : (d < -lateral_step_*10);
  }

  // 6D 力中位数滤波
  Eigen::Matrix<double,6,1> median6(const std::deque<Eigen::Matrix<double,6,1>>& w) const {
    Eigen::Matrix<double,6,1> m;
    for (int i=0; i<6; ++i) {
      std::vector<double> v; v.reserve(w.size());
      for (auto &f: w) v.push_back(f(i));
      size_t k = v.size()/2;
      std::nth_element(v.begin(), v.begin()+k, v.end());
      m(i) = v[k];
    }
    return m;
  }

  // 成员变量
  ScanState state_;

  // 在 INIT_SCAN 入口保存一次方向
  Eigen::Vector3d init_scan_dir_;
  bool init_scan_dir_computed_{false};

  bool initialized_{false}, have_js_{false}, have_jacobian_{false}, have_wrench_{false}, have_pose_{false};
  bool admittance_frozen_;
  double f_high_, f_low_, step_length_, lateral_step_;
  double axis_speed_, lateral_speed_, T_pause_, f_eps_, theta_tol_, k_theta_;
  int N_min_;
  const double dt_{0.01}, lambda_{0.1};
  Eigen::DiagonalMatrix<double,6> B_inv_, B_inv_saved_;
  Eigen::Matrix<double,6,1> f_des_, f_des_saved_, f_ext6_;
  Eigen::Matrix<double,6,7> J_;
  Eigen::Matrix<double,7,1> q_curr_, q_des_;
  Eigen::Vector3d ee_pos_, contact_pt_, n_hat_;
  Eigen::Matrix3d R_t_;
  Eigen::Vector2d main_axis_, lateral_axis_, object_center_;
  std::string ee_pose_topic_;
  std::deque<Eigen::Matrix<double,6,1>> f_window_;
  rclcpp::Time pause_start_;
  bool bounds_found_[2];
  int dir_lat_;
  size_t loop_count_;
  std::vector<std::string> joint_names_{"fr3_joint1","fr3_joint2","fr3_joint3","fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr    sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  sub_pose_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr   pub_cmd_;
  rclcpp::TimerBase::SharedPtr                                     timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HybridForceTangentialController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
