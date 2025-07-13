#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>
#include <vector>
#include <algorithm>

using std::placeholders::_1;

class HybridForceTangentialController : public rclcpp::Node
{
public:
  // 扫描状态机
  enum class ScanState {
    INIT_SCAN,         // 初次沿主扫描轴探测
    HORIZONTAL_SCAN,   // 在接触点做横向来回扫描
    ADVANCE_ON_AXIS,   // 沿主扫描轴推进一步
    DONE               // 扫描完成
  };

  HybridForceTangentialController()
  : Node("hybrid_force_tangential_controller"),

    // 初始化各类标志位
    have_js_(false),
    have_jacobian_(false),
    have_wrench_(false),
    have_pose_(false),
    initialized_(false),
    loop_count_(0),
    state_(ScanState::INIT_SCAN)
  {
    // 1）声明并获取参数（默认值及含义）  
    this->declare_parameter<double>("lambda_force", 0.1);  
    // λ_f：法向正则化系数，影响伪逆映射的稳定性，默认 0.1，防止奇异  

    this->declare_parameter<std::vector<double>>("B_inv_diag",  
      std::vector<double>{1.0/50,1.0/50,1.0/50,1.0/5,1.0/5,1.0/5});  
    /* B_inv_diag：虚拟阻尼逆矩阵对角线，  
       前三个元素对应 XYZ 力维度，这里 1/50→阻尼较大，法向导纳响应较慢；  
       后三个元素对应 Roll/Pitch/Yaw 力矩维度，这里 1/5→阻尼更小，姿态响应更快 */

    this->declare_parameter<std::vector<double>>("f_des",  
      std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});  
    /* f_des：期望六维力/力矩，  
       默认 [0, -5, 0, 0, 0, 0] 表示沿 Y 方向保持 5N 推压力，其余方向和力矩为 0 */

    this->declare_parameter<double>("f_high", 2.0);  
    // f_high：接触阈值，当法向力 ≥ 5N 时视为已接触  

    this->declare_parameter<double>("f_low", 0.1);  
    // f_low：离开阈值，当法向力 ≤ 2N 时视为已离开，用于去抖  

    this->declare_parameter<double>("step_length", 0.02);  
    // step_length：每次沿主扫描轴推进距离，默认 0.02m（2cm）  

    this->declare_parameter<double>("axis_speed", 0.01);  
    // axis_speed：沿主扫描轴匀速移动速度，默认 0.01m/s  

    this->declare_parameter<double>("lateral_speed", 0.05);  
    // lateral_speed：沿横向扫描轴匀速移动速度，默认 0.005m/s  

    this->declare_parameter<double>("axis_max_dist", 0.8);  
    // axis_max_dist：主扫描轴最大可达距离，默认 0.3m  

    this->declare_parameter<double>("lateral_max_dist", 0.5);  
    // lateral_max_dist：横向扫描最大可达距离，默认 0.1m  

    this->declare_parameter<double>("object_center_x", 0.0);  
    // object_center_x：物体中心点在 X 轴上的预估坐标，默认 0.5m  

    this->declare_parameter<double>("object_center_y", 1.0);  
    // object_center_y：物体中心点在 Y 轴上的预估坐标，默认 0.0m  

    this->declare_parameter<std::string>("ee_pose_topic", "/ee_pose");  
    // ee_pose_topic：末端位姿话题名称，默认 "/ee_pose"，用于获取当前 XY 平面位置  

    // 获取参数值
    this->get_parameter("lambda_force",     lambda_f_);
    std::vector<double> bdiag, fdes_vec;
    this->get_parameter("B_inv_diag",       bdiag);
    this->get_parameter("f_des",            fdes_vec);
    this->get_parameter("f_high",           f_high_);
    this->get_parameter("f_low",            f_low_);
    this->get_parameter("step_length",      step_length_);
    this->get_parameter("axis_speed",       axis_speed_);
    this->get_parameter("lateral_speed",    lateral_speed_);
    this->get_parameter("axis_max_dist",    axis_max_dist_);
    this->get_parameter("lateral_max_dist", lateral_max_dist_);
    double obj_x, obj_y;
    this->get_parameter("object_center_x", obj_x);
    this->get_parameter("object_center_y", obj_y);
    this->get_parameter("ee_pose_topic",   ee_pose_topic_);

    // 构造 B_inv 和 f_des
    B_inv_ = Eigen::DiagonalMatrix<double,6>(
      bdiag[0], bdiag[1], bdiag[2],
      bdiag[3], bdiag[4], bdiag[5]);
    f_des_.setZero();
    for(int i = 0; i < 6; ++i) {
      f_des_(i) = fdes_vec[i];
    }

    // 计算主扫描轴和横向扫描轴（仅在 XY 平面）
    Eigen::Vector2d base(0.0, 0.0);
    Eigen::Vector2d center(obj_x, obj_y);
    main_axis_    = (center - base).normalized();                            // 主向量指向物体中心
    lateral_axis_ = Eigen::Vector2d(-main_axis_.y(), main_axis_.x());         // 顺时针旋转90°作为横向

    // 关节名称列表
    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    // 初始化期望关节位置
    q_curr_.setZero();
    q_des_.setZero();

    // 打印关键信息
    RCLCPP_INFO(get_logger(), "HybridForceTangentialController 启动");
    RCLCPP_INFO_STREAM(get_logger(),
      "f_high="<<f_high_<<", f_low="<<f_low_
      <<", step_length="<<step_length_);
    RCLCPP_INFO_STREAM(get_logger(),
      "axis_speed="<<axis_speed_
      <<", lateral_speed="<<lateral_speed_);
    RCLCPP_INFO_STREAM(get_logger(),
      "main_axis=["<<main_axis_.transpose()
      <<"], lateral_axis=["<<lateral_axis_.transpose()<<"]");

    // 2）订阅：关节状态、雅可比、触觉、末端位姿
    sub_js_      = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, std::bind(&HybridForceTangentialController::on_js, this, _1));
    sub_jacobian_= create_subscription<std_msgs::msg::Float64MultiArray>(
      "/jacobian",     10, std::bind(&HybridForceTangentialController::on_jac, this, _1));
    sub_wrench_  = create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/touch_tip/wrench", rclcpp::SensorDataQoS(),
      std::bind(&HybridForceTangentialController::on_wrench, this, _1));
    sub_pose_    = create_subscription<geometry_msgs::msg::PoseStamped>(
      ee_pose_topic_, 10, std::bind(&HybridForceTangentialController::on_pose, this, _1));

    // 3）发布 & 定时器
    pub_cmd_     = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10);
    timer_       = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&HybridForceTangentialController::control_loop, this));
  }

private:
  // —— 回调函数 —— 
  void on_js(const sensor_msgs::msg::JointState::SharedPtr msg) {
    // 接收并保存当前关节位置
    have_js_ = true;
    for(size_t i = 0; i < msg->name.size(); ++i) {
      auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[i]);
      if(it == joint_names_.end()) continue;
      size_t idx = std::distance(joint_names_.begin(), it);
      q_curr_(idx) = msg->position[i];
    }
    // 首次初始化期望位置 q_des_
    if(!initialized_) {
      q_des_ = q_curr_;
      initialized_ = true;
      RCLCPP_INFO(get_logger(), "q_des_ 已初始化为当前关节位置");
    }
  }

  void on_jac(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    // 确认雅可比矩阵维度为 6×7
    if(msg->layout.dim.size()!=2 ||
       msg->layout.dim[0].size!=6 ||
       msg->layout.dim[1].size!=7) {
      RCLCPP_WARN(get_logger(), "接收到错误维度的雅可比");
      return;
    }
    // 填充到 Eigen::Matrix
    for(int i = 0; i < 6; ++i)
      for(int j = 0; j < 7; ++j)
        J_(i,j) = msg->data[i*7 + j];
    have_jacobian_ = true;
  }

  void on_wrench(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    RCLCPP_INFO(get_logger(),
    "[DEBUG on_wrench] frame_id=%s, fx=%.3f, fy=%.3f, fz=%.3f",
    msg->header.frame_id.c_str(),
    msg->wrench.force.x,
    msg->wrench.force.y,
    msg->wrench.force.z);
    if (msg->header.frame_id != "touch_tip") {
        RCLCPP_WARN(get_logger(),
        "[DEBUG on_wrench] ignored non-touch_tip frame: %s",
        msg->header.frame_id.c_str());
        return;
    }
    // 仅处理 touch_tip 坐标系下的力
    if(msg->header.frame_id != "touch_tip") return;
    // 保存三维力和六维力
    f_ext3_ = Eigen::Vector3d(
      msg->wrench.force.x,
      msg->wrench.force.y,
      msg->wrench.force.z);
    f_ext6_.head<3>() = f_ext3_;
    f_ext6_.tail<3>() = Eigen::Vector3d(
      msg->wrench.torque.x,
      msg->wrench.torque.y,
      msg->wrench.torque.z);
    have_wrench_ = true;
  }

  void on_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "[DEBUG on_pose] frame_id=%s, x=%.3f, y=%.3f",
              msg->header.frame_id.c_str(),
              msg->pose.position.x,
              msg->pose.position.y);
    // 获取末端在世界坐标系下的 XY 平面位置
    ee_pos_(0) = msg->pose.position.x;
    ee_pos_(1) = msg->pose.position.y;
    have_pose_ = true;
  }

  // —— 辅助函数 —— 
  // 判断是否到达主轴可达范围边界
  bool reached_axis_limit() const {
    double dist = ee_pos_.dot(main_axis_);
    return dist < 0.0 || dist > axis_max_dist_;
  }
  // 判断是否到达横向可达范围边界
  bool reached_lateral_limit(int dir) const {
    double d = (ee_pos_ - contact_pt_).dot(lateral_axis_);
    return dir>0 ? (d > lateral_max_dist_) : (d < -lateral_max_dist_);
  }

  // —— 主控制循环 —— 
  void control_loop() {
    // 新增调试：打印各个 have_* 的状态
    RCLCPP_DEBUG(get_logger(),
    "[DEBUG control_loop] state=%d, have_js=%d, have_jac=%d, inited=%d, have_wrench=%d, have_pose=%d",
    static_cast<int>(state_), have_js_, have_jacobian_, initialized_,
    have_wrench_, have_pose_);
    // 只要 joint_states、jacobian、q_des_ 就能启动 INIT_SCAN
    if (!(have_js_ && have_jacobian_ && initialized_)) return;

    // 除了 INIT_SCAN，其他阶段都要检查 wrench+pose
    if (state_ != ScanState::INIT_SCAN && !(have_wrench_ && have_pose_)) return;

    ++loop_count_;
    double f_norm = f_ext3_.norm();

    // 1) 计算法向导纳速度 x_dot_n = B_inv * (f_ext6 - f_des)
    Eigen::Matrix<double,6,6> W = J_ * J_.transpose()
      + lambda_f_*lambda_f_ * Eigen::Matrix<double,6,6>::Identity();
    Eigen::Matrix<double,6,1> delta_f = f_ext6_ - f_des_;
    Eigen::Matrix<double,6,1> x_dot_n = B_inv_ * delta_f;
    Eigen::Matrix<double,6,6> W_inv = W.inverse();

    // 2) 根据状态机决定平面速度 v_xy
    Eigen::Vector2d v_xy = Eigen::Vector2d::Zero();
    switch(state_) {
      case ScanState::INIT_SCAN:
        // 沿主扫描轴匀速前进
        v_xy = main_axis_ * axis_speed_;
        if(f_norm >= f_high_) {
          // 首次接触，记录位置，进入横向扫描
          contact_pt_ = ee_pos_;
          bounds_found_[0] = bounds_found_[1] = false;
          state_ = ScanState::HORIZONTAL_SCAN;
          RCLCPP_INFO(get_logger(), "进入 HORIZONTAL_SCAN");
        } else if(reached_axis_limit()) {
          state_ = ScanState::DONE;
          RCLCPP_WARN(get_logger(), "未检测到物体，结束扫描");
        }
        break;

      case ScanState::HORIZONTAL_SCAN: {
        // 横向扫描：先正向(+1)，找到边界后反向(-1)
        int dir = bounds_found_[0] ? -1 : +1;
        v_xy = lateral_axis_ * lateral_speed_ * dir;
        // 正向边界检测
        if(!bounds_found_[0] &&
           (f_norm <= f_low_ || reached_lateral_limit(+1))) {
          bounds_found_[0] = true;
          RCLCPP_INFO(get_logger(), "找到横向正边界");
        }
        // 负向边界检测
        else if(bounds_found_[0] && !bounds_found_[1] &&
                (f_norm <= f_low_ || reached_lateral_limit(-1))) {
          bounds_found_[1] = true;
          state_ = ScanState::ADVANCE_ON_AXIS;
          RCLCPP_INFO(get_logger(), "横向双边界完成，进入 ADVANCE_ON_AXIS");
        }
        break;
      }

      case ScanState::ADVANCE_ON_AXIS:
        // 沿主轴推进一个步长
        v_xy = main_axis_ * axis_speed_;
        if((ee_pos_ - contact_pt_).dot(main_axis_) >= step_length_) {
          contact_pt_ += main_axis_ * step_length_;
          state_ = ScanState::HORIZONTAL_SCAN;
          RCLCPP_INFO(get_logger(), "主轴推进一步，返回 HORIZONTAL_SCAN");
        }
        break;

      case ScanState::DONE:
        // 扫描完成，速度置零
        v_xy.setZero();
        break;
    }

    // 3) 构造切向速度向量 x_dot_t = [v_x, v_y, 0, 0, 0, 0]^T
    Eigen::Matrix<double,6,1> x_dot_t = Eigen::Matrix<double,6,1>::Zero();
    x_dot_t(0) = v_xy.x();
    x_dot_t(1) = v_xy.y();

    // 4) 合成速度并映射到关节 q_dot，积分更新 q_des_
    Eigen::Matrix<double,6,1> x_dot = x_dot_n + x_dot_t;
    Eigen::Matrix<double,7,1> q_dot = J_.transpose() * W_inv * x_dot;
    constexpr double dt = 0.01;
    q_des_ += q_dot * dt;

    // 发布期望关节位置
    std_msgs::msg::Float64MultiArray cmd;
    cmd.data.assign(q_des_.data(), q_des_.data() + 7);
    pub_cmd_->publish(cmd);

    // 5) 诊断信息打印
    RCLCPP_INFO_STREAM(get_logger(),
      "[Loop " << loop_count_ << "] state=" << static_cast<int>(state_)
      << " |f|=" << f_norm
      << " v_xy=[" << v_xy.transpose() << "]"
      << " ee_xy=[" << ee_pos_.transpose() << "]");
  }
  size_t                          loop_count_;
  ScanState    state_;           // 当前扫描状态
  std::string  ee_pose_topic_;   // 末端位姿话题名
  // —— 成员变量 —— 
  double lambda_f_;                                    // 法向正则化系数
  Eigen::DiagonalMatrix<double,6> B_inv_;              // 虚拟阻尼逆矩阵
  Eigen::Matrix<double,6,7>       J_;                  // 6×7 雅可比
  Eigen::Matrix<double,7,1>       q_curr_, q_des_;     // 当前 / 期望 关节角
  Eigen::Matrix<double,6,1>       f_ext6_;             // 六维外力/力矩
  Eigen::Vector3d                 f_ext3_;             // 三维法向力
  Eigen::Matrix<double,6,1>       f_des_;              // 期望六维力
  Eigen::Vector2d                 main_axis_, lateral_axis_;  // 主/横向扫描轴
  Eigen::Vector2d                 contact_pt_, ee_pos_;       // 接触点 / 末端 XY
  bool                            have_js_, have_jacobian_, have_wrench_;
  bool                            have_pose_, initialized_;

  bool                            bounds_found_[2];     // 横向两侧边界标志
  double                          f_high_, f_low_;      // 力阈值
  double                          step_length_;         // 主轴推进步长
  double                          axis_speed_, lateral_speed_;  // 线速度
  double                          axis_max_dist_, lateral_max_dist_; // 可达边界
  std::vector<std::string>        joint_names_;

  // ROS 接口
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr    sub_js_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_jacobian_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  sub_pose_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr   pub_cmd_;
  rclcpp::TimerBase::SharedPtr                                     timer_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HybridForceTangentialController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
