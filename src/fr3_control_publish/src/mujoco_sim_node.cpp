// File: src/mujoco_sim_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <string>

using std::placeholders::_1;

class MujocoSimNode : public rclcpp::Node {
public:
  MujocoSimNode()
  : Node("mujoco_sim_node")
  {
    // 1) Load MuJoCo model
    std::string model_path = this->declare_parameter<std::string>(
      "model_path",
      "/home/mscrobotics2425laptop16/mujoco_ros_ws/src/mujoco_ros2_control/"
      "mujoco_ros2_control_demos/mujoco_models/scene.xml");
    char error[1000] = "Could not load XML model";
    m_ = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
    if (!m_) {
      RCLCPP_FATAL(get_logger(), "mj_loadXML error: %s", error);
      rclcpp::shutdown();
      return;
    }
    d_ = mj_makeData(m_);

    // 2) Initialize control variables
    n_ctrl_ = m_->nu;
    cmd_.assign(n_ctrl_, 0.0);

    // 3) Query site and sensor IDs
    joint_names_ = {
      "fr3_joint1","fr3_joint2","fr3_joint3",
      "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    site_id_ = mj_name2id(m_, mjOBJ_SITE, "touch_tip");
    if (site_id_ < 0) {
      RCLCPP_FATAL(get_logger(), "Could not find site 'touch_tip'");
      rclcpp::shutdown();
      return;
    }

    int id_f = mj_name2id(m_, mjOBJ_SENSOR, "touch_tip_force");
    int id_t = mj_name2id(m_, mjOBJ_SENSOR, "touch_tip_torque");
    if (id_f < 0 || id_t < 0) {
      RCLCPP_FATAL(get_logger(),
                   "Could not find sensors 'touch_tip_force'/'touch_tip_torque'");
      rclcpp::shutdown();
      return;
    }
    adr_force_  = m_->sensor_adr[id_f];
    adr_torque_ = m_->sensor_adr[id_t];

    RCLCPP_INFO(get_logger(),
                "Loaded model '%s' with %d ctrl dims, site_id=%d",
                m_->names + m_->name_bodyadr[0], n_ctrl_, site_id_);

    // 4) Set up ROS interfaces
    sub_cmd_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_position_controller/commands", 10,
      std::bind(&MujocoSimNode::on_command, this, _1));

    pub_js_ = create_publisher<sensor_msgs::msg::JointState>(
      "/joint_states", 10);

    pub_jac_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      "/jacobian", 10);

    pub_wrench_ = create_publisher<geometry_msgs::msg::WrenchStamped>(
      "/touch_tip/wrench", rclcpp::SensorDataQoS());

    timer_ = create_wall_timer(
      std::chrono::milliseconds(10),  // 100 Hz
      std::bind(&MujocoSimNode::update, this));

    // 5) Initialize GLFW + MuJoCo rendering
    if (!glfwInit()) {
      RCLCPP_FATAL(get_logger(), "Could not initialize GLFW");
      rclcpp::shutdown();
      return;
    }
    window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulation", nullptr, nullptr);
    if (!window_) {
      RCLCPP_FATAL(get_logger(), "Could not create GLFW window");
      glfwTerminate();
      rclcpp::shutdown();
      return;
    }
    glfwMakeContextCurrent(window_);

    // ——— 初始化 MuJoCo 渲染结构 ———
    mjv_defaultCamera(&cam_);          // camera defaults
    // 手动设置一个合理的视角和距离，避免 mjv_room2model 报错
    cam_.type      = mjCAMERA_FREE;
    cam_.trackbodyid = -1;
    cam_.fixedcamid = -1;

    // 想 “拉远一点，看到更多” 的示例：
    cam_.lookat[0] = 0.0;      // 相机中心点 X
    cam_.lookat[1] = 0.0;      // 相机中心点 Y
    cam_.lookat[2] = 0.3;      // 相机中心点 Z
    cam_.distance  = 3;      // 把镜头拉远到 2.0m（原来是 1.2）
    cam_.azimuth   = 90;       // 水平方向不变
    cam_.elevation = -20;      // 俯视角度变小，能看到更远的地平面

    mjv_defaultOption(&opt_);          // visualization options

    // 默认 scene 的 maxgeom=0，要自己分配足够的槽
    mjv_defaultScene(&scn_);
    // 这里给 1000 个 geom 槽，够绝大多数模型使用
    mjv_makeScene(m_, &scn_, 1000);
    mjr_defaultContext(&con_);         // custom GPU context
    mjr_makeContext(m_, &con_, mjFONTSCALE_150);
  }

  ~MujocoSimNode() override {
    // Cleanup MuJoCo rendering
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    if (window_) {
      glfwDestroyWindow(window_);
      glfwTerminate();
    }
    // Cleanup MuJoCo model
    mj_deleteData(d_);
    mj_deleteModel(m_);
  }

private:
  void on_command(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if ((int)msg->data.size() != n_ctrl_) {
      RCLCPP_WARN(get_logger(),
                  "Command size %zu, expected %d", msg->data.size(), n_ctrl_);
      return;
    }
    for (int i = 0; i < n_ctrl_; i++) {
      cmd_[i] = msg->data[i];
    }
  }

  void update() {
    // Write controls
    for (int i = 0; i < n_ctrl_; i++) {
      d_->ctrl[i] = cmd_[i];
    }

    // Step simulation
    mj_step(m_, d_);

    // Publish joint_states
    auto js = sensor_msgs::msg::JointState();
    js.header.stamp = now();
    js.name  = joint_names_;
    js.position.resize(n_ctrl_);
    js.velocity.resize(n_ctrl_);
    for (int i = 0; i < n_ctrl_; i++) {
      js.position[i] = d_->qpos[i];
      js.velocity[i] = d_->qvel[i];
    }
    pub_js_->publish(js);

    // Publish Jacobian (6×7)
    std_msgs::msg::Float64MultiArray jac;
    jac.layout.dim.resize(2);
    jac.layout.dim[0].label    = "rows";
    jac.layout.dim[0].size     = 6;
    jac.layout.dim[0].stride   = 6 * n_ctrl_;
    jac.layout.dim[1].label    = "cols";
    jac.layout.dim[1].size     = n_ctrl_;
    jac.layout.dim[1].stride   = n_ctrl_;
    jac.layout.data_offset     = 0;
    jac.data.resize(6 * n_ctrl_);

    std::vector<mjtNum> jacp(3*m_->nv), jacr(3*m_->nv);
    mj_jacSite(m_, d_, jacp.data(), jacr.data(), site_id_);
    for (int col = 0; col < n_ctrl_; col++) {
      for (int row = 0; row < 3; row++) {
        jac.data[row    * n_ctrl_ + col] = jacp[3*col + row];
        jac.data[(row+3)*n_ctrl_ + col] = jacr[3*col + row];
      }
    }
    pub_jac_->publish(jac);

    // Publish wrench
    geometry_msgs::msg::WrenchStamped w;
    w.header.stamp    = now();
    w.header.frame_id = "touch_tip";
    w.wrench.force.x  = d_->sensordata[adr_force_ + 0];
    w.wrench.force.y  = d_->sensordata[adr_force_ + 1];
    w.wrench.force.z  = d_->sensordata[adr_force_ + 2];
    w.wrench.torque.x = d_->sensordata[adr_torque_ + 0];
    w.wrench.torque.y = d_->sensordata[adr_torque_ + 1];
    w.wrench.torque.z = d_->sensordata[adr_torque_ + 2];
    pub_wrench_->publish(w);

    // Render
    if (glfwWindowShouldClose(window_)) {
      rclcpp::shutdown();
      return;
    }
    mjrRect vp = {0,0,0,0};
    glfwGetFramebufferSize(window_, &vp.width, &vp.height);
    mjv_updateScene(m_, d_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    mjr_render(vp, &scn_, &con_);
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }

  // MuJoCo data
  mjModel* m_ = nullptr;
  mjData*  d_ = nullptr;

  int               n_ctrl_;
  std::vector<double> cmd_;
  std::vector<std::string> joint_names_;

  int site_id_;
  int adr_force_, adr_torque_;

  // ROS interfaces
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr        pub_js_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr    pub_jac_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr   pub_wrench_;
  rclcpp::TimerBase::SharedPtr                                      timer_;

  // Rendering
  GLFWwindow* window_ = nullptr;
  mjvCamera   cam_;
  mjvOption   opt_;
  mjvScene    scn_;
  mjrContext  con_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MujocoSimNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
