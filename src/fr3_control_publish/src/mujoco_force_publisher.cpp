// src/mujoco_force_publisher.cpp

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <mujoco/mujoco.h>
#include <chrono>
#include <string>

class MujocoForcePublisher : public rclcpp::Node {
public:
  MujocoForcePublisher(const std::string & model_path,
                       double publish_rate = 100.0)
  : Node("mujoco_force_publisher"), publish_rate_(publish_rate)
  {
    // 加载模型
    char err[1000] = "Could not load model";
    model_ = mj_loadXML(model_path.c_str(), nullptr, err, sizeof(err));
    if (!model_) {
      RCLCPP_FATAL(this->get_logger(), "Load model error: %s", err);
      rclcpp::shutdown();
      return;
    }

    // 分配 Data
    data_ = mj_makeData(model_);
    if (!data_) {
      RCLCPP_FATAL(this->get_logger(), "Failed to allocate mjData");
      mj_deleteModel(model_);
      rclcpp::shutdown();
      return;
    }

    // 找到传感器 ID
    force_id_  = mj_name2id(model_, mjOBJ_SENSOR, "touch_force");
    torque_id_ = mj_name2id(model_, mjOBJ_SENSOR, "touch_torque");
    if (force_id_ < 0 || torque_id_ < 0) {
      RCLCPP_WARN(this->get_logger(),
        "Could not find sensors (force_id=%d, torque_id=%d)",
        force_id_, torque_id_);
    }

    // 创建 Publisher
    pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
             "/mujoco/wrench", 10);

    // 定时器：以 publish_rate_ 调用 timer_callback()
    auto period = std::chrono::duration<double>(1.0 / publish_rate_);
    timer_ = this->create_wall_timer(
      period, std::bind(&MujocoForcePublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
      "Initialized. Publishing at %.1f Hz", publish_rate_);
  }

  ~MujocoForcePublisher() override {
    if (data_)  mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

private:
  void timer_callback() {
    // 推演一步仿真
    mj_step(model_, data_);

    // 构造并发布消息
    geometry_msgs::msg::WrenchStamped msg;

    // 手动填充 ROS 时间戳
    auto now = this->get_clock()->now();
    int64_t ns = now.nanoseconds();
    msg.header.stamp.sec = static_cast<decltype(msg.header.stamp.sec)>(ns / 1000000000LL);
    msg.header.stamp.nanosec = static_cast<decltype(msg.header.stamp.nanosec)>(ns % 1000000000LL);
    msg.header.frame_id = "touch_tip";

    // 如果找到传感器，则从 data_->sensordata 和 model_->sensor_adr 读取
    if (force_id_ >= 0) {
      int adr = model_->sensor_adr[force_id_];
      msg.wrench.force.x = data_->sensordata[adr + 0];
      msg.wrench.force.y = data_->sensordata[adr + 1];
      msg.wrench.force.z = data_->sensordata[adr + 2];
    }
    if (torque_id_ >= 0) {
      int adr = model_->sensor_adr[torque_id_];
      msg.wrench.torque.x = data_->sensordata[adr + 0];
      msg.wrench.torque.y = data_->sensordata[adr + 1];
      msg.wrench.torque.z = data_->sensordata[adr + 2];
    }

    pub_->publish(msg);
    RCLCPP_DEBUG(this->get_logger(),
      "Published WrenchStamped: [%.3f,%.3f,%.3f; %.3f,%.3f,%.3f]",
      msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
      msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z);
  }

  // MuJoCo
  mjModel* model_{nullptr};
  mjData*  data_{nullptr};
  int force_id_{-1}, torque_id_{-1};

  // ROS2
  double publish_rate_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
  
    // 直接硬编码路径
    std::string model_path = "src/mujoco_ros2_control/mujoco_ros2_control_demos/mujoco_models/fr3.xml";
    double rate = 100.0;
  
    auto node = std::make_shared<MujocoForcePublisher>(model_path, rate);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
  }
  
