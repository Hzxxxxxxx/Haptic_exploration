// ee_pose_bridge.cpp

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

class EEPosePublisher : public rclcpp::Node
{
public:
  EEPosePublisher()
  : Node("ee_pose_publisher"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // 声明参数并设置默认值
    this->declare_parameter<std::string>("base_frame", "fr3_link0");
    this->declare_parameter<std::string>("ee_frame",   "touch_tip");
    this->declare_parameter<std::string>("output_topic", "/ee_pose");
    this->declare_parameter<double>("publish_rate", 100.0);

    // 获取参数
    this->get_parameter("base_frame", base_frame_);
    this->get_parameter("ee_frame",     ee_frame_);
    this->get_parameter("output_topic", output_topic_);
    double rate;
    this->get_parameter("publish_rate", rate);
    auto period = std::chrono::duration<double>(1.0 / rate);

    // 创建发布器
    pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      output_topic_, 10);

    // 定时器
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&EEPosePublisher::timerCallback, this));

    RCLCPP_INFO(this->get_logger(),
      "EEPosePublisher initialized: base_frame=%s, ee_frame=%s, topic=%s @ %.1fHz",
      base_frame_.c_str(), ee_frame_.c_str(), output_topic_.c_str(), rate);
  }

private:
  void timerCallback()
  {
    geometry_msgs::msg::TransformStamped transform;
    try {
      // lookupTransform(target_frame, source_frame)
      transform = tf_buffer_.lookupTransform(
        base_frame_, ee_frame_, rclcpp::Time(0));
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Could not transform %s to %s: %s",
        base_frame_.c_str(), ee_frame_.c_str(), ex.what());
      return;
    }

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = transform.header;
    pose_msg.pose.position.x    = transform.transform.translation.x;
    pose_msg.pose.position.y    = transform.transform.translation.y;
    pose_msg.pose.position.z    = transform.transform.translation.z;
    pose_msg.pose.orientation    = transform.transform.rotation;

    pub_->publish(pose_msg);
  }

  std::string base_frame_, ee_frame_, output_topic_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EEPosePublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
