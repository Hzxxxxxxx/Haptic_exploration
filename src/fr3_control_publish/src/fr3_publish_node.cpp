#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;      // 支持 200ms 字面量

class SimpleControlNode : public rclcpp::Node
{
public:
  SimpleControlNode()
  : Node("simple_control")
  {
    RCLCPP_INFO(get_logger(), "simple_control start.");

    // ① 发布者：话题名按实际控制器配置修改
    command_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_position_controller/joint_trajectory",
      rclcpp::QoS(10));                      // 深度 10，Reliable（默认）

    // ② 200 ms 定时器
    timer_ = create_wall_timer(500ms, [this]() { timer_callback(); });
  }

private:
  // ③ 回调：让 fr3_joint1 做 0.5 rad 正弦摆动
  void timer_callback()
    {
    static size_t step = 0;
    double theta = 2.0 * std::sin(step * 0.05);   // 幅值 2 rad

    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = now();
    traj.joint_names = {
        "fr3_joint1","fr3_joint2","fr3_joint3",
        "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
    };

    trajectory_msgs::msg::JointTrajectoryPoint p;
    p.positions = {theta, theta, theta, theta, theta, theta, theta};
    p.time_from_start = rclcpp::Duration(200ms);   // 0.2 s

    traj.points.push_back(p);
    command_pub_->publish(traj);

    // 打印调试信息
    RCLCPP_INFO(get_logger(),
                "step=%zu, theta=%.3f, joint7=%.3f",
                step, theta, 2*theta);

    ++step;
    }


  // ④ 成员
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr command_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleControlNode>());
  rclcpp::shutdown();
  return 0;
}
