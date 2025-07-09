// fr3_monitor_node.cpp — monitor FR3 controller state
// -------------------------------------------------------------
// Listens to /joint_position_controller/state and publishes diagnostics,
// also prints received joint data to console.
// -------------------------------------------------------------

#include <rclcpp/rclcpp.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <sstream>

using control_msgs::msg::JointTrajectoryControllerState;
using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using diagnostic_msgs::msg::KeyValue;

class FR3MonitorNode : public rclcpp::Node
{
public:
  FR3MonitorNode() : Node("fr3_monitor")
  {
    state_sub_ = create_subscription<JointTrajectoryControllerState>(
      "/joint_position_controller/state", 20,
      std::bind(&FR3MonitorNode::state_cb, this, std::placeholders::_1));

    diag_pub_ = create_publisher<DiagnosticArray>(
      "/fr3_monitor/diagnostics", 10);

    RCLCPP_INFO(get_logger(),
                "FR3 monitor node started, listening to /joint_position_controller/state");
  }

private:
  void state_cb(const JointTrajectoryControllerState::SharedPtr msg)
  {
    DiagnosticArray array;
    array.header.stamp = now();

    const auto & names = msg->joint_names;
    size_t n = names.size();

    // ---- 打印收到的实际位置到控制台 ----
    std::ostringstream oss;
    oss << "Actual positions: ";
    for (size_t i = 0; i < n; ++i) {
      oss << names[i] << "=" << msg->actual.positions[i] << (i + 1 < n ? ", " : "");
    }
    RCLCPP_INFO_STREAM(get_logger(), oss.str());

    for (size_t i = 0; i < n; ++i)
    {
      DiagnosticStatus st;
      st.level = DiagnosticStatus::OK;
      st.name = names[i];
      st.hardware_id = "fr3_arm";
      st.message = "controller_state";

      KeyValue kv_pos;
      kv_pos.key = "actual_position (rad)";
      kv_pos.value = std::to_string(msg->actual.positions[i]);
      st.values.emplace_back(kv_pos);

      KeyValue kv_vel;
      kv_vel.key = "actual_velocity (rad/s)";
      kv_vel.value = (i < msg->actual.velocities.size()) ?
                     std::to_string(msg->actual.velocities[i]) : "N/A";
      st.values.emplace_back(kv_vel);

      KeyValue kv_err;
      kv_err.key = "position_error (rad)";
      kv_err.value = (i < msg->error.positions.size()) ?
                     std::to_string(msg->error.positions[i]) : "N/A";
      st.values.emplace_back(kv_err);

      KeyValue kv_des;
      kv_des.key = "desired_position (rad)";
      kv_des.value = (i < msg->desired.positions.size()) ?
                     std::to_string(msg->desired.positions[i]) : "N/A";
      st.values.emplace_back(kv_des);

      array.status.emplace_back(std::move(st));
    }

    diag_pub_->publish(array);
  }

  rclcpp::Subscription<JointTrajectoryControllerState>::SharedPtr state_sub_;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr diag_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FR3MonitorNode>());
  rclcpp::shutdown();
  return 0;
}
