// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#ifndef FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__BUILDER_HPP_
#define FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fr3_control_publish/msg/detail/touch_tip_wrench__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fr3_control_publish
{

namespace msg
{

namespace builder
{

class Init_TouchTipWrench_torque
{
public:
  explicit Init_TouchTipWrench_torque(::fr3_control_publish::msg::TouchTipWrench & msg)
  : msg_(msg)
  {}
  ::fr3_control_publish::msg::TouchTipWrench torque(::fr3_control_publish::msg::TouchTipWrench::_torque_type arg)
  {
    msg_.torque = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fr3_control_publish::msg::TouchTipWrench msg_;
};

class Init_TouchTipWrench_force
{
public:
  explicit Init_TouchTipWrench_force(::fr3_control_publish::msg::TouchTipWrench & msg)
  : msg_(msg)
  {}
  Init_TouchTipWrench_torque force(::fr3_control_publish::msg::TouchTipWrench::_force_type arg)
  {
    msg_.force = std::move(arg);
    return Init_TouchTipWrench_torque(msg_);
  }

private:
  ::fr3_control_publish::msg::TouchTipWrench msg_;
};

class Init_TouchTipWrench_frame_id
{
public:
  explicit Init_TouchTipWrench_frame_id(::fr3_control_publish::msg::TouchTipWrench & msg)
  : msg_(msg)
  {}
  Init_TouchTipWrench_force frame_id(::fr3_control_publish::msg::TouchTipWrench::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_TouchTipWrench_force(msg_);
  }

private:
  ::fr3_control_publish::msg::TouchTipWrench msg_;
};

class Init_TouchTipWrench_stamp
{
public:
  Init_TouchTipWrench_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TouchTipWrench_frame_id stamp(::fr3_control_publish::msg::TouchTipWrench::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_TouchTipWrench_frame_id(msg_);
  }

private:
  ::fr3_control_publish::msg::TouchTipWrench msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fr3_control_publish::msg::TouchTipWrench>()
{
  return fr3_control_publish::msg::builder::Init_TouchTipWrench_stamp();
}

}  // namespace fr3_control_publish

#endif  // FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__BUILDER_HPP_
