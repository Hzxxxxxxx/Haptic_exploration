// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#ifndef FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__TRAITS_HPP_
#define FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fr3_control_publish/msg/detail/touch_tip_wrench__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'force'
// Member 'torque'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace fr3_control_publish
{

namespace msg
{

inline void to_flow_style_yaml(
  const TouchTipWrench & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: frame_id
  {
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << ", ";
  }

  // member: force
  {
    out << "force: ";
    to_flow_style_yaml(msg.force, out);
    out << ", ";
  }

  // member: torque
  {
    out << "torque: ";
    to_flow_style_yaml(msg.torque, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TouchTipWrench & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: frame_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "frame_id: ";
    rosidl_generator_traits::value_to_yaml(msg.frame_id, out);
    out << "\n";
  }

  // member: force
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "force:\n";
    to_block_style_yaml(msg.force, out, indentation + 2);
  }

  // member: torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "torque:\n";
    to_block_style_yaml(msg.torque, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TouchTipWrench & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace fr3_control_publish

namespace rosidl_generator_traits
{

[[deprecated("use fr3_control_publish::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const fr3_control_publish::msg::TouchTipWrench & msg,
  std::ostream & out, size_t indentation = 0)
{
  fr3_control_publish::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fr3_control_publish::msg::to_yaml() instead")]]
inline std::string to_yaml(const fr3_control_publish::msg::TouchTipWrench & msg)
{
  return fr3_control_publish::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fr3_control_publish::msg::TouchTipWrench>()
{
  return "fr3_control_publish::msg::TouchTipWrench";
}

template<>
inline const char * name<fr3_control_publish::msg::TouchTipWrench>()
{
  return "fr3_control_publish/msg/TouchTipWrench";
}

template<>
struct has_fixed_size<fr3_control_publish::msg::TouchTipWrench>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<fr3_control_publish::msg::TouchTipWrench>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<fr3_control_publish::msg::TouchTipWrench>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__TRAITS_HPP_
