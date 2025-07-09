// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#ifndef FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_HPP_
#define FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'force'
// Member 'torque'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__fr3_control_publish__msg__TouchTipWrench __attribute__((deprecated))
#else
# define DEPRECATED__fr3_control_publish__msg__TouchTipWrench __declspec(deprecated)
#endif

namespace fr3_control_publish
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TouchTipWrench_
{
  using Type = TouchTipWrench_<ContainerAllocator>;

  explicit TouchTipWrench_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    force(_init),
    torque(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
    }
  }

  explicit TouchTipWrench_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    frame_id(_alloc),
    force(_alloc, _init),
    torque(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _force_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _force_type force;
  using _torque_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _torque_type torque;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__force(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->force = _arg;
    return *this;
  }
  Type & set__torque(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->torque = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> *;
  using ConstRawPtr =
    const fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fr3_control_publish__msg__TouchTipWrench
    std::shared_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fr3_control_publish__msg__TouchTipWrench
    std::shared_ptr<fr3_control_publish::msg::TouchTipWrench_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TouchTipWrench_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->force != other.force) {
      return false;
    }
    if (this->torque != other.torque) {
      return false;
    }
    return true;
  }
  bool operator!=(const TouchTipWrench_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TouchTipWrench_

// alias to use template instance with default allocator
using TouchTipWrench =
  fr3_control_publish::msg::TouchTipWrench_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fr3_control_publish

#endif  // FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_HPP_
