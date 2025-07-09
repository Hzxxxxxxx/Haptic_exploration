// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#ifndef FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_H_
#define FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'frame_id'
#include "rosidl_runtime_c/string.h"
// Member 'force'
// Member 'torque'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/TouchTipWrench in the package fr3_control_publish.
/**
  * msg/TouchTipWrench.msg
 */
typedef struct fr3_control_publish__msg__TouchTipWrench
{
  builtin_interfaces__msg__Time stamp;
  /// 建议仍然写 "touch_tip"
  rosidl_runtime_c__String frame_id;
  /// N
  geometry_msgs__msg__Vector3 force;
  /// N·m
  geometry_msgs__msg__Vector3 torque;
} fr3_control_publish__msg__TouchTipWrench;

// Struct for a sequence of fr3_control_publish__msg__TouchTipWrench.
typedef struct fr3_control_publish__msg__TouchTipWrench__Sequence
{
  fr3_control_publish__msg__TouchTipWrench * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fr3_control_publish__msg__TouchTipWrench__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__STRUCT_H_
