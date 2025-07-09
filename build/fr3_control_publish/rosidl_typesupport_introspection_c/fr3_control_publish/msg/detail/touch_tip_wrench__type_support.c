// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "fr3_control_publish/msg/detail/touch_tip_wrench__rosidl_typesupport_introspection_c.h"
#include "fr3_control_publish/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "fr3_control_publish/msg/detail/touch_tip_wrench__functions.h"
#include "fr3_control_publish/msg/detail/touch_tip_wrench__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `force`
// Member `torque`
#include "geometry_msgs/msg/vector3.h"
// Member `force`
// Member `torque`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  fr3_control_publish__msg__TouchTipWrench__init(message_memory);
}

void fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_fini_function(void * message_memory)
{
  fr3_control_publish__msg__TouchTipWrench__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_member_array[4] = {
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fr3_control_publish__msg__TouchTipWrench, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "frame_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fr3_control_publish__msg__TouchTipWrench, frame_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "force",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fr3_control_publish__msg__TouchTipWrench, force),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fr3_control_publish__msg__TouchTipWrench, torque),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_members = {
  "fr3_control_publish__msg",  // message namespace
  "TouchTipWrench",  // message name
  4,  // number of fields
  sizeof(fr3_control_publish__msg__TouchTipWrench),
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_member_array,  // message members
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_init_function,  // function to initialize message memory (memory has to be allocated)
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_type_support_handle = {
  0,
  &fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_fr3_control_publish
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fr3_control_publish, msg, TouchTipWrench)() {
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_type_support_handle.typesupport_identifier) {
    fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &fr3_control_publish__msg__TouchTipWrench__rosidl_typesupport_introspection_c__TouchTipWrench_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
