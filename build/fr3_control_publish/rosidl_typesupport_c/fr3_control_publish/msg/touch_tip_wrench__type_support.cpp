// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "fr3_control_publish/msg/detail/touch_tip_wrench__struct.h"
#include "fr3_control_publish/msg/detail/touch_tip_wrench__type_support.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace fr3_control_publish
{

namespace msg
{

namespace rosidl_typesupport_c
{

typedef struct _TouchTipWrench_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _TouchTipWrench_type_support_ids_t;

static const _TouchTipWrench_type_support_ids_t _TouchTipWrench_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _TouchTipWrench_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _TouchTipWrench_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _TouchTipWrench_type_support_symbol_names_t _TouchTipWrench_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, fr3_control_publish, msg, TouchTipWrench)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fr3_control_publish, msg, TouchTipWrench)),
  }
};

typedef struct _TouchTipWrench_type_support_data_t
{
  void * data[2];
} _TouchTipWrench_type_support_data_t;

static _TouchTipWrench_type_support_data_t _TouchTipWrench_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _TouchTipWrench_message_typesupport_map = {
  2,
  "fr3_control_publish",
  &_TouchTipWrench_message_typesupport_ids.typesupport_identifier[0],
  &_TouchTipWrench_message_typesupport_symbol_names.symbol_name[0],
  &_TouchTipWrench_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t TouchTipWrench_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_TouchTipWrench_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace msg

}  // namespace fr3_control_publish

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, fr3_control_publish, msg, TouchTipWrench)() {
  return &::fr3_control_publish::msg::rosidl_typesupport_c::TouchTipWrench_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
