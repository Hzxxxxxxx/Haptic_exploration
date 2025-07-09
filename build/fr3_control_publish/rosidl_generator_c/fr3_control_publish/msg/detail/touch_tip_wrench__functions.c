// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice
#include "fr3_control_publish/msg/detail/touch_tip_wrench__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `force`
// Member `torque`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
fr3_control_publish__msg__TouchTipWrench__init(fr3_control_publish__msg__TouchTipWrench * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    fr3_control_publish__msg__TouchTipWrench__fini(msg);
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__init(&msg->frame_id)) {
    fr3_control_publish__msg__TouchTipWrench__fini(msg);
    return false;
  }
  // force
  if (!geometry_msgs__msg__Vector3__init(&msg->force)) {
    fr3_control_publish__msg__TouchTipWrench__fini(msg);
    return false;
  }
  // torque
  if (!geometry_msgs__msg__Vector3__init(&msg->torque)) {
    fr3_control_publish__msg__TouchTipWrench__fini(msg);
    return false;
  }
  return true;
}

void
fr3_control_publish__msg__TouchTipWrench__fini(fr3_control_publish__msg__TouchTipWrench * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // frame_id
  rosidl_runtime_c__String__fini(&msg->frame_id);
  // force
  geometry_msgs__msg__Vector3__fini(&msg->force);
  // torque
  geometry_msgs__msg__Vector3__fini(&msg->torque);
}

bool
fr3_control_publish__msg__TouchTipWrench__are_equal(const fr3_control_publish__msg__TouchTipWrench * lhs, const fr3_control_publish__msg__TouchTipWrench * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->frame_id), &(rhs->frame_id)))
  {
    return false;
  }
  // force
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->force), &(rhs->force)))
  {
    return false;
  }
  // torque
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->torque), &(rhs->torque)))
  {
    return false;
  }
  return true;
}

bool
fr3_control_publish__msg__TouchTipWrench__copy(
  const fr3_control_publish__msg__TouchTipWrench * input,
  fr3_control_publish__msg__TouchTipWrench * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->frame_id), &(output->frame_id)))
  {
    return false;
  }
  // force
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->force), &(output->force)))
  {
    return false;
  }
  // torque
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->torque), &(output->torque)))
  {
    return false;
  }
  return true;
}

fr3_control_publish__msg__TouchTipWrench *
fr3_control_publish__msg__TouchTipWrench__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fr3_control_publish__msg__TouchTipWrench * msg = (fr3_control_publish__msg__TouchTipWrench *)allocator.allocate(sizeof(fr3_control_publish__msg__TouchTipWrench), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fr3_control_publish__msg__TouchTipWrench));
  bool success = fr3_control_publish__msg__TouchTipWrench__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fr3_control_publish__msg__TouchTipWrench__destroy(fr3_control_publish__msg__TouchTipWrench * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fr3_control_publish__msg__TouchTipWrench__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fr3_control_publish__msg__TouchTipWrench__Sequence__init(fr3_control_publish__msg__TouchTipWrench__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fr3_control_publish__msg__TouchTipWrench * data = NULL;

  if (size) {
    data = (fr3_control_publish__msg__TouchTipWrench *)allocator.zero_allocate(size, sizeof(fr3_control_publish__msg__TouchTipWrench), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fr3_control_publish__msg__TouchTipWrench__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fr3_control_publish__msg__TouchTipWrench__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
fr3_control_publish__msg__TouchTipWrench__Sequence__fini(fr3_control_publish__msg__TouchTipWrench__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      fr3_control_publish__msg__TouchTipWrench__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

fr3_control_publish__msg__TouchTipWrench__Sequence *
fr3_control_publish__msg__TouchTipWrench__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fr3_control_publish__msg__TouchTipWrench__Sequence * array = (fr3_control_publish__msg__TouchTipWrench__Sequence *)allocator.allocate(sizeof(fr3_control_publish__msg__TouchTipWrench__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fr3_control_publish__msg__TouchTipWrench__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fr3_control_publish__msg__TouchTipWrench__Sequence__destroy(fr3_control_publish__msg__TouchTipWrench__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fr3_control_publish__msg__TouchTipWrench__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fr3_control_publish__msg__TouchTipWrench__Sequence__are_equal(const fr3_control_publish__msg__TouchTipWrench__Sequence * lhs, const fr3_control_publish__msg__TouchTipWrench__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fr3_control_publish__msg__TouchTipWrench__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fr3_control_publish__msg__TouchTipWrench__Sequence__copy(
  const fr3_control_publish__msg__TouchTipWrench__Sequence * input,
  fr3_control_publish__msg__TouchTipWrench__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fr3_control_publish__msg__TouchTipWrench);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fr3_control_publish__msg__TouchTipWrench * data =
      (fr3_control_publish__msg__TouchTipWrench *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fr3_control_publish__msg__TouchTipWrench__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fr3_control_publish__msg__TouchTipWrench__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fr3_control_publish__msg__TouchTipWrench__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
