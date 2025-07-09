// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from fr3_control_publish:msg/TouchTipWrench.idl
// generated code does not contain a copyright notice

#ifndef FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__FUNCTIONS_H_
#define FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "fr3_control_publish/msg/rosidl_generator_c__visibility_control.h"

#include "fr3_control_publish/msg/detail/touch_tip_wrench__struct.h"

/// Initialize msg/TouchTipWrench message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * fr3_control_publish__msg__TouchTipWrench
 * )) before or use
 * fr3_control_publish__msg__TouchTipWrench__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__init(fr3_control_publish__msg__TouchTipWrench * msg);

/// Finalize msg/TouchTipWrench message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
void
fr3_control_publish__msg__TouchTipWrench__fini(fr3_control_publish__msg__TouchTipWrench * msg);

/// Create msg/TouchTipWrench message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * fr3_control_publish__msg__TouchTipWrench__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
fr3_control_publish__msg__TouchTipWrench *
fr3_control_publish__msg__TouchTipWrench__create();

/// Destroy msg/TouchTipWrench message.
/**
 * It calls
 * fr3_control_publish__msg__TouchTipWrench__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
void
fr3_control_publish__msg__TouchTipWrench__destroy(fr3_control_publish__msg__TouchTipWrench * msg);

/// Check for msg/TouchTipWrench message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__are_equal(const fr3_control_publish__msg__TouchTipWrench * lhs, const fr3_control_publish__msg__TouchTipWrench * rhs);

/// Copy a msg/TouchTipWrench message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__copy(
  const fr3_control_publish__msg__TouchTipWrench * input,
  fr3_control_publish__msg__TouchTipWrench * output);

/// Initialize array of msg/TouchTipWrench messages.
/**
 * It allocates the memory for the number of elements and calls
 * fr3_control_publish__msg__TouchTipWrench__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__Sequence__init(fr3_control_publish__msg__TouchTipWrench__Sequence * array, size_t size);

/// Finalize array of msg/TouchTipWrench messages.
/**
 * It calls
 * fr3_control_publish__msg__TouchTipWrench__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
void
fr3_control_publish__msg__TouchTipWrench__Sequence__fini(fr3_control_publish__msg__TouchTipWrench__Sequence * array);

/// Create array of msg/TouchTipWrench messages.
/**
 * It allocates the memory for the array and calls
 * fr3_control_publish__msg__TouchTipWrench__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
fr3_control_publish__msg__TouchTipWrench__Sequence *
fr3_control_publish__msg__TouchTipWrench__Sequence__create(size_t size);

/// Destroy array of msg/TouchTipWrench messages.
/**
 * It calls
 * fr3_control_publish__msg__TouchTipWrench__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
void
fr3_control_publish__msg__TouchTipWrench__Sequence__destroy(fr3_control_publish__msg__TouchTipWrench__Sequence * array);

/// Check for msg/TouchTipWrench message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__Sequence__are_equal(const fr3_control_publish__msg__TouchTipWrench__Sequence * lhs, const fr3_control_publish__msg__TouchTipWrench__Sequence * rhs);

/// Copy an array of msg/TouchTipWrench messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fr3_control_publish
bool
fr3_control_publish__msg__TouchTipWrench__Sequence__copy(
  const fr3_control_publish__msg__TouchTipWrench__Sequence * input,
  fr3_control_publish__msg__TouchTipWrench__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // FR3_CONTROL_PUBLISH__MSG__DETAIL__TOUCH_TIP_WRENCH__FUNCTIONS_H_
