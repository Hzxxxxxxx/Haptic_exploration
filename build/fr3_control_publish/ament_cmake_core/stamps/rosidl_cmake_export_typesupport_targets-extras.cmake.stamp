# generated from
# rosidl_cmake/cmake/template/rosidl_cmake_export_typesupport_targets.cmake.in

set(_exported_typesupport_targets
  "__rosidl_generator_c:fr3_control_publish__rosidl_generator_c;__rosidl_typesupport_fastrtps_c:fr3_control_publish__rosidl_typesupport_fastrtps_c;__rosidl_typesupport_introspection_c:fr3_control_publish__rosidl_typesupport_introspection_c;__rosidl_typesupport_c:fr3_control_publish__rosidl_typesupport_c;__rosidl_generator_cpp:fr3_control_publish__rosidl_generator_cpp;__rosidl_typesupport_fastrtps_cpp:fr3_control_publish__rosidl_typesupport_fastrtps_cpp;__rosidl_typesupport_introspection_cpp:fr3_control_publish__rosidl_typesupport_introspection_cpp;__rosidl_typesupport_cpp:fr3_control_publish__rosidl_typesupport_cpp;__rosidl_generator_py:fr3_control_publish__rosidl_generator_py")

# populate fr3_control_publish_TARGETS_<suffix>
if(NOT _exported_typesupport_targets STREQUAL "")
  # loop over typesupport targets
  foreach(_tuple ${_exported_typesupport_targets})
    string(REPLACE ":" ";" _tuple "${_tuple}")
    list(GET _tuple 0 _suffix)
    list(GET _tuple 1 _target)

    set(_target "fr3_control_publish::${_target}")
    if(NOT TARGET "${_target}")
      # the exported target must exist
      message(WARNING "Package 'fr3_control_publish' exports the typesupport target '${_target}' which doesn't exist")
    else()
      list(APPEND fr3_control_publish_TARGETS${_suffix} "${_target}")
    endif()
  endforeach()
endif()
