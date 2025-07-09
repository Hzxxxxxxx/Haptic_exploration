# msg/TouchTipWrench.msg
builtin_interfaces/Time stamp
string                frame_id           # 建议仍然写 "touch_tip"
geometry_msgs/Vector3 force              # N
geometry_msgs/Vector3 torque             # N·m
