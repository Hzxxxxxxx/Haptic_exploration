#!/usr/bin/env python3
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('mujoco_ros2_control_demos')

    # ---------- 1. 机器人 URDF ----------
    urdf_path = os.path.join(pkg_share, 'urdf', 'fr3_nohand.urdf')
    doc = xacro.parse(open(urdf_path))
    xacro.process_doc(doc)
    robot_description = {'robot_description': doc.toxml()}

    # ---------- 2. MuJoCo-ros2_control ----------
    controller_yaml = os.path.join(pkg_share, 'config', 'fr3_controllers.yaml')
    scene_xml       = os.path.join(pkg_share, 'mujoco_models', 'scene.xml')

    mujoco_node = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[robot_description,
                    controller_yaml,
                    {'mujoco_model_path': scene_xml},]
    )

    # ---------- 3. Robot State Publisher ----------
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # ---------- 4. 依次激活控制器 ----------
    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )
    load_jpc = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_position_controller'],
        output='screen'
    )

    

    return LaunchDescription([
        RegisterEventHandler(OnProcessStart(
            target_action=mujoco_node,
            on_start=[load_jsb])),
        RegisterEventHandler(OnProcessExit(
            target_action=load_jsb,
            on_exit=[load_jpc])),
        mujoco_node,
        rsp_node,
    ])
