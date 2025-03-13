import os
import json
from launch.actions import DeclareLaunchArgument
import numpy as np
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.time_source import USE_SIM_TIME_NAME


def generate_launch_description():
    # ------------------------------------------------------------
    # 1. Locate config file and define a default LaunchConfiguration
    # ------------------------------------------------------------
    config_dir = os.path.join(get_package_share_directory('formation_control'), 'config')
    config_arg = DeclareLaunchArgument(
        'config',
        description='Name of the parameter file (with extension)'
    )
    param_file = PathJoinSubstitution([
        config_dir,
        LaunchConfiguration('config')
    ])
    # ------------------------------------------------------------
    # 2. Read environment variables for UAV name, ID, run_type, etc.
    # ------------------------------------------------------------
    uav_name = EnvironmentVariable('UAV_NAME')
    run_type = EnvironmentVariable('RUN_TYPE')
    uav_id = EnvironmentVariable('UAV_ID')
    uav_name_param = LaunchConfiguration('uav_name', default=uav_name)
    uav_id_param = LaunchConfiguration('uav_id', default=uav_id)


    uav_name_arg = DeclareLaunchArgument(
        'uav_name',
        default_value=uav_name,
        description='Namespace for the UAV'
    )
    # ------------------------------------------------------------
    # 3. Decide if we are in simulation mode
    # ------------------------------------------------------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # ------------------------------------------------------------
    # 4. Rewrite YAML with the environment-based substitutions
    # ------------------------------------------------------------
    substitutions = {
        'uav_name': uav_name_param,
        'uav_id': uav_id_param,
        'use_sim_time': use_sim_time,
    }


    node =  Node(
        package='formation_control',
        executable='formation_control_node',
        namespace=uav_name_param,
        name='formation_node',
        parameters=[substitutions,param_file]
    )

    return LaunchDescription([
        config_arg,
        uav_name_arg,
        node
    ])
