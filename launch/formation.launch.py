import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.time_source import USE_SIM_TIME_NAME
def generate_launch_description():
    # Set the namespace to the UAV name
    ns = LaunchConfiguration('uav_name')
    # If debug mode is enabled, start the node with gdb
    # prefix = ''
    # env_param = ''
    # for key, value in os.environ.items():
    #     if not key.startswith('BASH_FUNC_'):
    #         env_param += f' -e "{key}={value}"'
    # prefix = f'tmux split-window {env_param} gdb -ex run --args'
    return LaunchDescription([
        Node(
            package='formation_control',
            executable='formation_control_node',
            namespace=ns,
            name="formation_node",
            output="screen",
            #prefix=prefix,
            parameters=[{USE_SIM_TIME_NAME:True}]
        ),
        Node(
            package='formation_control',
            namespace=ns,
            executable='sim_neighbors_sensing_node.py',
            parameters=[{USE_SIM_TIME_NAME:True}]
        )
    ])
