import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.time_source import USE_SIM_TIME_NAME
from ament_index_python.packages import get_package_share_directory
# params
TARGET_R = 5.0
TARGET_T = 60.0


def generate_launch_description():
    ns = LaunchConfiguration('uav_name')
    # read params from file
    path = get_package_share_directory("formation_control")
    config_path = os.path.join(path, "config", "config.yaml")
    
    return LaunchDescription([
       Node(
            package='formation_control',
            executable='target_node',
            name="target_node",
            output="screen",
            namespace=ns,
            parameters=[{USE_SIM_TIME_NAME:True},config_path]
        )
    ])
