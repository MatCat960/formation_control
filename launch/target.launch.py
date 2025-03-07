import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.time_source import USE_SIM_TIME_NAME

# params
TARGET_R = 5.0
TARGET_T = 60.0


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='formation_control',
            executable='target_node',
            name="target_node",
            output="screen",
            #prefix=prefix,
            parameters=[{USE_SIM_TIME_NAME:True, 
                        'target_radius':TARGET_R,
                        'target_period':TARGET_T
                         }]
        )
    ])
