import os
import numpy as np
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.time_source import USE_SIM_TIME_NAME

# params
obstacles = np.array([[3.0, 0.0],
                      [0.0, 3.0],
                      [-5.0, 0.0]])


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='formation_control',
            executable='obstacles_node',
            name="obstacles_node",
            output="screen",
            #prefix=prefix,
            parameters=[{USE_SIM_TIME_NAME:True, 
                        'x_obstacles':obstacles[:, 0].tolist(),
                        'y_obstacles':obstacles[:, 1].tolist()
                         }]
        )
    ])
