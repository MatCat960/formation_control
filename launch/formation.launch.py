import os
import json
import numpy as np
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.time_source import USE_SIM_TIME_NAME

# params
MAX_AGENTS = 4
MAX_VEL = 3.0
ROBOT_SAFE_DIST = 5.0
OBSTACLE_SAFE_DIST = 5.0
CLF_ENABLED = True
FORMATION_TYPE = 0

# read obstacles
path = get_package_share_directory("formation_control")
config_path = os.path.join(path, "config", "obstacles.json")
with open(config_path, "r") as file:
    data = json.load(file)
obstacles = np.array(data["obstacles"])

MAX_OBSTACLES = obstacles.shape[0]

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
            parameters=[{USE_SIM_TIME_NAME:True, 
                        'max_agents':MAX_AGENTS,
                        'max_velocity':MAX_VEL,
                        'max_obstacles':MAX_OBSTACLES,
                        'robot_safe_distance':ROBOT_SAFE_DIST,
                        'obstacle_safe_distance':OBSTACLE_SAFE_DIST,
                        'clf_enabled':CLF_ENABLED,
                        'formation_type':FORMATION_TYPE
                         }]
        ),
        Node(
            package='formation_control',
            namespace=ns,
            executable='sim_neighbors_sensing_node.py',
            parameters=[{USE_SIM_TIME_NAME:True}]
        ),
        Node(
            package='formation_control',
            executable='obstacles_node',
            name="obstacles_node",
            output="screen",
            namespace=ns,
            parameters=[{USE_SIM_TIME_NAME:True, 
                        'x_obstacles':obstacles[:, 0].tolist(),
                        'y_obstacles':obstacles[:, 1].tolist(),
                        'z_obstacles':obstacles[:, 2].tolist()
                         }]
        )
    ])
