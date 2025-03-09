import os
import json
import numpy as np
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.time_source import USE_SIM_TIME_NAME

# read params from file
path = get_package_share_directory("formation_control")
config_path = os.path.join(path, "config", "config.json")
with open(config_path, "r") as file:
    data = json.load(file)
obstacles = np.array(data["obstacles"])

NUM_OBSTACLES = obstacles.shape[0]
MAX_OBSTACLES = data["max_obstacles"]
MAX_AGENTS = data["max_agents"]
MAX_VEL = data["max_vel"]
ROBOT_SAFE_DIST = data["robot_safety_dist"]
OBSTACLE_SAFE_DIST = data["obstacles_safety_dist"]
CLF_ENABLED = data["clf_enabled"]
FORMATION_TYPE = data["formation_type"]
FORMATION_RADIUS = data["formation_radius"]
TARGET_R = data["target_trajectory_r"]
TARGET_T = data["target_trajectory_t"]
FORMATION_CLF_GAIN = data["formation_clf_gain"]
OBSTACLE_AVOIDANCE_GAIN = data["obstacle_avoidance_gain"]
ROBOT_AVOIDANCE_GAIN = data["robot_avoidance_gain"]


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
                        'num_obstacles':NUM_OBSTACLES,
                        'robot_safe_distance':ROBOT_SAFE_DIST,
                        'obstacle_safe_distance':OBSTACLE_SAFE_DIST,
                        "robot_avoidance_gain":ROBOT_AVOIDANCE_GAIN,
                        "obstacle_avoidance_gain":OBSTACLE_AVOIDANCE_GAIN,
                        'clf_enabled':CLF_ENABLED,
                        'formation_type':FORMATION_TYPE,
                        "formation_radius":FORMATION_RADIUS,
                        "formation_clf_gain":FORMATION_CLF_GAIN
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
        ),
        Node(
            package='formation_control',
            executable='target_node',
            name="target_node",
            output="screen",
            namespace=ns,
            parameters=[{USE_SIM_TIME_NAME:True, 
                        'target_radius':TARGET_R,
                        'target_period':TARGET_T
                         }]
        )
    ])
