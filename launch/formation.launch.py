import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rclpy.time_source import USE_SIM_TIME_NAME

# params
MAX_AGENTS = 4
MAX_VEL = 3.0
MAX_OBSTACLES = 10
ROBOT_SAFE_DIST = 5.0
OBSTACLE_SAFE_DIST = 5.0
CLF_ENABLED = True
FORMATION_TYPE = 0


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
        )
    ])
