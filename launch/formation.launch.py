from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Set the namespace to the UAV name
    ns = LaunchConfiguration('uav_name')
    return LaunchDescription([
        Node(
            package='formation_control',
            executable='formation_control_node',
            namespace=ns,
            name="formation_node",
            output="screen"
        ),
        Node(
            package='formation_control',
            namespace=ns,
            executable='sim_neighbors_sensing_node.py',
        )
    ])
