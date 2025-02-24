from launch import LaunchDescription
from launch_ros.actions import Node

ROBOTS_NUM = 3


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='formation_control',
            namespace='',
            executable='formation_node',
            name=f"formation_node",
            parameters=[{"ROBOTS_NUM": ROBOTS_NUM}],
            output="screen"
        ),
    ])