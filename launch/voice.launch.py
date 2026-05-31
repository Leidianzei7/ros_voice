from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros_voice",
            executable="voice_node",
            name="voice_node",
            output="screen",
        ),
        Node(
            package="ros_voice",
            executable="brain_node",
            name="brain_node",
            output="screen",
        ),
        Node(
            package="ros_voice",
            executable="control_node",
            name="control_node",
            output="screen",
        ),
    ])
