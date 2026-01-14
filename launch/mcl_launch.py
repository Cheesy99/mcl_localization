from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mcl_localization',
            executable='mcl_node',
            name='mcl_node',
            output='screen',
            parameters=[
                {'particle_count': 200} 
            ]
        )
    ])