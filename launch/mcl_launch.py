from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('particle_count', default_value='1000'),
        DeclareLaunchArgument('noise_v', default_value='0.05'), 
        DeclareLaunchArgument('noise_w', default_value='0.02'), 
        DeclareLaunchArgument('sensor_std', default_value='0.316'),
        Node(
            package='mcl_localization',
            executable='mcl_node',
            parameters=[{
                'particle_count': LaunchConfiguration('particle_count'),
                'noise_v': LaunchConfiguration('noise_v'),
                'noise_w': LaunchConfiguration('noise_w'),
                'sensor_std': LaunchConfiguration('sensor_std')
            }]
        )
    ])