import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    
    # --- DYNAMIC PATH FINDING ---
    # This finds the installed directory regardless of where your workspace is
    pkg_share = get_package_share_directory('my_bot')
    
    pkg_share = get_package_share_directory('mybot_2wd')
    urdf_file = os.path.join(pkg_share, 'description', 'robot.urdf.xacro')
    robot_desc = Command(['xacro ', urdf_file])
    config_file = os.path.join(pkg_share, 'config', 'lidar_config.yaml')

    parameters=[{'robot_description': robot_desc}]

    # Read the URDF file
    # with open(urdf_file, 'r') as infp:
    #     robot_desc = infp.read()

    return LaunchDescription([
        # 1. Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),
        
        # 2. Servo Driver
        Node(
            package='my_bot', 
            executable='servo_driver', # Note: No .py extension needed if setup.py is correct
            name='servo_driver',
            parameters=[config_file]
        ),
        
        # 3. Scan Accumulator
        Node(
            package='my_bot',
            executable='scan_accumulator',
            name='scan_accumulator',
            parameters=[config_file]
        ),
        
        # 4. Smart Wiper
        Node(
            package='my_bot',
            executable='smart_wiper',
            name='smart_wiper',
            parameters=[config_file]
        ),
        
        # # 5. RViz2 (Optional)
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     # arguments=['-d', os.path.join(pkg_share, 'config', 'my_rviz_config.rviz')] # Uncomment if you save a config
        # )
    ])