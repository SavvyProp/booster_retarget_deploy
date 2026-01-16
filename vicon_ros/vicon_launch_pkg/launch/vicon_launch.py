from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution  # <-- pull in from launch.substitutions

def generate_launch_description():

    # --- 1) Vicon bridge node ----------------------------------
    vicon_params = {
        'host_name': '192.168.50.19',
        'stream_mode': 'ClientPull',
        'update_rate_hz': 300.0, # More than 2x VICON publish rate
        'world_frame_id': 'world',
        'tf_namespace': 'vicon',
        'publish_specific_segment': False,
        'target_subject_name': 'tape',
        'target_segment_name': 'tape'
    }
    vicon_node = Node(
        package='vicon_bridge',
        executable='vicon_bridge',
        name='vicon_bridge',
        output='screen',
        parameters=[vicon_params]
    )

    return LaunchDescription([
        vicon_node
    ])

