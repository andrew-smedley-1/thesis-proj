"""Two tables/items, fixed known map, actual Nav2 and MoveIt. No real hardware."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='/agent/worlds/two_items_19.sdf'),
        DeclareLaunchArgument('headless', default_value='false'),
        ExecuteProcess(cmd=['python3','/agent/scripts/two_table_map.py'],output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('tm12s_moveit_config'),'launch','sim.launch.py')),
            launch_arguments={'world':LaunchConfiguration('world'),'spawn_x':'-0.8',
                'headless':LaunchConfiguration('headless'),'rviz':'false','perception':'false','slam':'false',
                'gui_config':'/agent/config/two_table.gui.config',
                'params_file':'/agent/config/two_table_nav2.yaml'}.items())
    ])
