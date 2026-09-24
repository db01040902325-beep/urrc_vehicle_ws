import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    use_rviz = LaunchConfiguration('rviz')

    pkg_share = get_package_share_directory('urrc_lidar_racer')

    rviz_config = os.path.join(
        pkg_share,
        'config',
        'race.rviz'
    )

    racer_config = os.path.join(
        pkg_share,
        'config',
        'racer.yaml'
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz2'
        ),

        # =========================
        # LiDAR TF
        # =========================
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_static_tf',
            arguments=[
                '--x', '0.272',
                '--y', '0.0',
                '--z', '-0.040',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id',
                'urrc_car/lidar_link/race_lidar'
            ],
            output='screen'
        ),

        # =========================
        # Gazebo -> ROS LiDAR
        # =========================
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='scan_bridge',
            arguments=[
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            ],
            output='screen'
        ),

        # =========================
        # ROS -> Gazebo cmd_vel
        # =========================
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='cmd_vel_bridge',
            arguments=[
                '/model/urrc_car/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            output='screen'
        ),

        # =========================
        # LiDAR autonomous racer
        # =========================
        Node(
            package='urrc_lidar_racer',
            executable='lidar_racer',
            name='lidar_racer',
            parameters=[racer_config],
            output='screen'
        ),

        # =========================
        # Gazebo odometry -> ROS 2
        # =========================
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='odom_bridge',
            arguments=[
                '/model/urrc_car/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry'
            ],
            output='screen'
        ),

        # =========================
        # Lap timer
        # =========================
        Node(
            package='urrc_lidar_racer',
            executable='lap_timer',
            name='lap_timer',
            output='screen'
        ),

        # =========================
        # Race statistics / CSV logger
        # =========================
        Node(
            package='urrc_lidar_racer',
            executable='race_stats',
            name='race_stats',
            output='screen'
        ),

        # =========================
        # LiDAR Bird-eye image
        # =========================
        Node(
            package='urrc_lidar_racer',
            executable='birdseye_view',
            name='birdseye_view',
            output='screen'
        ),

        # =========================
        # RViz
        # =========================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d',
                rviz_config
            ],
            condition=IfCondition(use_rviz),
            output='screen'
        ),

    ])
