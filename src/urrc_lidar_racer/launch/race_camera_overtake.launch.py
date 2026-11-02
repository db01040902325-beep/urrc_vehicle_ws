import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    use_rviz = LaunchConfiguration('rviz')

    pkg_share = get_package_share_directory(
        'urrc_lidar_racer'
    )

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

    fusion_config = os.path.join(
        pkg_share,
        'config',
        'camera_overtake.yaml'
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz2'
        ),

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

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rear_lidar_static_tf',
            arguments=[
                '--x', '-0.20',
                '--y', '0.0',
                '--z', '-0.040',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '3.14159265',
                '--frame-id', 'base_link',
                '--child-frame-id',
                'urrc_car/rear_lidar_link/rear_lidar'
            ],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_static_tf',
            arguments=[
                '--x', '0.16',
                '--y', '0.0',
                '--z', '0.075',
                '--roll', '0.0',
                '--pitch', '0.0',
                '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id',
                'urrc_car/camera_link/front_camera'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='scan_bridge',
            arguments=[
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='rear_scan_bridge',
            arguments=[
                '/rear_scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='imu_bridge',
            arguments=[
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_bridge',
            arguments=[
                '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='left_wheel_velocity_bridge',
            arguments=[
                '/model/urrc_car/left_wheel_velocity@std_msgs/msg/Float64]gz.msgs.Double'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='right_wheel_velocity_bridge',
            arguments=[
                '/model/urrc_car/right_wheel_velocity@std_msgs/msg/Float64]gz.msgs.Double'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='steer_angle_bridge',
            arguments=[
                '/model/urrc_car/steer_angle@std_msgs/msg/Float64]gz.msgs.Double'
            ],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='odom_bridge',
            arguments=[
                '/model/urrc_car/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry'
            ],
            output='screen'
        ),

        # Existing LiDAR racer becomes the base-path controller.
        Node(
            package='urrc_lidar_racer',
            executable='lidar_racer',
            name='lidar_racer',
            parameters=[racer_config],
            remappings=[
                (
                    '/model/urrc_car/cmd_vel',
                    '/race/lidar_cmd_vel'
                )
            ],
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='camera_perception',
            name='camera_perception',
            parameters=[fusion_config],
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='overtake_fusion',
            name='overtake_fusion',
            parameters=[fusion_config],
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='drive_actuator',
            name='drive_actuator',
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='lap_timer',
            name='lap_timer',
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='race_stats',
            name='race_stats',
            output='screen'
        ),

        Node(
            package='urrc_lidar_racer',
            executable='birdseye_view',
            name='birdseye_view',
            output='screen'
        ),

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
