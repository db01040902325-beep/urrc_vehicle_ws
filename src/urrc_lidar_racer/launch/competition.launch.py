import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
)


def generate_launch_description():

    rviz = LaunchConfiguration('rviz')
    home = os.path.expanduser('~')

    vehicle_file = os.path.join(
        home,
        'urrc_vehicle_ws',
        'models',
        'urrc_car',
        'model_f1_lidar_audi_clean.sdf'
    )

    racer_share = get_package_share_directory(
        'urrc_lidar_racer'
    )

    track_share = get_package_share_directory(
        'urrc_track_gazebo'
    )

    run_world = os.path.join(
        track_share,
        'scripts',
        'run_world.py'
    )

    traffic_lights = os.path.join(
        track_share,
        'scripts',
        'traffic_lights_v10.py'
    )

    gui_config = os.path.join(
        track_share,
        'worlds',
        'monza_grid.gui.config'
    )

    track_models_root = os.path.join(
        track_share,
        'models'
    )

    vehicle_model_root = os.path.join(
        home,
        'urrc_vehicle_ws',
        'models'
    )

    race_launch = os.path.join(
        racer_share,
        'launch',
        'race_camera_overtake.launch.py'
    )

    partition = f'urrc_practice_{os.getuid()}'

    # Latest urrc_2 Monza server, with the start-light gantry.
    # Auto lights are delayed until the vehicle/camera have spawned.
    gazebo_server = ExecuteProcess(
        cmd=[
            sys.executable,
            run_world,
            'monza',
            '--server',
            '--view',
            'grid',
            '--no-auto-start',
        ],
        output='screen'
    )

    spawn_command = f"""
set -e

echo "[URRC] Waiting for /world/monza/create ..."

for i in $(seq 1 160); do

    if gz service -l | grep -Fxq "/world/monza/create"; then

        echo "[URRC] Gazebo server READY"
        echo "[URRC] Spawning LiDAR + camera vehicle at grid 1 ..."

        exec ros2 launch ros_gz_sim gz_spawn_model.launch.py \
            world:=monza \
            file:={vehicle_file} \
            entity_name:=urrc_car \
            x:=0.3786723464 \
            y:=-2.0560353977 \
            z:=1.2435265615 \
            R:=0.0 \
            P:=0.0 \
            Y:=1.4573844161
    fi

    sleep 0.25
done

echo "[URRC][FAIL] Gazebo create service timeout"
exit 1
"""

    spawn_vehicle = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            spawn_command
        ],
        output='screen'
    )

    gazebo_gui = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-g',
            '-v',
            '3',
            '--gui-config',
            gui_config,
        ],
        output='screen'
    )

    race = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            race_launch
        ),
        launch_arguments={
            'rviz': rviz
        }.items()
    )

    start_lights = ExecuteProcess(
        cmd=[
            sys.executable,
            traffic_lights,
            '--track',
            'monza',
            '--skip-camera-check',
        ],
        output='screen'
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'rviz',
            default_value='true'
        ),

        SetEnvironmentVariable(
            'GZ_PARTITION',
            partition
        ),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [
                track_models_root,
                ':',
                track_share,
                ':',
                vehicle_model_root,
                ':',
                EnvironmentVariable(
                    'GZ_SIM_RESOURCE_PATH',
                    default_value=''
                )
            ]
        ),

        gazebo_server,
        spawn_vehicle,

        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_vehicle,
                on_exit=[
                    gazebo_gui,

                    # Camera bridge and fusion start first.
                    TimerAction(
                        period=1.0,
                        actions=[race]
                    ),

                    # Then run 5-red sequence.
                    TimerAction(
                        period=3.0,
                        actions=[start_lights]
                    ),
                ]
            )
        ),
    ])
