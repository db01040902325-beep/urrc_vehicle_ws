import os

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
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
)


def generate_launch_description():

    rviz = LaunchConfiguration('rviz')
    home = os.path.expanduser('~')

    world_file = os.path.join(
        home,
        'urrc_vehicle_ws',
        'worlds',
        'monza_lidar.sdf'
    )

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

    track_resource_root = os.path.dirname(track_share)

    vehicle_model_root = os.path.join(
        home,
        'urrc_vehicle_ws',
        'models'
    )

    race_launch = os.path.join(
        racer_share,
        'launch',
        'race.launch.py'
    )

    # ==========================================
    # 1. Gazebo SERVER ONLY
    # ==========================================
    gazebo_server = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-s',
            '-r',
            '-v',
            '3',
            world_file
        ],
        output='screen'
    )

    # ==========================================
    # 2. World create 서비스 확인 후 차량 Spawn
    # ==========================================
    spawn_command = f"""
set -e

echo "[URRC] Waiting for /world/monza/create ..."

for i in $(seq 1 120); do

    if gz service -l | grep -Fxq "/world/monza/create"; then

        echo "[URRC] Gazebo server READY"
        echo "[URRC] Spawning urrc_car ..."

        exec ros2 launch ros_gz_sim gz_spawn_model.launch.py \
            world:=monza \
            file:={vehicle_file} \
            entity_name:=urrc_car \
            x:=39.59260 \
            y:=-27.38488 \
            z:=0.120 \
            R:=0.0 \
            P:=0.0 \
            Y:=1.5707963
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

    # ==========================================
    # 3. 차량이 존재한 다음 GUI 실행
    # ==========================================
    gazebo_gui = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-g',
            '-v',
            '3'
        ],
        output='screen'
    )

    # ==========================================
    # 4. 자율주행 시스템
    # ==========================================
    race = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            race_launch
        ),
        launch_arguments={
            'rviz': rviz
        }.items()
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'rviz',
            default_value='true'
        ),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [
                track_resource_root,
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

        # 서버부터 시작
        gazebo_server,

        # 서버 준비 후 차량 Spawn
        spawn_vehicle,

        # 차량 Spawn 완료 후
        # GUI를 열고, 2초 뒤 자율주행 시작
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_vehicle,
                on_exit=[
                    gazebo_gui,

                    TimerAction(
                        period=2.0,
                        actions=[race]
                    )
                ]
            )
        ),
    ])
