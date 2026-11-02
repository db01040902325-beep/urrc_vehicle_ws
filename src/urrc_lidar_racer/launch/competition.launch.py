import json
import os
import sys
import tempfile

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

    world_file = os.path.join(
        track_share,
        'worlds',
        'monza_race.sdf'
    )

    traffic_lights = os.path.join(
        track_share,
        'scripts',
        'traffic_lights_v10.py'
    )

    grid_data_file = os.path.join(
        track_share,
        'tracks',
        'processed',
        'monza.json'
    )

    with open(grid_data_file, 'r', encoding='utf-8') as handle:
        monza_data = json.load(handle)

    grid_1 = monza_data['grid'][0]
    spawn_x = float(grid_1['x'])
    spawn_y = float(grid_1['y'])
    spawn_yaw = float(grid_1['yaw'])
    road_z = float(grid_1['road_z'])

    # Wheel center is 0.075 m below model origin and radius is 0.045 m.
    # Add 5 mm clearance so the tires settle onto the road instead of
    # spawning inside the collision mesh.
    spawn_z = road_z + 0.075 + 0.045 + 0.005

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

    rolling_rosbag = os.path.join(
        home,
        'urrc_vehicle_ws',
        'scripts',
        'rolling_rosbag.sh'
    )

    patched_world = os.path.join(
        tempfile.gettempdir(),
        f'urrc_monza_vehicle_{os.getuid()}.sdf'
    )

    with open(world_file, 'r', encoding='utf-8') as handle:
        world_text = handle.read()

    if 'gz-sim-imu-system' not in world_text:
        imu_plugin = (
            '    <plugin filename="gz-sim-imu-system" '
            'name="gz::sim::systems::Imu" />\n'
        )
        scene_marker = '    <scene>'
        if scene_marker not in world_text:
            raise RuntimeError(
                'Monza world has no <scene> marker for IMU patch'
            )
        world_text = world_text.replace(
            scene_marker,
            imu_plugin + scene_marker,
            1,
        )

    with open(patched_world, 'w', encoding='utf-8') as handle:
        handle.write(world_text)

    partition = f'urrc_practice_{os.getuid()}'

    # Start the race world directly WITHOUT run_world.py so the official
    # light sequence cannot begin before the vehicle camera is ready.
    gazebo_server = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            '-v',
            '3',
            '-s',
            patched_world,
        ],
        output='screen'
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
            x:={spawn_x} \
            y:={spawn_y} \
            z:={spawn_z} \
            R:=0.0 \
            P:=0.0 \
            Y:={spawn_yaw}
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

    rosbag_recorder = ExecuteProcess(
        cmd=[
            'bash',
            rolling_rosbag,
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

                    # Bring vehicle sensors and control online immediately
                    # after the model exists. run_world.py independently
                    # waits for Gazebo readiness before its official lights.
                    TimerAction(
                        period=0.2,
                        actions=[race]
                    ),

                    # Save one ROS bag per run. The recorder script keeps
                    # only the newest 10 runs and removes the oldest first.
                    TimerAction(
                        period=1.0,
                        actions=[rosbag_recorder]
                    ),

                    # Camera / perception nodes must be alive before the
                    # five-light sequence begins. This removes WAIT_START
                    # races caused by run_world.py starting lights too early.
                    TimerAction(
                        period=3.0,
                        actions=[start_lights]
                    ),
                ]
            )
        ),
    ])
