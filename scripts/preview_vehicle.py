#!/usr/bin/env python3
"""Connect vehicle spawning/control to an existing urrc_2 preview without editing maps."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PREVIEW_PARTITION = re.compile(r'^urrc_v19_preview_[0-9a-f]{10}$')


def find_partition(track):
    candidates = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            entries = (process / 'environ').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if entry.startswith(b'GZ_PARTITION='):
                value = entry.split(b'=', 1)[1].decode('ascii', 'ignore')
                if PREVIEW_PARTITION.fullmatch(value):
                    candidates.add(value)
    active = []
    for partition in sorted(candidates):
        env = dict(os.environ, GZ_PARTITION=partition)
        try:
            result = subprocess.run(['gz', 'service', '-l'], env=env,
                                    capture_output=True, text=True, timeout=6)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and f'/world/{track}/create' in result.stdout.splitlines():
            active.append(partition)
    if not active:
        raise RuntimeError(f'{track} preview의 /world/{track}/create 서비스가 없습니다. '
                           f'먼저 bash race_v24.sh preview qualifying --track {track}를 실행하십시오.')
    if len(active) != 1:
        raise RuntimeError(f'{track} preview가 여러 개 실행 중입니다: {active}')
    return active[0]


def main():
    parser = argparse.ArgumentParser(description='기존 urrc_2 v24 preview에 차량 수동 연결')
    parser.add_argument('action', choices=('check', 'spawn', 'start', 'signal'))
    parser.add_argument('--track', choices=('monza', 'hungaroring', 'monaco'), required=True)
    parser.add_argument('--grid', type=int, default=1)
    args = parser.parse_args()
    if args.grid < 1:
        parser.error('--grid는 1 이상이어야 합니다')
    try:
        partition = find_partition(args.track)
        print(f'[URRC VEHICLE] Gazebo connected: GZ_PARTITION={partition}', flush=True)
        if args.action == 'check':
            return 0

        home = Path.home()
        vehicle_root = home / 'urrc_vehicle_ws'
        map_root = home / 'urrc_2'
        env = os.environ.copy()
        env['GZ_PARTITION'] = partition
        env['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join(filter(None, (
            str(vehicle_root / 'models'),
            str(map_root / 'src/urrc_track_gazebo/models'),
            env.get('GZ_SIM_RESOURCE_PATH', ''),
        )))
        if args.action == 'signal':
            # Replay the existing map's official five-red model; do not modify urrc_2.
            signal_file = map_root / f'src/urrc_track_gazebo/worlds/{args.track}_signal_active.sdf'
            if not signal_file.is_file():
                raise RuntimeError(f'신호등 템플릿이 없습니다: {signal_file}')
            nodes = subprocess.run(['ros2', 'node', 'list'], env=env,
                                   capture_output=True, text=True, timeout=10)
            running = set(nodes.stdout.splitlines()) if nodes.returncode == 0 else set()
            required = {'/camera_perception', '/overtake_fusion'}
            missing = required - running
            if missing:
                raise RuntimeError('출발 노드를 먼저 실행하십시오: ' + ', '.join(sorted(missing)))
            sys.path.insert(0, str(map_root / 'race/scripts'))
            from atomic_signal_v10 import AtomicFiveRed
            # The map's signal implementation uses the inherited partition.
            os.environ['GZ_PARTITION'] = partition
            gate = AtomicFiveRed(map_root, args.track)
            try:
                gate.start()
                for index in range(1, 6):
                    gate.red(index)
                    print(f'[URRC VEHICLE] RED {index}/5', flush=True)
                    time.sleep(1.0)
                time.sleep(1.0)
                gate.all_off()
                print('[URRC VEHICLE] LIGHTS OUT', flush=True)
            except BaseException:
                gate.cleanup_on_error()
                raise
            return 0
        if args.action == 'start':
            cmd = ['ros2', 'launch', 'urrc_lidar_racer',
                   'race_camera_overtake.launch.py', 'rviz:=false']
            print('[URRC VEHICLE] Starting camera + LiDAR driving nodes', flush=True)
        else:
            model = vehicle_root / 'models/urrc_car/model_f1_lidar_audi_clean.sdf'
            track_file = map_root / f'src/urrc_track_gazebo/tracks/processed/{args.track}.json'
            if not model.is_file():
                raise RuntimeError(f'차량 모델 파일 없음: {model}')
            slots = json.loads(track_file.read_text(encoding='utf-8'))['grid']
            if args.grid > len(slots):
                raise RuntimeError(f'{args.track} 그리드 범위: 1~{len(slots)}')
            slot = slots[args.grid - 1]
            z = float(slot['road_z']) + 0.125
            cmd = ['ros2', 'launch', 'ros_gz_sim', 'gz_spawn_model.launch.py',
                   f'world:={args.track}', f'file:={model}', 'entity_name:=urrc_car',
                   f"x:={float(slot['x'])}", f"y:={float(slot['y'])}",
                   f'z:={z}', 'R:=0.0', 'P:=0.0', f"Y:={float(slot['yaw'])}"]
            print(f'[URRC VEHICLE] Spawning {args.track} grid {args.grid}', flush=True)
        os.execvpe(cmd[0], cmd, env)
    except (OSError, RuntimeError, KeyError, ValueError, IndexError) as exc:
        print(f'[URRC VEHICLE][ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
