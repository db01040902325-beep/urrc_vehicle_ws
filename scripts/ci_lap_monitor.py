#!/usr/bin/env python3
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int32, String


class Monitor(Node):
    def __init__(self, timeout):
        super().__init__('ci_lap_monitor')
        self.timeout = timeout
        self.wall_start = time.monotonic()
        self.last_report = self.wall_start

        self.current_lap = 1
        self.last_lap = None
        self.current_lap_time = 0.0

        self.start_allowed = None
        self.red_ratio = None
        self.road_confidence = None
        self.overtake_state = 'UNKNOWN'
        self.controller_mode = 'UNKNOWN'
        self.boundary_confidence = None
        self.imu_yaw_rate = None
        self.imu_lateral_accel = None
        self.imu_roll_deg = None
        self.stability_factor = None
        self.curve_prediction_deg = None
        self.curve_speed_limit = None
        self.lidar_weight = None
        self.camera_weight = None
        self.front_closing_speed = None
        self.front_vehicle_ttc = None
        self.front_vehicle_speed = None
        self.sensor_health = 'UNKNOWN'
        self.overtake_allowed = None
        self.rear_closing_speed = None
        self.rear_vehicle_ttc = None
        self.left_blindspot = None
        self.right_blindspot = None

        self.odom_count = 0
        self.last_xy = None
        self.distance = 0.0
        self.speed_sum = 0.0
        self.speed_samples = 0
        self.max_speed = 0.0
        self.max_distance_from_origin = 0.0

        self.create_subscription(Int32, '/race/lap/current', self.on_lap, 10)
        self.create_subscription(Float32, '/race/lap/last_time_sec', self.on_last_lap, 10)
        self.create_subscription(Float32, '/race/lap/current_time_sec', self.on_lap_time, 10)
        self.create_subscription(Bool, '/camera/start_allowed', self.on_start, 10)
        self.create_subscription(Float32, '/camera/red_ratio', self.on_red, 10)
        self.create_subscription(Float32, '/camera/road_confidence', self.on_conf, 10)
        self.create_subscription(String, '/race/overtake_state', self.on_state, 10)
        self.create_subscription(String, '/race/controller_mode', self.on_mode, 10)
        self.create_subscription(Float32, '/race/continuous_boundary_confidence', self.on_boundary, 10)
        self.create_subscription(Float32, '/race/imu_yaw_rate_rad_s', self.on_imu_yaw, 20)
        self.create_subscription(Float32, '/race/imu_lateral_accel_mps2', self.on_imu_lat, 20)
        self.create_subscription(Float32, '/race/imu_roll_deg', self.on_imu_roll, 20)
        self.create_subscription(Float32, '/race/stability_factor', self.on_stability, 20)
        self.create_subscription(Float32, '/race/curve_prediction_deg', self.on_curve_prediction, 20)
        self.create_subscription(Float32, '/race/curve_speed_limit_mps', self.on_curve_speed, 20)
        self.create_subscription(Float32, '/race/lidar_fusion_weight', self.on_lidar_weight, 20)
        self.create_subscription(Float32, '/race/camera_fusion_weight', self.on_camera_weight, 20)
        self.create_subscription(Float32, '/race/front_closing_speed_mps', self.on_closing_speed, 20)
        self.create_subscription(Float32, '/race/front_vehicle_ttc_sec', self.on_vehicle_ttc, 20)
        self.create_subscription(Float32, '/race/front_vehicle_estimated_speed_mps', self.on_vehicle_speed, 20)
        self.create_subscription(String, '/race/sensor_health', self.on_sensor_health, 20)
        self.create_subscription(Bool, '/race/overtake_allowed', self.on_overtake_allowed, 20)
        self.create_subscription(Float32, '/race/rear_closing_speed_mps', self.on_rear_closing, 20)
        self.create_subscription(Float32, '/race/rear_vehicle_ttc_sec', self.on_rear_ttc, 20)
        self.create_subscription(Bool, '/race/left_blindspot', self.on_left_blindspot, 20)
        self.create_subscription(Bool, '/race/right_blindspot', self.on_right_blindspot, 20)
        self.create_subscription(Odometry, '/model/urrc_car/odometry', self.on_odom, 50)

        self.done = False
        self.success = False
        self.create_timer(1.0, self.tick)

    def on_lap(self, msg):
        self.current_lap = int(msg.data)
        if self.current_lap >= 2:
            self.success = True
            self.done = True

    def on_last_lap(self, msg):
        self.last_lap = float(msg.data)

    def on_lap_time(self, msg):
        self.current_lap_time = float(msg.data)

    def on_start(self, msg):
        self.start_allowed = bool(msg.data)

    def on_red(self, msg):
        self.red_ratio = float(msg.data)

    def on_conf(self, msg):
        self.road_confidence = float(msg.data)

    def on_state(self, msg):
        self.overtake_state = msg.data

    def on_mode(self, msg):
        self.controller_mode = msg.data

    def on_boundary(self, msg):
        self.boundary_confidence = float(msg.data)

    def on_imu_yaw(self, msg):
        self.imu_yaw_rate = float(msg.data)

    def on_imu_lat(self, msg):
        self.imu_lateral_accel = float(msg.data)

    def on_imu_roll(self, msg):
        self.imu_roll_deg = float(msg.data)

    def on_stability(self, msg):
        self.stability_factor = float(msg.data)

    def on_curve_prediction(self, msg):
        self.curve_prediction_deg = float(msg.data)

    def on_curve_speed(self, msg):
        self.curve_speed_limit = float(msg.data)

    def on_lidar_weight(self, msg):
        self.lidar_weight = float(msg.data)

    def on_camera_weight(self, msg):
        self.camera_weight = float(msg.data)

    def on_closing_speed(self, msg):
        self.front_closing_speed = float(msg.data)

    def on_vehicle_ttc(self, msg):
        self.front_vehicle_ttc = float(msg.data)

    def on_vehicle_speed(self, msg):
        self.front_vehicle_speed = float(msg.data)

    def on_sensor_health(self, msg):
        self.sensor_health = msg.data

    def on_overtake_allowed(self, msg):
        self.overtake_allowed = bool(msg.data)

    def on_rear_closing(self, msg):
        self.rear_closing_speed = float(msg.data)

    def on_rear_ttc(self, msg):
        self.rear_vehicle_ttc = float(msg.data)

    def on_left_blindspot(self, msg):
        self.left_blindspot = bool(msg.data)

    def on_right_blindspot(self, msg):
        self.right_blindspot = bool(msg.data)

    def on_odom(self, msg):
        self.odom_count += 1
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        d0 = math.hypot(x, y)
        self.max_distance_from_origin = max(self.max_distance_from_origin, d0)

        if self.last_xy is not None:
            step = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
            if step < 2.0:
                self.distance += step
        self.last_xy = (x, y)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)
        self.max_speed = max(self.max_speed, speed)
        self.speed_sum += speed
        self.speed_samples += 1

    def tick(self):
        now = time.monotonic()
        wall = now - self.wall_start

        if now - self.last_report >= 10.0:
            avg = self.speed_sum / self.speed_samples if self.speed_samples else 0.0
            print(
                f'[MONITOR] wall={wall:.1f}s lap={self.current_lap} '
                f'lap_time={self.current_lap_time:.1f}s '
                f'distance={self.distance:.1f}m avg={avg:.2f}m/s '
                f'max={self.max_speed:.2f}m/s max_r={self.max_distance_from_origin:.1f}m '
                f'start={self.start_allowed} red={self.red_ratio} '
                f'road_conf={self.road_confidence} boundary={self.boundary_confidence} '
                f'imu_yaw={self.imu_yaw_rate} imu_lat={self.imu_lateral_accel} '
                f'imu_roll={self.imu_roll_deg} stability={self.stability_factor} '
                f'curve={self.curve_prediction_deg} curve_limit={self.curve_speed_limit} '
                f'lidar_w={self.lidar_weight} camera_w={self.camera_weight} '
                f'closing={self.front_closing_speed} ttc={self.front_vehicle_ttc} '
                f'lead_v={self.front_vehicle_speed} '
                f'health={self.sensor_health} overtake_ok={self.overtake_allowed} '
                f'rear_closing={self.rear_closing_speed} rear_ttc={self.rear_vehicle_ttc} '
                f'blind_l={self.left_blindspot} blind_r={self.right_blindspot} '
                f'mode={self.controller_mode} overtake={self.overtake_state}',
                flush=True,
            )
            self.last_report = now

        if wall >= self.timeout:
            self.done = True

    def summary(self):
        avg = self.speed_sum / self.speed_samples if self.speed_samples else 0.0
        print('=== CI_RACE_SUMMARY ===')
        print(f'success={self.success}')
        print(f'current_lap={self.current_lap}')
        print(f'last_lap_sec={self.last_lap}')
        print(f'current_lap_time_sec={self.current_lap_time:.3f}')
        print(f'odom_samples={self.odom_count}')
        print(f'distance_travelled_m={self.distance:.3f}')
        print(f'max_distance_from_origin_m={self.max_distance_from_origin:.3f}')
        print(f'avg_speed_mps={avg:.3f}')
        print(f'max_speed_mps={self.max_speed:.3f}')
        print(f'camera_start_allowed={self.start_allowed}')
        print(f'camera_red_ratio={self.red_ratio}')
        print(f'camera_road_confidence={self.road_confidence}')
        print(f'overtake_state={self.overtake_state}')
        print(f'controller_mode={self.controller_mode}')
        print(f'boundary_confidence={self.boundary_confidence}')
        print(f'imu_yaw_rate_rad_s={self.imu_yaw_rate}')
        print(f'imu_lateral_accel_mps2={self.imu_lateral_accel}')
        print(f'imu_roll_deg={self.imu_roll_deg}')
        print(f'stability_factor={self.stability_factor}')
        print(f'curve_prediction_deg={self.curve_prediction_deg}')
        print(f'curve_speed_limit_mps={self.curve_speed_limit}')
        print(f'lidar_fusion_weight={self.lidar_weight}')
        print(f'camera_fusion_weight={self.camera_weight}')
        print(f'front_closing_speed_mps={self.front_closing_speed}')
        print(f'front_vehicle_ttc_sec={self.front_vehicle_ttc}')
        print(f'front_vehicle_estimated_speed_mps={self.front_vehicle_speed}')
        print(f'sensor_health={self.sensor_health}')
        print(f'overtake_allowed={self.overtake_allowed}')
        print(f'rear_closing_speed_mps={self.rear_closing_speed}')
        print(f'rear_vehicle_ttc_sec={self.rear_vehicle_ttc}')
        print(f'left_blindspot={self.left_blindspot}')
        print(f'right_blindspot={self.right_blindspot}')
        print('=== END_CI_RACE_SUMMARY ===')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=330.0)
    args = parser.parse_args()

    rclpy.init()
    node = Monitor(args.timeout)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.summary()
        ok = node.success
        node.destroy_node()
        rclpy.shutdown()

    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
