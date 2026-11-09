#!/usr/bin/env python3
import math
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String


class OvertakeState(Enum):
    CRUISE = auto()
    DEFEND = auto()
    SHIFT_OUT = auto()
    PASS = auto()
    RETURN = auto()


class OvertakeFusion(Node):
    """Final LiDAR + camera controller with safe overtaking."""

    def __init__(self):
        super().__init__('overtake_fusion')

        defaults = {
            'raw_cmd_topic': '/race/lidar_cmd_vel',
            'final_cmd_topic': '/model/urrc_car/cmd_vel',
            'scan_topic': '/scan',
            'rear_scan_topic': '/rear_scan',
            'imu_topic': '/imu',
            'wheelbase': 0.330,

            # IMU high-speed stabilization.
            'imu_stability_enable': True,
            'imu_timeout_sec': 0.30,
            'imu_fail_speed_mps': 8.0,
            'imu_lateral_soft_mps2': 12.0,
            'imu_lateral_hard_mps2': 20.0,
            'imu_roll_soft_deg': 7.0,
            'imu_roll_hard_deg': 14.0,
            'imu_yaw_overshoot_ratio': 1.35,
            'imu_yaw_margin_rad_s': 0.80,
            'imu_spin_yaw_rate_rad_s': 6.0,
            'imu_min_speed_factor': 0.35,
            'imu_steer_damping': 0.70,

            'camera_gain_deg': 12.0,
            'camera_min_confidence': 0.35,
            'camera_fallback_enable': True,

            # Confidence-based LiDAR + camera steering fusion.
            'boundary_confidence_topic': '/race/continuous_boundary_confidence',
            'fusion_lidar_min_confidence': 0.25,
            'fusion_camera_min_weight': 0.20,
            'sensor_degraded_confidence': 0.20,
            'sensor_degraded_speed_mps': 4.0,

            # Sensor health / fail-safe.
            'camera_timeout_sec': 0.60,
            'boundary_timeout_sec': 0.50,
            'rear_scan_timeout_sec': 0.50,
            'camera_fail_speed_mps': 12.0,
            'rear_lidar_fail_speed_mps': 10.0,
            'multi_sensor_fail_speed_mps': 4.0,

            'camera_fallback_speed_mps': 3.00,
            'camera_fallback_gain_scale': 1.45,

            # Wall danger-zone avoidance only.
            # Outside the danger zone this layer adds zero steering.
            'center_align_enable': True,
            'center_wall_angle_min_deg': 75.0,
            'center_wall_angle_max_deg': 88.0,
            'center_wall_percentile': 55.0,
            'center_wall_min_m': 0.20,
            'center_wall_max_m': 4.50,
            'wall_danger_enter_m': 0.50,
            'wall_danger_release_m': 0.68,
            'wall_danger_min_steer_deg': 2.0,
            'wall_danger_max_steer_deg': 7.0,
            'wall_danger_speed_mps': 2.5,
            'wall_danger_critical_m': 0.30,
            'wall_danger_critical_speed_mps': 1.2,
            'wall_danger_rate_deg_s': 6.0,
            'wall_danger_confirm_frames': 8,
            'wall_danger_width_min_m': 0.90,
            'wall_danger_width_max_m': 3.50,

            'max_steer_deg': 30.0,
            'straight_steer_deadband_deg': 2.5,
            'max_speed_mps': 15.00,
            'raw_cmd_timeout_sec': 0.35,
            'scan_timeout_sec': 0.25,
            'raw_cmd_soft_timeout_sec': 0.35,
            'raw_cmd_hard_timeout_sec': 1.00,
            'scan_soft_timeout_sec': 0.35,
            'scan_hard_timeout_sec': 1.00,
            'sensor_soft_speed_mps': 2.50,
            'lateral_limit_timeout_sec': 0.50,

            'emergency_stop_m': 1.00,
            'front_min_valid_range_m': 0.25,
            'front_sector_deg': 7.0,
            'front_percentile': 25.0,
            'front_emergency_confirm_frames': 3,
            'hard_ttc_sec': 0.30,
            'reaction_time_sec': 0.25,
            'brake_decel_mps2': 10.0,
            'safety_margin_m': 0.50,

            'overtake_trigger_m': 17.50,
            'overtake_trigger_margin_m': 2.00,
            'overtake_front_clear_m': 12.00,

            # Relative-speed / TTC based active overtaking.
            'relative_speed_filter_alpha': 0.35,
            'vehicle_track_timeout_sec': 0.40,
            'overtake_ttc_trigger_sec': 2.20,
            'overtake_ttc_urgent_sec': 1.20,
            'follow_ttc_target_sec': 2.80,
            'follow_min_gap_m': 2.00,
            'follow_speed_margin_mps': 0.40,
            'overtake_side_clear_m': 0.82,
            'overtake_abort_side_m': 0.55,
            'rear_return_clear_m': 2.50,

            # Rear approach / side blind-spot protection.
            'rear_relative_speed_filter_alpha': 0.35,
            'rear_vehicle_track_timeout_sec': 0.45,
            'rear_ttc_block_sec': 2.00,
            'rear_ttc_urgent_sec': 1.10,
            'rear_vehicle_min_distance_m': 1.50,
            'side_blindspot_m': 0.70,
            'side_abort_m': 0.55,

            # Safe single-move defensive driving.
            'defense_enable': True,
            'defense_ttc_trigger_sec': 1.80,
            'defense_ttc_release_sec': 2.60,
            'defense_min_closing_speed_mps': 0.50,
            'defense_lane_target': 0.12,
            'defense_max_sec': 1.00,
            'defense_cooldown_sec': 2.00,
            'defense_speed_mps': 12.0,
            'defense_front_clear_m': 8.0,
            'defense_side_clear_m': 1.00,
            'defense_max_base_steer_deg': 7.5,
            'defense_side_margin_m': 0.30,

            # Front-car discrimination: reject walls / guardrails.
            'vehicle_cluster_gap_m': 0.22,
            'vehicle_min_width_m': 0.10,
            'vehicle_max_width_m': 0.60,
            'vehicle_max_depth_m': 1.00,
            'vehicle_min_points': 4,
            'vehicle_corridor_half_width_m': 0.85,

            # Pass corridor and repeated-overtake protection.
            'pass_corridor_min_m': 1.00,
            'overtake_cooldown_sec': 1.50,

            # Final actuator smoothing.
            'final_accel_limit_mps2': 10.0,
            'final_decel_limit_mps2': 10.0,
            'final_steer_rate_deg_s': 45.0,
            'stop_hold_sec': 0.40,
            'stop_speed_deadband_mps': 0.20,

            'overtake_max_base_steer_deg': 10.0,
            'overtake_detect_hold_sec': 0.25,

            'shift_out_sec': 0.85,
            'pass_min_sec': 0.45,
            'return_sec': 0.95,
            'pass_lane_target': 0.30,

            'fallback_shift_bias_deg': 3.5,
            'fallback_return_bias_deg': 3.0,

            'shift_speed_mps': 10.00,
            'overtake_speed_mps': 15.00,
            'return_speed_mps': 10.00,

            'start_gate_enable': True,
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.raw_cmd = Twist()
        self.raw_cmd_stamp = 0.0
        self.scan = None
        self.scan_stamp = 0.0
        self.rear_scan = None
        self.rear_scan_stamp = 0.0

        self.imu_stamp = 0.0
        self.imu_yaw_rate = 0.0
        self.imu_lateral_accel = 0.0
        self.imu_roll_rad = 0.0

        self.camera_offset = 0.0
        self.camera_confidence = 0.0
        self.camera_stamp = 0.0
        self.boundary_confidence = 0.0
        self.boundary_stamp = 0.0
        self.lateral_speed_limit = float(
            self.get_parameter('max_speed_mps').value
        )
        self.lateral_limit_stamp = 0.0
        self.start_allowed = not bool(
            self.get_parameter('start_gate_enable').value
        )

        self.state = OvertakeState.CRUISE
        self.state_start = self.now_sec()
        self.detect_start = None
        self.pass_side = 1.0
        self.front_clear_since = None
        self.last_overtake_end = -1e9

        self.last_vehicle_distance = None
        self.last_vehicle_stamp = 0.0
        self.filtered_closing_speed = 0.0

        self.last_rear_vehicle_distance = None
        self.last_rear_vehicle_stamp = 0.0
        self.filtered_rear_closing_speed = 0.0

        self.defend_side = 0.0
        self.last_defense_end = -1e9
        self.front_emergency_count = 0
        self.center_align_active = False
        self.center_error_m = 0.0
        self.center_filtered_correction = 0.0
        self.center_last_update_time = self.now_sec()
        self.wall_danger_side = 0
        self.wall_left_danger_frames = 0
        self.wall_right_danger_frames = 0
        self.stop_hold_until = 0.0

        self.output_speed = 0.0
        self.output_steer = 0.0
        self.last_output_time = self.now_sec()

        self.create_subscription(
            Twist,
            str(self.get_parameter('raw_cmd_topic').value),
            self.on_raw_cmd,
            20,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self.on_scan,
            20,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter('rear_scan_topic').value),
            self.on_rear_scan,
            20,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter('imu_topic').value),
            self.on_imu,
            100,
        )
        self.create_subscription(
            Float32,
            '/camera/road_center_offset',
            self.on_camera_offset,
            10,
        )
        self.create_subscription(
            Float32,
            '/camera/road_confidence',
            self.on_camera_confidence,
            10,
        )
        self.create_subscription(
            Float32,
            str(
                self.get_parameter(
                    'boundary_confidence_topic'
                ).value
            ),
            self.on_boundary_confidence,
            10,
        )
        self.create_subscription(
            Float32,
            '/race/lateral_speed_limit_mps',
            self.on_lateral_speed_limit,
            20,
        )
        self.create_subscription(
            Bool,
            '/camera/start_allowed',
            self.on_start_allowed,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('final_cmd_topic').value),
            20,
        )
        self.state_pub = self.create_publisher(
            String, '/race/overtake_state', 10
        )
        self.mode_pub = self.create_publisher(
            String, '/race/controller_mode', 10
        )
        self.safe_speed_pub = self.create_publisher(
            Float32, '/race/safe_speed_limit_mps', 10
        )
        self.front_object_pub = self.create_publisher(
            Float32, '/race/front_vehicle_distance_m', 10
        )
        self.front_object_detected_pub = self.create_publisher(
            Bool, '/race/front_vehicle_detected', 10
        )
        self.closing_speed_pub = self.create_publisher(
            Float32, '/race/front_closing_speed_mps', 20
        )
        self.vehicle_ttc_pub = self.create_publisher(
            Float32, '/race/front_vehicle_ttc_sec', 20
        )
        self.lead_speed_pub = self.create_publisher(
            Float32, '/race/front_vehicle_estimated_speed_mps', 20
        )
        self.rear_closing_speed_pub = self.create_publisher(
            Float32, '/race/rear_closing_speed_mps', 20
        )
        self.rear_vehicle_ttc_pub = self.create_publisher(
            Float32, '/race/rear_vehicle_ttc_sec', 20
        )
        self.left_blindspot_pub = self.create_publisher(
            Bool, '/race/left_blindspot', 20
        )
        self.right_blindspot_pub = self.create_publisher(
            Bool, '/race/right_blindspot', 20
        )
        self.imu_yaw_pub = self.create_publisher(
            Float32, '/race/imu_yaw_rate_rad_s', 20
        )
        self.imu_lat_pub = self.create_publisher(
            Float32, '/race/imu_lateral_accel_mps2', 20
        )
        self.imu_roll_pub = self.create_publisher(
            Float32, '/race/imu_roll_deg', 20
        )
        self.stability_factor_pub = self.create_publisher(
            Float32, '/race/stability_factor', 20
        )
        self.lidar_weight_pub = self.create_publisher(
            Float32, '/race/lidar_fusion_weight', 20
        )
        self.camera_weight_pub = self.create_publisher(
            Float32, '/race/camera_fusion_weight', 20
        )
        self.sensor_health_pub = self.create_publisher(
            String, '/race/sensor_health', 20
        )
        self.overtake_allowed_pub = self.create_publisher(
            Bool, '/race/overtake_allowed', 20
        )
        self.defense_active_pub = self.create_publisher(
            Bool, '/race/defense_active', 20
        )
        self.defense_side_pub = self.create_publisher(
            Float32, '/race/defense_side', 20
        )
        self.center_error_pub = self.create_publisher(
            Float32, '/race/center_error_m', 20
        )
        self.left_wall_pub = self.create_publisher(
            Float32, '/race/left_wall_distance_m', 20
        )
        self.right_wall_pub = self.create_publisher(
            Float32, '/race/right_wall_distance_m', 20
        )
        self.center_active_pub = self.create_publisher(
            Bool, '/race/center_align_active', 20
        )
        self.front_clearance_pub = self.create_publisher(
            Float32, '/race/front_clearance_m', 20
        )
        self.stop_reason_pub = self.create_publisher(
            String, '/race/stop_reason', 20
        )

        self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            'Fusion controller: wall tracking + IMU + overtaking + safe defense'
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_raw_cmd(self, msg):
        self.raw_cmd = msg
        self.raw_cmd_stamp = self.now_sec()

    def on_scan(self, msg):
        self.scan = msg
        self.scan_stamp = self.now_sec()

    def on_rear_scan(self, msg):
        self.rear_scan = msg
        self.rear_scan_stamp = self.now_sec()

    def on_imu(self, msg):
        self.imu_stamp = self.now_sec()
        self.imu_yaw_rate = float(msg.angular_velocity.z)
        self.imu_lateral_accel = float(msg.linear_acceleration.y)

        q = msg.orientation
        sinr_cosp = 2.0 * (
            q.w * q.x + q.y * q.z
        )
        cosr_cosp = 1.0 - 2.0 * (
            q.x * q.x + q.y * q.y
        )
        self.imu_roll_rad = math.atan2(
            sinr_cosp,
            cosr_cosp,
        )

        self.imu_yaw_pub.publish(
            Float32(data=self.imu_yaw_rate)
        )
        self.imu_lat_pub.publish(
            Float32(data=self.imu_lateral_accel)
        )
        self.imu_roll_pub.publish(
            Float32(
                data=math.degrees(self.imu_roll_rad)
            )
        )

    def on_camera_offset(self, msg):
        self.camera_offset = float(np.clip(msg.data, -1.0, 1.0))
        self.camera_stamp = self.now_sec()

    def on_camera_confidence(self, msg):
        self.camera_confidence = float(np.clip(msg.data, 0.0, 1.0))
        self.camera_stamp = self.now_sec()

    def on_boundary_confidence(self, msg):
        self.boundary_confidence = float(
            np.clip(msg.data, 0.0, 1.0)
        )
        self.boundary_stamp = self.now_sec()

    def on_lateral_speed_limit(self, msg):
        self.lateral_speed_limit = float(np.clip(
            msg.data,
            0.0,
            float(self.get_parameter('max_speed_mps').value),
        ))
        self.lateral_limit_stamp = self.now_sec()

    def on_start_allowed(self, msg):
        # Start permission is a one-way latch for the current race.
        # Once the camera has confirmed LIGHTS OUT, later stale/duplicate
        # false messages must never put the vehicle back into WAIT_START.
        if self.start_allowed:
            return

        if bool(msg.data):
            self.start_allowed = True
            self.stop_hold_until = 0.0
            self.front_emergency_count = 0
            self.get_logger().info(
                'Start gate latched OPEN'
            )

    @staticmethod
    def sector_values(scan, deg_min, deg_max):
        if scan is None or not scan.ranges:
            return np.array([], dtype=np.float32)

        values = np.asarray(scan.ranges, dtype=np.float32)
        i0 = int(math.floor(
            (math.radians(deg_min) - scan.angle_min)
            / scan.angle_increment
        ))
        i1 = int(math.ceil(
            (math.radians(deg_max) - scan.angle_min)
            / scan.angle_increment
        ))

        i0 = max(0, min(len(values) - 1, i0))
        i1 = max(0, min(len(values), i1))
        if i1 <= i0:
            return np.array([], dtype=np.float32)

        values = values[i0:i1]
        values = values[np.isfinite(values)]
        values = values[
            (values > max(0.03, scan.range_min))
            & (values < scan.range_max)
        ]
        return values

    def wall_center_measurement(self):
        """Measure perpendicular distance to both guardrails.

        Positive center error means the left wall is farther away than the
        right wall, so the vehicle is too far right. Negative means too far
        left. The correction uses the same sign convention as the existing
        LiDAR wall-centering controller.
        """
        scan = self.scan
        if scan is None or not scan.ranges:
            return None

        angle_min = math.radians(
            float(
                self.get_parameter(
                    'center_wall_angle_min_deg'
                ).value
            )
        )
        angle_max = math.radians(
            float(
                self.get_parameter(
                    'center_wall_angle_max_deg'
                ).value
            )
        )
        minimum = float(
            self.get_parameter('center_wall_min_m').value
        )
        maximum = float(
            self.get_parameter('center_wall_max_m').value
        )

        left = []
        right = []

        for i, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            abs_angle = abs(angle)
            if abs_angle < angle_min or abs_angle > angle_max:
                continue

            lateral = abs(distance * math.sin(angle))
            if lateral < minimum or lateral > maximum:
                continue

            if angle > 0.0:
                left.append(lateral)
            elif angle < 0.0:
                right.append(lateral)

        if len(left) < 4 or len(right) < 4:
            return None

        percentile = float(
            self.get_parameter('center_wall_percentile').value
        )
        left_distance = float(np.percentile(left, percentile))
        right_distance = float(np.percentile(right, percentile))

        center_error = 0.5 * (
            left_distance - right_distance
        )

        return left_distance, right_distance, center_error

    def center_alignment(self):
        """Wall danger-zone avoidance.

        This no longer tries to center the vehicle. When both walls are
        outside the danger zone, this layer contributes exactly zero steering.
        """
        if not bool(
            self.get_parameter('center_align_enable').value
        ):
            return 0.0, float('inf'), False

        measurement = self.wall_center_measurement()
        if measurement is None:
            self.center_align_active = False
            self.wall_danger_side = 0
            self.center_error_m = 0.0
            self.center_filtered_correction = 0.0
            self.center_error_pub.publish(Float32(data=0.0))
            self.left_wall_pub.publish(Float32(data=-1.0))
            self.right_wall_pub.publish(Float32(data=-1.0))
            self.center_active_pub.publish(Bool(data=False))
            return 0.0, float('inf'), False

        left_distance, right_distance, center_error = measurement
        self.center_error_m = center_error

        self.center_error_pub.publish(
            Float32(data=float(center_error))
        )
        self.left_wall_pub.publish(
            Float32(data=float(left_distance))
        )
        self.right_wall_pub.publish(
            Float32(data=float(right_distance))
        )

        enter = float(
            self.get_parameter('wall_danger_enter_m').value
        )
        release = float(
            self.get_parameter('wall_danger_release_m').value
        )

        width = left_distance + right_distance
        width_ok = (
            float(
                self.get_parameter(
                    'wall_danger_width_min_m'
                ).value
            )
            <= width
            <= float(
                self.get_parameter(
                    'wall_danger_width_max_m'
                ).value
            )
        )

        if width_ok and left_distance < enter and right_distance >= enter:
            self.wall_left_danger_frames += 1
        else:
            self.wall_left_danger_frames = 0

        if width_ok and right_distance < enter and left_distance >= enter:
            self.wall_right_danger_frames += 1
        else:
            self.wall_right_danger_frames = 0

        confirm_frames = int(
            self.get_parameter(
                'wall_danger_confirm_frames'
            ).value
        )

        # Hysteresis + temporal confirmation. A single odd LiDAR return must
        # never cause a lane-change-like steering event.
        if self.center_align_active:
            tracked = (
                left_distance
                if self.wall_danger_side < 0
                else right_distance
            )
            if tracked >= release or not width_ok:
                self.center_align_active = False
                self.wall_danger_side = 0
                self.center_filtered_correction = 0.0
        else:
            if self.wall_left_danger_frames >= confirm_frames:
                # Left wall danger -> steer right (negative).
                self.wall_danger_side = -1
                self.center_align_active = True
                self.wall_left_danger_frames = 0
            elif self.wall_right_danger_frames >= confirm_frames:
                # Right wall danger -> steer left (positive).
                self.wall_danger_side = 1
                self.center_align_active = True
                self.wall_right_danger_frames = 0

        if not self.center_align_active:
            self.center_filtered_correction = 0.0
            self.center_active_pub.publish(Bool(data=False))
            return 0.0, float('inf'), False

        danger_distance = (
            left_distance
            if self.wall_danger_side < 0
            else right_distance
        )

        minimum = float(
            self.get_parameter('center_wall_min_m').value
        )
        severity = float(np.clip(
            (enter - danger_distance)
            / max(enter - minimum, 0.05),
            0.0,
            1.0,
        ))

        min_steer = math.radians(
            float(
                self.get_parameter(
                    'wall_danger_min_steer_deg'
                ).value
            )
        )
        max_steer = math.radians(
            float(
                self.get_parameter(
                    'wall_danger_max_steer_deg'
                ).value
            )
        )
        target = (
            min_steer
            + severity * (max_steer - min_steer)
        )
        if self.wall_danger_side < 0:
            target = -target

        now = self.now_sec()
        dt = float(np.clip(
            now - self.center_last_update_time,
            0.01,
            0.10,
        ))
        self.center_last_update_time = now

        rate = math.radians(
            float(
                self.get_parameter(
                    'wall_danger_rate_deg_s'
                ).value
            )
        )
        max_delta = rate * dt
        delta = float(np.clip(
            target - self.center_filtered_correction,
            -max_delta,
            max_delta,
        ))
        self.center_filtered_correction += delta

        speed_limit = float(
            self.get_parameter('wall_danger_speed_mps').value
        )
        critical = float(
            self.get_parameter('wall_danger_critical_m').value
        )
        if danger_distance < critical:
            speed_limit = min(
                speed_limit,
                float(
                    self.get_parameter(
                        'wall_danger_critical_speed_mps'
                    ).value
                ),
            )

        self.center_active_pub.publish(Bool(data=True))

        return (
            float(np.clip(
                self.center_filtered_correction,
                -self.max_steer_rad(),
                self.max_steer_rad(),
            )),
            speed_limit,
            True,
        )

    def front_distance(self):
        sector = float(
            self.get_parameter('front_sector_deg').value
        )
        values = self.sector_values(
            self.scan,
            -sector,
            sector,
        )
        if values.size == 0:
            return float('inf')

        minimum_valid = float(
            self.get_parameter(
                'front_min_valid_range_m'
            ).value
        )
        values = values[values >= minimum_valid]

        if values.size == 0:
            return float('inf')

        percentile = float(
            self.get_parameter('front_percentile').value
        )
        return float(np.percentile(values, percentile))

    def side_clearance(self, left):
        if left:
            values = self.sector_values(self.scan, 14.0, 68.0)
        else:
            values = self.sector_values(self.scan, -68.0, -14.0)
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, 28))

    def rear_distance(self):
        values = self.sector_values(self.rear_scan, -22.0, 22.0)
        if values.size == 0:
            return float('inf')
        return float(np.percentile(values, 18))

    def front_vehicle_candidate(self):
        """Return nearest compact object that resembles another 1/10 car.

        Long continuous surfaces are rejected so guardrails and walls do not
        directly trigger overtaking.
        """
        scan = self.scan
        if scan is None or not scan.ranges:
            return None

        half_width = float(
            self.get_parameter('vehicle_corridor_half_width_m').value
        )
        gap_limit = float(
            self.get_parameter('vehicle_cluster_gap_m').value
        )
        min_width = float(
            self.get_parameter('vehicle_min_width_m').value
        )
        max_width = float(
            self.get_parameter('vehicle_max_width_m').value
        )
        max_depth = float(
            self.get_parameter('vehicle_max_depth_m').value
        )
        min_points = int(
            self.get_parameter('vehicle_min_points').value
        )

        points = []
        for i, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            if x <= 0.20:
                continue
            if abs(y) > half_width:
                continue

            points.append((i, x, y))

        if not points:
            return None

        clusters = []
        current = [points[0]]

        for point in points[1:]:
            prev = current[-1]
            index_gap = point[0] - prev[0]
            euclidean = math.hypot(
                point[1] - prev[1],
                point[2] - prev[2],
            )

            if index_gap <= 2 and euclidean <= gap_limit:
                current.append(point)
            else:
                clusters.append(current)
                current = [point]

        if current:
            clusters.append(current)

        candidates = []

        for cluster in clusters:
            if len(cluster) < min_points:
                continue

            xs = [p[1] for p in cluster]
            ys = [p[2] for p in cluster]

            lateral_width = max(ys) - min(ys)
            depth = max(xs) - min(xs)

            if lateral_width < min_width or lateral_width > max_width:
                continue
            if depth > max_depth:
                continue

            center_x = float(np.median(xs))
            center_y = float(np.median(ys))

            # Keep the object mostly in the forward driving corridor.
            if abs(center_y) > half_width:
                continue

            candidates.append((
                center_x,
                center_y,
                lateral_width,
                depth,
                len(cluster),
            ))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])

    def rear_vehicle_candidate(self):
        """Return nearest compact object directly behind the car."""
        scan = self.rear_scan
        if scan is None or not scan.ranges:
            return None

        half_width = float(
            self.get_parameter('vehicle_corridor_half_width_m').value
        )
        gap_limit = float(
            self.get_parameter('vehicle_cluster_gap_m').value
        )
        min_width = float(
            self.get_parameter('vehicle_min_width_m').value
        )
        max_width = float(
            self.get_parameter('vehicle_max_width_m').value
        )
        max_depth = float(
            self.get_parameter('vehicle_max_depth_m').value
        )
        min_points = int(
            self.get_parameter('vehicle_min_points').value
        )

        points = []
        for i, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            if x <= 0.20 or abs(y) > half_width:
                continue

            points.append((i, x, y))

        if not points:
            return None

        clusters = []
        current = [points[0]]

        for point in points[1:]:
            prev = current[-1]
            gap = math.hypot(
                point[1] - prev[1],
                point[2] - prev[2],
            )
            if point[0] - prev[0] <= 2 and gap <= gap_limit:
                current.append(point)
            else:
                clusters.append(current)
                current = [point]

        if current:
            clusters.append(current)

        candidates = []
        for cluster in clusters:
            if len(cluster) < min_points:
                continue

            xs = [p[1] for p in cluster]
            ys = [p[2] for p in cluster]
            width = max(ys) - min(ys)
            depth = max(xs) - min(xs)

            if not min_width <= width <= max_width:
                continue
            if depth > max_depth:
                continue

            candidates.append((
                float(np.median(xs)),
                float(np.median(ys)),
            ))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])

    def update_rear_motion(
        self,
        rear_vehicle_distance,
        now,
    ):
        timeout = float(
            self.get_parameter(
                'rear_vehicle_track_timeout_sec'
            ).value
        )
        alpha = float(
            self.get_parameter(
                'rear_relative_speed_filter_alpha'
            ).value
        )

        if not math.isfinite(rear_vehicle_distance):
            if now - self.last_rear_vehicle_stamp > timeout:
                self.last_rear_vehicle_distance = None
                self.filtered_rear_closing_speed = 0.0
            return 0.0, float('inf')

        raw_closing = 0.0

        if (
            self.last_rear_vehicle_distance is not None
            and self.last_rear_vehicle_stamp > 0.0
        ):
            dt = now - self.last_rear_vehicle_stamp
            if 0.01 <= dt <= timeout:
                raw_closing = (
                    self.last_rear_vehicle_distance
                    - rear_vehicle_distance
                ) / dt

        self.last_rear_vehicle_distance = rear_vehicle_distance
        self.last_rear_vehicle_stamp = now

        raw_closing = float(np.clip(
            raw_closing,
            -20.0,
            20.0,
        ))

        self.filtered_rear_closing_speed = (
            alpha * raw_closing
            + (1.0 - alpha)
            * self.filtered_rear_closing_speed
        )

        closing = max(0.0, self.filtered_rear_closing_speed)

        if closing > 0.05:
            ttc = rear_vehicle_distance / closing
        else:
            ttc = float('inf')

        return closing, ttc

    def rear_side_distance(self, left):
        """Distance to a rear vehicle approaching on either side."""
        if left:
            values = self.sector_values(
                self.rear_scan, -58.0, -8.0
            )
        else:
            values = self.sector_values(
                self.rear_scan, 8.0, 58.0
            )

        if values.size == 0:
            return float('inf')

        return float(np.percentile(values, 18))

    def blindspot_distance(self, left):
        """Estimate immediate side occupancy from front + rear LiDAR."""
        if left:
            front_values = self.sector_values(
                self.scan, 58.0, 105.0
            )
            # Rear sensor faces backward, so base-left maps to negative angles.
            rear_values = self.sector_values(
                self.rear_scan, -105.0, -58.0
            )
        else:
            front_values = self.sector_values(
                self.scan, -105.0, -58.0
            )
            rear_values = self.sector_values(
                self.rear_scan, 58.0, 105.0
            )

        distances = []
        for values in (front_values, rear_values):
            if values.size > 0:
                distances.append(
                    float(np.percentile(values, 12))
                )

        if not distances:
            return float('inf')

        return min(distances)

    def update_vehicle_motion(
        self,
        vehicle_distance,
        now,
    ):
        """Estimate closing speed, lead speed and TTC from distance history."""
        timeout = float(
            self.get_parameter(
                'vehicle_track_timeout_sec'
            ).value
        )
        alpha = float(
            self.get_parameter(
                'relative_speed_filter_alpha'
            ).value
        )

        if not math.isfinite(vehicle_distance):
            if now - self.last_vehicle_stamp > timeout:
                self.last_vehicle_distance = None
                self.filtered_closing_speed = 0.0
            return 0.0, float('inf'), 0.0

        raw_closing = 0.0

        if (
            self.last_vehicle_distance is not None
            and self.last_vehicle_stamp > 0.0
        ):
            dt = now - self.last_vehicle_stamp
            if 0.01 <= dt <= timeout:
                raw_closing = (
                    self.last_vehicle_distance
                    - vehicle_distance
                ) / dt

        self.last_vehicle_distance = vehicle_distance
        self.last_vehicle_stamp = now

        raw_closing = float(np.clip(
            raw_closing,
            -20.0,
            20.0,
        ))

        self.filtered_closing_speed = (
            alpha * raw_closing
            + (1.0 - alpha)
            * self.filtered_closing_speed
        )

        closing = max(
            0.0,
            self.filtered_closing_speed,
        )

        if closing > 0.05:
            ttc = vehicle_distance / closing
        else:
            ttc = float('inf')

        estimated_lead_speed = max(
            0.0,
            self.output_speed
            - self.filtered_closing_speed,
        )

        return (
            closing,
            ttc,
            estimated_lead_speed,
        )

    def follow_speed_limit(
        self,
        vehicle_distance,
        lead_speed,
        vehicle_ttc,
    ):
        if not math.isfinite(vehicle_distance):
            return float(
                self.get_parameter('max_speed_mps').value
            )

        target_ttc = float(
            self.get_parameter(
                'follow_ttc_target_sec'
            ).value
        )
        minimum_gap = float(
            self.get_parameter(
                'follow_min_gap_m'
            ).value
        )
        margin = float(
            self.get_parameter(
                'follow_speed_margin_mps'
            ).value
        )

        if (
            vehicle_distance <= minimum_gap
            or vehicle_ttc < target_ttc
        ):
            return max(
                0.0,
                lead_speed + margin,
            )

        return float(
            self.get_parameter('max_speed_mps').value
        )

    def pass_corridor_clearance(self, left):
        """Conservative free-space estimate for the intended pass side."""
        if left:
            sectors = [(10.0, 30.0), (30.0, 52.0), (52.0, 72.0)]
        else:
            sectors = [(-30.0, -10.0), (-52.0, -30.0), (-72.0, -52.0)]

        estimates = []
        for lo, hi in sectors:
            values = self.sector_values(self.scan, lo, hi)
            if values.size == 0:
                continue
            estimates.append(float(np.percentile(values, 22)))

        if not estimates:
            return 0.0

        return min(estimates)

    def limit_final_command(self, requested_speed, requested_steer):
        now = self.now_sec()
        dt = float(np.clip(
            now - self.last_output_time,
            0.005,
            0.10,
        ))
        self.last_output_time = now

        accel = float(
            self.get_parameter('final_accel_limit_mps2').value
        )
        decel = float(
            self.get_parameter('final_decel_limit_mps2').value
        )

        speed_delta = requested_speed - self.output_speed
        if speed_delta >= 0.0:
            speed_delta = min(speed_delta, accel * dt)
        else:
            speed_delta = max(speed_delta, -decel * dt)

        self.output_speed += speed_delta

        steer_rate = math.radians(
            float(
                self.get_parameter('final_steer_rate_deg_s').value
            )
        )
        steer_delta = requested_steer - self.output_steer
        steer_delta = float(np.clip(
            steer_delta,
            -steer_rate * dt,
            steer_rate * dt,
        ))
        self.output_steer += steer_delta

        return self.output_speed, self.output_steer

    def apply_imu_stability(
        self,
        speed,
        steer,
    ):
        if not bool(
            self.get_parameter(
                'imu_stability_enable'
            ).value
        ):
            return speed, steer, 1.0, 'IMU_OFF'

        now = self.now_sec()
        timeout = float(
            self.get_parameter('imu_timeout_sec').value
        )

        if (
            self.imu_stamp <= 0.0
            or now - self.imu_stamp > timeout
        ):
            fail_speed = float(
                self.get_parameter(
                    'imu_fail_speed_mps'
                ).value
            )
            return (
                min(speed, fail_speed),
                steer,
                min(
                    1.0,
                    fail_speed / max(speed, 0.05)
                ),
                'IMU_FAILSAFE',
            )

        factor = 1.0
        steer_factor = 1.0
        status = 'IMU_OK'

        lateral = abs(self.imu_lateral_accel)
        lat_soft = float(
            self.get_parameter(
                'imu_lateral_soft_mps2'
            ).value
        )
        lat_hard = float(
            self.get_parameter(
                'imu_lateral_hard_mps2'
            ).value
        )

        if lateral > lat_soft:
            factor = min(
                factor,
                math.sqrt(
                    lat_soft
                    / max(lateral, 0.01)
                ),
            )
            status = 'IMU_LAT_LIMIT'

        if lateral > lat_hard:
            factor = min(factor, 0.50)
            steer_factor = min(steer_factor, 0.75)
            status = 'IMU_LAT_HARD'

        roll_deg = abs(
            math.degrees(self.imu_roll_rad)
        )
        roll_soft = float(
            self.get_parameter(
                'imu_roll_soft_deg'
            ).value
        )
        roll_hard = float(
            self.get_parameter(
                'imu_roll_hard_deg'
            ).value
        )

        if roll_deg > roll_soft:
            factor = min(
                factor,
                max(
                    0.50,
                    roll_soft / max(roll_deg, 0.1),
                ),
            )
            status = 'IMU_ROLL_LIMIT'

        if roll_deg > roll_hard:
            factor = min(factor, 0.40)
            steer_factor = min(steer_factor, 0.65)
            status = 'IMU_ROLL_HARD'

        wheelbase = float(
            self.get_parameter('wheelbase').value
        )
        expected_yaw = 0.0
        if speed > 0.05:
            expected_yaw = (
                speed
                * math.tan(steer)
                / max(wheelbase, 1e-3)
            )

        measured_yaw = self.imu_yaw_rate
        yaw_ratio = float(
            self.get_parameter(
                'imu_yaw_overshoot_ratio'
            ).value
        )
        yaw_margin = float(
            self.get_parameter(
                'imu_yaw_margin_rad_s'
            ).value
        )
        spin_limit = float(
            self.get_parameter(
                'imu_spin_yaw_rate_rad_s'
            ).value
        )

        overshoot_limit = max(
            spin_limit,
            abs(expected_yaw) * yaw_ratio + yaw_margin,
        )

        if abs(measured_yaw) > overshoot_limit:
            factor = min(factor, 0.60)
            steer_factor = min(
                steer_factor,
                float(
                    self.get_parameter(
                        'imu_steer_damping'
                    ).value
                ),
            )
            status = 'IMU_YAW_LIMIT'

        if (
            abs(expected_yaw) > 0.50
            and abs(measured_yaw) > 0.50
            and expected_yaw * measured_yaw < 0.0
        ):
            factor = min(factor, 0.50)
            steer_factor = min(steer_factor, 0.55)
            status = 'IMU_YAW_MISMATCH'

        min_factor = float(
            self.get_parameter(
                'imu_min_speed_factor'
            ).value
        )
        factor = float(np.clip(
            factor,
            min_factor,
            1.0,
        ))

        return (
            speed * factor,
            steer * steer_factor,
            factor,
            status,
        )

    def max_steer_rad(self):
        return math.radians(
            float(self.get_parameter('max_steer_deg').value)
        )

    def raw_steer_rad(self, speed):
        if abs(speed) < 0.05:
            return 0.0
        wheelbase = float(self.get_parameter('wheelbase').value)
        steer = math.atan(
            self.raw_cmd.angular.z * wheelbase / speed
        )
        return float(np.clip(
            steer, -self.max_steer_rad(), self.max_steer_rad()
        ))

    def sensor_health(self, now):
        camera_timeout = float(
            self.get_parameter('camera_timeout_sec').value
        )
        boundary_timeout = float(
            self.get_parameter('boundary_timeout_sec').value
        )
        rear_timeout = float(
            self.get_parameter('rear_scan_timeout_sec').value
        )
        imu_timeout = float(
            self.get_parameter('imu_timeout_sec').value
        )

        camera_ok = (
            self.camera_stamp > 0.0
            and now - self.camera_stamp <= camera_timeout
        )
        boundary_ok = (
            self.boundary_stamp > 0.0
            and now - self.boundary_stamp <= boundary_timeout
        )
        rear_ok = (
            self.rear_scan_stamp > 0.0
            and now - self.rear_scan_stamp <= rear_timeout
        )
        imu_ok = (
            self.imu_stamp > 0.0
            and now - self.imu_stamp <= imu_timeout
        )

        failed = []
        if not camera_ok:
            failed.append('CAMERA')
        if not boundary_ok:
            failed.append('BOUNDARY')
        if not rear_ok:
            failed.append('REAR_LIDAR')
        if not imu_ok:
            failed.append('IMU')

        health = 'OK' if not failed else 'DEGRADED:' + '+'.join(failed)

        return {
            'camera_ok': camera_ok,
            'boundary_ok': boundary_ok,
            'rear_ok': rear_ok,
            'imu_ok': imu_ok,
            'health': health,
        }

    def fail_safe_speed_limit(self, health):
        limits = [
            float(self.get_parameter('max_speed_mps').value)
        ]

        if not health['camera_ok']:
            limits.append(
                float(
                    self.get_parameter(
                        'camera_fail_speed_mps'
                    ).value
                )
            )

        if not health['rear_ok']:
            limits.append(
                float(
                    self.get_parameter(
                        'rear_lidar_fail_speed_mps'
                    ).value
                )
            )

        failed_count = sum(
            1
            for key in (
                'camera_ok',
                'boundary_ok',
                'rear_ok',
                'imu_ok',
            )
            if not health[key]
        )

        if failed_count >= 2:
            limits.append(
                float(
                    self.get_parameter(
                        'multi_sensor_fail_speed_mps'
                    ).value
                )
            )

        return min(limits)

    def fusion_weights(self):
        now = self.now_sec()
        boundary_fresh = (
            self.boundary_stamp > 0.0
            and now - self.boundary_stamp
            <= float(
                self.get_parameter(
                    'boundary_timeout_sec'
                ).value
            )
        )
        camera_fresh = (
            self.camera_stamp > 0.0
            and now - self.camera_stamp
            <= float(
                self.get_parameter(
                    'camera_timeout_sec'
                ).value
            )
        )

        lidar_weight = (
            float(np.clip(self.boundary_confidence, 0.0, 1.0))
            if boundary_fresh
            else 0.0
        )

        minimum_camera_weight = float(
            self.get_parameter(
                'fusion_camera_min_weight'
            ).value
        )

        # Camera influence grows as LiDAR confidence drops.
        camera_scale = max(
            minimum_camera_weight,
            1.0 - lidar_weight,
        )
        camera_weight = (
            float(
                np.clip(
                    self.camera_confidence * camera_scale,
                    0.0,
                    1.0,
                )
            )
            if camera_fresh
            else 0.0
        )

        return lidar_weight, camera_weight

    def camera_correction(self, target_offset=0.0, scale=1.0):
        if (
            self.camera_stamp <= 0.0
            or self.now_sec() - self.camera_stamp
            > float(
                self.get_parameter(
                    'camera_timeout_sec'
                ).value
            )
        ):
            return None

        minimum = float(
            self.get_parameter('camera_min_confidence').value
        )
        if self.camera_confidence < minimum:
            return None

        gain = math.radians(
            float(self.get_parameter('camera_gain_deg').value)
        )
        error = self.camera_offset - target_offset

        _, camera_weight = self.fusion_weights()

        return (
            -gain
            * error
            * camera_weight
            * scale
        )

    def dynamic_stop_distance(self, speed):
        reaction = float(
            self.get_parameter('reaction_time_sec').value
        )
        decel = max(
            0.5,
            float(self.get_parameter('brake_decel_mps2').value),
        )
        margin = float(
            self.get_parameter('safety_margin_m').value
        )
        speed = max(0.0, float(speed))
        return (
            margin
            + speed * reaction
            + speed * speed / (2.0 * decel)
        )

    def safe_speed_for_distance(self, distance):
        if not math.isfinite(distance):
            return float(
                self.get_parameter('max_speed_mps').value
            )

        reaction = float(
            self.get_parameter('reaction_time_sec').value
        )
        decel = max(
            0.5,
            float(self.get_parameter('brake_decel_mps2').value),
        )
        margin = float(
            self.get_parameter('safety_margin_m').value
        )

        available = max(0.0, distance - margin)
        # v^2/(2a) + reaction*v <= available
        root = math.sqrt(
            (decel * reaction) ** 2
            + 2.0 * decel * available
        )
        return max(0.0, -decel * reaction + root)

    def set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self.state_start = self.now_sec()
        self.front_clear_since = None
        self.get_logger().info(
            f'Overtake state -> {state.name}'
        )

    def publish_stop(self, mode):
        now = self.now_sec()
        was_moving = self.output_speed > float(
            self.get_parameter('stop_speed_deadband_mps').value
        )

        # WAIT_START and ordinary data waits must not arm a persistent hold.
        # Only a stop that interrupts actual motion gets a short brake hold.
        physical_hold_modes = {
            'EMERGENCY_STOP',
            'FRONT_LIDAR_TIMEOUT',
            'PATH_SENSORS_LOST',
        }
        if was_moving and mode in physical_hold_modes:
            self.stop_hold_until = (
                now
                + float(self.get_parameter('stop_hold_sec').value)
            )
        elif mode in {'WAIT_START', 'RAW_CMD_TIMEOUT'}:
            self.stop_hold_until = 0.0

        self.output_speed = 0.0
        self.output_steer = 0.0
        self.cmd_pub.publish(Twist())
        self.state_pub.publish(String(data=self.state.name))
        self.mode_pub.publish(String(data=mode))
        self.stop_reason_pub.publish(String(data=mode))
        self.safe_speed_pub.publish(Float32(data=0.0))
        self.stability_factor_pub.publish(
            Float32(data=0.0)
        )

    def publish_command(self, speed, steer, mode):
        now = self.now_sec()

        if now < self.stop_hold_until:
            self.output_speed = 0.0
            self.output_steer = 0.0
            self.cmd_pub.publish(Twist())
            self.state_pub.publish(String(data=self.state.name))
            self.mode_pub.publish(String(data=f'{mode}:STOP_HOLD'))
            self.safe_speed_pub.publish(Float32(data=0.0))
            return

        deadband = float(
            self.get_parameter('stop_speed_deadband_mps').value
        )
        if 0.0 <= speed < deadband:
            speed = 0.0

        speed = float(np.clip(
            speed,
            0.0,
            float(self.get_parameter('max_speed_mps').value),
        ))
        steer = float(np.clip(
            steer,
            -self.max_steer_rad(),
            self.max_steer_rad(),
        ))

        (
            speed,
            steer,
            stability_factor,
            imu_status,
        ) = self.apply_imu_stability(
            speed,
            steer,
        )

        self.stability_factor_pub.publish(
            Float32(data=float(stability_factor))
        )

        if imu_status != 'IMU_OK':
            mode = f'{mode}:{imu_status}'

        speed, steer = self.limit_final_command(
            speed,
            steer,
        )

        command = Twist()
        command.linear.x = speed

        if speed > 0.05:
            wheelbase = float(
                self.get_parameter('wheelbase').value
            )
            command.angular.z = (
                speed * math.tan(steer) / wheelbase
            )

        self.cmd_pub.publish(command)
        self.state_pub.publish(String(data=self.state.name))
        self.mode_pub.publish(String(data=mode))
        self.stop_reason_pub.publish(String(data='NONE'))

    def control_loop(self):
        now = self.now_sec()

        raw_soft = float(
            self.get_parameter('raw_cmd_soft_timeout_sec').value
        )
        raw_hard = float(
            self.get_parameter('raw_cmd_hard_timeout_sec').value
        )
        scan_soft = float(
            self.get_parameter('scan_soft_timeout_sec').value
        )
        scan_hard = float(
            self.get_parameter('scan_hard_timeout_sec').value
        )

        raw_age = (
            float('inf')
            if self.raw_cmd_stamp <= 0.0
            else max(0.0, now - self.raw_cmd_stamp)
        )
        scan_age = (
            float('inf')
            if self.scan_stamp <= 0.0
            else max(0.0, now - self.scan_stamp)
        )

        raw_degraded = raw_age > raw_soft
        scan_degraded = scan_age > scan_soft

        camera_timeout = float(
            self.get_parameter('camera_timeout_sec').value
        )
        camera_fresh = (
            self.camera_stamp > 0.0
            and now - self.camera_stamp <= camera_timeout
        )
        camera_can_recover = (
            self.start_allowed
            and camera_fresh
            and self.camera_confidence
            >= float(
                self.get_parameter(
                    'camera_min_confidence'
                ).value
            )
        )

        if raw_age > raw_hard:
            if not camera_can_recover:
                self.publish_stop('RAW_CMD_TIMEOUT')
                return

            # Hard raw-command timeout can still recover from camera, but the
            # vehicle must re-enter through the deliberate low-speed fallback.
            self.raw_cmd = Twist()
            raw_degraded = True

        if scan_age > scan_hard:
            self.publish_stop('FRONT_LIDAR_TIMEOUT')
            return

        if (
            bool(self.get_parameter('start_gate_enable').value)
            and not self.start_allowed
        ):
            self.publish_stop('WAIT_START')
            return

        health = self.sensor_health(now)
        self.sensor_health_pub.publish(
            String(data=health['health'])
        )

        overtake_allowed = bool(
            health['rear_ok']
            and health['imu_ok']
            and health['boundary_ok']
            and not raw_degraded
            and not scan_degraded
        )
        self.overtake_allowed_pub.publish(
            Bool(data=overtake_allowed)
        )

        # No reliable path source: stop instead of guessing.
        if (
            not health['boundary_ok']
            and not health['camera_ok']
        ):
            self.publish_stop('PATH_SENSORS_LOST')
            return

        base_speed = max(0.0, float(self.raw_cmd.linear.x))
        steer = self.raw_steer_rad(base_speed)

        lidar_weight, camera_weight = self.fusion_weights()
        self.lidar_weight_pub.publish(
            Float32(data=float(lidar_weight))
        )
        self.camera_weight_pub.publish(
            Float32(data=float(camera_weight))
        )

        front = self.front_distance()
        self.front_clearance_pub.publish(
            Float32(
                data=(
                    float(front)
                    if math.isfinite(front)
                    else -1.0
                )
            )
        )
        left = self.side_clearance(True)
        right = self.side_clearance(False)
        rear = self.rear_distance()

        (
            center_correction,
            center_speed_limit,
            center_active,
        ) = self.center_alignment()

        vehicle = self.front_vehicle_candidate()
        if vehicle is None:
            vehicle_distance = float('inf')
            self.front_object_detected_pub.publish(Bool(data=False))
            self.front_object_pub.publish(Float32(data=-1.0))
        else:
            vehicle_distance = float(vehicle[0])
            self.front_object_detected_pub.publish(Bool(data=True))
            self.front_object_pub.publish(
                Float32(data=vehicle_distance)
            )

        (
            closing_speed,
            vehicle_ttc,
            lead_speed,
        ) = self.update_vehicle_motion(
            vehicle_distance,
            now,
        )

        self.closing_speed_pub.publish(
            Float32(data=float(closing_speed))
        )
        self.vehicle_ttc_pub.publish(
            Float32(
                data=(
                    float(vehicle_ttc)
                    if math.isfinite(vehicle_ttc)
                    else -1.0
                )
            )
        )
        self.lead_speed_pub.publish(
            Float32(data=float(lead_speed))
        )

        rear_vehicle = self.rear_vehicle_candidate()
        if rear_vehicle is None:
            rear_vehicle_distance = float('inf')
            rear_vehicle_lateral = 0.0
        else:
            rear_vehicle_distance = float(rear_vehicle[0])
            rear_vehicle_lateral = float(rear_vehicle[1])

        (
            rear_closing_speed,
            rear_vehicle_ttc,
        ) = self.update_rear_motion(
            rear_vehicle_distance,
            now,
        )

        self.rear_closing_speed_pub.publish(
            Float32(data=float(rear_closing_speed))
        )
        self.rear_vehicle_ttc_pub.publish(
            Float32(
                data=(
                    float(rear_vehicle_ttc)
                    if math.isfinite(rear_vehicle_ttc)
                    else -1.0
                )
            )
        )

        left_blindspot_distance = self.blindspot_distance(True)
        right_blindspot_distance = self.blindspot_distance(False)

        blindspot_limit = float(
            self.get_parameter('side_blindspot_m').value
        )
        left_blindspot = left_blindspot_distance < blindspot_limit
        right_blindspot = right_blindspot_distance < blindspot_limit

        self.left_blindspot_pub.publish(
            Bool(data=left_blindspot)
        )
        self.right_blindspot_pub.publish(
            Bool(data=right_blindspot)
        )

        rear_left_distance = self.rear_side_distance(True)
        rear_right_distance = self.rear_side_distance(False)

        self.defense_active_pub.publish(
            Bool(data=self.state == OvertakeState.DEFEND)
        )
        self.defense_side_pub.publish(
            Float32(data=float(self.defend_side))
        )

        left_corridor = self.pass_corridor_clearance(True)
        right_corridor = self.pass_corridor_clearance(False)

        emergency = float(
            self.get_parameter('emergency_stop_m').value
        )
        hard_ttc = float(
            self.get_parameter('hard_ttc_sec').value
        )

        if math.isfinite(front):
            ttc = front / max(base_speed, 0.05)
        else:
            ttc = float('inf')

        emergency_condition = (
            front < emergency
            or (
                base_speed > 0.5
                and ttc < hard_ttc
            )
        )

        if emergency_condition:
            self.front_emergency_count += 1
        else:
            self.front_emergency_count = 0

        emergency_frames = int(
            self.get_parameter(
                'front_emergency_confirm_frames'
            ).value
        )

        if self.front_emergency_count >= emergency_frames:
            self.publish_stop('EMERGENCY_STOP')
            return

        # If LiDAR path generation has dropped to zero but the camera still
        # sees a reliable road, continue at a deliberately low speed.
        fallback_enabled = bool(
            self.get_parameter('camera_fallback_enable').value
        )
        fallback_speed = float(
            self.get_parameter('camera_fallback_speed_mps').value
        )
        minimum_conf = float(
            self.get_parameter('camera_min_confidence').value
        )

        if (
            self.state == OvertakeState.CRUISE
            and fallback_enabled
            and base_speed <= 0.05
            and self.camera_confidence >= minimum_conf
        ):
            correction = self.camera_correction(
                0.0,
                float(
                    self.get_parameter(
                        'camera_fallback_gain_scale'
                    ).value
                ),
            )
            steer = 0.0 if correction is None else correction
            steer += center_correction
            steer = float(np.clip(
                steer,
                -self.max_steer_rad(),
                self.max_steer_rad(),
            ))
            safe_limit = self.safe_speed_for_distance(front)
            speed = min(
                fallback_speed,
                safe_limit,
                center_speed_limit,
            )
            self.safe_speed_pub.publish(
                Float32(data=float(safe_limit))
            )
            self.publish_command(
                speed, steer, 'CAMERA_FALLBACK'
            )
            return

        speed = base_speed
        mode = self.state.name

        if self.state == OvertakeState.CRUISE:
            # Outside a wall danger zone, ignore tiny steering noise and go
            # straight. Camera correction is only added when the base path
            # already indicates a real corner.
            if not center_active:
                straight_deadband = math.radians(
                    float(
                        self.get_parameter(
                            'straight_steer_deadband_deg'
                        ).value
                    )
                )
                if abs(steer) < straight_deadband:
                    steer = 0.0
                else:
                    correction = self.camera_correction()
                    if correction is not None:
                        steer += correction

            steer += center_correction
            steer = float(np.clip(
                steer,
                -self.max_steer_rad(),
                self.max_steer_rad(),
            ))

            if center_active:
                speed = min(speed, center_speed_limit)
                mode = 'WALL_AVOID'

            safe_limit = self.safe_speed_for_distance(front)
            self.safe_speed_pub.publish(
                Float32(data=float(safe_limit))
            )

            trigger = max(
                float(
                    self.get_parameter('overtake_trigger_m').value
                ),
                self.dynamic_stop_distance(base_speed)
                + float(
                    self.get_parameter(
                        'overtake_trigger_margin_m'
                    ).value
                ),
            )

            side_need = float(
                self.get_parameter('overtake_side_clear_m').value
            )
            max_base = math.radians(
                float(
                    self.get_parameter(
                        'overtake_max_base_steer_deg'
                    ).value
                )
            )

            corridor_need = max(
                side_need,
                float(
                    self.get_parameter(
                        'pass_corridor_min_m'
                    ).value
                ),
            )
            cooldown_ok = (
                now - self.last_overtake_end
                >= float(
                    self.get_parameter(
                        'overtake_cooldown_sec'
                    ).value
                )
            )

            ttc_trigger = float(
                self.get_parameter(
                    'overtake_ttc_trigger_sec'
                ).value
            )
            ttc_urgent = float(
                self.get_parameter(
                    'overtake_ttc_urgent_sec'
                ).value
            )

            approaching = (
                closing_speed > 0.10
                and vehicle_ttc < ttc_trigger
            )
            urgent = (
                closing_speed > 0.10
                and vehicle_ttc < ttc_urgent
            )

            rear_ttc_block = float(
                self.get_parameter('rear_ttc_block_sec').value
            )
            rear_min_distance = float(
                self.get_parameter(
                    'rear_vehicle_min_distance_m'
                ).value
            )
            rear_safe_for_lane_change = (
                not math.isfinite(rear_vehicle_distance)
                or (
                    rear_vehicle_distance > rear_min_distance
                    and (
                        not math.isfinite(rear_vehicle_ttc)
                        or rear_vehicle_ttc > rear_ttc_block
                    )
                )
            )

            left_lane_safe = (
                not left_blindspot
                and left_corridor > corridor_need
            )
            right_lane_safe = (
                not right_blindspot
                and right_corridor > corridor_need
            )

            defense_enabled = bool(
                self.get_parameter('defense_enable').value
            )
            defense_trigger = float(
                self.get_parameter(
                    'defense_ttc_trigger_sec'
                ).value
            )
            defense_closing = float(
                self.get_parameter(
                    'defense_min_closing_speed_mps'
                ).value
            )
            defense_front_clear = float(
                self.get_parameter(
                    'defense_front_clear_m'
                ).value
            )
            defense_max_steer = math.radians(
                float(
                    self.get_parameter(
                        'defense_max_base_steer_deg'
                    ).value
                )
            )
            defense_cooldown_ok = (
                now - self.last_defense_end
                >= float(
                    self.get_parameter(
                        'defense_cooldown_sec'
                    ).value
                )
            )

            side_margin = float(
                self.get_parameter(
                    'defense_side_margin_m'
                ).value
            )

            # Rear LiDAR faces backward: negative sensor-y corresponds
            # to the vehicle's left side, positive sensor-y to right.
            attack_left = rear_vehicle_lateral < -side_margin
            attack_right = rear_vehicle_lateral > side_margin

            defense_side_clear = float(
                self.get_parameter(
                    'defense_side_clear_m'
                ).value
            )

            left_defense_safe = (
                attack_left
                and not left_blindspot
                and left_corridor > defense_side_clear
            )
            right_defense_safe = (
                attack_right
                and not right_blindspot
                and right_corridor > defense_side_clear
            )

            front_safe_for_defense = (
                front > defense_front_clear
                and (
                    not math.isfinite(vehicle_distance)
                    or vehicle_distance > defense_front_clear
                )
            )

            defense_candidate = (
                defense_enabled
                and health['rear_ok']
                and health['imu_ok']
                and health['boundary_ok']
                and health['camera_ok']
                and self.camera_confidence
                >= float(
                    self.get_parameter(
                        'camera_min_confidence'
                    ).value
                )
                and rear_closing_speed >= defense_closing
                and math.isfinite(rear_vehicle_ttc)
                and rear_vehicle_ttc < defense_trigger
                and front_safe_for_defense
                and abs(steer) < defense_max_steer
                and defense_cooldown_ok
                and (left_defense_safe or right_defense_safe)
            )

            if defense_candidate:
                self.defend_side = (
                    1.0 if left_defense_safe else -1.0
                )
                self.set_state(OvertakeState.DEFEND)
                mode = 'DEFEND_ENTER'

            candidate = (
                math.isfinite(vehicle_distance)
                and (
                    approaching
                    or (
                        vehicle_distance < trigger
                        and closing_speed > 0.25
                    )
                )
                and (left_lane_safe or right_lane_safe)
                and rear_safe_for_lane_change
                and abs(steer) < max_base
                and base_speed > 0.2
                and cooldown_ok
                and overtake_allowed
            )

            if self.state == OvertakeState.DEFEND:
                self.detect_start = None

            elif candidate:
                if self.detect_start is None:
                    self.detect_start = now
                elif (
                    now - self.detect_start
                    >= float(
                        self.get_parameter(
                            'overtake_detect_hold_sec'
                        ).value
                    )
                ):
                    if left_lane_safe and right_lane_safe:
                        self.pass_side = (
                            1.0
                            if left_corridor >= right_corridor
                            else -1.0
                        )
                    elif left_lane_safe:
                        self.pass_side = 1.0
                    else:
                        self.pass_side = -1.0
                    self.detect_start = None
                    self.set_state(OvertakeState.SHIFT_OUT)
            else:
                self.detect_start = None

                follow_limit = self.follow_speed_limit(
                    vehicle_distance,
                    lead_speed,
                    vehicle_ttc,
                )

                speed = min(
                    speed,
                    safe_limit,
                    follow_limit,
                )

                if urgent and math.isfinite(vehicle_distance):
                    mode = f'{mode}:TTC_URGENT_FOLLOW'
                elif (
                    math.isfinite(vehicle_distance)
                    and follow_limit
                    < float(
                        self.get_parameter(
                            'max_speed_mps'
                        ).value
                    )
                ):
                    mode = f'{mode}:FOLLOW_LEAD'

        elif self.state == OvertakeState.DEFEND:
            elapsed = now - self.state_start

            speed = min(
                base_speed,
                float(
                    self.get_parameter(
                        'defense_speed_mps'
                    ).value
                ),
            )

            target = (
                self.defend_side
                * float(
                    self.get_parameter(
                        'defense_lane_target'
                    ).value
                )
            )

            correction = self.camera_correction(
                target,
                0.85,
            )
            correction_valid = correction is not None
            if correction_valid:
                steer += correction

            selected_blindspot = (
                left_blindspot
                if self.defend_side > 0.0
                else right_blindspot
            )
            selected_clear = (
                left_corridor
                if self.defend_side > 0.0
                else right_corridor
            )

            release_ttc = float(
                self.get_parameter(
                    'defense_ttc_release_sec'
                ).value
            )
            max_duration = float(
                self.get_parameter(
                    'defense_max_sec'
                ).value
            )
            side_need = float(
                self.get_parameter(
                    'defense_side_clear_m'
                ).value
            )
            front_need = float(
                self.get_parameter(
                    'defense_front_clear_m'
                ).value
            )

            release_defense = (
                not correction_valid
                or selected_blindspot
                or selected_clear < side_need
                or front < front_need
                or (
                    math.isfinite(vehicle_distance)
                    and vehicle_distance < front_need
                )
                or rear_closing_speed < 0.20
                or (
                    math.isfinite(rear_vehicle_ttc)
                    and rear_vehicle_ttc >= release_ttc
                )
                or elapsed >= max_duration
            )

            if release_defense:
                self.last_defense_end = now
                self.defend_side = 0.0
                self.set_state(OvertakeState.CRUISE)
                mode = 'DEFEND_RELEASE'
            else:
                mode = 'DEFEND_HOLD'

        elif self.state == OvertakeState.SHIFT_OUT:
            elapsed = now - self.state_start
            speed = min(
                float(
                    self.get_parameter('shift_speed_mps').value
                ),
                max(1.0, base_speed),
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter('pass_lane_target').value
                )
            )
            correction = self.camera_correction(target, 1.15)
            if correction is not None:
                steer += correction
            else:
                steer += (
                    self.pass_side
                    * math.radians(
                        float(
                            self.get_parameter(
                                'fallback_shift_bias_deg'
                            ).value
                        )
                    )
                )

            chosen_clear = (
                left_corridor
                if self.pass_side > 0.0
                else right_corridor
            )
            selected_blindspot = (
                left_blindspot
                if self.pass_side > 0.0
                else right_blindspot
            )
            if (
                selected_blindspot
                or chosen_clear
                < float(
                    self.get_parameter(
                        'side_abort_m'
                    ).value
                )
            ):
                self.set_state(OvertakeState.RETURN)
            elif elapsed >= float(
                self.get_parameter('shift_out_sec').value
            ):
                self.set_state(OvertakeState.PASS)

        elif self.state == OvertakeState.PASS:
            elapsed = now - self.state_start

            if not health['rear_ok']:
                speed = min(
                    float(
                        self.get_parameter(
                            'rear_lidar_fail_speed_mps'
                        ).value
                    ),
                    max(1.0, base_speed),
                )
                mode = 'REAR_LIDAR_FAIL_HOLD_PASS'
                self.publish_command(
                    speed,
                    steer,
                    mode,
                )
                return

            speed = float(
                self.get_parameter('overtake_speed_mps').value
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter('pass_lane_target').value
                )
            )
            correction = self.camera_correction(target, 1.0)
            if correction is not None:
                steer += correction

            chosen_clear = (
                left_corridor
                if self.pass_side > 0.0
                else right_corridor
            )
            selected_pass_blindspot = (
                left_blindspot
                if self.pass_side > 0.0
                else right_blindspot
            )
            if (
                selected_pass_blindspot
                or chosen_clear
                < float(
                    self.get_parameter(
                        'overtake_abort_side_m'
                    ).value
                )
            ):
                speed = min(speed, 2.0)

            front_clear = float(
                self.get_parameter('overtake_front_clear_m').value
            )
            pass_front = (
                vehicle_distance
                if math.isfinite(vehicle_distance)
                else front
            )
            if pass_front > front_clear:
                if self.front_clear_since is None:
                    self.front_clear_since = now
            else:
                self.front_clear_since = None

            stable_clear = (
                self.front_clear_since is not None
                and now - self.front_clear_since > 0.35
            )
            rear_ok = (
                rear
                > float(
                    self.get_parameter('rear_return_clear_m').value
                )
            )
            rear_ttc_return_ok = (
                not math.isfinite(rear_vehicle_ttc)
                or rear_vehicle_ttc
                > float(
                    self.get_parameter('rear_ttc_block_sec').value
                )
            )

            if (
                elapsed
                >= float(
                    self.get_parameter('pass_min_sec').value
                )
                and stable_clear
                and rear_ok
                and rear_ttc_return_ok
            ):
                self.set_state(OvertakeState.RETURN)

        elif self.state == OvertakeState.RETURN:
            if not health['rear_ok']:
                # Never merge back when rear perception is stale.
                self.set_state(OvertakeState.PASS)
                self.publish_command(
                    min(
                        float(
                            self.get_parameter(
                                'rear_lidar_fail_speed_mps'
                            ).value
                        ),
                        max(1.0, base_speed),
                    ),
                    steer,
                    'REAR_LIDAR_STALE_HOLD_PASS',
                )
                return

            rear_ok = (
                rear
                > float(
                    self.get_parameter('rear_return_clear_m').value
                )
            )
            rear_ttc_urgent = float(
                self.get_parameter('rear_ttc_urgent_sec').value
            )
            rear_threat = (
                math.isfinite(rear_vehicle_ttc)
                and rear_vehicle_ttc < rear_ttc_urgent
            )

            if not rear_ok or rear_threat:
                # Stay in the passing lane until the overtaken vehicle has
                # dropped safely behind.
                self.set_state(OvertakeState.PASS)
                self.publish_command(
                    min(
                        float(
                            self.get_parameter(
                                'return_speed_mps'
                            ).value
                        ),
                        max(1.0, base_speed),
                    ),
                    steer,
                    'WAIT_REAR_CLEAR',
                )
                return

            elapsed = now - self.state_start
            duration = max(
                0.1,
                float(self.get_parameter('return_sec').value),
            )
            progress = float(np.clip(
                elapsed / duration, 0.0, 1.0
            ))
            speed = min(
                float(
                    self.get_parameter('return_speed_mps').value
                ),
                max(1.0, base_speed),
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter('pass_lane_target').value
                )
                * (1.0 - progress)
            )
            correction = self.camera_correction(target, 1.15)
            if correction is not None:
                steer += correction
            else:
                steer -= (
                    self.pass_side
                    * math.radians(
                        float(
                            self.get_parameter(
                                'fallback_return_bias_deg'
                            ).value
                        )
                    )
                )

            if elapsed >= duration:
                self.last_overtake_end = now
                self.set_state(OvertakeState.CRUISE)

        if self.state == OvertakeState.CRUISE and center_active:
            speed = min(speed, center_speed_limit)

        if raw_degraded or scan_degraded:
            speed = min(
                speed,
                float(
                    self.get_parameter(
                        'sensor_soft_speed_mps'
                    ).value
                ),
            )
            degraded = []
            if raw_degraded:
                degraded.append('RAW')
            if scan_degraded:
                degraded.append('SCAN')
            mode = f"{mode}:SOFT_TIMEOUT_{'+'.join(degraded)}"

        lateral_limit_fresh = (
            self.lateral_limit_stamp > 0.0
            and now - self.lateral_limit_stamp
            <= float(
                self.get_parameter(
                    'lateral_limit_timeout_sec'
                ).value
            )
        )
        if lateral_limit_fresh and speed > self.lateral_speed_limit:
            speed = self.lateral_speed_limit
            mode = f'{mode}:LATERAL_LIMIT'

        health_speed_limit = self.fail_safe_speed_limit(
            health
        )
        if speed > health_speed_limit:
            speed = health_speed_limit
            mode = f'{mode}:FAILSAFE_LIMIT'

        degraded_confidence = float(
            self.get_parameter(
                'sensor_degraded_confidence'
            ).value
        )
        if (
            self.boundary_confidence < degraded_confidence
            and self.camera_confidence < degraded_confidence
        ):
            speed = min(
                speed,
                float(
                    self.get_parameter(
                        'sensor_degraded_speed_mps'
                    ).value
                ),
            )
            mode = f'{mode}:SENSOR_DEGRADED'

        self.publish_command(speed, steer, mode)


def main(args=None):
    rclpy.init(args=args)
    node = OvertakeFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
