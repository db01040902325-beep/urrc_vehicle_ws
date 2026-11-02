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
            'imu_timeout_sec': 0.15,
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

            'camera_gain_deg': 5.5,
            'camera_min_confidence': 0.35,
            'camera_fallback_enable': True,

            # Confidence-based LiDAR + camera steering fusion.
            'boundary_confidence_topic': '/race/continuous_boundary_confidence',
            'fusion_lidar_min_confidence': 0.25,
            'fusion_camera_min_weight': 0.20,
            'sensor_degraded_confidence': 0.20,
            'sensor_degraded_speed_mps': 4.0,
            'camera_fallback_speed_mps': 3.00,
            'camera_fallback_gain_scale': 1.35,

            'max_steer_deg': 28.0,
            'max_speed_mps': 15.00,
            'raw_cmd_timeout_sec': 0.35,
            'scan_timeout_sec': 0.25,

            'emergency_stop_m': 1.00,
            'hard_ttc_sec': 0.30,
            'reaction_time_sec': 0.25,
            'brake_decel_mps2': 10.0,
            'safety_margin_m': 0.50,

            'overtake_trigger_m': 17.50,
            'overtake_trigger_margin_m': 2.00,
            'overtake_front_clear_m': 12.00,
            'overtake_side_clear_m': 0.82,
            'overtake_abort_side_m': 0.55,
            'rear_return_clear_m': 2.50,

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
            'final_steer_rate_deg_s': 85.0,

            'overtake_max_base_steer_deg': 8.0,
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
        self.boundary_confidence = 0.0
        self.start_allowed = not bool(
            self.get_parameter('start_gate_enable').value
        )

        self.state = OvertakeState.CRUISE
        self.state_start = self.now_sec()
        self.detect_start = None
        self.pass_side = 1.0
        self.front_clear_since = None
        self.last_overtake_end = -1e9

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

        self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            'Fusion controller: wall tracking support + IMU stabilization + overtaking'
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

    def on_camera_confidence(self, msg):
        self.camera_confidence = float(np.clip(msg.data, 0.0, 1.0))

    def on_boundary_confidence(self, msg):
        self.boundary_confidence = float(
            np.clip(msg.data, 0.0, 1.0)
        )

    def on_start_allowed(self, msg):
        self.start_allowed = bool(msg.data)

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

    def front_distance(self):
        values = self.sector_values(self.scan, -9.0, 9.0)
        if values.size == 0:
            return float('inf')
        return float(np.percentile(values, 12))

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

    def fusion_weights(self):
        lidar_weight = float(
            np.clip(self.boundary_confidence, 0.0, 1.0)
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
        camera_weight = float(
            np.clip(
                self.camera_confidence * camera_scale,
                0.0,
                1.0,
            )
        )

        return lidar_weight, camera_weight

    def camera_correction(self, target_offset=0.0, scale=1.0):
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
        self.output_speed = 0.0
        self.cmd_pub.publish(Twist())
        self.state_pub.publish(String(data=self.state.name))
        self.mode_pub.publish(String(data=mode))
        self.safe_speed_pub.publish(Float32(data=0.0))
        self.stability_factor_pub.publish(
            Float32(data=0.0)
        )

    def publish_command(self, speed, steer, mode):
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

    def control_loop(self):
        now = self.now_sec()

        raw_timeout = float(
            self.get_parameter('raw_cmd_timeout_sec').value
        )
        if (
            self.raw_cmd_stamp <= 0.0
            or now - self.raw_cmd_stamp > raw_timeout
        ):
            self.publish_stop('RAW_CMD_TIMEOUT')
            return

        scan_timeout = float(
            self.get_parameter('scan_timeout_sec').value
        )
        if (
            self.scan_stamp <= 0.0
            or now - self.scan_stamp > scan_timeout
        ):
            self.publish_stop('FRONT_LIDAR_TIMEOUT')
            return

        if (
            bool(self.get_parameter('start_gate_enable').value)
            and not self.start_allowed
        ):
            self.publish_stop('WAIT_START')
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
        left = self.side_clearance(True)
        right = self.side_clearance(False)
        rear = self.rear_distance()

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

        if front < emergency or (
            base_speed > 0.5 and ttc < hard_ttc
        ):
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
            and front > self.dynamic_stop_distance(fallback_speed)
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
            safe_limit = self.safe_speed_for_distance(front)
            speed = min(fallback_speed, safe_limit)
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
            correction = self.camera_correction()
            if correction is not None:
                steer += correction

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

            candidate = (
                math.isfinite(vehicle_distance)
                and vehicle_distance < trigger
                and max(left_corridor, right_corridor) > corridor_need
                and abs(steer) < max_base
                and base_speed > 0.2
                and cooldown_ok
            )

            if candidate:
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
                    self.pass_side = (
                        1.0
                        if left_corridor >= right_corridor
                        else -1.0
                    )
                    self.detect_start = None
                    self.set_state(OvertakeState.SHIFT_OUT)
            else:
                self.detect_start = None
                speed = min(speed, safe_limit)

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
            if (
                chosen_clear
                < float(
                    self.get_parameter(
                        'overtake_abort_side_m'
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
            if (
                chosen_clear
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

            if (
                elapsed
                >= float(
                    self.get_parameter('pass_min_sec').value
                )
                and stable_clear
                and rear_ok
            ):
                self.set_state(OvertakeState.RETURN)

        elif self.state == OvertakeState.RETURN:
            rear_ok = (
                rear
                > float(
                    self.get_parameter('rear_return_clear_m').value
                )
            )
            if not rear_ok:
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
