import math

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, Float32

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from .steering_guard import SteeringReversalGuard


class LidarRacer(Node):

    def __init__(self):
        super().__init__('lidar_racer')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.declare_parameter(
            'cmd_topic',
            '/race/lidar_cmd_vel',
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('cmd_topic').value),
            10
        )

        # 사용자 기준:
        # + = 우회전
        # - = 좌회전
        self.steer_pub = self.create_publisher(
            Float32,
            '/race/steering_deg',
            10
        )


        # 속도 m/s
        self.speed_pub = self.create_publisher(
            Float32,
            '/race/speed_mps',
            10
        )

        # 전방 코너 예측 각도
        self.preview_pub = self.create_publisher(
            Float32,
            '/race/preview_angle_deg',
            10
        )

        self.curve_prediction_pub = self.create_publisher(
            Float32,
            '/race/curve_prediction_deg',
            10
        )

        self.curve_speed_pub = self.create_publisher(
            Float32,
            '/race/curve_speed_limit_mps',
            10
        )

        self.lateral_speed_pub = self.create_publisher(
            Float32,
            '/race/lateral_speed_limit_mps',
            10
        )

        self.boundary_mode_pub = self.create_publisher(
            Float32,
            '/race/continuous_boundary_confidence',
            10
        )
        # Signal path validity separately from a numeric zero-speed command.
        self.path_valid_pub = self.create_publisher(
            Bool, '/race/path_valid', 10
        )
        # RViz Marker
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/race/markers',
            10
        )

        # =========================
        # ROS parameters
        # =========================

        defaults = {
            'wheelbase': 0.330,
            'track_width': 2.40,
            'max_steer_deg': 30.0,

            'steer_ema_alpha': 0.35,
            'max_steer_step_deg': 3.0,
            'steer_reversal_deadband_deg': 3.0,
            'steer_reversal_confirm_frames': 3,

            # Steering uses road-direction change, never lateral centering.
            'direction_near_x_m': 2.5,
            'direction_far_x_m': 6.0,
            'direction_steer_gain': 1.10,
            'direction_deadband_deg': 1.50,
            'direction_max_steer_deg': 24.0,

            'speed_straight': 5.00,
            'speed_curve_1': 3.00,
            'speed_curve_2': 2.50,
            'speed_curve_3': 2.00,
            'speed_sharp': 1.50,

            'speed_threshold_1_deg': 2.0,
            'speed_threshold_2_deg': 5.0,
            'speed_threshold_3_deg': 10.0,
            'speed_threshold_4_deg': 18.0,

            'lateral_accel_limit_mps2': 6.0,
            'lateral_stability_deadband_deg': 0.0,
            'high_speed_steer_soft_limit_deg': 10.0,
            'high_speed_steer_threshold_mps': 12.0,

            'preview_speed_4deg': 3.00,
            'preview_speed_7deg': 2.50,
            'preview_speed_12deg': 2.00,
            'preview_speed_18deg': 1.50,

            # Long-range curve prediction for current high-speed operation.
            'curve_prediction_enable': True,
            'curve_near_distance': 4.0,
            'curve_mid_distance': 8.0,
            'curve_far_distance': 12.0,
            'curve_soft_deg': 4.0,
            'curve_medium_deg': 8.0,
            'curve_hard_deg': 14.0,
            'curve_speed_soft': 3.00,
            'curve_speed_medium': 2.50,
            'curve_speed_hard': 2.00,
            'curve_min_confidence': 0.25,

            # Hairpin / sharp-corner detector.
            # Detect a broad wall ahead and turn toward the side with more
            # free space before the normal x-binned boundary tracker fails.
            'hairpin_enable': True,
            'hairpin_front_sector_deg': 14.0,
            'hairpin_front_trigger_m': 10.0,
            'hairpin_front_release_m': 12.0,
            'hairpin_front_hit_ratio': 0.35,
            'hairpin_side_inner_deg': 20.0,
            'hairpin_side_outer_deg': 72.0,
            'hairpin_open_margin_m': 0.50,
            'hairpin_min_open_m': 2.00,
            'hairpin_steer_min_deg': 10.0,
            'hairpin_steer_max_deg': 26.0,
            'hairpin_speed_mps': 1.50,
            'hairpin_confirm_frames': 2,
            'hairpin_release_frames': 4,
            'hairpin_min_speed_mps': 1.0,
            'hairpin_open_ratio': 1.25,
            'hairpin_direction_agreement_deg': 2.0,

            'lidar_max_use_range': 36.0,
            'cluster_gap': 0.18,
            'cluster_max_width': 0.22,

            'detection_min_x': 0.15,
            'detection_max_x': 32.0,
            'detection_max_abs_y': 1.50,
            'side_deadband_y': 0.15,

            # Continuous wall / guardrail tracking for the new Monza map.
            'continuous_boundary_enable': True,
            'boundary_bin_size_m': 0.65,
            'boundary_min_points_per_bin': 2,
            'boundary_min_abs_y_m': 0.25,
            'boundary_max_abs_y_m': 3.50,
            'boundary_min_x_m': 0.35,
            'boundary_max_x_m': 32.0,
            'boundary_pair_min_width_m': 1.80,
            'boundary_pair_max_width_m': 3.00,
            'boundary_near_window_m': 2.00,

            'loss_hold_frames': 2,
            'loss_slow_frames': 5,
            'loss_hold_speed': 1.50,
            'loss_slow_speed': 0.80,
            'loss_recovery_speed': 0.00,
            'loss_recovery_front_clear_m': 1.25,
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        for name in defaults:
            setattr(
                self,
                name,
                self.get_parameter(name).value
            )

        self.wheelbase = float(self.wheelbase)
        self.track_width = float(self.track_width)
        self.max_steer_deg = float(self.max_steer_deg)

        # Steering smoothing
        self.filtered_steer_deg = 0.0
        self.last_speed = 0.0
        self.reversal_guard = SteeringReversalGuard(
            self.steer_reversal_deadband_deg,
            self.steer_reversal_confirm_frames,
        )

        # EMA: 클수록 새 조향값을 빠르게 반영
        self.steer_ema_alpha = float(self.steer_ema_alpha)

        # 한 LiDAR frame당 최대 조향 변화량
        self.max_steer_step_deg = float(self.max_steer_step_deg)

        # 순간적인 LiDAR 경계 손실 복구
        self.miss_count = 0

        self.hairpin_active = False
        self.hairpin_direction = 0
        self.hairpin_confirm_count = 0
        self.hairpin_release_count = 0

        self.hairpin_pub = self.create_publisher(
            Float32,
            '/race/hairpin_direction',
            10
        )

        self.frame_id = 'urrc_car/lidar_link/race_lidar'

        # 런타임 파라미터 변경
        self.add_on_set_parameters_callback(
            self.parameter_callback
        )

        self.get_logger().info(
            'LiDAR direction follower started: straight=0deg, RIGHT=+, LEFT=-'
        )

    def parameter_callback(self, params):

        for param in params:

            if not hasattr(self, param.name):
                continue

            value = param.value

            # 주요 안전 조건
            if param.name == 'speed_straight':
                if value <= 0.0 or value > 15.0:
                    return SetParametersResult(
                        successful=False,
                        reason='speed_straight must be 0~15 m/s'
                    )

            if param.name == 'max_steer_deg':
                if value <= 0.0 or value > 35.0:
                    return SetParametersResult(
                        successful=False,
                        reason='max_steer_deg must be 0~35 deg'
                    )

            if param.name == 'track_width':
                if value < 0.5 or value > 3.0:
                    return SetParametersResult(
                        successful=False,
                        reason='track_width out of range'
                    )

            setattr(
                self,
                param.name,
                value
            )

            self.get_logger().info(
                f'Parameter changed: '
                f'{param.name} = {value}'
            )

        return SetParametersResult(
            successful=True
        )


    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def scan_callback(self, msg):

        continuous_left = []
        continuous_right = []
        boundary_confidence = 0.0

        if bool(self.continuous_boundary_enable):
            (
                continuous_left,
                continuous_right,
                boundary_confidence,
            ) = self.extract_continuous_boundaries(msg)

        # Prefer the new continuous wall / guardrail tracker when it has
        # sufficient evidence. Keep the original compact-cluster detector as
        # a fallback for cone-style practice layouts.
        if (
            len(continuous_left) + len(continuous_right) >= 3
            and boundary_confidence >= 0.20
        ):
            left = continuous_left
            right = continuous_right
        else:
            clusters = self.extract_clusters(msg)
            left = []
            right = []

            for x, y, size in clusters:
                if x < self.detection_min_x or x > self.detection_max_x:
                    continue

                if abs(y) > self.detection_max_abs_y:
                    continue

                if y > self.side_deadband_y:
                    left.append((x, y))
                elif y < -self.side_deadband_y:
                    right.append((x, y))

        self.boundary_mode_pub.publish(
            Float32(data=float(boundary_confidence))
        )

        (
            hairpin_steer_deg,
            hairpin_speed_limit,
        ) = self.detect_hairpin(msg)

        self.hairpin_pub.publish(
            Float32(data=float(self.hairpin_direction))
        )

        # ==========================================
        # Direction-only path steering
        #
        # The vehicle is NOT pulled toward the road center. Two valid
        # left/right wall pairs are sampled at different forward distances
        # and only the change in road direction is used for steering.
        # Therefore an off-center grid start still commands 0 deg on a
        # straight.
        # ==========================================

        direction_result = self.directional_curve_steer(
            left,
            right,
        )

        if direction_result is None:
            target = None
            path_heading_deg = 0.0
            steer_user_deg = 0.0
        else:
            (
                steer_user_deg,
                path_heading_deg,
                target,
            ) = direction_result

        preview_angle_deg = float(path_heading_deg)
        self.preview_pub.publish(
            Float32(data=preview_angle_deg)
        )

        (
            curve_prediction_deg,
            curve_speed_limit,
        ) = self.predict_long_curve(
            left,
            right,
            boundary_confidence,
        )

        self.curve_prediction_pub.publish(
            Float32(data=float(curve_prediction_deg))
        )
        self.curve_speed_pub.publish(
            Float32(data=float(curve_speed_limit))
        )

        self.publish_markers(
            left,
            right,
            target,
        )

        self.path_valid_pub.publish(Bool(data=direction_result is not None))

        if direction_result is None:
            if self.hairpin_active and hairpin_steer_deg is not None:
                target = (3.0, 0.0)
                steer_user_deg = float(hairpin_steer_deg)
            else:
                self.handle_track_loss(msg)
                return
        elif self.hairpin_active and hairpin_steer_deg is not None:
            steer_user_deg = float(hairpin_steer_deg)

        self.miss_count = 0

        # Reject isolated opposite-sign sensor noise on ordinary corners.
        # A true sustained direction change is accepted after 3 scans.
        # Hairpin detection overrides this guard for prompt safe turning.
        if self.hairpin_active:
            self.reversal_guard.reset()
        else:
            self.reversal_guard.deadband_deg = float(
                self.steer_reversal_deadband_deg
            )
            self.reversal_guard.confirm_frames = max(
                1, int(self.steer_reversal_confirm_frames)
            )
            steer_user_deg = self.reversal_guard.filter(steer_user_deg)

        steer_user_deg = max(
            -self.max_steer_deg,
            min(self.max_steer_deg, steer_user_deg)
        )

        # ==========================================
        # Steering EMA
        # ==========================================

        ema_steer = (
            self.steer_ema_alpha * steer_user_deg
            + (1.0 - self.steer_ema_alpha)
            * self.filtered_steer_deg
        )

        # ==========================================
        # Steering Rate Limit
        # ==========================================

        change = ema_steer - self.filtered_steer_deg

        change = max(
            -self.max_steer_step_deg,
            min(self.max_steer_step_deg, change)
        )

        steer_user_deg = (
            self.filtered_steer_deg + change
        )

        self.filtered_steer_deg = steer_user_deg

        abs_steer = abs(steer_user_deg)

        # 현재 속도 설정
        # ==========================================
        # Curvature-based speed control
        # ==========================================

        if abs_steer > self.speed_threshold_4_deg:
            speed = self.speed_sharp

        elif abs_steer > self.speed_threshold_3_deg:
            speed = self.speed_curve_3

        elif abs_steer > self.speed_threshold_2_deg:
            speed = self.speed_curve_2

        elif abs_steer > self.speed_threshold_1_deg:
            speed = self.speed_curve_1

        else:
            speed = self.speed_straight

        if self.hairpin_active:
            speed = min(
                speed,
                float(hairpin_speed_limit),
            )

        # ==========================================
        # Preview pre-braking
        # 현재 조향이 작아도 앞쪽에 코너가 보이면 미리 감속
        # ==========================================

        preview_abs = abs(preview_angle_deg)

        if preview_abs > 18.0:
            speed = min(speed, self.preview_speed_18deg)

        elif preview_abs > 12.0:
            speed = min(speed, self.preview_speed_12deg)

        elif preview_abs > 7.0:
            speed = min(speed, self.preview_speed_7deg)

        elif preview_abs > 4.0:
            speed = min(speed, self.preview_speed_4deg)

        # Long-range predicted curvature braking.
        # This acts before the vehicle reaches the actual steering point.
        if bool(self.curve_prediction_enable):
            speed = min(
                speed,
                float(curve_speed_limit),
            )

        # ==========================================
        # High-speed stability
        # Limit speed from steering curvature so lateral acceleration
        # remains below the configured threshold.
        # a_lat = v^2 * curvature, curvature = tan(delta) / wheelbase
        # ==========================================
        stability_deadband = float(
            self.lateral_stability_deadband_deg
        )
        stability_steer_deg = max(
            0.0,
            abs(steer_user_deg) - stability_deadband,
        )
        delta_for_stability = math.radians(
            stability_steer_deg
        )
        curvature = abs(
            math.tan(delta_for_stability)
            / max(self.wheelbase, 1e-3)
        )

        lateral_accel_limit = float(
            self.lateral_accel_limit_mps2
        )

        if curvature > 1e-5:
            lateral_speed_limit = math.sqrt(
                max(0.0, lateral_accel_limit / curvature)
            )
        else:
            lateral_speed_limit = self.speed_straight

        # Above 10 m/s, avoid large instantaneous steering commands.
        if speed >= self.high_speed_steer_threshold_mps:
            soft_limit = float(
                self.high_speed_steer_soft_limit_deg
            )
            if abs(steer_user_deg) > soft_limit:
                speed = min(
                    speed,
                    max(
                        self.speed_sharp,
                        lateral_speed_limit
                    )
                )

        speed = min(speed, lateral_speed_limit)

        lateral_msg = Float32()
        lateral_msg.data = float(lateral_speed_limit)
        self.lateral_speed_pub.publish(lateral_msg)

        # Front collision / TTC safety is intentionally owned only by
        # overtake_fusion. The base path controller must not independently
        # stop and accidentally trigger fallback logic.

        # 사용자 조향 -> Gazebo 좌표계
        delta_gazebo = math.radians(
            -steer_user_deg
        )

        if abs(delta_gazebo) < 0.001 or speed == 0.0:
            yaw_rate = 0.0

        else:
            yaw_rate = (
                speed
                * math.tan(delta_gazebo)
                / self.wheelbase
            )

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = yaw_rate

        self.cmd_pub.publish(cmd)

        speed_msg = Float32()
        speed_msg.data = float(speed)
        self.speed_pub.publish(speed_msg)

        self.last_speed = speed

        steer_msg = Float32()
        steer_msg.data = float(
            steer_user_deg
        )

        self.steer_pub.publish(
            steer_msg
        )

    # =====================================================
    # Hairpin / sharp-corner detection
    # =====================================================

    def sector_scan_stats(self, msg, deg_min, deg_max):
        total = 0
        hits = []

        for i, distance in enumerate(msg.ranges):
            angle_deg = math.degrees(
                msg.angle_min + i * msg.angle_increment
            )
            if angle_deg < deg_min or angle_deg > deg_max:
                continue

            total += 1
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min:
                continue
            if distance > min(
                msg.range_max,
                float(self.lidar_max_use_range),
            ):
                continue
            hits.append(float(distance))

        if total <= 0:
            return float('inf'), 0.0

        if not hits:
            return float('inf'), 0.0

        return (
            self.percentile(hits, 50.0),
            len(hits) / float(total),
        )

    def sector_near_ratio(
        self,
        msg,
        deg_min,
        deg_max,
        maximum_distance,
    ):
        total = 0
        near = 0

        for i, distance in enumerate(msg.ranges):
            angle_deg = math.degrees(
                msg.angle_min + i * msg.angle_increment
            )
            if angle_deg < deg_min or angle_deg > deg_max:
                continue

            total += 1
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min:
                continue
            if distance <= maximum_distance:
                near += 1

        if total <= 0:
            return 0.0

        return near / float(total)

    def detect_hairpin(self, msg):
        if not bool(self.hairpin_enable):
            self.hairpin_active = False
            self.hairpin_direction = 0
            return None, float(self.speed_straight)

        if self.last_speed < float(self.hairpin_min_speed_mps):
            self.hairpin_active = False
            self.hairpin_direction = 0
            self.hairpin_confirm_count = 0
            self.hairpin_release_count = 0
            return None, float(self.speed_straight)

        front_half = float(self.hairpin_front_sector_deg)
        front_distance, _ = self.sector_scan_stats(
            msg,
            -front_half,
            front_half,
        )
        front_near_ratio = self.sector_near_ratio(
            msg,
            -front_half,
            front_half,
            float(self.hairpin_front_trigger_m),
        )

        inner = float(self.hairpin_side_inner_deg)
        outer = float(self.hairpin_side_outer_deg)

        left_distance, _ = self.sector_scan_stats(
            msg,
            inner,
            outer,
        )
        right_distance, _ = self.sector_scan_stats(
            msg,
            -outer,
            -inner,
        )

        # No return in a side sector means that side is very open.
        open_cap = min(
            float(self.lidar_max_use_range),
            float(self.hairpin_front_trigger_m) * 2.0,
        )
        left_open = (
            open_cap
            if not math.isfinite(left_distance)
            else min(left_distance, open_cap)
        )
        right_open = (
            open_cap
            if not math.isfinite(right_distance)
            else min(right_distance, open_cap)
        )

        trigger = float(self.hairpin_front_trigger_m)
        release = float(self.hairpin_front_release_m)
        hit_ratio_required = float(self.hairpin_front_hit_ratio)
        margin = float(self.hairpin_open_margin_m)
        minimum_open = float(self.hairpin_min_open_m)

        candidate_direction = 0
        broad_wall = (
            math.isfinite(front_distance)
            and front_distance <= trigger
            and front_near_ratio >= hit_ratio_required
        )

        open_ratio = max(
            1.0,
            float(self.hairpin_open_ratio),
        )

        if broad_wall:
            if (
                right_open >= minimum_open
                and right_open - left_open >= margin
                and right_open >= left_open * open_ratio
            ):
                candidate_direction = 1
            elif (
                left_open >= minimum_open
                and left_open - right_open >= margin
                and left_open >= right_open * open_ratio
            ):
                candidate_direction = -1

        agreement_deg = float(
            self.hairpin_direction_agreement_deg
        )
        if (
            candidate_direction != 0
            and abs(self.filtered_steer_deg) >= agreement_deg
        ):
            current_direction = (
                1 if self.filtered_steer_deg > 0.0 else -1
            )
            if current_direction != candidate_direction:
                candidate_direction = 0

        if not self.hairpin_active:
            if candidate_direction != 0:
                if candidate_direction == self.hairpin_direction:
                    self.hairpin_confirm_count += 1
                else:
                    self.hairpin_direction = candidate_direction
                    self.hairpin_confirm_count = 1
            else:
                self.hairpin_direction = 0
                self.hairpin_confirm_count = 0

            if self.hairpin_confirm_count >= int(
                self.hairpin_confirm_frames
            ):
                self.hairpin_active = True
                self.hairpin_release_count = 0
        else:
            if (
                math.isfinite(front_distance)
                and front_distance < release
            ):
                self.hairpin_release_count = 0
            else:
                self.hairpin_release_count += 1

            if self.hairpin_release_count >= int(
                self.hairpin_release_frames
            ):
                self.hairpin_active = False
                self.hairpin_direction = 0
                self.hairpin_confirm_count = 0
                self.hairpin_release_count = 0

        if not self.hairpin_active or self.hairpin_direction == 0:
            return None, float(self.speed_straight)

        opening_delta = abs(right_open - left_open)
        severity = max(
            0.0,
            min(
                1.0,
                opening_delta / max(trigger, 0.1),
            ),
        )

        steer_min = float(self.hairpin_steer_min_deg)
        steer_max = float(self.hairpin_steer_max_deg)
        steer = steer_min + severity * (steer_max - steer_min)
        steer *= float(self.hairpin_direction)

        return (
            float(steer),
            float(self.hairpin_speed_mps),
        )

    # =====================================================
    # Temporary track-loss recovery
    # =====================================================

    def handle_track_loss(self, msg):
        """No blind straight-line motion when both track boundaries are unverified."""
        self.miss_count += 1
        self.path_valid_pub.publish(Bool(data=False))
        self.reversal_guard.reset()
        self.filtered_steer_deg = 0.0
        self.last_speed = 0.0

        self.cmd_pub.publish(Twist())
        self.speed_pub.publish(Float32(data=0.0))
        self.steer_pub.publish(Float32(data=0.0))

        self.get_logger().warn(
            f'Track boundary lost: frame={self.miss_count}; commanding STOP',
            throttle_duration_sec=0.5,
        )


    # =====================================================
    # RViz markers
    # =====================================================

    def publish_markers(
        self,
        left,
        right,
        target
    ):

        msg = MarkerArray()

        # 이전 marker 제거
        delete = Marker()
        delete.action = Marker.DELETEALL

        msg.markers.append(delete)

        # -------------------------
        # LEFT cones
        # -------------------------

        left_marker = Marker()

        left_marker.header.frame_id = self.frame_id
        left_marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        left_marker.ns = 'left_cones'
        left_marker.id = 0

        left_marker.type = Marker.SPHERE_LIST
        left_marker.action = Marker.ADD

        left_marker.pose.orientation.w = 1.0

        left_marker.scale.x = 0.12
        left_marker.scale.y = 0.12
        left_marker.scale.z = 0.12

        # 파란색
        left_marker.color.r = 0.0
        left_marker.color.g = 0.3
        left_marker.color.b = 1.0
        left_marker.color.a = 1.0

        for x, y in left:

            p = Point()

            p.x = float(x)
            p.y = float(y)
            p.z = 0.0

            left_marker.points.append(p)

        msg.markers.append(
            left_marker
        )

        # -------------------------
        # RIGHT cones
        # -------------------------

        right_marker = Marker()

        right_marker.header.frame_id = self.frame_id
        right_marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        right_marker.ns = 'right_cones'
        right_marker.id = 1

        right_marker.type = Marker.SPHERE_LIST
        right_marker.action = Marker.ADD

        right_marker.pose.orientation.w = 1.0

        right_marker.scale.x = 0.12
        right_marker.scale.y = 0.12
        right_marker.scale.z = 0.12

        # 빨간색
        right_marker.color.r = 1.0
        right_marker.color.g = 0.1
        right_marker.color.b = 0.1
        right_marker.color.a = 1.0

        for x, y in right:

            p = Point()

            p.x = float(x)
            p.y = float(y)
            p.z = 0.0

            right_marker.points.append(p)

        msg.markers.append(
            right_marker
        )

        # -------------------------
        # TARGET
        # -------------------------

        if target is not None:

            tx, ty = target

            target_marker = Marker()

            target_marker.header.frame_id = self.frame_id
            target_marker.header.stamp = (
                self.get_clock().now().to_msg()
            )

            target_marker.ns = 'target'
            target_marker.id = 2

            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD

            target_marker.pose.position.x = float(tx)
            target_marker.pose.position.y = float(ty)
            target_marker.pose.position.z = 0.0

            target_marker.pose.orientation.w = 1.0

            target_marker.scale.x = 0.18
            target_marker.scale.y = 0.18
            target_marker.scale.z = 0.18

            # 초록색
            target_marker.color.r = 0.0
            target_marker.color.g = 1.0
            target_marker.color.b = 0.1
            target_marker.color.a = 1.0

            msg.markers.append(
                target_marker
            )

            # -------------------------
            # 차량 -> 목표점 방향선
            # -------------------------

            line = Marker()

            line.header.frame_id = self.frame_id
            line.header.stamp = (
                self.get_clock().now().to_msg()
            )

            line.ns = 'target_line'
            line.id = 3

            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            line.pose.orientation.w = 1.0

            line.scale.x = 0.04

            line.color.r = 0.0
            line.color.g = 1.0
            line.color.b = 0.1
            line.color.a = 1.0

            origin = Point()
            origin.x = 0.0
            origin.y = 0.0
            origin.z = 0.0

            goal = Point()
            goal.x = float(tx)
            goal.y = float(ty)
            goal.z = 0.0

            line.points = [
                origin,
                goal
            ]

            msg.markers.append(
                line
            )

        self.marker_pub.publish(
            msg
        )

    # =====================================================
    # Long-range curve prediction
    # =====================================================

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def predict_long_curve(
        self,
        left,
        right,
        boundary_confidence,
    ):
        if not bool(self.curve_prediction_enable):
            return 0.0, float(self.speed_straight)

        if boundary_confidence < float(self.curve_min_confidence):
            return 0.0, float(self.speed_straight)

        distances = [
            float(self.curve_near_distance),
            float(self.curve_mid_distance),
            float(self.curve_far_distance),
        ]

        points = []
        previous_x = -1.0

        for distance in distances:
            point = self.center_from_boundaries(
                left,
                right,
                distance,
            )
            if point is None:
                continue

            # A requested farther preview point must actually be farther
            # forward. Reusing a nearer fallback point creates a fake
            # ~180-degree turn when headings are differenced.
            if point[0] <= previous_x + 0.50:
                continue

            if abs(point[0] - distance) > max(
                3.0,
                0.35 * distance,
            ):
                continue

            points.append(point)
            previous_x = point[0]

        if len(points) < 3:
            return 0.0, float(self.speed_straight)

        headings = []

        for index in range(1, len(points)):
            dx = points[index][0] - points[index - 1][0]
            dy = points[index][1] - points[index - 1][1]
            if dx <= 0.20 or math.hypot(dx, dy) < 0.50:
                continue
            headings.append(math.atan2(dy, dx))

        if len(headings) < 2:
            return 0.0, float(self.speed_straight)

        candidates = []

        for index in range(1, len(headings)):
            candidates.append(
                self.normalize_angle(
                    headings[index] - headings[index - 1]
                )
            )

        dominant = max(
            candidates,
            key=lambda value: abs(value),
        )
        prediction_deg = math.degrees(dominant)
        abs_prediction = abs(prediction_deg)

        if abs_prediction >= float(self.curve_hard_deg):
            speed_limit = float(self.curve_speed_hard)
        elif abs_prediction >= float(self.curve_medium_deg):
            speed_limit = float(self.curve_speed_medium)
        elif abs_prediction >= float(self.curve_soft_deg):
            speed_limit = float(self.curve_speed_soft)
        else:
            speed_limit = float(self.speed_straight)

        return prediction_deg, speed_limit

    # =====================================================
    # Continuous wall / guardrail tracking
    # =====================================================

    def extract_continuous_boundaries(self, msg):
        """Extract left/right continuous track boundaries from raw LiDAR.

        The new Monza uses long collision meshes for the inner and outer
        walls. Points are grouped by forward-distance bins. For each bin,
        an outer-side percentile is used so compact cars and small objects
        near the lane center do not replace the wall estimate.
        """
        bin_size = max(0.25, float(self.boundary_bin_size_m))
        min_points = max(1, int(self.boundary_min_points_per_bin))
        min_abs_y = float(self.boundary_min_abs_y_m)
        max_abs_y = float(self.boundary_max_abs_y_m)
        min_x = float(self.boundary_min_x_m)
        max_x = min(
            float(self.boundary_max_x_m),
            float(self.lidar_max_use_range),
        )

        left_bins = {}
        right_bins = {}
        valid_points = 0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > min(msg.range_max, self.lidar_max_use_range):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if x < min_x or x > max_x:
                continue
            if abs(y) < min_abs_y or abs(y) > max_abs_y:
                continue

            key = int(x / bin_size)
            valid_points += 1

            if y > 0.0:
                left_bins.setdefault(key, []).append((x, y))
            else:
                right_bins.setdefault(key, []).append((x, y))

        def summarize(bins, left_side):
            output = []
            for key in sorted(bins):
                points = bins[key]
                if len(points) < min_points:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                # On the left choose a high lateral percentile; on the right
                # choose a low percentile. This follows the outer wall rather
                # than a car located closer to the lane center.
                percentile = 72.0 if left_side else 28.0
                boundary_y = self.percentile(ys, percentile)
                boundary_x = self.percentile(xs, 50.0)

                output.append((
                    float(boundary_x),
                    float(boundary_y),
                ))
            return output

        left = summarize(left_bins, True)
        right = summarize(right_bins, False)

        occupied_bins = len(left) + len(right)
        expected_bins = max(
            1,
            int((max_x - min_x) / bin_size),
        )
        confidence = min(
            1.0,
            occupied_bins / max(4.0, 0.65 * expected_bins),
        )

        if valid_points < 8:
            confidence *= valid_points / 8.0

        return left, right, confidence

    @staticmethod
    def percentile(values, percentile):
        if not values:
            return 0.0

        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]

        rank = (
            max(0.0, min(100.0, float(percentile)))
            / 100.0
            * (len(ordered) - 1)
        )
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return ordered[lo]

        t = rank - lo
        return ordered[lo] * (1.0 - t) + ordered[hi] * t

    def boundary_near_distance(self, points, target_x):
        if not points:
            return None

        window = max(0.3, float(self.boundary_near_window_m))
        candidates = [
            p for p in points
            if abs(p[0] - target_x) <= window
        ]

        # Do not extrapolate a nonexistent wall segment to a requested
        # preview distance: that generated phantom headings near Monaco walls.
        if not candidates:
            return None

        weights = [
            1.0 / (0.20 + abs(p[0] - target_x))
            for p in candidates
        ]
        total = sum(weights)
        if total <= 0.0:
            return candidates[0]

        x = sum(
            p[0] * w
            for p, w in zip(candidates, weights)
        ) / total
        y = sum(
            p[1] * w
            for p, w in zip(candidates, weights)
        ) / total

        return float(x), float(y)

    def center_from_boundaries(
        self,
        left,
        right,
        target_x,
    ):
        """Return road center only when both walls form a plausible pair.

        Single-wall extrapolation was removed because it can create a phantom
        center roughly half a track width toward a wall and cause an abrupt
        steering command immediately after the start.
        """
        lp = self.boundary_near_distance(left, target_x)
        rp = self.boundary_near_distance(right, target_x)

        if lp is None or rp is None:
            return None

        width = lp[1] - rp[1]
        min_width = float(self.boundary_pair_min_width_m)
        max_width = float(self.boundary_pair_max_width_m)

        if not min_width <= width <= max_width:
            return None

        return (
            0.5 * (lp[0] + rp[0]),
            0.5 * (lp[1] + rp[1]),
        )

    def directional_curve_steer(self, left, right):
        """Steer from road direction change, not from lateral road-center error."""
        near = self.center_from_boundaries(
            left,
            right,
            float(self.direction_near_x_m),
        )
        far = self.center_from_boundaries(
            left,
            right,
            float(self.direction_far_x_m),
        )

        if near is None or far is None:
            return None

        dx = far[0] - near[0]
        dy = far[1] - near[1]

        if dx <= 0.50:
            return None

        path_heading_deg = math.degrees(
            math.atan2(dy, dx)
        )

        deadband = float(self.direction_deadband_deg)
        if abs(path_heading_deg) <= deadband:
            steer = 0.0
        else:
            # LiDAR +y is vehicle-left while user steering convention is
            # RIGHT=+, LEFT=-.
            steer = (
                -float(self.direction_steer_gain)
                * path_heading_deg
            )

        max_direction_steer = min(
            float(self.max_steer_deg),
            float(self.direction_max_steer_deg),
        )
        steer = max(
            -max_direction_steer,
            min(max_direction_steer, steer),
        )

        return (
            float(steer),
            float(path_heading_deg),
            far,
        )

    # =====================================================
    # LiDAR clustering
    # =====================================================

    def extract_clusters(self, msg):

        clusters = []

        current = []
        previous = None

        for i, r in enumerate(
            msg.ranges
        ):

            if not math.isfinite(r):

                if current:
                    self.add_cluster(
                        current,
                        clusters
                    )

                    current = []

                previous = None
                continue

            if (
                r < msg.range_min
                or r > min(
                    msg.range_max,
                    self.lidar_max_use_range
                )
            ):

                if current:
                    self.add_cluster(
                        current,
                        clusters
                    )

                    current = []

                previous = None
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # 앞쪽만 사용
            if x <= 0.10:

                if current:
                    self.add_cluster(
                        current,
                        clusters
                    )

                    current = []

                previous = None
                continue

            point = (
                x,
                y
            )

            if previous is None:

                current = [
                    point
                ]

            else:

                dx = (
                    point[0]
                    - previous[0]
                )

                dy = (
                    point[1]
                    - previous[1]
                )

                distance = math.hypot(
                    dx,
                    dy
                )

                if distance < self.cluster_gap:

                    current.append(
                        point
                    )

                else:

                    if current:
                        self.add_cluster(
                            current,
                            clusters
                        )

                    current = [
                        point
                    ]

            previous = point

        if current:
            self.add_cluster(
                current,
                clusters
            )

        return clusters

    def add_cluster(
        self,
        points,
        output
    ):

        if len(points) < 2:
            return

        xs = [
            p[0]
            for p in points
        ]

        ys = [
            p[1]
            for p in points
        ]

        width = max(
            max(xs) - min(xs),
            max(ys) - min(ys)
        )

        # 라바콘보다 지나치게 큰 물체 제외
        if width > self.cluster_max_width:
            return

        cx = (
            sum(xs)
            / len(xs)
        )

        cy = (
            sum(ys)
            / len(ys)
        )

        output.append(
            (
                cx,
                cy,
                len(points)
            )
        )

    # =====================================================
    # Track center
    # =====================================================

    # =====================================================
    # Front safety
    # =====================================================

    def front_distance(
        self,
        msg
    ):

        minimum = float('inf')

        for i, r in enumerate(
            msg.ranges
        ):

            if not math.isfinite(r):
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            if abs(angle) <= math.radians(
                10.0
            ):

                minimum = min(
                    minimum,
                    r
                )

        return minimum


def main(args=None):

    rclpy.init(
        args=args
    )

    node = LidarRacer()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        node.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
