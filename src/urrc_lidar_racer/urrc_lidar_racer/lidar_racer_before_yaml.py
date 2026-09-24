import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class LidarRacer(Node):

    def __init__(self):
        super().__init__('lidar_racer')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/model/urrc_car/cmd_vel',
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

        # RViz Marker
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/race/markers',
            10
        )

        self.wheelbase = 0.330
        self.track_width = 1.50

        self.max_steer_deg = 28.0

        # Adaptive lookahead
        self.lookahead = 1.70

        # Steering smoothing
        self.filtered_steer_deg = 0.0
        self.last_speed = 0.0

        # EMA: 클수록 새 조향값을 빠르게 반영
        self.steer_ema_alpha = 0.45

        # 한 LiDAR frame당 최대 조향 변화량
        self.max_steer_step_deg = 7.0

        # 순간적인 LiDAR 경계 손실 복구
        self.miss_count = 0

        self.frame_id = 'urrc_car/lidar_link/race_lidar'

        self.get_logger().info(
            'LiDAR racer started: RIGHT=+, LEFT=-'
        )

    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def scan_callback(self, msg):

        clusters = self.extract_clusters(msg)

        left = []
        right = []

        for x, y, size in clusters:

            if x < 0.15 or x > 4.0:
                continue

            if abs(y) > 1.50:
                continue

            if y > 0.15:
                left.append((x, y))

            elif y < -0.15:
                right.append((x, y))

        # ==========================================
        # Adaptive Lookahead
        # 직선에서는 멀리 보고,
        # 급코너에서는 가까이 봄
        # ==========================================

        previous_steer = abs(self.filtered_steer_deg)

        if previous_steer < 8.0:
            self.lookahead = 2.00

        elif previous_steer < 18.0:
            self.lookahead = 1.55

        else:
            self.lookahead = 1.15

        target = self.find_center(left, right)

        # ==========================================
        # 2~4m 전방 코너 예측
        # ==========================================
        preview_target = self.find_preview_target(
            left,
            right
        )

        if preview_target is not None:
            preview_x, preview_y = preview_target

            preview_angle_deg = math.degrees(
                math.atan2(
                    preview_y,
                    preview_x
                )
            )
        else:
            preview_angle_deg = 0.0

        preview_msg = Float32()
        preview_msg.data = float(preview_angle_deg)
        self.preview_pub.publish(preview_msg)

        # RViz 표시
        self.publish_markers(
            left,
            right,
            target
        )

        if target is None:
            self.handle_track_loss()
            return

        # 정상 검출로 복귀
        self.miss_count = 0

        target_x, target_y = target

        # Pure Pursuit
        ld2 = target_x * target_x + target_y * target_y

        if ld2 < 0.05:
            self.stop()
            return

        delta_left = math.atan2(
            2.0 * self.wheelbase * target_y,
            ld2
        )

        delta_left_deg = math.degrees(delta_left)

        # 사용자 기준:
        # + = 우회전
        # - = 좌회전
        steer_user_deg = -delta_left_deg

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

        if abs_steer > 22.0:
            speed = 0.90

        elif abs_steer > 15.0:
            speed = 1.15

        elif abs_steer > 8.0:
            speed = 1.45

        elif abs_steer > 4.0:
            speed = 1.85

        else:
            speed = 2.30

        # ==========================================
        # Preview pre-braking
        # 현재 조향이 작아도 앞쪽에 코너가 보이면 미리 감속
        # ==========================================

        preview_abs = abs(preview_angle_deg)

        if preview_abs > 18.0:
            speed = min(speed, 0.95)

        elif preview_abs > 12.0:
            speed = min(speed, 1.15)

        elif preview_abs > 7.0:
            speed = min(speed, 1.45)

        elif preview_abs > 4.0:
            speed = min(speed, 1.85)

        front_min = self.front_distance(msg)

        # ==========================================
        # TTC + 거리 기반 안전제어
        # ==========================================

        if math.isfinite(front_min):
            ttc = front_min / max(speed, 0.05)
        else:
            ttc = float('inf')

        # 즉시 정지
        if front_min < 0.50 or ttc < 0.35:
            speed = 0.0

        # 위험 접근
        elif front_min < 0.90 or ttc < 0.65:
            speed = min(speed, 0.60)

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
    # Temporary track-loss recovery
    # =====================================================

    def handle_track_loss(self):

        self.miss_count += 1

        # 1~2 frame:
        # 직전 조향을 잠깐 유지하면서 감속
        if self.miss_count <= 2:
            speed = min(self.last_speed, 0.50)
            steer = self.filtered_steer_deg

        # 3~5 frame:
        # 매우 저속으로 직전 방향 유지
        elif self.miss_count <= 5:
            speed = 0.30
            steer = self.filtered_steer_deg

        # 장시간 검출 실패:
        # 완전 정지
        else:
            speed = 0.0
            steer = 0.0
            self.filtered_steer_deg = 0.0

        delta_gazebo = math.radians(-steer)

        if speed <= 0.0:
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
        steer_msg.data = float(steer)
        self.steer_pub.publish(steer_msg)

        self.last_speed = speed

        self.get_logger().warn(
            f'Track boundary lost: frame={self.miss_count}',
            throttle_duration_sec=0.5
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
                    5.0
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

                if distance < 0.12:

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
        if width > 0.22:
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

    def find_preview_target(
        self,
        left,
        right
    ):
        # 약 3m 앞의 트랙 중앙점을 찾음
        preview_distance = 3.0
        candidates = []

        # 좌우 라바콘 쌍
        for lx, ly in left:
            for rx, ry in right:

                width = ly - ry

                if width < 0.60 or width > 2.20:
                    continue

                # 서로 너무 앞뒤로 떨어진 라바콘은
                # 같은 횡단면의 쌍으로 사용하지 않음
                if abs(lx - rx) > 0.80:
                    continue

                cx = (lx + rx) * 0.5
                cy = (ly + ry) * 0.5

                if 1.20 <= cx <= 4.00:
                    candidates.append(
                        (cx, cy)
                    )

        if candidates:
            return min(
                candidates,
                key=lambda p:
                abs(p[0] - preview_distance)
            )

        # 왼쪽 경계만 보이는 경우
        far_left = [
            p for p in left
            if 1.20 <= p[0] <= 4.00
        ]

        if far_left:
            lx, ly = min(
                far_left,
                key=lambda p:
                abs(p[0] - preview_distance)
            )

            return (
                lx,
                ly - self.track_width / 2.0
            )

        # 오른쪽 경계만 보이는 경우
        far_right = [
            p for p in right
            if 1.20 <= p[0] <= 4.00
        ]

        if far_right:
            rx, ry = min(
                far_right,
                key=lambda p:
                abs(p[0] - preview_distance)
            )

            return (
                rx,
                ry + self.track_width / 2.0
            )

        return None


    def find_center(
        self,
        left,
        right
    ):

        best = None
        best_cost = float('inf')

        for lx, ly in left:

            for rx, ry in right:

                width = (
                    ly - ry
                )

                if (
                    width < 0.60
                    or width > 2.20
                ):
                    continue

                center_x = (
                    lx + rx
                ) * 0.5

                center_y = (
                    ly + ry
                ) * 0.5

                longitudinal_difference = abs(
                    lx - rx
                )

                cost = (
                    2.0
                    * abs(
                        width
                        - self.track_width
                    )
                    + longitudinal_difference
                    + 0.4
                    * abs(
                        center_x
                        - self.lookahead
                    )
                )

                if cost < best_cost:

                    best_cost = cost

                    best = (
                        center_x,
                        center_y
                    )

        if best is not None:
            return best

        # 한쪽만 보이는 경우

        if left:

            lx, ly = min(
                left,
                key=lambda p:
                abs(
                    p[0]
                    - self.lookahead
                )
            )

            return (
                lx,
                ly
                - self.track_width
                / 2.0
            )

        if right:

            rx, ry = min(
                right,
                key=lambda p:
                abs(
                    p[0]
                    - self.lookahead
                )
            )

            return (
                rx,
                ry
                + self.track_width
                / 2.0
            )

        return None

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
