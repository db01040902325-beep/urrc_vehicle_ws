import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


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

        self.wheelbase = 0.330

        # 트랙 폭
        self.track_width = 1.50

        # 차량 자체 최대 조향 35도지만
        # 레이싱 제어에서는 28도로 제한
        self.max_steer_deg = 28.0

        self.lookahead = 1.70

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

        target = self.find_center(left, right)

        if target is None:
            self.stop()

            self.get_logger().warn(
                'Track boundary not detected',
                throttle_duration_sec=1.0
            )
            return

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

        # 사용자 지정 조향 부호
        # Gazebo + = 좌회전
        # 사용자 + = 우회전
        steer_user_deg = -delta_left_deg

        steer_user_deg = max(
            -self.max_steer_deg,
            min(self.max_steer_deg, steer_user_deg)
        )

        # 급회전 시 감속
        abs_steer = abs(steer_user_deg)

        if abs_steer > 22.0:
            speed = 0.80

        elif abs_steer > 12.0:
            speed = 1.10

        else:
            speed = 1.80

        # 정면 충돌 안전 정지
        front_min = self.front_distance(msg)

        if front_min < 0.50:
            speed = 0.0

        elif front_min < 0.90:
            speed = min(speed, 0.80)

        # 사용자 조향 -> Gazebo 좌표계
        delta_gazebo = math.radians(-steer_user_deg)

        if abs(delta_gazebo) < 0.001 or speed == 0.0:
            yaw_rate = 0.0
        else:
            yaw_rate = (
                speed *
                math.tan(delta_gazebo) /
                self.wheelbase
            )

        cmd = Twist()

        cmd.linear.x = speed
        cmd.angular.z = yaw_rate

        self.cmd_pub.publish(cmd)

        steer_msg = Float32()
        steer_msg.data = float(steer_user_deg)

        self.steer_pub.publish(steer_msg)

    def extract_clusters(self, msg):

        clusters = []
        current = []
        previous = None

        for i, r in enumerate(msg.ranges):

            if not math.isfinite(r):
                if current:
                    self.add_cluster(current, clusters)
                    current = []

                previous = None
                continue

            if r < msg.range_min or r > min(msg.range_max, 5.0):
                if current:
                    self.add_cluster(current, clusters)
                    current = []

                previous = None
                continue

            angle = msg.angle_min + i * msg.angle_increment

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # 차량 앞쪽만 사용
            if x <= 0.10:
                if current:
                    self.add_cluster(current, clusters)
                    current = []

                previous = None
                continue

            point = (x, y)

            if previous is None:
                current = [point]

            else:
                dx = point[0] - previous[0]
                dy = point[1] - previous[1]

                distance = math.hypot(dx, dy)

                # 같은 물체로 판단
                if distance < 0.12:
                    current.append(point)

                else:
                    if current:
                        self.add_cluster(
                            current,
                            clusters
                        )

                    current = [point]

            previous = point

        if current:
            self.add_cluster(current, clusters)

        return clusters

    def add_cluster(self, points, output):

        if len(points) < 2:
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        width = max(
            max(xs) - min(xs),
            max(ys) - min(ys)
        )

        # 라바콘보다 지나치게 큰 물체 제외
        if width > 0.22:
            return

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        output.append(
            (cx, cy, len(points))
        )

    def find_center(self, left, right):

        best = None
        best_cost = float('inf')

        # 좌우 라바콘 쌍 찾기
        for lx, ly in left:
            for rx, ry in right:

                width = ly - ry

                if width < 0.60 or width > 2.20:
                    continue

                center_x = (lx + rx) * 0.5
                center_y = (ly + ry) * 0.5

                longitudinal_difference = abs(lx - rx)

                cost = (
                    2.0 * abs(
                        width - self.track_width
                    )
                    + longitudinal_difference
                    + 0.4 * abs(
                        center_x - self.lookahead
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

        # 한쪽 라바콘만 보일 경우
        if left:

            lx, ly = min(
                left,
                key=lambda p:
                abs(p[0] - self.lookahead)
            )

            return (
                lx,
                ly - self.track_width / 2.0
            )

        if right:

            rx, ry = min(
                right,
                key=lambda p:
                abs(p[0] - self.lookahead)
            )

            return (
                rx,
                ry + self.track_width / 2.0
            )

        return None

    def front_distance(self, msg):

        minimum = float('inf')

        for i, r in enumerate(msg.ranges):

            if not math.isfinite(r):
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            # 정면 ±10도
            if abs(angle) <= math.radians(10.0):
                minimum = min(
                    minimum,
                    r
                )

        return minimum


def main(args=None):

    rclpy.init(args=args)

    node = LidarRacer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
