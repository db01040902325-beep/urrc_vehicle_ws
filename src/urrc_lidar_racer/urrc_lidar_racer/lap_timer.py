import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32


class LapTimer(Node):

    def __init__(self):
        super().__init__('lap_timer')

        # 시작점 = odom 기준 (0, 0)
        self.finish_radius = 1.20
        self.arm_radius = 3.00

        # 출발 방향과 비슷한 방향으로 들어올 때만 랩 인정
        self.heading_tolerance_deg = 40.0

        # 시작 직후 오검출 방지
        self.min_lap_time = 10.0

        self.current_lap = 1
        self.best_lap = float('inf')
        self.last_lap = 0.0

        self.lap_start_time = None
        self.armed = False

        self.create_subscription(
            Odometry,
            '/model/urrc_car/odometry',
            self.odom_cb,
            10
        )

        self.current_lap_pub = self.create_publisher(
            Int32,
            '/race/lap/current',
            10
        )

        self.current_time_pub = self.create_publisher(
            Float32,
            '/race/lap/current_time_sec',
            10
        )

        self.last_time_pub = self.create_publisher(
            Float32,
            '/race/lap/last_time_sec',
            10
        )

        self.best_time_pub = self.create_publisher(
            Float32,
            '/race/lap/best_time_sec',
            10
        )

        log_dir = os.path.expanduser(
            '~/urrc_vehicle_ws/race_logs'
        )
        os.makedirs(log_dir, exist_ok=True)

        stamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        self.csv_path = os.path.join(
            log_dir,
            f'laps_{stamp}.csv'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.writer = csv.writer(
            self.csv_file
        )

        self.writer.writerow([
            'lap',
            'lap_time_sec',
            'best_lap_sec'
        ])

        self.get_logger().info(
            'Lap timer started'
        )

        self.get_logger().info(
            'Start/finish = odom (0,0), '
            f'finish radius={self.finish_radius:.2f} m'
        )

    def stamp_to_sec(self, stamp):
        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    def quaternion_to_yaw(self, q):

        siny = (
            2.0 *
            (
                q.w * q.z
                + q.x * q.y
            )
        )

        cosy = (
            1.0 -
            2.0 *
            (
                q.y * q.y
                + q.z * q.z
            )
        )

        return math.atan2(
            siny,
            cosy
        )

    def normalize_angle(self, angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def odom_cb(self, msg):

        now = self.stamp_to_sec(
            msg.header.stamp
        )

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        distance_from_start = math.hypot(
            x,
            y
        )

        yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        # odom 초기 방향 = 0 rad
        heading_error = abs(
            math.degrees(
                self.normalize_angle(yaw)
            )
        )

        # 첫 odom 수신 시점부터 Lap 1 시작
        if self.lap_start_time is None:
            self.lap_start_time = now

        elapsed = max(
            0.0,
            now - self.lap_start_time
        )

        # 실시간 현재 랩
        lap_msg = Int32()
        lap_msg.data = self.current_lap
        self.current_lap_pub.publish(lap_msg)

        # 현재 랩 경과시간
        time_msg = Float32()
        time_msg.data = float(elapsed)
        self.current_time_pub.publish(time_msg)

        # 출발점에서 충분히 멀어지면 랩 판정 ARM
        if (
            not self.armed
            and distance_from_start >= self.arm_radius
        ):
            self.armed = True

            self.get_logger().info(
                f'Lap {self.current_lap} armed'
            )

        # 출발점으로 복귀
        if (
            self.armed
            and distance_from_start <= self.finish_radius
            and heading_error <= self.heading_tolerance_deg
            and elapsed >= self.min_lap_time
        ):
            self.complete_lap(
                now,
                elapsed
            )

    def complete_lap(
        self,
        now,
        lap_time
    ):

        self.last_lap = lap_time

        if lap_time < self.best_lap:
            self.best_lap = lap_time

        last_msg = Float32()
        last_msg.data = float(
            self.last_lap
        )
        self.last_time_pub.publish(
            last_msg
        )

        best_msg = Float32()
        best_msg.data = float(
            self.best_lap
        )
        self.best_time_pub.publish(
            best_msg
        )

        self.writer.writerow([
            self.current_lap,
            round(lap_time, 3),
            round(self.best_lap, 3)
        ])

        self.csv_file.flush()

        self.get_logger().info(
            f'LAP {self.current_lap} = '
            f'{lap_time:.3f} sec | '
            f'BEST = {self.best_lap:.3f} sec'
        )

        self.current_lap += 1

        self.lap_start_time = now

        # 다음 랩을 위해 다시 출발점에서
        # 3m 이상 벗어나야 함
        self.armed = False

    def destroy_node(self):

        if hasattr(self, 'csv_file'):
            self.csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = LapTimer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
