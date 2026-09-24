import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Int32


class RaceStats(Node):

    def __init__(self):
        super().__init__('race_stats')

        self.start_time = time.monotonic()

        self.current_speed = 0.0
        self.max_speed = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0

        self.max_abs_steer = 0.0

        self.track_loss_count = 0
        self.was_track_lost = False

        self.create_subscription(
            Float32,
            '/race/speed_mps',
            self.speed_cb,
            10
        )

        self.create_subscription(
            Float32,
            '/race/steering_deg',
            self.steer_cb,
            10
        )

        # 통계 출력 토픽
        self.avg_speed_pub = self.create_publisher(
            Float32,
            '/race/stats/avg_speed_mps',
            10
        )

        self.max_speed_pub = self.create_publisher(
            Float32,
            '/race/stats/max_speed_mps',
            10
        )

        self.max_steer_pub = self.create_publisher(
            Float32,
            '/race/stats/max_steering_deg',
            10
        )

        self.runtime_pub = self.create_publisher(
            Float32,
            '/race/stats/runtime_sec',
            10
        )

        self.loss_pub = self.create_publisher(
            Int32,
            '/race/stats/track_loss_count',
            10
        )

        # 10 Hz 기록
        self.timer = self.create_timer(
            0.1,
            self.timer_cb
        )

        # 로그 폴더
        log_dir = os.path.expanduser(
            '~/urrc_vehicle_ws/race_logs'
        )
        os.makedirs(log_dir, exist_ok=True)

        stamp = datetime.now().strftime(
            '%Y%m%d_%H%M%S'
        )

        self.csv_path = os.path.join(
            log_dir,
            f'race_{stamp}.csv'
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
            'runtime_sec',
            'speed_mps',
            'avg_speed_mps',
            'max_speed_mps',
            'steering_deg',
            'max_abs_steering_deg',
            'track_loss_count'
        ])

        self.last_steer = 0.0

        self.get_logger().info(
            f'Race stats logging: {self.csv_path}'
        )

    def speed_cb(self, msg):

        speed = float(msg.data)

        self.current_speed = speed

        if speed > self.max_speed:
            self.max_speed = speed

        self.speed_sum += speed
        self.speed_count += 1

    def steer_cb(self, msg):

        steer = float(msg.data)

        self.last_steer = steer

        abs_steer = abs(steer)

        if abs_steer > self.max_abs_steer:
            self.max_abs_steer = abs_steer

    def timer_cb(self):

        runtime = (
            time.monotonic()
            - self.start_time
        )

        if self.speed_count > 0:
            avg_speed = (
                self.speed_sum
                / self.speed_count
            )
        else:
            avg_speed = 0.0

        msg = Float32()

        msg.data = float(avg_speed)
        self.avg_speed_pub.publish(msg)

        msg = Float32()
        msg.data = float(self.max_speed)
        self.max_speed_pub.publish(msg)

        msg = Float32()
        msg.data = float(self.max_abs_steer)
        self.max_steer_pub.publish(msg)

        msg = Float32()
        msg.data = float(runtime)
        self.runtime_pub.publish(msg)

        loss_msg = Int32()
        loss_msg.data = self.track_loss_count
        self.loss_pub.publish(loss_msg)

        self.writer.writerow([
            round(runtime, 3),
            round(self.current_speed, 3),
            round(avg_speed, 3),
            round(self.max_speed, 3),
            round(self.last_steer, 3),
            round(self.max_abs_steer, 3),
            self.track_loss_count
        ])

        self.csv_file.flush()

    def destroy_node(self):

        if hasattr(self, 'csv_file'):
            self.csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = RaceStats()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
