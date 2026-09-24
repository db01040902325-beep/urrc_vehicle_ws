import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge


class BirdseyeView(Node):

    def __init__(self):
        super().__init__('birdseye_view')

        self.bridge = CvBridge()

        self.width = 800
        self.height = 800

        # 화면에 표시할 전방 거리
        self.max_forward = 6.0

        # 좌우 표시 범위
        self.max_side = 3.0

        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/race/birdseye_image',
            10
        )

        self.get_logger().info(
            'Birdseye view -> /race/birdseye_image'
        )

    def world_to_pixel(self, x, y):

        # 차량은 화면 아래 중앙
        px = int(
            self.width / 2
            - y / self.max_side
            * (self.width / 2)
        )

        py = int(
            self.height
            - x / self.max_forward
            * self.height
        )

        return px, py

    def scan_callback(self, msg):

        image = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # -----------------------------
        # 거리 Grid
        # -----------------------------
        for meter in range(1, 7):

            y = int(
                self.height
                - meter / self.max_forward
                * self.height
            )

            if 0 <= y < self.height:
                cv2.line(
                    image,
                    (0, y),
                    (self.width, y),
                    (50, 50, 50),
                    1
                )

                cv2.putText(
                    image,
                    f'{meter}m',
                    (10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (150, 150, 150),
                    1
                )

        # 중앙선
        cv2.line(
            image,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (70, 70, 70),
            1
        )

        # -----------------------------
        # LiDAR points
        # -----------------------------
        for i, r in enumerate(msg.ranges):

            if not math.isfinite(r):
                continue

            if r < msg.range_min or r > min(
                msg.range_max,
                self.max_forward
            ):
                continue

            angle = (
                msg.angle_min
                + i * msg.angle_increment
            )

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if x < 0.0:
                continue

            px, py = self.world_to_pixel(
                x,
                y
            )

            if (
                0 <= px < self.width
                and 0 <= py < self.height
            ):
                cv2.circle(
                    image,
                    (px, py),
                    4,
                    (0, 255, 255),
                    -1
                )

        # -----------------------------
        # 차량 표시
        # -----------------------------
        vehicle_x = self.width // 2
        vehicle_y = self.height - 25

        cv2.rectangle(
            image,
            (vehicle_x - 18, vehicle_y - 30),
            (vehicle_x + 18, vehicle_y + 10),
            (255, 255, 255),
            2
        )

        cv2.arrowedLine(
            image,
            (vehicle_x, vehicle_y - 10),
            (vehicle_x, vehicle_y - 65),
            (0, 255, 0),
            3,
            tipLength=0.25
        )

        cv2.putText(
            image,
            'URRC LiDAR Bird Eye View',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        ros_image = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        ros_image.header = msg.header

        self.pub.publish(
            ros_image
        )


def main(args=None):

    rclpy.init(args=args)

    node = BirdseyeView()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
