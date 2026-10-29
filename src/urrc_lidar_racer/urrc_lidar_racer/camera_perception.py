#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


class CameraPerception(Node):
    """Front camera perception for road-center correction and start lights."""

    def __init__(self):
        super().__init__('camera_perception')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('road_confidence_min_area_ratio', 0.08)
        self.declare_parameter('start_gate_enable', True)
        self.declare_parameter('red_on_ratio', 0.0015)
        self.declare_parameter('red_off_ratio', 0.00045)
        self.declare_parameter('red_seen_frames', 3)
        self.declare_parameter('red_off_frames', 4)

        self.bridge = CvBridge()
        self.red_seen_count = 0
        self.red_off_count = 0
        self.red_was_seen = False
        self.start_allowed = not bool(
            self.get_parameter('start_gate_enable').value
        )

        image_topic = str(self.get_parameter('image_topic').value)
        self.create_subscription(Image, image_topic, self.on_image, 10)

        self.offset_pub = self.create_publisher(
            Float32, '/camera/road_center_offset', 10
        )
        self.conf_pub = self.create_publisher(
            Float32, '/camera/road_confidence', 10
        )
        self.start_pub = self.create_publisher(
            Bool, '/camera/start_allowed', 10
        )
        self.red_pub = self.create_publisher(
            Float32, '/camera/red_ratio', 10
        )

        self.get_logger().info(
            f'Camera perception started: {image_topic}'
        )

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as exc:
            self.get_logger().warning(
                f'cv_bridge conversion failed: {exc}'
            )
            return

        if frame.shape[0] < 80 or frame.shape[1] < 80:
            return

        offset, confidence = self.estimate_road_center(frame)
        self.offset_pub.publish(Float32(data=float(offset)))
        self.conf_pub.publish(Float32(data=float(confidence)))

        red_ratio = self.estimate_red_ratio(frame)
        self.red_pub.publish(Float32(data=float(red_ratio)))
        self.update_start_gate(red_ratio)
        self.start_pub.publish(
            Bool(data=bool(self.start_allowed))
        )

    def estimate_road_center(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Monza asphalt: low saturation / medium brightness.
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 28], dtype=np.uint8),
            np.array([179, 105, 235], dtype=np.uint8),
        )

        roi = np.zeros_like(mask)
        polygon = np.array([[
            (int(0.05 * w), h - 1),
            (int(0.95 * w), h - 1),
            (int(0.70 * w), int(0.45 * h)),
            (int(0.30 * w), int(0.45 * h)),
        ]], dtype=np.int32)
        cv2.fillPoly(roi, polygon, 255)
        mask = cv2.bitwise_and(mask, roi)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, 8
        )
        if count <= 1:
            return 0.0, 0.0

        center_x = 0.5 * w
        bottom_y = int(0.88 * h)
        candidates = []

        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 0.015 * w * h:
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            ww = int(stats[idx, cv2.CC_STAT_WIDTH])
            hh = int(stats[idx, cv2.CC_STAT_HEIGHT])

            score = float(area)
            if y <= bottom_y <= y + hh:
                score += 0.25 * w * h
            if x <= center_x <= x + ww:
                score += 0.20 * w * h
            candidates.append((score, idx, area))

        if not candidates:
            return 0.0, 0.0

        _, idx, area = max(candidates)
        ys, xs = np.nonzero(labels == idx)
        if len(xs) == 0:
            return 0.0, 0.0

        weights = 0.35 + 0.65 * (
            ys.astype(np.float32) / max(1.0, float(h - 1))
        )
        cx = float(
            np.average(xs.astype(np.float32), weights=weights)
        )

        offset = (cx - center_x) / max(1.0, center_x)
        offset = float(np.clip(offset, -1.0, 1.0))

        area_ratio = area / float(w * h)
        min_ratio = float(
            self.get_parameter(
                'road_confidence_min_area_ratio'
            ).value
        )
        confidence = float(np.clip(
            (area_ratio - min_ratio)
            / max(0.01, 0.35 - min_ratio),
            0.0,
            1.0,
        ))

        return offset, confidence

    @staticmethod
    def estimate_red_ratio(frame):
        h, w = frame.shape[:2]

        # Start gantry should appear in the upper-middle image.
        roi = frame[
            0:int(0.52 * h),
            int(0.12 * w):int(0.88 * w),
        ]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_1 = cv2.inRange(
            hsv,
            np.array([0, 120, 120], np.uint8),
            np.array([11, 255, 255], np.uint8),
        )
        red_2 = cv2.inRange(
            hsv,
            np.array([169, 120, 120], np.uint8),
            np.array([179, 255, 255], np.uint8),
        )
        red = cv2.bitwise_or(red_1, red_2)
        red = cv2.morphologyEx(
            red, cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        return (
            float(np.count_nonzero(red))
            / float(red.size)
        )

    def update_start_gate(self, red_ratio):
        enabled = bool(
            self.get_parameter('start_gate_enable').value
        )
        if not enabled:
            self.start_allowed = True
            return

        on_ratio = float(
            self.get_parameter('red_on_ratio').value
        )
        off_ratio = float(
            self.get_parameter('red_off_ratio').value
        )
        seen_need = int(
            self.get_parameter('red_seen_frames').value
        )
        off_need = int(
            self.get_parameter('red_off_frames').value
        )

        if red_ratio >= on_ratio:
            self.red_seen_count += 1
            self.red_off_count = 0

            if self.red_seen_count >= seen_need:
                self.red_was_seen = True
                self.start_allowed = False

        elif self.red_was_seen and red_ratio <= off_ratio:
            self.red_off_count += 1

            if self.red_off_count >= off_need:
                self.start_allowed = True

        else:
            self.red_seen_count = max(
                0, self.red_seen_count - 1
            )
            self.red_off_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerception()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
