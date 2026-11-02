#!/usr/bin/env python3
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32


class CameraPerception(Node):
    """Front camera perception for road-center correction and five-light start."""

    def __init__(self):
        super().__init__('camera_perception')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('road_confidence_min_area_ratio', 0.08)
        self.declare_parameter('start_gate_enable', True)
        self.declare_parameter('start_full_on_min_count', 4)
        self.declare_parameter('start_off_max_count', 1)
        self.declare_parameter('start_seen_frames', 3)
        self.declare_parameter('start_off_frames', 4)
        self.declare_parameter('lamp_roi_y_max_ratio', 0.48)
        self.declare_parameter('lamp_roi_x_min_ratio', 0.15)
        self.declare_parameter('lamp_roi_x_max_ratio', 0.85)

        self.bridge = CvBridge()
        self.full_on_frames = 0
        self.off_frames = 0
        self.full_on_seen = False
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
        self.red_count_pub = self.create_publisher(
            Int32, '/camera/red_light_count', 10
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

        lamp_count, red_ratio = self.detect_start_lamps(frame)
        self.red_pub.publish(Float32(data=float(red_ratio)))
        self.red_count_pub.publish(Int32(data=int(lamp_count)))

        self.update_start_gate(lamp_count)
        self.start_pub.publish(Bool(data=bool(self.start_allowed)))

    def estimate_road_center(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
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
        cx = float(np.average(xs.astype(np.float32), weights=weights))

        offset = (cx - center_x) / max(1.0, center_x)
        offset = float(np.clip(offset, -1.0, 1.0))

        area_ratio = area / float(w * h)
        min_ratio = float(
            self.get_parameter('road_confidence_min_area_ratio').value
        )
        confidence = float(np.clip(
            (area_ratio - min_ratio)
            / max(0.01, 0.35 - min_ratio),
            0.0,
            1.0,
        ))

        return offset, confidence

    def detect_start_lamps(self, frame):
        """Count compact red lamp blobs in the upper gantry region.

        Large red surfaces are intentionally rejected. The gate only arms
        after the row of five lamps has actually been observed.
        """
        h, w = frame.shape[:2]

        x0 = int(
            float(self.get_parameter('lamp_roi_x_min_ratio').value) * w
        )
        x1 = int(
            float(self.get_parameter('lamp_roi_x_max_ratio').value) * w
        )
        y1 = int(
            float(self.get_parameter('lamp_roi_y_max_ratio').value) * h
        )

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(1, min(h, y1))

        roi = frame[0:y1, x0:x1]
        if roi.size == 0:
            return 0, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_1 = cv2.inRange(
            hsv,
            np.array([0, 135, 145], np.uint8),
            np.array([11, 255, 255], np.uint8),
        )
        red_2 = cv2.inRange(
            hsv,
            np.array([169, 135, 145], np.uint8),
            np.array([179, 255, 255], np.uint8),
        )
        red = cv2.bitwise_or(red_1, red_2)
        red = cv2.morphologyEx(
            red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )

        roi_area = float(red.size)
        raw_ratio = float(np.count_nonzero(red)) / max(1.0, roi_area)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(red, 8)
        blobs = []

        min_area = max(5.0, roi_area * 0.00004)
        max_area = roi_area * 0.006

        for idx in range(1, count):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            bw = float(stats[idx, cv2.CC_STAT_WIDTH])
            bh = float(stats[idx, cv2.CC_STAT_HEIGHT])

            if area < min_area or area > max_area:
                continue
            if bw < 2.0 or bh < 2.0:
                continue

            aspect = bw / bh
            fill = area / max(1.0, bw * bh)

            if not 0.45 <= aspect <= 2.20:
                continue
            if fill < 0.35:
                continue

            cx, cy = centroids[idx]
            blobs.append((float(cx), float(cy), area))

        if not blobs:
            return 0, raw_ratio

        # Five start lights are approximately on one horizontal row.
        row_tol = max(8.0, 0.08 * roi.shape[0])
        best_row = []

        for seed in blobs:
            row = [b for b in blobs if abs(b[1] - seed[1]) <= row_tol]
            if len(row) > len(best_row):
                best_row = row

        best_row.sort(key=lambda b: b[0])

        if len(best_row) >= 2:
            span = best_row[-1][0] - best_row[0][0]
            if span < 0.10 * roi.shape[1]:
                return 0, raw_ratio

        return min(5, len(best_row)), raw_ratio

    def update_start_gate(self, lamp_count):
        if not bool(self.get_parameter('start_gate_enable').value):
            self.start_allowed = True
            return

        # Once released, do not re-arm during the race.
        if self.start_allowed:
            return

        full_on_min = int(
            self.get_parameter('start_full_on_min_count').value
        )
        off_max = int(
            self.get_parameter('start_off_max_count').value
        )
        seen_need = int(
            self.get_parameter('start_seen_frames').value
        )
        off_need = int(
            self.get_parameter('start_off_frames').value
        )

        if not self.full_on_seen:
            if lamp_count >= full_on_min:
                self.full_on_frames += 1
            else:
                self.full_on_frames = max(0, self.full_on_frames - 1)

            if self.full_on_frames >= seen_need:
                self.full_on_seen = True
                self.off_frames = 0
                self.get_logger().info(
                    'Five-light start sequence armed'
                )
            return

        if lamp_count <= off_max:
            self.off_frames += 1
        else:
            self.off_frames = 0

        if self.off_frames >= off_need:
            self.start_allowed = True
            self.get_logger().info(
                'Start lights OFF -> GO'
            )


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
