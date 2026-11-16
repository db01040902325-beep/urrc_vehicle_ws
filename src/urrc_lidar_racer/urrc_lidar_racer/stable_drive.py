"""Stability-first autonomous driving, independent of the legacy race controllers.

Input: front LiDAR, IMU, camera's five-red start gate.
Output: one and only one /model/urrc_car/cmd_vel producer.
No movement on uncertain geometry, stale sensors, or incorrect steering response.
"""
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String


def clamp(value, low, high):
    return max(low, min(high, value))


def median_wall(points, x0, x1, side):
    values = [y for x, y in points if x0 <= x <= x1
              and (y > 0 if side == 'left' else y < 0)
              and 0.22 <= abs(y) <= 2.7]
    if len(values) < 3:
        return None
    # Use the outer half of each side; nearby cars / debris are nearer the center.
    values.sort()
    if side == 'left':
        values = values[len(values) // 2:]
    else:
        values = values[:max(1, (len(values) + 1) // 2)]
    return statistics.median(values)


def corridor(points, x0, x1):
    left = median_wall(points, x0, x1, 'left')
    right = median_wall(points, x0, x1, 'right')
    if left is None or right is None:
        return None
    width = left - right
    if not 1.65 <= width <= 3.10:
        return None
    return (0.5 * (left + right), left, right, width)


def choose_command(points, previous_steer, previous_speed, dt,
                   wheelbase=0.330):
    """Return (speed, steering radians, reason). This routine has no ROS dependency."""
    near = corridor(points, 0.8, 2.4)
    far = corridor(points, 3.0, 5.8)
    if near is None or far is None:
        return 0.0, 0.0, 'PATH_UNVERIFIED'

    # Change of corridor direction rather than aggressive recentering.
    heading = math.atan2(far[0] - near[0], 2.8)
    target = clamp(0.65 * heading, -math.radians(15), math.radians(15))

    # Correct lateral position only when a wall is close; otherwise go straight.
    clearance_left = near[1] - 0.13
    clearance_right = -near[2] - 0.13
    if min(clearance_left, clearance_right) < 0.38:
        target += clamp(-0.30 * near[0], -0.10, 0.10)
    target = clamp(target, -math.radians(15), math.radians(15))

    # Stop before a front wall, not after steering toward it.
    front = [math.hypot(x, y) for x, y in points
             if x > 0 and abs(y) < 0.20]
    if front and min(front) < 1.15:
        return 0.0, 0.0, 'FRONT_BLOCKED'
    if min(clearance_left, clearance_right) < 0.20:
        return 0.0, 0.0, 'SIDE_WALL_TOO_CLOSE'

    # Low-speed baseline until a complete lap has been demonstrated.
    target_speed = 2.2 if abs(target) < math.radians(3) else 1.4
    if front and min(front) < 2.0:
        target_speed = min(target_speed, 0.8)
    if min(clearance_left, clearance_right) < 0.45:
        target_speed = min(target_speed, 0.8)

    # Slew limits apply to acceleration and steering. Emergency stops bypass.
    steer = previous_steer + clamp(target - previous_steer,
                                  -math.radians(25) * dt,
                                  math.radians(25) * dt)
    speed = previous_speed + clamp(target_speed - previous_speed,
                                  -1.5 * dt, 1.5 * dt)
    speed = clamp(speed, 0.0, 2.2)
    if speed * speed * abs(math.tan(steer)) / wheelbase > 3.0:
        speed = min(speed, math.sqrt(3.0 * wheelbase /
                                   max(abs(math.tan(steer)), 1e-6)))
    return speed, steer, 'DRIVING'


class StableDrive(Node):
    def __init__(self):
        super().__init__('stable_drive')
        self.scan = None
        self.scan_time = 0.0
        self.imu_time = 0.0
        self.yaw_rate = 0.0
        self.gate_open = False
        self.speed = 0.0
        self.steer = 0.0
        self.last_tick = time.monotonic()
        self.path_streak = 0
        self.bad_yaw_streak = 0
        self.latched_fault = ''
        self.last_requested_yaw = 0.0
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 20)
        self.create_subscription(Bool, '/camera/start_allowed', self.on_gate, 10)
        self.cmd_pub = self.create_publisher(Twist, '/model/urrc_car/cmd_vel', 10)
        self.mode_pub = self.create_publisher(String, '/race/stop_reason', 10)
        self.speed_pub = self.create_publisher(Float32, '/race/speed_mps', 10)
        self.steer_pub = self.create_publisher(Float32, '/race/steering_deg', 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info('StableDrive active: max 2.2 m/s; start gate required')

    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = time.monotonic()

    def on_imu(self, msg):
        self.imu_time = time.monotonic()
        self.yaw_rate = float(msg.angular_velocity.z)

    def on_gate(self, msg):
        if msg.data:
            self.gate_open = True

    def stop(self, reason):
        self.speed = 0.0
        self.steer = 0.0
        self.last_requested_yaw = 0.0
        self.cmd_pub.publish(Twist())
        self.mode_pub.publish(String(data=reason))
        self.speed_pub.publish(Float32(data=0.0))
        self.steer_pub.publish(Float32(data=0.0))

    def tick(self):
        now = time.monotonic()
        dt = clamp(now - self.last_tick, 0.01, 0.10)
        self.last_tick = now
        if self.latched_fault:
            self.stop(self.latched_fault)
            return
        if not self.gate_open:
            self.stop('WAIT_START')
            return
        if self.scan is None or now - self.scan_time > 0.30:
            self.path_streak = 0
            self.stop('LIDAR_STALE')
            return
        if self.imu_time == 0.0 or now - self.imu_time > 0.30:
            self.path_streak = 0
            self.stop('IMU_STALE')
            return

        pts = []
        scan = self.scan
        for i, distance in enumerate(scan.ranges):
            if not math.isfinite(distance) or not scan.range_min <= distance <= scan.range_max:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            if -0.15 <= x <= 8.0:
                pts.append((x, y))

        speed, steer, reason = choose_command(pts, self.steer, self.speed, dt)
        if reason != 'DRIVING':
            self.path_streak = 0
            self.stop(reason)
            return

        self.path_streak += 1
        if self.path_streak < 5:
            self.stop('PATH_CONFIRMING')
            return

        # Sensor sanity check: steering sign and observed yaw must agree.
        if (self.speed > 0.7 and abs(self.last_requested_yaw) > 0.20
                and abs(self.yaw_rate) > 0.08
                and self.last_requested_yaw * self.yaw_rate < 0):
            self.bad_yaw_streak += 1
        else:
            self.bad_yaw_streak = 0
        if self.bad_yaw_streak >= 5:
            self.latched_fault = 'STEERING_SIGN_MISMATCH'
            self.stop(self.latched_fault)
            return

        self.speed, self.steer = speed, steer
        yaw_command = speed * math.tan(steer) / 0.330
        self.last_requested_yaw = yaw_command
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = yaw_command
        self.cmd_pub.publish(cmd)
        self.mode_pub.publish(String(data='DRIVING'))
        self.speed_pub.publish(Float32(data=speed))
        self.steer_pub.publish(Float32(data=math.degrees(steer)))


def main(args=None):
    rclpy.init(args=args)
    node = StableDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop('SHUTDOWN')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
