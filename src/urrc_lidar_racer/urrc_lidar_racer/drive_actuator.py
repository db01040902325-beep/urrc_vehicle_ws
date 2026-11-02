#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64


class DriveActuator(Node):
    """Split final Twist into wheel angular speeds and Ackermann steer angle."""

    def __init__(self):
        super().__init__('drive_actuator')

        self.declare_parameter(
            'cmd_topic',
            '/model/urrc_car/cmd_vel',
        )
        self.declare_parameter('wheelbase_m', 0.330)
        self.declare_parameter('track_width_m', 0.190)
        self.declare_parameter('wheel_radius_m', 0.045)
        self.declare_parameter(
            'steer_limit_rad',
            0.610865,
        )
        self.declare_parameter(
            'max_wheel_speed_rad_s',
            340.0,
        )

        self.left_pub = self.create_publisher(
            Float64,
            '/model/urrc_car/left_wheel_velocity',
            20,
        )
        self.right_pub = self.create_publisher(
            Float64,
            '/model/urrc_car/right_wheel_velocity',
            20,
        )
        self.steer_pub = self.create_publisher(
            Float64,
            '/model/urrc_car/steer_angle',
            20,
        )

        self.create_subscription(
            Twist,
            str(self.get_parameter('cmd_topic').value),
            self.on_command,
            20,
        )

        self.get_logger().info(
            'Direct wheel actuator enabled: '
            'Twist -> wheel velocity + steer angle'
        )

    def on_command(self, msg):
        speed = float(msg.linear.x)
        yaw_rate = float(msg.angular.z)

        wheelbase = float(
            self.get_parameter('wheelbase_m').value
        )
        track = float(
            self.get_parameter('track_width_m').value
        )
        radius = float(
            self.get_parameter('wheel_radius_m').value
        )
        steer_limit = float(
            self.get_parameter('steer_limit_rad').value
        )
        max_wheel = float(
            self.get_parameter(
                'max_wheel_speed_rad_s'
            ).value
        )

        if abs(speed) > 0.05:
            steer = math.atan(
                wheelbase * yaw_rate / speed
            )
        else:
            steer = 0.0

        steer = max(
            -steer_limit,
            min(steer_limit, steer),
        )

        # Approximate left/right path-speed difference from yaw rate.
        left_linear = speed - yaw_rate * track * 0.5
        right_linear = speed + yaw_rate * track * 0.5

        left_wheel = left_linear / max(radius, 1e-4)
        right_wheel = right_linear / max(radius, 1e-4)

        left_wheel = max(
            -max_wheel,
            min(max_wheel, left_wheel),
        )
        right_wheel = max(
            -max_wheel,
            min(max_wheel, right_wheel),
        )

        self.left_pub.publish(
            Float64(data=float(left_wheel))
        )
        self.right_pub.publish(
            Float64(data=float(right_wheel))
        )
        self.steer_pub.publish(
            Float64(data=float(steer))
        )


def main(args=None):
    rclpy.init(args=args)
    node = DriveActuator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
