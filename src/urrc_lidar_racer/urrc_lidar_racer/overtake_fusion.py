#!/usr/bin/env python3
import math
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


class OvertakeState(Enum):
    CRUISE = auto()
    SHIFT_OUT = auto()
    PASS = auto()
    RETURN = auto()


class OvertakeFusion(Node):
    """LiDAR + camera final command controller with overtaking."""

    def __init__(self):
        super().__init__('overtake_fusion')

        defaults = {
            'raw_cmd_topic': '/race/lidar_cmd_vel',
            'final_cmd_topic': '/model/urrc_car/cmd_vel',
            'scan_topic': '/scan',
            'wheelbase': 0.330,
            'camera_gain_deg': 5.5,
            'camera_min_confidence': 0.35,
            'max_steer_deg': 28.0,
            'raw_cmd_timeout_sec': 0.35,
            'emergency_stop_m': 0.45,
            'overtake_trigger_m': 1.35,
            'overtake_front_clear_m': 1.80,
            'overtake_side_clear_m': 0.82,
            'overtake_abort_side_m': 0.55,
            'overtake_max_base_steer_deg': 8.0,
            'overtake_detect_hold_sec': 0.35,
            'shift_out_sec': 0.70,
            'pass_min_sec': 0.55,
            'pass_max_sec': 2.20,
            'return_sec': 0.85,
            'pass_lane_target': 0.30,
            'fallback_shift_bias_deg': 3.5,
            'fallback_return_bias_deg': 3.0,
            'shift_speed_mps': 2.35,
            'overtake_speed_mps': 2.90,
            'return_speed_mps': 2.35,
            'start_gate_enable': True,
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.raw_cmd = Twist()
        self.raw_cmd_stamp = 0.0
        self.scan = None

        self.camera_offset = 0.0
        self.camera_confidence = 0.0
        self.start_allowed = not bool(
            self.get_parameter('start_gate_enable').value
        )

        self.state = OvertakeState.CRUISE
        self.state_start = self.now_sec()
        self.detect_start = None
        self.pass_side = 1.0
        self.front_clear_since = None

        self.create_subscription(
            Twist,
            str(self.get_parameter('raw_cmd_topic').value),
            self.on_raw_cmd,
            20,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self.on_scan,
            20,
        )
        self.create_subscription(
            Float32,
            '/camera/road_center_offset',
            self.on_camera_offset,
            10,
        )
        self.create_subscription(
            Float32,
            '/camera/road_confidence',
            self.on_camera_confidence,
            10,
        )
        self.create_subscription(
            Bool,
            '/camera/start_allowed',
            self.on_start_allowed,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('final_cmd_topic').value),
            20,
        )
        self.state_pub = self.create_publisher(
            String, '/race/overtake_state', 10
        )

        self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            'LiDAR + camera fusion/overtake controller started'
        )

    def now_sec(self):
        return (
            self.get_clock().now().nanoseconds
            * 1e-9
        )

    def on_raw_cmd(self, msg):
        self.raw_cmd = msg
        self.raw_cmd_stamp = self.now_sec()

    def on_scan(self, msg):
        self.scan = msg

    def on_camera_offset(self, msg):
        self.camera_offset = float(
            np.clip(msg.data, -1.0, 1.0)
        )

    def on_camera_confidence(self, msg):
        self.camera_confidence = float(
            np.clip(msg.data, 0.0, 1.0)
        )

    def on_start_allowed(self, msg):
        self.start_allowed = bool(msg.data)

    def sector_values(self, deg_min, deg_max):
        if self.scan is None or not self.scan.ranges:
            return np.array([], dtype=np.float32)

        amin = self.scan.angle_min
        inc = self.scan.angle_increment
        values = np.asarray(
            self.scan.ranges, dtype=np.float32
        )

        i0 = int(math.floor(
            (math.radians(deg_min) - amin) / inc
        ))
        i1 = int(math.ceil(
            (math.radians(deg_max) - amin) / inc
        ))

        i0 = max(0, min(len(values) - 1, i0))
        i1 = max(0, min(len(values), i1))

        if i1 <= i0:
            return np.array([], dtype=np.float32)

        values = values[i0:i1]
        values = values[np.isfinite(values)]
        values = values[
            (values > max(0.03, self.scan.range_min))
            & (values < self.scan.range_max)
        ]

        return values

    def front_distance(self):
        values = self.sector_values(-9.0, 9.0)

        if values.size == 0:
            return float('inf')

        return float(np.percentile(values, 12))

    def side_clearance(self, left):
        if left:
            values = self.sector_values(14.0, 68.0)
        else:
            values = self.sector_values(-68.0, -14.0)

        if values.size == 0:
            return 0.0

        return float(np.percentile(values, 28))

    def max_steer_rad(self):
        return math.radians(
            float(
                self.get_parameter(
                    'max_steer_deg'
                ).value
            )
        )

    def raw_steer_rad(self, speed):
        if abs(speed) < 0.05:
            return 0.0

        wheelbase = float(
            self.get_parameter('wheelbase').value
        )

        steer = math.atan(
            self.raw_cmd.angular.z
            * wheelbase
            / speed
        )

        return float(np.clip(
            steer,
            -self.max_steer_rad(),
            self.max_steer_rad(),
        ))

    def camera_correction(
        self,
        target_offset=0.0,
        scale=1.0,
    ):
        minimum = float(
            self.get_parameter(
                'camera_min_confidence'
            ).value
        )

        if self.camera_confidence < minimum:
            return None

        gain = math.radians(
            float(
                self.get_parameter(
                    'camera_gain_deg'
                ).value
            )
        )

        error = (
            self.camera_offset
            - target_offset
        )

        return (
            -gain
            * error
            * self.camera_confidence
            * scale
        )

    def set_state(self, state):
        if state == self.state:
            return

        self.state = state
        self.state_start = self.now_sec()
        self.front_clear_since = None

        self.get_logger().info(
            f'Overtake state -> {state.name}'
        )

    def publish_stop(self, state_name):
        self.cmd_pub.publish(Twist())
        self.state_pub.publish(
            String(data=state_name)
        )

    def control_loop(self):
        now = self.now_sec()

        timeout = float(
            self.get_parameter(
                'raw_cmd_timeout_sec'
            ).value
        )

        if (
            self.raw_cmd_stamp <= 0.0
            or now - self.raw_cmd_stamp > timeout
        ):
            self.publish_stop('RAW_CMD_TIMEOUT')
            return

        if (
            bool(
                self.get_parameter(
                    'start_gate_enable'
                ).value
            )
            and not self.start_allowed
        ):
            self.publish_stop('WAIT_START')
            return

        base_speed = max(
            0.0,
            float(self.raw_cmd.linear.x),
        )
        steer = self.raw_steer_rad(base_speed)

        front = self.front_distance()
        left = self.side_clearance(True)
        right = self.side_clearance(False)

        emergency = float(
            self.get_parameter(
                'emergency_stop_m'
            ).value
        )

        if front < emergency:
            self.publish_stop('EMERGENCY_STOP')
            return

        speed = base_speed

        if self.state == OvertakeState.CRUISE:
            correction = self.camera_correction()

            if correction is not None:
                steer += correction

            trigger = float(
                self.get_parameter(
                    'overtake_trigger_m'
                ).value
            )
            side_need = float(
                self.get_parameter(
                    'overtake_side_clear_m'
                ).value
            )
            max_base = math.radians(
                float(
                    self.get_parameter(
                        'overtake_max_base_steer_deg'
                    ).value
                )
            )

            candidate = (
                front < trigger
                and max(left, right) > side_need
                and abs(steer) < max_base
                and base_speed > 0.2
            )

            if candidate:
                if self.detect_start is None:
                    self.detect_start = now

                elif (
                    now - self.detect_start
                    >= float(
                        self.get_parameter(
                            'overtake_detect_hold_sec'
                        ).value
                    )
                ):
                    self.pass_side = (
                        1.0
                        if left >= right
                        else -1.0
                    )
                    self.detect_start = None
                    self.set_state(
                        OvertakeState.SHIFT_OUT
                    )

            else:
                self.detect_start = None

        elif self.state == OvertakeState.SHIFT_OUT:
            elapsed = now - self.state_start

            speed = min(
                float(
                    self.get_parameter(
                        'shift_speed_mps'
                    ).value
                ),
                max(1.0, base_speed),
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter(
                        'pass_lane_target'
                    ).value
                )
            )

            correction = self.camera_correction(
                target, 1.15
            )

            if correction is not None:
                steer += correction
            else:
                steer += (
                    self.pass_side
                    * math.radians(
                        float(
                            self.get_parameter(
                                'fallback_shift_bias_deg'
                            ).value
                        )
                    )
                )

            chosen_clear = (
                left
                if self.pass_side > 0.0
                else right
            )

            if (
                chosen_clear
                < float(
                    self.get_parameter(
                        'overtake_abort_side_m'
                    ).value
                )
            ):
                self.set_state(
                    OvertakeState.RETURN
                )

            elif (
                elapsed
                >= float(
                    self.get_parameter(
                        'shift_out_sec'
                    ).value
                )
            ):
                self.set_state(
                    OvertakeState.PASS
                )

        elif self.state == OvertakeState.PASS:
            elapsed = now - self.state_start

            speed = float(
                self.get_parameter(
                    'overtake_speed_mps'
                ).value
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter(
                        'pass_lane_target'
                    ).value
                )
            )

            correction = self.camera_correction(
                target, 1.0
            )

            if correction is not None:
                steer += correction

            chosen_clear = (
                left
                if self.pass_side > 0.0
                else right
            )

            if (
                chosen_clear
                < float(
                    self.get_parameter(
                        'overtake_abort_side_m'
                    ).value
                )
            ):
                self.set_state(
                    OvertakeState.RETURN
                )

            else:
                front_clear = float(
                    self.get_parameter(
                        'overtake_front_clear_m'
                    ).value
                )

                if front > front_clear:
                    if self.front_clear_since is None:
                        self.front_clear_since = now
                else:
                    self.front_clear_since = None

                stable_clear = (
                    self.front_clear_since is not None
                    and (
                        now
                        - self.front_clear_since
                    ) > 0.35
                )

                min_time = float(
                    self.get_parameter(
                        'pass_min_sec'
                    ).value
                )
                max_time = float(
                    self.get_parameter(
                        'pass_max_sec'
                    ).value
                )

                if (
                    (
                        elapsed >= min_time
                        and stable_clear
                    )
                    or elapsed >= max_time
                ):
                    self.set_state(
                        OvertakeState.RETURN
                    )

        elif self.state == OvertakeState.RETURN:
            elapsed = now - self.state_start
            duration = max(
                0.1,
                float(
                    self.get_parameter(
                        'return_sec'
                    ).value
                ),
            )
            progress = float(np.clip(
                elapsed / duration,
                0.0,
                1.0,
            ))

            speed = min(
                float(
                    self.get_parameter(
                        'return_speed_mps'
                    ).value
                ),
                max(1.0, base_speed),
            )

            target = (
                self.pass_side
                * float(
                    self.get_parameter(
                        'pass_lane_target'
                    ).value
                )
                * (1.0 - progress)
            )

            correction = self.camera_correction(
                target, 1.15
            )

            if correction is not None:
                steer += correction
            else:
                steer -= (
                    self.pass_side
                    * math.radians(
                        float(
                            self.get_parameter(
                                'fallback_return_bias_deg'
                            ).value
                        )
                    )
                )

            if elapsed >= duration:
                self.set_state(
                    OvertakeState.CRUISE
                )

        steer = float(np.clip(
            steer,
            -self.max_steer_rad(),
            self.max_steer_rad(),
        ))
        speed = float(np.clip(
            speed, 0.0, 3.0
        ))

        command = Twist()
        command.linear.x = speed

        if speed > 0.05:
            wheelbase = float(
                self.get_parameter(
                    'wheelbase'
                ).value
            )
            command.angular.z = (
                speed
                * math.tan(steer)
                / wheelbase
            )

        self.cmd_pub.publish(command)
        self.state_pub.publish(
            String(data=self.state.name)
        )


def main(args=None):
    rclpy.init(args=args)
    node = OvertakeFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
